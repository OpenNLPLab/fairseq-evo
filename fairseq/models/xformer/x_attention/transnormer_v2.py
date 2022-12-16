# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (FairseqEncoder, FairseqEncoderDecoderModel,
                            FairseqIncrementalDecoder, register_model,
                            register_model_architecture)
from fairseq.models.transformer import (DEFAULT_MAX_SOURCE_POSITIONS,
                                        DEFAULT_MAX_TARGET_POSITIONS,
                                        DEFAULT_MIN_PARAMS_TO_WRAP,
                                        TransformerDecoder, TransformerEncoder,
                                        TransformerModel, base_architecture)
from fairseq.modules import (AdaptiveInput, CharacterTokenEmbedder,
                             TransnormerV2DecoderLayer,
                             TransnormerV2EncoderLayer)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.helpers import get_norm_fn, logging_info
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from omegaconf import II
from torch import Tensor


class TransnormerV2Encoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.chunk_size = getattr(args, "chunk_size", -1)
        self.attention_types = getattr(args, "encoder_attention_types", [])
        self.local_layer = 0
        for attention_type in self.attention_types:
            self.local_layer += (attention_type == 1)
        self.attn_heads = args.encoder_attention_heads
        norm_type = getattr(args, 'norm_type', 'layernorm')
        embed_dim = args.encoder_embed_dim
        self.layer_norm = get_norm_fn(norm_type)(embed_dim)
        logging_info(f"chunk_size {self.chunk_size}")
        logging_info(f"local_layer {self.local_layer}")
        logging_info(f"attn_heads {self.attn_heads}")

    def build_encoder_layer(self, args):
        layer = TransnormerV2EncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def transform(self, x):
        x = rearrange(x, 'b (l c) e -> b l c e', c=self.chunk_size)

        return x
    
    def reverse_transform(self, x):
        x = rearrange(x, 'b l c e -> b (l c) e', c=self.chunk_size)

        return x

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # x: B x T x C
        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)
            
        # norm attention
        n = x.shape[1]
        len_pad = (self.chunk_size - n % self.chunk_size) % self.chunk_size
        x = F.pad(x, (0, 0, 0, len_pad, 0, 0))
        # b n d -> b l c d
        x = self.transform(x)
    
        # norm layers
        for layer in self.layers[:self.local_layer]:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
        # b l c d -> b n d
        x = self.reverse_transform(x)
        x = x[:, :n]
        
        # linear layers
        for layer in self.layers[self.local_layer:]:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        # # T x B x C -> B x T x C
        # x = x.transpose(0, 1)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

class TransnormerV2Decoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.chunk_size = getattr(args, "chunk_size", -1)
        self.attention_types = getattr(args, "decoder_attention_types", [])
        self.local_layer = 0
        for attention_type in self.attention_types:
            self.local_layer += (attention_type == 1)
        self.attn_heads = args.decoder_attention_heads
        norm_type = getattr(args, 'norm_type', 'layernorm')
        embed_dim = args.decoder_embed_dim
        self.layer_norm = get_norm_fn(norm_type)(embed_dim)
        logging_info(f"chunk_size {self.chunk_size}")
        logging_info(f"local_layer {self.local_layer}")
        logging_info(f"attn_heads {self.attn_heads}")

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransnormerV2DecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def transform(self, x):
        x = rearrange(x, 'b (l c) e -> b l c e', c=self.chunk_size)

        return x
    
    def reverse_transform(self, x):
        x = rearrange(x, 'b l c e -> b (l c) e', c=self.chunk_size)

        return x

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        #logging_info('transformer decoder input:', prev_output_tokens.shape)
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None:
            enc = encoder_out["encoder_out"][0]
            padding_mask = encoder_out["encoder_padding_mask"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # if self.use_alibi or self.use_toep:
        #     if incremental_state is None and not full_context_alignment:
        #         self_attn_mask = self.buffered_future_mask(x)
        #     else:
        #         self_attn_mask = None
        # x: B x T x C
        # T x B x C -> B x T x C
        # 待处理, 可能不需要
        if enc != None:
            enc = enc.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        
        # norm attention
        n = x.shape[1]
        if self.chunk_size < n ** 2 - n:
            len_pad_x = (self.chunk_size - n % self.chunk_size) % self.chunk_size
            x = F.pad(x, (0, 0, 0, len_pad_x, 0, 0))
            if enc != None:
                m = enc.shape[1]
                len_pad_enc = (self.chunk_size - m % self.chunk_size) % self.chunk_size
                enc = F.pad(enc, (0, 0, 0, len_pad_enc, 0, 0))
            # b n d -> b l c d
            x = self.transform(x)
            if enc != None:
                enc = self.transform(enc)

        # local layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        # attn_mask
        # self_attn_mask = (torch.tril(torch.ones(self.chunk_size, self.chunk_size))).to(x)
        self_attn_mask = self.buffered_mask(self.attn_heads, x.shape[-2]).to(x)
        for idx, layer in enumerate(self.layers):
            if idx >= self.local_layer:
                break

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
        if self.chunk_size < n ** 2 - n:
            # b l c d -> b n d
            x = self.reverse_transform(x)
            x = x[:, :n]
            if enc != None:
                enc = self.reverse_transform(enc)
                enc = enc[:, :m]
        
        # linear layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        # attn_mask
        # self_attn_mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2])).to(x)
        self_attn_mask = torch.exp(self.buffered_mask(self.attn_heads, x.shape[-2]).to(x))
        for idx, layer in enumerate(self.layers):
            if idx < self.local_layer:
                continue
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
                
        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # # T x B x C -> B x T x C
        # x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def buffered_mask(self, h, n):
        # copy from alibi
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        slopes = torch.Tensor(get_slopes(h))
        weight = -slopes.unsqueeze(1) * torch.arange(n).unsqueeze(0).expand(h, -1)
        
        # build Toeplitz matrix
        c = weight
        r = torch.Tensor([0] + [float("-inf")] * (n - 1)).expand(h, -1)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-1)
        shape = h, n, n
        i, j = torch.ones(n, n).nonzero().T
        T = vals[:, j - i].reshape(h, n, -1)

        return T
