# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
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
                             CtnnDecoderLayer, CtnnEncoderLayer)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.helpers import logging_info
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from omegaconf import II
from torch import Tensor

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

class CtnnEncoder(TransformerEncoder):
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

    def build_encoder_layer(self, args):
        layer = CtnnEncoderLayer(args)
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

class CtnnDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        
        # max_len
        self.max_len = getattr(args, 'max_len', 512)
        # embed type
        self.embed_type = getattr(args, 'embed_type', -1)
        # causal
        self.causal = getattr(args, 'causal', True)
        # gamma
        self.gamma = getattr(args, 'gamma', 1)
        # max_seq_len
        self.max_seq = 0
        # cos
        k = getattr(args, 'k', 128)
        h = args.decoder_attention_heads
        d = args.decoder_embed_dim * args.expand_ratio // h
        self.h = h
        self.vander = torch.empty(0)
        # index
        self.index = torch.empty(0)
        # decay
        self.pos = torch.empty(0)
        self.neg = torch.empty(0)
        self.zero = torch.empty(0)
        # rpe input
        self.rpe_pos = torch.empty(0)
        self.rpe_neg = torch.empty(0)
        self.rpe_zero = torch.empty(0)
        # slope
        # (h, 1, 1)
        slope = torch.Tensor(get_slopes(h)).reshape(h, -1, 1)
        self.slope = nn.Parameter(torch.exp(-slope), requires_grad=False)

        logging_info(f"slope: {self.slope}")
        logging_info(f"causal: {self.causal}")
        logging_info(f"gamma: {self.gamma}")
        logging_info(f"k: {k}")
        logging_info(f"max_len: {self.max_len}")

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = CtnnDecoderLayer(args, no_encoder_attn)
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

    def build_encoder_layer(self, args):
        layer = CtnnEncoderLayer(args)
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

        self.update_cache(x)
        index = self.index
        vander = self.vander
        if not self.causal:
            # 1, n, 1
            decay = torch.cat([self.zero, self.pos, self.zero, self.neg], dim=1)
            # n, 1
            rpe_input = torch.cat([self.rpe_zero, self.rpe_pos, self.rpe_zero, self.rpe_neg], dim=0)
        else:
            # 1, n, 1
            decay = torch.cat([self.zero, self.pos], dim=1)
            # n, 1
            rpe_input = torch.cat([self.rpe_zero, self.rpe_pos], dim=0)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            # if not self.use_alibi and not self.use_toep and (not (self.rpe_type > 0)):
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=None,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                vander=vander,
                index=index,
                decay=decay,
                rpe_input=rpe_input,
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

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def update_cache(self, x):
        n = x.size(1)
        if self.max_seq < n:
            self.max_seq = n
            # index
            self.index = torch.tensor(range(self.max_seq)).to(x.device)
            # decay
            # 1, n - 1, 1
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            gamma = self.slope ** coef
            print(gamma.shape)
            self.zero = torch.ones(self.h, 1, 1).to(x)
            self.pos = gamma
            if self.causal:
                self.neg = torch.flip(gamma, dims=[1])
            # rpe input
            self.rpe_zero = torch.zeros(1, 1).to(x)
            self.rpe_pos = torch.arange(1, n).reshape(-1, 1).to(x)
            if self.causal:
                self.rpe_neg = torch.flip(self.rpe_pos, dims=[1])
