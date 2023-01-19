# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import repeat
from omegaconf import II
from torch import Tensor

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
                             MhaRpeDecoderLayer, MhaRpeEncoderLayer)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.helpers import logging_info
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


# from alibi
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

def toeplitz(c, r):
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)

def get_mask(n, type=-1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    
    if type == 1:
        # 0, -1, -2, ..., -n
        for i in range(n):
            mask[i, :i + 1] = -torch.arange(i + 1)
    elif type == 2:
        # -ln(1), -ln(2), ...
        for i in range(n):
            mask[i, :i + 1] = -torch.log(1 + torch.arange(i + 1) ** 2)
    elif type == 3:
        # -n, ..., -2, -1, 0
        for i in range(n):
            mask[i, :i + 1] = -torch.flip(torch.arange(i + 1), [0])
    elif type == 4:
        # -ln(n), ..., -ln(2), -ln(1), 0
        for i in range(n):
            mask[i, :i + 1] = -torch.log(1 + torch.flip(torch.arange(i + 1) ** 2, [0]))
    elif type == 5:
        # -n^2, ..., -2^2, -1^2, 0
        for i in range(n):
            mask[i, :i + 1] = -torch.flip(torch.arange(i + 1) ** 2, [0])
    elif type == 6:
        for i in range(n):
            mask[i, :i + 1] = -torch.flip(torch.arange(i + 1) ** 0.5, [0])
        
    return mask

def get_mask_k(n, type=-1, k=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    if type == 1:
        # 0, -1, -2, ..., -n
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -y
    elif type == 2:
        # -ln(1^2), -ln(2^2), ...
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.log(1 + y ** 2)
    elif type == 3:
        # -n, ..., -2, -1, 0
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.flip(y, [0])
    elif type == 4:
        # -ln(n^2), ..., -ln(2^2), -ln(1^2), 0
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.log(1 + torch.flip(y ** 2, [0]))
    elif type == 5:
        # -n^2, ..., -2^2, -1^2, 0
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.flip(y ** 2, [0])
    elif type == 6:
        # -n^0.5, ..., -2^0.5, -1^0.5, 0
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.flip(y ** 0.5, [0])
    elif type == 7:
        # -ln(n), ..., -ln(2), -ln(1), 0
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.log(1 + torch.flip(y, [0]))
    elif type == 8:
        # -n^1.25, ..., -2^1.25, -1^1.25, 0
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.flip(y ** 1.25, [0])
    elif type == 9:
        # -0.5 * n^0.75, ..., -0.5 * 2^0.75, -0.5 * 1^0.75, 0
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.flip(0.5 * y ** 0.75, [0])
    elif type == 10:
        # lnx/x^2
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = torch.flip(torch.log(1 + torch.log(1 + y)) - (torch.log((y + 1) ** 2)), [0])
    elif type == 11:
        # exp(-ln^2x)
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = torch.flip(-torch.log(3 + y) ** 2, [0])
    elif type == 12:
        # exp(-x^4)
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            mask[i, :i + 1] = -torch.flip(y ** 4, [0])
    elif type == 13:
        # 1/n(lnn) = exp(-(lnx + lnlnx))
        for i in range(n):
            x = torch.arange(i + 1)
            y = k * x
            z = torch.log(3 + y)
            mask[i, :i + 1] = -torch.flip(z + torch.log(z), [0])
            
    return mask

class TransformerRpeDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.rpe_type = getattr(args, 'rpe_type', -1)
        if self.rpe_type != -1:
            maxpos = args.tokens_per_sample
            attn_heads = args.decoder_attention_heads
            # h, 1, 1
            self.slopes = torch.Tensor(get_slopes(attn_heads)).reshape(attn_heads, 1, 1)
            self.buffered_future_mask = self.buffered_future_mask_rpe

        # kerple log
        self.kerple_log = getattr(args, 'kerple_log', -1)
        if self.kerple_log != -1:
            # Allocate weights and initialize.
            # The kernel has the form -p*log(1+a*|m-n|)
            # (h, 1, 1)
            attn_heads = args.decoder_attention_heads
            def get_parameter(scale, init_method):
                if init_method == 'ones':
                    return nn.Parameter(torch.ones(
                                attn_heads,
                                device=torch.cuda.current_device(),
                                )[:,None,None]*scale )
                elif init_method == 'uniform':
                    return nn.Parameter(torch.rand(
                                attn_heads,
                                device=torch.cuda.current_device(),
                                )[:,None,None]*scale )

            self.bias_p = get_parameter(2, 'uniform')
            self.bias_a = get_parameter(1, 'uniform')
            self.cache_matrix = nn.Parameter(self.get_kerple_cache_matrix(), requires_grad=False)
            self.causal_mask = nn.Parameter(self.get_causal_mask(), requires_grad=False)
            self.eps = 1e-2
            self.buffered_future_mask = self.buffered_future_mask_kerple_log
            logging_info(f'bias_p {self.bias_p}')
            logging_info(f'bias_a {self.bias_a}')
            
        # kerple power
        self.kerple_power = getattr(args, 'kerple_power', -1)
        if self.kerple_power != -1:
            # bias_kernel = -bias_a*|m-n|^bias_p
            # weight_kernel = exp(-wei_a*|m-n|^wei_p)
            # !!! we only use bias here !!!
            attn_heads = args.decoder_attention_heads
            def get_parameter(scale, init_method):
                if init_method == 'ones':
                    return nn.Parameter(torch.ones(
                                attn_heads,
                                device=torch.cuda.current_device(),
                                )[:,None,None]*scale )
                elif init_method == 'uniform':
                    return nn.Parameter(torch.rand(
                                attn_heads,
                                device=torch.cuda.current_device(),
                                )[:,None,None]*scale )
            
            self.bias_p = get_parameter(2, 'uniform')
            self.bias_a = get_parameter(1, 'uniform')
            self.cache_matrix = nn.Parameter(self.get_kerple_cache_matrix(), requires_grad=False)
            self.causal_mask = nn.Parameter(self.get_causal_mask(), requires_grad=False)
            self.eps = 1e-2
            self.buffered_future_mask = self.buffered_future_mask_kerple_power
            logging_info(f'bias_p {self.bias_p}')
            logging_info(f'bias_a {self.bias_a}')
            
        # sandwich
        self.sandwich = getattr(args, 'sandwich', -1)
        if self.sandwich != -1:
            maxpos = args.tokens_per_sample
            attn_heads = args.decoder_attention_heads
            # h, 1, 1
            slopes = torch.arange(1, attn_heads + 1) / attn_heads * 8
            self.slopes = nn.Parameter(1 / slopes.reshape(attn_heads, 1, 1), requires_grad=False)
            # self.slopes = nn.Parameter(slopes.reshape(attn_heads, 1, 1), requires_grad=False)
            dim = args.decoder_embed_dim
            # for test
            # dim = dim // attn_heads
            # for test
            dim = dim // attn_heads * 2
            # compute 10000 ^ (2* i / d)
            half_dim = dim // 2
            emb = math.log(10000) / half_dim
            # d, 1, 1
            emb = torch.exp(torch.arange(1, half_dim + 1, dtype=torch.float) * -emb).reshape(half_dim, 1, -1)
            self.emb = nn.Parameter(emb, requires_grad=False)
            self.half_dim = half_dim
            self.buffered_future_mask = self.buffered_future_mask_sandwich

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = MhaRpeDecoderLayer(args, no_encoder_attn)
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

    # v2
    # def buffered_future_mask_rpe(self, tensor):
    #     # l
    #     dim = tensor.size(1)
    #     # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
    #     if (
    #         self._future_mask.size(0) == 0
    #         or (not self._future_mask.device == tensor.device)
    #         or self._future_mask.size(1) < self.args.tokens_per_sample
    #     ):
    #         self._future_mask = get_mask(self.args.tokens_per_sample, self.rpe_type)
    #         # slopes: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    #         # 1, n, n; h, 1, 1 -> h, n, n
    #         if self.rpe_type in [2, 4]:
    #             # kx = exp(ln(k) + ln(x))
    #             self._future_mask = (self._future_mask + self.slopes).to(tensor)
    #         else:
    #             self._future_mask = (self._future_mask * self.slopes).to(tensor)

    #     return self._future_mask[:, :dim, :dim]
            
    # v3
    def buffered_future_mask_rpe(self, tensor):
        # n
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(1) < self.args.tokens_per_sample
        ):
            # slopes: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
            arr = []
            for k in self.slopes:
                arr.append(get_mask_k(self.args.tokens_per_sample, self.rpe_type, k.item()))
            self._future_mask = torch.stack(arr, dim=0).to(tensor)
            # mask1 = get_mask(self.args.tokens_per_sample, self.rpe_type) * self.slopes
            # mask1 = mask1.to(tensor)
            # print(self._future_mask[0])
            # print(mask1[0])
            # print(torch.norm(mask1 - self._future_mask))

        return self._future_mask[:, :dim, :dim]
    
    def get_kerple_cache_matrix(self):
        n = self.args.tokens_per_sample
        # 1, n, n
        diff = torch.tril(
            torch.arange(n).view(n, 1).repeat(1, n)
            + torch.arange(0, -n, -1)
        ).unsqueeze(0)
        
        return diff
    
    def get_causal_mask(self):
        n = self.args.tokens_per_sample
        # 1, n, n
        mask = torch.triu(
            utils.fill_with_neg_inf(torch.zeros([n, n])), 1
        ).unsqueeze(0)

        return mask
    
    # kerple log
    def buffered_future_mask_kerple_log(self, tensor):
        # n
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self.cache_matrix.size(0) == 0
            or (not self.cache_matrix.device == tensor.device)
            or self.cache_matrix.size(-1) < self.args.tokens_per_sample
        ):
            self.cache_matrix = nn.Parameter(self.get_kerple_cache_matrix(), requires_grad=False)
            self.causal_mask = nn.Parameter(self.get_causal_mask(), requires_grad=False)
        
        diff = self.cache_matrix
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p*torch.log(1+self.bias_a*diff) # log kernel
        
        # causal mask
        bias += self.causal_mask

        return bias[:, :dim, :dim]
    
    # kerple power
    def buffered_future_mask_kerple_power(self, tensor):
        # n
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self.cache_matrix.size(0) == 0
            or (not self.cache_matrix.device == tensor.device)
            or self.cache_matrix.size(-1) < self.args.tokens_per_sample
        ):
            self.cache_matrix = nn.Parameter(self.get_kerple_cache_matrix(), requires_grad=False)
            self.causal_mask = nn.Parameter(self.get_causal_mask(), requires_grad=False)
        # 1, n, n
        diff = self.cache_matrix
        
        if self.bias_p is not None:
            self.bias_p.data = self.bias_p.data.clamp(min=self.eps, max=2)
            bias = diff.pow(self.bias_p)
        else:
            bias = diff
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            bias = -bias*self.bias_a
        else:
            bias = -bias
            
        # causal mask
        bias += self.causal_mask
            
        return bias[:, :dim, :dim]

    # v3
    def buffered_future_mask_sandwich(self, tensor):
        # n
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(1) < self.args.tokens_per_sample
        ):
            n = self.args.tokens_per_sample
            # 1, n, n
            diff = self.get_kerple_cache_matrix().to(tensor)
            # sum cos
            # # 1, n, n; d, 1, 1 -> d, n, n
            # diff_cos = torch.cos(diff * self.emb)
            # # d, n, n -> 1, n, n
            # cos = torch.sum(diff_cos, dim=0, keepdim=True) - self.half_dim
            cos = torch.zeros_like(diff)
            d = self.emb.shape[0]
            for i in range(d):
                cos += torch.cos(diff * self.emb[i])
            eps = 1e-4
            cos -= (self.half_dim + eps)
                
            # 1, n, n; h, 1, 1 -> h, n, n
            bias = self.slopes * cos
            self._future_mask = bias + self.get_causal_mask().to(tensor)
            # for i in range(8):
            #     print(torch.exp(self._future_mask)[i, -1])

        return self._future_mask[:, :dim, :dim]

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

        if self.use_alibi or self.use_toep or self.rpe_type > 0 or self.kerple_log > 0 or self.kerple_power > 0 or self.sandwich > 0:
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

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
            if not(self.use_alibi or self.use_toep or self.rpe_type > 0 or self.kerple_log > 0 or self.kerple_power > 0 or self.sandwich > 0):
                if incremental_state is None and not full_context_alignment:
                    self_attn_mask = self.buffered_future_mask(x)
                else:
                    self_attn_mask = None

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
        assert False

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
