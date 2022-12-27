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
            mask[i, :i + 1] = torch.flip(torch.exp(-torch.log(3 + y) ** 2), [0])
    
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
        # l
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