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
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from omegaconf import II
from torch import Tensor
import math
from einops import repeat

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
        # -n, ..., -2, -1, 0
        for i in range(n):
            mask[i, :i + 1] = -torch.log(1 + torch.flip(torch.arange(i + 1) ** 2, [0]))
        
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
            batch_size = args.max_tokens // maxpos
            # adapt to fairseq attention
            self.slopes = repeat(self.slopes, 'h 1 1 -> (b h) 1 1', b=batch_size)
            self.buffered_future_mask = self.buffered_future_mask_rpe
            
    def buffered_future_mask_rpe(self, tensor):
        # l
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(1) < self.args.tokens_per_sample
        ):
            self._future_mask = get_mask(self.args.tokens_per_sample, self.rpe_type)
        
        # 1, n, n; b, 1, 1 -> b, n, n
        self._future_mask = self._future_mask * self.slopes
        # b * h, l, l
        return self._future_mask[:tensor.shape[0]*self.args.decoder_attention_heads, :dim, :dim].to(tensor)
    
            
    