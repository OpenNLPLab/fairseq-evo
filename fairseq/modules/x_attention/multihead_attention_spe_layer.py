# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.modules import (LayerNorm, TransformerDecoderLayer,
                             TransformerEncoderLayer)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from .multihead_attention_spe import MultiheadAttentionSpe

def get_data_save_dir(args):
    rpe_type = getattr(args, 'rpe_type', -1)
    kerple_log = getattr(args, 'kerple_log', -1)
    kerple_power = getattr(args, 'kerple_power', -1)
    sandwich = getattr(args, 'sandwich', -1)
    data_save_dir = "rpe"
    if rpe_type != -1:
        # for save
        data_save_dir = rpe_type
    elif kerple_log != -1:
        # for save
        data_save_dir = "kerple_log"
    elif kerple_power != -1:
        # for save
        data_save_dir = "kerple_power"
    elif sandwich != -1:
        # for save
        data_save_dir = "sandwich"
    
    return data_save_dir
    

class MhaSpeEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttentionSpe(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            index=args.index,
            data_save_dir=get_data_save_dir(args),
        )

class MhaSpeDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttentionSpe(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            index=args.index,
            data_save_dir=get_data_save_dir(args),
            max_seq=getattr(args, "max_seq", 512)
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttentionSpe(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            index=args.index,
            data_save_dir=get_data_save_dir(args),
            max_seq=getattr(args, "max_seq", 512)
        )
