# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadRfaAttention, MultiheadRfaCausalAttention, TransformerEncoderLayer, TransformerDecoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

# class TransformerRfaEncoderLayer(TransformerEncoderLayer):
#     def __init__(self, args):
#         super().__init__(args)

#     def build_self_attention(self, embed_dim, args):
#         return MultiheadRfaAttention(
#             embed_dim,
#             args.encoder_attention_heads,
#             dropout=args.attention_dropout,
#             self_attention=True,
#             q_noise=self.quant_noise,
#             qn_block_size=self.quant_noise_block_size,
#         )

# class TransformerRfaDecoderLayer(TransformerDecoderLayer):
#     def __init__(
#         elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
#     ):
#         super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

#     def build_self_attention(
#         self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
#     ):
#         return MultiheadRfaAttention(
#             embed_dim,
#             args.decoder_attention_heads,
#             dropout=args.attention_dropout,
#             add_bias_kv=add_bias_kv,
#             add_zero_attn=add_zero_attn,
#             self_attention=not getattr(args, "cross_self_attention", False),
#             q_noise=self.quant_noise,
#             qn_block_size=self.quant_noise_block_size,
#         )

class TransformerRfaEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return MultiheadRfaCausalAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            proj_dim=args.proj_dim,
            tau=args.tau,
            reparam_proj=args.reparam_proj,
            cuda_causal_rfa=args.cuda_causal_rfa
        )

class TransformerRfaDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadRfaCausalAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            proj_dim=args.proj_dim,
            tau=args.tau,
            reparam_proj=args.reparam_proj,
            cuda_causal_rfa=args.cuda_causal_rfa
        )