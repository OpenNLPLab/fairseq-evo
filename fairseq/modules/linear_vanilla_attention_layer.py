# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import MultiheadAttention, LayerNorm, TransformerEncoderLayer, TransformerDecoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
# merge attention
from fairseq.modules import MultiheadCosformerAttention, MultiheadPerformerAttention, LinearKernelAttention

class LinearVanillaEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        if args.attention_type == 1:
            print("===========")
            print("cos")
            print("===========")
            return MultiheadCosformerAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                has_out=getattr(args, "has_out", False),
                use_relu=getattr(args, "use_relu", False),
                use_elu=getattr(args, "use_elu", False),
                use_leak=getattr(args, "use_leak", False),
                index=args.index,
                max_l=getattr(args, "max_l", 1024),
                causal=getattr(args, "causal", False),
                resi=getattr(args, "resi", False),
            )
        elif args.attention_type == 2:
            print("===========")
            print("performer")
            print("===========")
            return MultiheadPerformerAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                approx_attn_dim=getattr(args, 'approx_attn_dim', 64),
                causal=getattr(args, 'causal', False),
            )
        elif args.attention_type == 3:
            print("===========")
            print("1+elu")
            print("===========")
            return LinearKernelAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                causal=getattr(args, "causal", False),
                use_orpe=getattr(args, "use_orpe", True),
                kernel_type=getattr(args, "kernel_type", "1+elu"),
                core_matrix=getattr(args, "core_matrix", 1),
                p_matrix=getattr(args, "p_matrix", 1),
                max_positions=getattr(args, "max_positions", 512),
                theta_type=getattr(args, "theta_type", "a"),
                theta_learned=getattr(args, "theta_learned", False), 
                householder_learned=getattr(args, "householder_learned", False),
                use_rope=getattr(args, "use_rope", False),
                use_spe=getattr(args, "use_spe", False),
                use_permutate=getattr(args, "use_permutate", False),
                max_seq_len=getattr(args, "max_seq_len", 512),
                # index
                index=args.index
            )
        elif args.attention_type == -1:
            print("===========")
            print("vanilla")
            print("===========")
            return MultiheadAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                index=args.index,
            )