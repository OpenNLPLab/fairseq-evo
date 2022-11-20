# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import (LayerNorm, TransformerDecoderLayer,
                             TransformerEncoderLayer)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

from .weight_linear_attention import WeightLinearAttention


class WeightLinearEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return WeightLinearAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            index=args.index,
            act_fun=getattr(args, "act_fun", "gelu"),
            weight_type=getattr(args, "weight_type", -1),
            # add
            causal=getattr(args, "causal", True),
            use_urpe=getattr(args, "use_urpe", False),
            core_matrix=getattr(args, "core_matrix", 1),
            p_matrix=getattr(args, "p_matrix", 1),
            theta_type=getattr(args, "theta_type", "a"),
            theta_learned=getattr(args, "theta_learned", False), 
            householder_learned=getattr(args, "householder_learned", False),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            use_sigmoid=getattr(args, "use_sigmoid", False),
            # cos
            cos_prenorm=getattr(args, "cos_prenorm", False),
            cos_postnorm=getattr(args, "cos_postnorm", False),
        )

class WeightLinearDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return WeightLinearAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            index=args.index,
            act_fun=getattr(args, "act_fun", "gelu"),
            weight_type=getattr(args, "weight_type", -1),
            # add
            causal=True,
            use_urpe=getattr(args, "use_urpe", False),
            core_matrix=getattr(args, "core_matrix", 1),
            p_matrix=getattr(args, "p_matrix", 1),
            theta_type=getattr(args, "theta_type", "a"),
            theta_learned=getattr(args, "theta_learned", False), 
            householder_learned=getattr(args, "householder_learned", False),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            use_sigmoid=getattr(args, "use_sigmoid", False),
            # cos
            cos_prenorm=getattr(args, "cos_prenorm", False),
            cos_postnorm=getattr(args, "cos_postnorm", False),
        )

    def build_encoder_attention(self, embed_dim, args):
        return WeightLinearAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            index=args.index,
            act_fun=getattr(args, "act_fun", "gelu"),
            weight_type=getattr(args, "weight_type", -1),
            # add
            causal=False,
            use_urpe=getattr(args, "use_urpe", False),
            core_matrix=getattr(args, "core_matrix", 1),
            p_matrix=getattr(args, "p_matrix", 1),
            theta_type=getattr(args, "theta_type", "a"),
            theta_learned=getattr(args, "theta_learned", False), 
            householder_learned=getattr(args, "householder_learned", False),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            use_sigmoid=getattr(args, "use_sigmoid", False),
            # cos
            cos_prenorm=getattr(args, "cos_prenorm", False),
            cos_postnorm=getattr(args, "cos_postnorm", False),
        )
