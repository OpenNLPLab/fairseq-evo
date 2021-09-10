# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, TransformerEncoderLayer, TransformerDecoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
# merge attention
from fairseq.modules import MultiheadCosAttention

class TransformerCosEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return MultiheadCosAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            is_base=getattr(args, "is_base", True),
            is_ada_q=getattr(args, "is_ada_q", False),
            is_ada_k=getattr(args, "is_ada_k", False),
            lambda_=getattr(args, "lambda_", 0.99),
            up_fq=getattr(args, "up_fq", 16),
            dropout_before=getattr(args, "dropout_before", False),
            use_q=getattr(args, "use_q", False),
            use_k=getattr(args, "use_k", False),
            # add
            low_d=getattr(args, "low_d", False),
            has_out=getattr(args, "has_out", False),
            do_scale=getattr(args, "do_scale", True),
            norm_taylor=getattr(args, "norm_taylor", True),
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            use_square=getattr(args, "use_square", False),
            use_sigmoid=getattr(args, "use_sigmoid", False),
            use_l2=getattr(args, "use_l2", False),
            dim_scale=getattr(args, "dim_scale", -1),
            sparse=getattr(args, "sparse", False),
            d1=getattr(args, "d1", 32),
            d2=getattr(args, "d2", 8),
            has_res=getattr(args, "has_res", False),
            has_right_weight=getattr(args, "has_right_weight", False),
            do_softmax=getattr(args, "do_softmax", False),
            has_right_weight_not_share=getattr(args, "has_right_weight_not_share", False),
            index=args.index,
            alpha_beta=getattr(args, "alpha_beta", False),
            max_l=getattr(args, "max_l", 1024),
        )

class TransformerCosDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadCosAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            is_base=getattr(args, "is_base", True),
            is_ada_q=getattr(args, "is_ada_q", False),
            is_ada_k=getattr(args, "is_ada_k", False),
            lambda_=getattr(args, "lambda_", 0.99),
            up_fq=getattr(args, "up_fq", 16),
            dropout_before=getattr(args, "dropout_before", False),
            use_q=getattr(args, "use_q", False),
            use_k=getattr(args, "use_k", False),
            # add
            low_d=getattr(args, "low_d", False),
            has_out=getattr(args, "has_out", False),
            do_scale=getattr(args, "do_scale", True),
            norm_taylor=getattr(args, "norm_taylor", True),
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            use_square=getattr(args, "use_square", False),
            use_sigmoid=getattr(args, "use_sigmoid", False),
            use_l2=getattr(args, "use_l2", False),
            dim_scale=getattr(args, "dim_scale", -1),
            sparse=getattr(args, "sparse", False),
            d1=getattr(args, "d1", 32),
            d2=getattr(args, "d2", 8),
            has_res=getattr(args, "has_res", False),
            has_right_weight=getattr(args, "has_right_weight", False),
            do_softmax=getattr(args, "do_softmax", False),
            has_right_weight_not_share=getattr(args, "has_right_weight_not_share", False),
            index=args.index,
            alpha_beta=getattr(args, "alpha_beta", False),
            max_l=getattr(args, "max_l", 1024),
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadCosAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            is_base=getattr(args, "is_base", True),
            is_ada_q=getattr(args, "is_ada_q", False),
            is_ada_k=getattr(args, "is_ada_k", False),
            lambda_=getattr(args, "lambda_", 0.99),
            up_fq=getattr(args, "up_fq", 16),
            dropout_before=getattr(args, "dropout_before", False),
            use_q=getattr(args, "use_q", False),
            use_k=getattr(args, "use_k", False),
            # add
            low_d=getattr(args, "low_d", False),
            has_out=getattr(args, "has_out", False),
            do_scale=getattr(args, "do_scale", True),
            norm_taylor=getattr(args, "norm_taylor", True),
            use_relu=getattr(args, "use_relu", False),
            use_elu=getattr(args, "use_elu", False),
            use_leak=getattr(args, "use_leak", False),
            use_square=getattr(args, "use_square", False),
            use_sigmoid=getattr(args, "use_sigmoid", False),
            use_l2=getattr(args, "use_l2", False),
            dim_scale=getattr(args, "dim_scale", -1),
            sparse=getattr(args, "sparse", False),
            d1=getattr(args, "d1", 32),
            d2=getattr(args, "d2", 8),
            has_res=getattr(args, "has_res", False),
            has_right_weight=getattr(args, "has_right_weight", False),
            do_softmax=getattr(args, "do_softmax", False),
            has_right_weight_not_share=getattr(args, "has_right_weight_not_share", False),
            index=args.index,
            alpha_beta=getattr(args, "alpha_beta", False),
            max_l=getattr(args, "max_l", 1024),
        )