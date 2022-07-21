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
from .tno import TNO

class TNOFFNEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return TNO(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            index=args.index,
            act_fun=getattr(args, "act_fun", "silu"),
            causal=getattr(args, "causal", True),
            expand_ratio=getattr(args, "expand_ratio", 2),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            # Toeplizt
            use_exp=getattr(args, "use_exp", False),
            toep_type=getattr(args, "toep_type", 1),
            max_l=getattr(args, "max_l", 512),
            use_decay=getattr(args, "use_decay", False),
            use_dynamic=getattr(args, "use_dynamic", False),
            dpb_embedding=getattr(args, "dpb_embedding", 512),
            # se
            use_se=getattr(args, "use_se", False),
            se_ratio=getattr(args, "se_ratio", 16),
        )

class TNOFFNDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return TNO(
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
            act_fun=getattr(args, "act_fun", "silu"),
            causal=True,
            expand_ratio=getattr(args, "expand_ratio", 2),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            # Toeplizt
            use_exp=getattr(args, "use_exp", False),
            toep_type=getattr(args, "toep_type", 1),
            max_l=getattr(args, "max_l", 512),
            use_decay=getattr(args, "use_decay", False),
            use_dynamic=getattr(args, "use_dynamic", False),
            dpb_embedding=getattr(args, "dpb_embedding", 512),
            # se
            use_se=getattr(args, "use_se", False),
            se_ratio=getattr(args, "se_ratio", 16),
        )

    def build_encoder_attention(self, embed_dim, args):
        return TNO(
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
            act_fun=getattr(args, "act_fun", "silu"),
            causal=False,
            expand_ratio=getattr(args, "expand_ratio", 2),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            # Toeplizt
            use_exp=getattr(args, "use_exp", False),
            toep_type=getattr(args, "toep_type", 1),
            max_l=getattr(args, "max_l", 512),
            use_decay=getattr(args, "use_decay", False),
            use_dynamic=getattr(args, "use_dynamic", False),
            dpb_embedding=getattr(args, "dpb_embedding", 512),
            # se
            use_se=getattr(args, "use_se", False),
            se_ratio=getattr(args, "se_ratio", 16),
        )