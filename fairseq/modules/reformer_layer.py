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
# linear transformer
from fairseq.modules import ReformerAttention_
from fairseq.modules import LSHAttention

class ReformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        # 有bug
        # return ReformerAttention(
        #     d_model=embed_dim,
        #     n_heads=args.encoder_attention_heads
        # )

        return LSHAttention(
            dim=embed_dim,
            heads=args.encoder_attention_heads
        )

class ReformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        print("init")
        print(args)
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):  
        # 有bug
        # return ReformerAttention_(
        #     d_model=embed_dim,
        #     n_heads=args.decoder_attention_heads
        # )

        return LSHAttention(
            dim=embed_dim,
            heads=args.decoder_attention_heads,
            causal=args.causal,
            bucket_size=args.bucket_size,
            n_hashes=args.n_hashes,
            attn_chunks=args.attn_chunks
        )