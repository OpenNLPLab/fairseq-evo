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
from fairseq.modules import LSAttentionCausal, LSAttentionNonCausal, LongShortAttention

class LSAttentionEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        # return LSAttentionNonCausal(
        #     dim=embed_dim,
        #     num_heads=args.encoder_attention_heads,
        #     max_seq_len=getattr(args, "max_seq_len", 512),
        #     dropout=args.attention_dropout,
        #     num_landmarks=getattr(args, "num_landmarks", 32),
        #     window_size=getattr(args, "window_size", 8),
        # )
        # return LSAttentionNonCausal(
        #     dim=embed_dim,
        #     num_heads=args.encoder_attention_heads,
        #     attn_drop=args.attention_dropout,
        # )
        return LongShortAttention(
            dim=embed_dim,
            heads=args.encoder_attention_heads,
            causal=getattr(args, "causal", False),
            window_size=getattr(args, "window_size", 128),
            segment_size=getattr(args, "segment_size", 16),
            r=getattr(args, "r", 1),
        )

############# don't use
class LSAttentionDecoderLayer(TransformerDecoderLayer):
    def __init__(
        elf, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        # return LSAttentionCausal(
        #     d_model=embed_dim,
        #     n_head=args.encoder_attention_heads,
        #     chunk_size=getattr(args, "chunk_size", 16),
        #     chunk_rank=getattr(args, "chunk_rank", 1),
        #     window_len=getattr(args, "window_len", 512),
        #     attn_drop=args.attention_dropout,
        # )
        return LongShortAttention(
            dim=embed_dim,
            heads=args.decoder_attention_heads,
            causal=getattr(args, "causal", True),
            window_size=getattr(args, "window_size", 128),
            segment_size=getattr(args, "segment_size", 16),
            r=getattr(args, "r", 1),
        )

    def build_encoder_attention(self, embed_dim, args):
        # return LSAttentionNonCausal(
        #     dim=embed_dim,
        #     num_heads=args.encoder_attention_heads,
        #     attn_drop=args.attention_dropout,
        # )
        # return LSAttentionNonCausal(
        #     dim=embed_dim,
        #     num_heads=args.encoder_attention_heads,
        #     max_seq_len=getattr(args, "max_seq_len", 512),
        #     dropout=args.attention_dropout,
        #     num_landmarks=getattr(args, "num_landmarks", 32),
        #     window_size=getattr(args, "window_size", 8),
        # )
        return LongShortAttention(
            dim=embed_dim,
            heads=args.encoder_attention_heads,
            causal=getattr(args, "causal", False),
            window_size=getattr(args, "window_size", 128),
            segment_size=getattr(args, "segment_size", 16),
            r=getattr(args, "r", 1),
        )