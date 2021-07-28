# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

# from typed_ast.ast3 import arg

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, TransformerEncoderLayer, TransformerDecoderLayer, LongformerSelfAttention

# add
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor


class TransformerLongformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def build_self_attention(self, embed_dim, args):
        return LongformerSelfAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            attention_window= args.attention_window,
            attention_dilation= args.attention_dilation,
            autoregressive= args.autoregressive, 
            attention_mode= args.attention_mode,
            layer_id = 0
        )

class TransformerLongformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, args, layer_id = 0, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        self.layer_id = layer_id
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):

        return LongformerSelfAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            
            # attention_window= args.attention_window,
            # attention_dilation= args.attention_dilation,
            # autoregressive= args.autoregressive, 
            # attention_mode= args.attention_mode,
            # layer_id = 0
            # 暂时写死

            attention_window= args.attention_window,
            attention_dilation= args.attention_dilation,
            autoregressive= args.autoregressive, 
            attention_mode= args.attention_mode,
            layer_id = self.layer_id

        )
