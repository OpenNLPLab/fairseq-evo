# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from ..ffn import GLU
from ..helpers import get_norm_fn, logging_info
from .gtu_module_v2 import GtuV2Module


class TnnSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        args,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        
        # Initialize blocks
        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
        )

        norm_type = getattr(args, 'norm_type', 'layernorm')
        logging_info(f"Sentence Encoder Norm Type: {norm_type}")
        self.self_attn_layer_norm = get_norm_fn(norm_type)(self.embed_dim)

        # bias
        self.normalize_before = args.encoder_normalize_before
        logging_info(f"normalize_before {self.normalize_before}")
        self.glu_act = getattr(args, "glu_act", "silu")
        self.fina_act = getattr(args, "fina_act", "None")
        self.glu_dropout = getattr(args, "glu_dropout", 0.0)
        self.glu_dim = getattr(args, "glu_dim", -1)
        if self.glu_dim == -1:
            self.glu_dim = self.embed_dim
        # bias
        bias = getattr(args, "bias", True)
        self.glu = GLU(self.embed_dim, self.glu_dim, self.glu_act, self.fina_act, self.glu_dropout, bias)

        self.final_layer_norm = get_norm_fn(norm_type)(self.embed_dim)

    def build_self_attention(
        self,
        embed_dim,
        args,
    ):
        return GtuV2Module(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # gtu
            act_fun=getattr(args, "act_fun", "silu"),
            causal=getattr(args, "causal", False),
            expand_ratio=getattr(args, "expand_ratio", 3),
            resi_param=getattr(args, "resi_param", False),
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            use_decay=getattr(args, "use_decay", False),
            use_multi_decay=getattr(args, "use_multi_decay", False),
            rpe_layers=getattr(args, "rpe_layers", 6),
            rpe_embedding=getattr(args, "rpe_embedding", 64),
            rpe_act=getattr(args, "rpe_act", "relu"),
            normalize=getattr(args, "normalize", False),
            par_type=getattr(args, "par_type", 1),
            residual=getattr(args, "residual", False),
            gamma=getattr(args, "gamma", 0.99),
            act_type=getattr(args, "act_type", "none"),
        )
        
    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.glu(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, None

