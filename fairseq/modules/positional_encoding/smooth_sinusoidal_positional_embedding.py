# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional

import torch
import torch.onnx.operators
from fairseq import utils
from torch import Tensor, nn


class SmoothSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024, max_seq=512, method=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SmoothSinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)
        self.max_seq = max_seq
        self.method = method
        

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        pos_list, coef_list = utils.make_smooth_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace, max_seq=self.max_seq
        )

        if self.method == 1:
            pos_embedding = 0
            n = len(pos_list)
            
            for i in range(n):
                pos_embedding += coef_list[i].unsqueeze(-1) * \
                                (
                                    self.weights.index_select(0, pos_list[i].view(-1))
                                    .view(bsz, seq_len, -1)
                                    .detach()
                                )
            
            return pos_embedding
        elif self.method == 2:
            positions = utils.make_group_positions(
                input, self.padding_idx, onnx_trace=self.onnx_trace, max_seq=self.max_seq
            )
            
            return (
                self.weights.index_select(0, positions.view(-1))
                .view(bsz, seq_len, -1)
                .detach()
            )