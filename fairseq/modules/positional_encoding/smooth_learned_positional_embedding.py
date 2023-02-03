# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor


class SmoothLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, max_seq=512, method=1):
        super().__init__(max_seq + padding_idx + 1, embedding_dim, padding_idx)
        self.num_embeddings = num_embeddings
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings
        self.max_seq = max_seq
        self.method = method
        self.cnt = 1

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."
        
        if self.method == 1:
            pos_list, coef_list = utils.make_smooth_positions(
                input, self.padding_idx, onnx_trace=self.onnx_trace, max_seq=self.max_seq
            )
            pos_embedding = 0
            n = len(pos_list)
            
            for i in range(n):
                pos_embedding += coef_list[i].unsqueeze(-1) * \
                                F.embedding(
                                    pos_list[i],
                                    self.weight,
                                    self.padding_idx,
                                    self.max_norm,
                                    self.norm_type,
                                    self.scale_grad_by_freq,
                                    self.sparse,
                                )
            
            return pos_embedding
        elif self.method == 2:
            positions = utils.make_group_positions(
                input, self.padding_idx, onnx_trace=self.onnx_trace, max_seq=self.max_seq
            )

            return F.embedding(
                positions,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        elif self.method == 3:
            if self.training:
                positions = utils.make_group_positions_training(
                    input, self.padding_idx, onnx_trace=self.onnx_trace, group=self.cnt
                )
                self.cnt = (self.cnt + 1) % self.max_seq
            else:
                positions = utils.make_group_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace, max_seq=self.max_seq
                )

            return F.embedding(
                positions,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        elif self.method == 4:
            pos1, pos2, coef = utils.make_group_positions(
                input, self.padding_idx, onnx_trace=self.onnx_trace, max_seq=self.max_seq
            )
            
            pe1 = F.embedding(
                pos1,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            pe2 = F.embedding(
                pos2,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            
            return pe1 * coef + pe2 * (1 - coef)