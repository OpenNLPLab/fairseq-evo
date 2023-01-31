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

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, max_seq=512):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings
        self.max_seq = max_seq

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

        pos_list, coef_list = utils.make_smooth_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace, max_seq=self.max_seq
        )
        
        pos_embedding = 0
        n = len(pos_list)
        for i in range(n):
            pos_embedding += coef_list[i] * \
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