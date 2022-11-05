# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (LayerDropModuleList, RealNumberEmbedding,
                             TnnSentenceEncoderLayer, get_norm_fn)
from fairseq.modules.fairseq_dropout import FairseqDropout


class TnnLRAEncoder(nn.Module):
    """
    Implementation for a Bi-directional FLASH based Sentence Encoder used
    in masked pre-trained language models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_type: str = "sparse",
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        ffn_hidden_dim: int = 1024,
        activation: str = 'silu',
        attention_activation: str = 'softmax',
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        norm_type: str = 'layernorm',
        normalize_before: bool = False,
        normalize_embedding: bool = False,
        feature_dropout: bool = False,
        layerdrop: float = 0.0,
        truncation: int = None,
        rel_pos_bias: str = 'simple',
        max_seq_len: int = 256,
        export: bool = False,
        traceable: bool = False,
        sen_rep_type: str = 'cls',
        args=None,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.embedding_dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.embedding_dim,
                                                 self.vocab_size, self.padding_idx)

        assert not normalize_embedding or not normalize_before
        # self.embed_norm = SequenceNorm(norm_type, embedding_dim, export=export) if normalize_embedding else None
        self.embed_norm = get_norm_fn(norm_type)(embedding_dim) if normalize_embedding else None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        self.layers.extend([
            self.build_sentence_encoder_layer(
                args=args,
            )
            for _ in range(self.num_layers)
        ])

        if normalize_before:
            # self.final_norm = SequenceNorm(norm_type, embedding_dim, export=export)
            self.final_norm = get_norm_fn(norm_type)(embedding_dim)
        else:
            self.final_norm = None

    def build_embedding(self, embedding_type, embedding_dim, vocab_size, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens = Embedding(vocab_size, embedding_dim, padding_idx)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def build_sentence_encoder_layer(
        self,
        args,
    ):
        return TnnSentenceEncoderLayer(
            args=args,
        )

    def forward(
            self,
            tokens: torch.Tensor,
            src_lengths: torch.Tensor,
            last_state_only: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        if self.embedding_type == 'sparse':
            padding_mask = tokens.eq(self.padding_idx)
            if not self.traceable and not self.tpu and not padding_mask.any():
                padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)
        else:
            padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)

        if self.embed_norm is not None:
            x = self.embed_norm(x)

        x = self.embedding_dropout(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            # B x N
            inverse_mask = 1.0 - padding_mask.type_as(x)
            x = x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(self.num_layers):
            x, _ = self.layers[i](x, encoder_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=0) / src_lengths.unsqueeze(1)
        else:
            sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.reset_parameters(embedding_dim)

    def reset_parameters(self, embedding_dim):
        std = embedding_dim ** -0.5
        nn.init.normal_(self.embed.weight, mean=0, std=std)

    def forward(self, tokens):
        x = self.embed(tokens)
        return x
