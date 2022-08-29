# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging
from numpy import False_

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.roberta import RobertaEncoder, RobertaModel, base_architecture

from fairseq.models.xformer import SynthesizerEncoder

class RobertaSynthesizerEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = SynthesizerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_synthesizer")
class RobertaSynthesizer(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaSynthesizerEncoder(args, task.source_dictionary)
        return cls(args, encoder)

@register_model_architecture("roberta_synthesizer", "roberta_synthesizer_dense_base")
def roberta_synthesizer_dense_base(args):
    base_architecture(args)
    args.synthesizer_type = "dense"
    args.max_seq_len = 512
    args.causal = False
    args.encoder_layers = 16

@register_model_architecture("roberta_synthesizer", "roberta_synthesizer_random_base")
def roberta_synthesizer_random_base(args):
    base_architecture(args)
    args.synthesizer_type = "random"
    args.max_seq_len = 512
    args.causal = False
    args.encoder_layers = 18