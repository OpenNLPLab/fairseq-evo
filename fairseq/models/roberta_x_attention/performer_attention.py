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

from fairseq.models.xformer import PerformerEncoder

class RobertaPerformerEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = PerformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_performer")
class RobertaPerformerModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaPerformerEncoder(args, task.source_dictionary)
        return cls(args, encoder)

########## performer
@register_model_architecture("roberta_performer", "roberta_performer")
def roberta_base_architecture_performer(args):
    base_architecture(args)
    args.approx_attn_dim = 64
    args.causal = False
    
########## rebuttal
@register_model_architecture("roberta_performer", "roberta_performer_prenorm")
def roberta_base_architecture_performer_prenorm(args):
    base_architecture(args)
    args.approx_attn_dim = 64
    args.causal = False
    args.encoder_normalize_before = True
    
@register_model_architecture("roberta_performer", "roberta_performer_prenorm_1d_1")
def roberta_base_architecture_performer_prenorm(args):
    base_architecture(args)
    args.approx_attn_dim = 64
    args.causal = False
    args.encoder_normalize_before = True
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "relu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True