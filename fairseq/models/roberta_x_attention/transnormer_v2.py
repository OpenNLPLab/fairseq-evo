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

from fairseq.models.xformer import TransnormerV2Encoder

class RobertaTransnormerV2Encoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransnormerV2Encoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_transnormerv2")
class RobertaTransnormerV2Model(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaTransnormerV2Encoder(args, task.source_dictionary)
        return cls(args, encoder)
    
@register_model_architecture("roberta_transnormerv2", "roberta_transnormer_v2_t1_postnorm")
def roberta_transnormer_v2_t1_postnorm(args):
    base_architecture(args)
    # add
    args.chunk_size = 64
    args.encoder_layers = args.encoder_layers
    n = args.encoder_layers
    m = n // 2
    args.encoder_attention_types = [2 for _ in range(m)] + [1 for _ in range(n - m)]
    args.norm_type = "simplermsnorm"
    args.final_simplermsnorm = "simplermsnorm"
    args.causal = True
    args.local_act_fun = "relu"
    args.use_softmax = False
    args.linear_act_fun = "elu"
    args.uv_act_fun = "swish"
    args.hidden_dim = args.encoder_embed_dim
    # glu
    args.use_glu = True
    args.glu_act = "swish"
    
@register_model_architecture("roberta_transnormerv2", "roberta_transnormer_v2_t2_postnorm")
def roberta_transnormer_v2_t2_postnorm(args):
    base_architecture(args)
    # add
    args.chunk_size = 64
    args.encoder_layers = args.encoder_layers
    n = args.encoder_layers
    m = n // 2
    args.encoder_attention_types = [2 for _ in range(m)] + [1 for _ in range(n - m)]
    args.norm_type = "simplermsnorm"
    args.final_simplermsnorm = "simplermsnorm"
    args.causal = True
    args.local_act_fun = "relu"
    args.use_softmax = True
    args.linear_act_fun = "1+elu"
    args.uv_act_fun = "swish"
    args.hidden_dim = args.encoder_embed_dim
    # glu
    args.use_glu = True
    args.glu_act = "swish"
    
@register_model_architecture("roberta_transnormerv2", "roberta_transnormer_v2_t1_prenorm")
def roberta_transnormer_v2_t1_prenorm(args):
    base_architecture(args)
    # add
    args.chunk_size = 64
    args.encoder_layers = args.encoder_layers
    n = args.encoder_layers
    m = n // 2
    args.encoder_attention_types = [2 for _ in range(m)] + [1 for _ in range(n - m)]
    args.norm_type = "simplermsnorm"
    args.final_simplermsnorm = "simplermsnorm"
    args.causal = True
    args.local_act_fun = "relu"
    args.use_softmax = False
    args.linear_act_fun = "elu"
    args.uv_act_fun = "swish"
    args.encoder_normalize_before = True
    args.hidden_dim = args.encoder_embed_dim
    # glu
    args.use_glu = True
    args.glu_act = "swish"
    
@register_model_architecture("roberta_transnormerv2", "roberta_transnormer_v2_t2_prenorm")
def roberta_transnormer_v2_t2_prenorm(args):
    base_architecture(args)
    # add
    args.chunk_size = 64
    args.encoder_layers = args.encoder_layers
    n = args.encoder_layers
    m = n // 2
    args.encoder_attention_types = [2 for _ in range(m)] + [1 for _ in range(n - m)]
    args.norm_type = "simplermsnorm"
    args.final_simplermsnorm = "simplermsnorm"
    args.causal = True
    args.local_act_fun = "relu"
    args.use_softmax = True
    args.linear_act_fun = "1+elu"
    args.uv_act_fun = "swish"
    args.encoder_normalize_before = True
    args.hidden_dim = args.encoder_embed_dim
    # glu
    args.use_glu = True
    args.glu_act = "swish"