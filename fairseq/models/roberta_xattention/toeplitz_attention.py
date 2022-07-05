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

from fairseq.models.xformer import ToeplitzAttentionEncoder

class RobertaToeplitzAttentionEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = ToeplitzAttentionEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_toeplitz")
class RobertatoeplitzModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaToeplitzAttentionEncoder(args, task.source_dictionary)
        return cls(args, encoder)

##### only T
@register_model_architecture("roberta_toeplitz", "roberta_toeplitz_pure_linear_1+elu_AV+TV_exp")
def roberta_toeplitz_pure_linear_1_elu_AV_TV_exp(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.attention_use_layer_norm = False
    args.causal = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True

@register_model_architecture("roberta_toeplitz", "roberta_toeplitz_norm_linear_1+elu_AV+TV_exp")
def roberta_toeplitz_norm_linear_1_elu_AV_TV_exp(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.norm_type = "simplermsnorm"
    args.causal = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True

@register_model_architecture("roberta_toeplitz", "roberta_toeplitz_norm_linear_1+elu_AV_TV_no_exp")
def roberta_toeplitz_norm_linear_1_elu_AV_TV_no_exp(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.norm_type = "simplermsnorm"
    args.causal = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = False

##### urpe + T
@register_model_architecture("roberta_toeplitz", "roberta_toeplitz_pure_linear_1+elu_AV+TV_exp_urpe_1d_3")
def roberta_toeplitz_pure_linear_1_elu_AV_TV_exp_urpe_1d_3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.attention_use_layer_norm = False
    args.causal = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_toeplitz", "roberta_toeplitz_norm_linear_1+elu_AV+TV_exp_urpe_1d_3")
def roberta_toeplitz_norm_linear_1_elu_AV_TV_exp_urpe_1d_3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.norm_type = "simplermsnorm"
    args.causal = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_toeplitz", "roberta_toeplitz_norm_linear_1+elu_AV_TV_no_exp_urpe_1d_3")
def roberta_toeplitz_norm_linear_1_elu_AV_TV_no_exp_urpe_1d_3(args):
    base_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.norm_type = "simplermsnorm"
    args.causal = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = False
    ##### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True