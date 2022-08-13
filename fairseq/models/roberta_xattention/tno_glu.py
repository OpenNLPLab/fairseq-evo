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

from fairseq.models.xformer import TNOGLUEncoder

class RobertaTNOGLUEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TNOGLUEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_tno_glu")
class RobertaTNOGLU(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaTNOGLUEncoder(args, task.source_dictionary)
        return cls(args, encoder)

##### base
@register_model_architecture("roberta_tno_glu", "roberta_tno_exp_base_3_1")
def roberta_tno_exp_base_3_1(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 128
    # args.dpb_embedding = args.encoder_embed_dim // 4
    
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_3_1")
def roberta_tno_no_exp_base_3_1(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 128
    # args.dpb_embedding = args.encoder_embed_dim // 4
    
@register_model_architecture("roberta_tno_glu", "roberta_tno_exp_base_2_2")
def roberta_tno_exp_base_2_2(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 128
    # args.dpb_embedding = args.encoder_embed_dim // 4
    
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_2_2")
def roberta_tno_no_exp_base_2_2(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 128
    # args.dpb_embedding = args.encoder_embed_dim // 4

@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_e4_s4")
def roberta_tno_no_exp_base_e4_s4(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 4
    args.shrink_ratio = 4
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.encoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 128
##### base

##### no pos
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_3_1_no_pos")
def roberta_tno_no_exp_base_3_1_no_pos(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 128
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_2_2_no_pos")
def roberta_tno_no_exp_base_2_2_no_pos(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 128
    # pos
    args.no_token_positional_embeddings = True
##### no pos

##### standard
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_3_1_standard")
def roberta_tno_no_exp_base_3_1_standard(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4

@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_2_2_standard")
def roberta_tno_no_exp_base_2_2_standard(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_e4_s4_standard")
def roberta_tno_no_exp_base_e4_s4_standard(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 4
    args.shrink_ratio = 4
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.encoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
##### standard

##### standard pos
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_3_1_standard_no_pos")
def roberta_tno_no_exp_base_3_1_standard_no_pos(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
      
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_2_2_standard_no_pos")
def roberta_tno_no_exp_base_2_2_standard_no_pos(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_e4_s4_standard_no_pos")
def roberta_tno_no_exp_base_e4_s4_standard_no_pos(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 4
    args.shrink_ratio = 4
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.encoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### standard pos

##### single head
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_3_1_standard_no_pos_one_head")
def roberta_tno_no_exp_base_3_1_standard_no_pos_one_head(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.encoder_embed_dim
    args.encoder_attention_heads = 1
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
      
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_2_2_standard_no_pos_one_head")
def roberta_tno_no_exp_base_2_2_standard_no_pos_one_head(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.encoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.encoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("roberta_tno_glu", "roberta_tno_no_exp_base_e4_s4_standard_no_pos_one_head")
def roberta_tno_no_exp_base_e4_s4_standard_no_pos_one_head(args):
    base_architecture(args)
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 4
    args.shrink_ratio = 4
    args.encoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.encoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.encoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### single head