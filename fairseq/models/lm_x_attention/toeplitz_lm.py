# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import logging
from dataclasses import dataclass, field
from typing import Optional

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
logger = logging.getLogger(__name__)
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, TransformerDecoder
)

from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II
from typing import Dict, List, Optional
import torch

from fairseq.models.transformer_lm import (
    DEFAULT_MAX_TARGET_POSITIONS, 
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
    base_lm_architecture,
    transformer_lm_big,
)

from ..xformer import ToeplitzAttentionDecoder

@register_model("toeplitz_lm", dataclass=TransformerLanguageModelConfig)
class ToeplitzLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(ToeplitzLanguageModel, self).__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = ToeplitzAttentionDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

########## linear + toeplizt
@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_linear_1+elu_TV")
def transofrmer_toeplitz_lm_pure_linear_1_elu_TV(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    args.attention_use_layer_norm = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = -1

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_linear_1+elu_ATV")
def transofrmer_toeplitz_lm_pure_linear_1_elu_ATV(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    args.attention_use_layer_norm = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 1

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_linear_1+elu_TAV")
def transofrmer_toeplitz_lm_pure_linear_1_elu_ATV(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    args.attention_use_layer_norm = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 2
########## linear + toeplizt

########## linear + toeplizt
@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_TV")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_TV(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = -1

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_ATV")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_ATV(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 1

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_TAV")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_ATV(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 2
########## linear + toeplizt

########## pure linear AV + TV
@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_linear_1+elu_AV+TV_exp")
def transofrmer_toeplitz_lm_pure_linear_1_elu_AV_TV_exp(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.attention_use_layer_norm = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_linear_1+elu_AV+TV_no_exp")
def transofrmer_toeplitz_lm_pure_linear_1_elu_AV_TV_no_exp(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.attention_use_layer_norm = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = False
########## pure linear AV + TV

########## norm + toeplizt AV + TV
@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_AV+TV_exp")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_AV_TV_exp(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_AV_TV_no_exp")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_AV_TV_no_exp(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = False
########## norm + toeplizt AV + TV

########## urpe + toeplizt AV + TV
@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_linear_1+elu_AV+TV_exp_urpe_1")
def transofrmer_toeplitz_lm_pure_linear_1_elu_AV_TV_exp_urpe_1(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.attention_use_layer_norm = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_AV+TV_exp_urpe_1")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_AV_TV_exp_urpe_1(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_AV_TV_no_exp_urpe_1")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_AV_TV_no_exp_urpe_1(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = False
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_linear_1+elu_AV+TV_exp_urpe_1d_3")
def transofrmer_toeplitz_lm_pure_linear_1_elu_AV_TV_exp_urpe_1d_3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.attention_use_layer_norm = False
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_AV+TV_exp_urpe_1d_3")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_AV_TV_exp_urpe_1d_3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = True
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_AV_TV_no_exp_urpe_1d_3")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_AV_TV_no_exp_urpe_1d_3(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.use_exp = False
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
########## urpe + toeplizt AV + TV

########## norm + toeplizt AV + TV multi
@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_AV+TV_exp_multi")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_AV_TV_exp_multi(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.dynamic_type = 4
    args.use_exp = True

@register_model_architecture("toeplitz_lm", "toeplitz_lm_pure_norm_linear_1+elu_AV_TV_no_exp_multi")
def transofrmer_toeplitz_lm_pure_norm_linear_1_elu_AV_TV_no_exp_multi(args):
    base_lm_architecture(args)
    ##### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    args.norm_type = "simplermsnorm"
    ##### topelitz
    args.type_num = -1
    args.toep_type = 3
    args.dynamic_type = 4
    args.use_exp = False
########## norm + toeplizt AV + TV multi