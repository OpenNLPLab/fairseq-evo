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
    transformer_lm_baevski_wiki103,
)

from ..xformer import TNOGLUDecoder

@register_model("tno_glu_lm", dataclass=TransformerLanguageModelConfig)
class TNOGLULanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TNOGLULanguageModel, self).__init__(decoder)

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

        decoder = TNOGLUDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

########## large test
@register_model_architecture("tno_glu_lm", "tno_glu_silu_e3_g1_large_no_exp")
def tno_glu_silu_e3_g1_large_no_exp(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_layers = 12
    args.decoder_embed_dim = 1024
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_input_dim = args.decoder_embed_dim
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_e2_g2_large_no_exp")
def tno_glu_silu_e2_g2_large_no_exp(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.decoder_layers = 12
    args.decoder_embed_dim = 1024
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_input_dim = args.decoder_embed_dim
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_e4s4_g1_large_no_exp")
def tno_glu_silu_e4s4_g1_large_no_exp(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.decoder_layers = 12
    args.decoder_embed_dim = 1024
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_input_dim = args.decoder_embed_dim
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
########## large test

########## large model
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_ada")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_ada(args):
    transformer_lm_baevski_wiki103(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 1024
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_ada")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_ada(args):
    transformer_lm_baevski_wiki103(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 1024
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_ada")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_ada(args):
    transformer_lm_baevski_wiki103(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 1024
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_ada")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_ada(args):
    transformer_lm_baevski_wiki103(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 1024
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
########## large model

########## base
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_glu_1_dpb")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_glu_1_dpb(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_glu_1_dpb")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_glu_1_dpb(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_glu_1_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_glu_1_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_glu_1_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_glu_1_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_2_3_forward4_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_2_3_forward4_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = int(2 / 3 * args.decoder_embed_dim)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_glu_2_3_forward4_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_glu_2_3_forward4_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = int(2 / 3 * args.decoder_embed_dim)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_2_3_forward4_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_2_3_forward4_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = int(2 / 3 * args.decoder_embed_dim)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_glu_2_3_forward4_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_glu_2_3_forward4_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = int(2 / 3 * args.decoder_embed_dim)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_glu_2_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_glu_2_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_glu_2_dpb_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_glu_2_dpb_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
########## base

########## dpb_v2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
        
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_2_3_forward4_dpb_v2_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_2_3_forward4_dpb_v2_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = int(2 / 3 * args.decoder_embed_dim)
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
       
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_2_3_forward4_dpb_v2_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_2_3_forward4_dpb_v2_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = int(2 / 3 * args.decoder_embed_dim)
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
     
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
########## dpb_v2

########## no pos
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
##### v1

##### v2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
        
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
     
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
##### v2
########## no pos

########## decay
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_with_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_with_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_with_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # decay
    args.use_decay = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_with_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # decay
    args.use_decay = True
##### v1

##### v2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_with_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # decay
    args.use_decay = True
        
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_with_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_with_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # decay
    args.use_decay = True
     
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_with_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # decay
    args.use_decay = True
##### v2
########## decay

########## no pos decay
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
##### v1

##### v2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_with_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
        
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_with_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_with_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
     
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_with_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_with_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
##### v2
########## no pos decay

########## no pos with normalize
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_normalize")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_normalize(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # normalize
    args.normalize = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_normalize")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_normalize(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # normalize
    args.normalize = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_normalize")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_normalize(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # normalize
    args.normalize = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_normalize")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_normalize(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # normalize
    args.normalize = True
##### v1

##### v2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_normalize")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_normalize(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # normalize
    args.normalize = True
        
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_normalize")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_normalize(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # normalize
    args.normalize = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_normalize")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_normalize(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # normalize
    args.normalize = True
     
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_normalize")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_normalize(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # normalize
    args.normalize = True
##### v2
########## no pos with normalize

########## no pos multi_decay
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
##### v1

##### v2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_with_multi_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_with_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
        
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_with_multi_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_with_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_with_multi_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_with_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
     
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_with_multi_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_with_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
##### v2
########## no pos multi_decay

########## no pos
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_neg_exp")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_neg_exp(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.use_neg_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_neg_exp")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_neg_exp(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.use_neg_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    
##### v1

##### v2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_neg_exp")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v2_no_norm_no_pos_neg_exp(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.use_neg_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
        
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_neg_exp")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v2_no_norm_no_pos_neg_exp(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.use_neg_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb_v2
    args.use_dynamic_v2 = True
    args.dpb_embedding = 256
    args.dpb_act = "silu"
    # pos
    args.no_token_positional_embeddings = True
##### v2
########## no pos

########## multi dim v1
########## no pos
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
##### v1
########## no pos

########## no pos decay
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
##### v1
########## no pos decay

########## no pos multi_decay
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 1
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
##### v1
########## no pos multi_decay
########## multi dim v1

########## multi dim v2
########## no pos
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
##### v1
########## no pos

########## no pos decay
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_decay_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # decay
    args.use_decay = True
##### v1
########## no pos decay

########## no pos multi_decay
##### v1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_no_norm_no_pos_with_multi_decay_multi_dim_par_1_dpb_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.use_dynamic_v3 = True
    args.par_type = 1
    args.dpb_type = 2
    args.dpb_embedding = args.decoder_embed_dim
    # pos
    args.no_token_positional_embeddings = True
    # multi_decay
    args.use_multi_decay = True
##### v1
########## no pos multi_decay
########## multi dim v2

########## no pos v4
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## no pos v4

########## no pos v5
##### v5
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 1
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 1
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v5_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v5_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 1
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v5_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v5_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 1
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v5

##### v5_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v5_residual_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v5_residual_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 1
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_residual_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_residual_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 1
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v5_residual_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v5_residual_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 1
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v5_residual_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v5_residual_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 1
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v5_residual

##### v5
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v5_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v5_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v5_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v5_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v5

##### v5_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v5_residual_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v5_residual_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_residual_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_residual_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v5_residual_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v5_residual_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v5_residual_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v5_residual_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v5_residual
########## no pos v5

########## mutli decay
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_multi_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_multi_decay = True
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_multi_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_multi_decay = True
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_multi_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_multi_decay = True
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_multi_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_multi_decay = True
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_multi_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_multi_decay = True
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_multi_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_multi_decay = True
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_multi_decay")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_multi_decay = True
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_multi_decay")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_multi_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_multi_decay = True
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2
########## mutli decay

########## decay 0.999
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_999")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_999(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.999
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_999")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_999(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.999
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_999")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_999(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.999
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_999")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_999(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.999
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_999")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_999(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.999
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_999")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_999(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.999
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_999")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_999(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.999
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_999")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_999(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.999
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2
########## decay 0.999

########## decay 0.99
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_99")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_99(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.99
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_99")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_99(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.99
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_99")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_99(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.99
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_99")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_99(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.99
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_99")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_99(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.99
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_99")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_99(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.99
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_99")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_99(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.99
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_99")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_99(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.99
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2
########## decay 0.99

########## decay 0.9
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_9")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_9(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.9
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_9")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_decay_9(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.9
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_9")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_9(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.9
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_9")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_decay_9(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.9
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_9")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_9(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.9
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_9")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_decay_9(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.9
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_9")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_9(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.9
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_9")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_decay_9(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    args.gamma = 0.9
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2
########## decay 0.9

########## sigmoid
########## no pos v4
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos")
def tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos")
def tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos")
def tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos")
def tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos")
def tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos")
def tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos")
def tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos")
def tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_sigmoid_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_sigmoid_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "sigmoid"
    args.causal = True
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
    args.glu_act = "sigmoid"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## no pos v4
########## sigmoid

########## gelu
########## no pos v4
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos")
def tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos")
def tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos")
def tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos")
def tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos")
def tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos")
def tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos")
def tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos")
def tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2")
def tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_gelu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2")
def tno_glu_gelu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
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
    args.glu_act = "gelu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## no pos v4
########## gelu

########## silu forward5
########## no pos v4
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward5")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward5")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward5")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward5")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward5")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward5")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward5")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward5")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward5")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward5")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward5")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward5")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward5")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward5")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward5")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward5")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward5(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 5
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## no pos v4
########## silu forward5

########## no pos v4 no bias
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_no_bias")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_no_bias")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_no_bias")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_no_bias")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_no_bias")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_no_bias")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_no_bias")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_no_bias")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_no_bias")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_no_bias")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_no_bias")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_no_bias")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_no_bias")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_no_bias")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_no_bias")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_no_bias")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    # no bias
    args.bias = False
##### v4 par_type 2_residual
########## no pos v4 no bias

########## no pos v4 with coef
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_with_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_with_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_with_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.resi_param  = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_with_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.resi_param  = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_with_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_with_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_with_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.resi_param  = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_with_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.resi_param  = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_with_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_with_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_with_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.resi_param  = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_with_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.resi_param  = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_with_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_with_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_with_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.resi_param  = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_with_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_with_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.resi_param  = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## no pos v4 with coef


########## no pos v4 token shift
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_token_shift")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_token_shift")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_token_shift")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_token_shift")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_token_shift")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_token_shift")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_token_shift")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_token_shift")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_token_shift")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_token_shift")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_token_shift")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_token_shift")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 1
##### v4 par_type 2_residual
########## no pos v4 token shift

########## no pos v4 token shift 2
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_token_shift_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    args.token_shift_type = 2
##### v4 par_type 2_residual
########## no pos v4 token shift 2

########## forward 4
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
##### v4 par_type 2_residual
########## forward 4

########## forward 6
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward6")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward6")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward6")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward6")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward6")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward6")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward6")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward6")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward6")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward6")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward6")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward6")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward6")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward6")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward6")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward6")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward6(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 6
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## forward 6

########## forward 7
########## no pos v4
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## no pos v4
########## forward 7

########## forward 7 large
########## no pos v4
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 7
    args.max_l = 512
    # model
    args.expand_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## no pos v4
########## forward 7 large

########## forward 1 large
########## no pos v4
##### v4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4

##### v4_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4_residual

##### v4 par_type 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_par_type_2_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_par_type_2_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_par_type_2_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_par_type_2_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_par_type_2_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_par_type_2_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_par_type_2_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_no_norm_no_pos_par_type_2_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2

##### v4 par_type 2_residual
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_7_3_dpb_v4_residual_no_norm_no_pos_par_type_2_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 7 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.par_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.residual = True
    # pos
    args.no_token_positional_embeddings = True
##### v4 par_type 2_residual
########## no pos v4
########## forward 1 large

##### forward 1 large, coef
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### forward 1 large, coef

##### forward 1 large, coef, dpb large
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_large_no_norm_no_pos_forward1_large_coef")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v4_large_no_norm_no_pos_forward1_large_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 2
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_large_no_norm_no_pos_forward1_large_coef")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_large_no_norm_no_pos_forward1_large_coef(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 2
    # pos
    args.no_token_positional_embeddings = True
##### forward 1 large, coef, dpb large

##### one head
##### forward 1 large, coef
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_coef_one_head")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_coef_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_coef_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_coef_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.resi_param = True
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### forward 1 large, coef

##### forward 1 large, coef, dpb large
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### forward 1 large, coef, dpb large
##### one head

##### 2 2.5 large
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_5_2_dpb_v4_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_5_2_dpb_v4_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 5 * args.decoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_5_2_dpb_v4_no_norm_no_pos_forward1_large_one_head")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_2_glu_5_2_dpb_v4_no_norm_no_pos_forward1_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 5 * args.decoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_5_2_dpb_v4_no_norm_no_pos_forward1_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_5_2_dpb_v4_no_norm_no_pos_forward1_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 5 * args.decoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_5_2_dpb_v4_no_norm_no_pos_forward1_large_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_5_2_dpb_v4_no_norm_no_pos_forward1_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 5 * args.decoder_embed_dim // 2
    args.decoder_attention_heads = 1
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### 2 2.5 large

##### act test
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_cos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_cos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.tno_act_type = "cos"
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head_cos")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head_cos(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.tno_act_type = "cos"
    # model
    args.expand_ratio = 3
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_relu")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_relu(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.tno_act_type = "relu"
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head_relu")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head_relu(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.tno_act_type = "relu"
    # model
    args.expand_ratio = 3
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_relu2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_relu2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.tno_act_type = "relu2"
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head_relu2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head_relu2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.tno_act_type = "relu2"
    # model
    args.expand_ratio = 3
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_leak")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_leak(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.tno_act_type = "leak"
    # model
    args.expand_ratio = 3
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head_leak")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_4_3_dpb_v4_no_norm_no_pos_forward1_large_one_head_leak(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.tno_act_type = "leak"
    # model
    args.expand_ratio = 3
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 4 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### act test

##### expand 4 shrink 4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_4_glu_3_2_dpb_v4_no_norm_no_pos_forward1_shrink4_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_4_glu_3_2_dpb_v4_no_norm_no_pos_forward1_shrink4_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = 3 * args.decoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_4_glu_3_2_dpb_v4_no_norm_no_pos_forward1_shrink4_large_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_4_glu_3_2_dpb_v4_no_norm_no_pos_forward1_shrink4_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### expand 4 shrink 4

##### expand 2 shrink 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink2_large")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink2_large(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # layer
    args.decoder_layers = 8

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink2_large_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink2_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 2
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    # layer
    args.decoder_layers = 8

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 2
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink2_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink2_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 2
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### expand 2 shrink 2

##### expand 2 shrink 1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink1_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 1
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### expand 2 shrink 1

##### dpb test
# t1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = -1
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
# t2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = -1
    args.par_type = 2
    # pos
    args.no_token_positional_embeddings = True
    
# t3
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t3")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = -1
    args.par_type = 3
    # pos
    args.no_token_positional_embeddings = True
    
# t1 tran 1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 1
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
# t1 tran 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 2
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
# t1 tran 3
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran3")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 3
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True

# t1 tran 4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 4
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
##### dpb test

##### dpb test no exp
# t1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = -1
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
# t2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = -1
    args.par_type = 2
    # pos
    args.no_token_positional_embeddings = True
    
# t3
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t3")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = -1
    args.par_type = 3
    # pos
    args.no_token_positional_embeddings = True
    
# t1 tran 1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 1
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
# t1 tran 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 2
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
# t1 tran 3
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran3")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 3
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 4
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
##### dpb test no exp

##### dpb sin cos dim test
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran1_r_1_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran1_r_1_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 2
    args.transform_type = 1
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran1_r_1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t1_tran1_r_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim
    args.transform_type = 1
    args.par_type = 1
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2_r_1_4")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2_r_1_4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2_r_1_2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2_r_1_2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = args.decoder_embed_dim // 2
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2_r_1")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2_r_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.l = args.decoder_embed_dim
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### dpb sin cos dim test

##### [-1, 1] -> sin, cos
# t2 tran 1
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2_tran1")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2_tran1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 1
    args.par_type = 2
    # pos
    args.no_token_positional_embeddings = True
    
# t2 tran 2
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2_tran2")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2_tran2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 2
    args.par_type = 2
    # pos
    args.no_token_positional_embeddings = True
    
# t2 tran 3
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2_tran3")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2_tran3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 3
    args.par_type = 2
    # pos
    args.no_token_positional_embeddings = True

# t2 tran 4
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2_tran4")
def tno_glu_silu_simplermsnorm_toep_use_exp_1_rate_3_glu_1_dpb_v6_no_norm_no_pos_t2_tran4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 6
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.l = args.decoder_embed_dim // 4
    args.transform_type = 4
    args.par_type = 2
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2_par2")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v5_no_norm_no_pos_t2_par2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
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
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 5
    args.par_type = 2
    args.l = 10
    args.transform_type = 2
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### [-1. 1] -> sin, cos

##### head test
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_4_glu_2_3_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_4_glu_2_3_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 4
    args.shrink_ratio = 1
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim // 3
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_5_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_5_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 1
    args.decoder_attention_heads = 1
    # glu
    args.glu_act = "silu"
    args.glu_dim = 5 * args.decoder_embed_dim // 2
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True
##### head test

##### more layers
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward1_large_l7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward1_large_l7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_large_one_head_l7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_large_one_head_l7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.decoder_layers = 7
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    args.decoder_attention_heads = 1
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head_l7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head_l7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head_l7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_no_norm_no_pos_forward1_shrink1_large_one_head_l7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 1
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True
##### more layers

##### more layers dpb small
@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_64_no_norm_no_pos_forward1_large_l7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_64_no_norm_no_pos_forward1_large_l7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 64
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_64_no_norm_no_pos_forward1_large_one_head_l7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_64_no_norm_no_pos_forward1_large_one_head_l7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.decoder_layers = 7
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    args.decoder_attention_heads = 1
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 64
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_64_no_norm_no_pos_forward1_shrink1_large_one_head_l7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_3_glu_1_dpb_v4_64_no_norm_no_pos_forward1_shrink1_large_one_head_l7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 64
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_glu_lm", "tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_64_no_norm_no_pos_forward1_shrink1_large_one_head_l7")
def tno_glu_silu_simplermsnorm_toep_no_use_exp_1_rate_2_glu_2_dpb_v4_64_no_norm_no_pos_forward1_shrink1_large_one_head_l7(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.shrink_ratio = 1
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # glu
    args.glu_act = "silu"
    args.glu_dim = 2 * args.decoder_embed_dim
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = 64
    args.dpb_use_pad = False
    # pos
    args.no_token_positional_embeddings = True
##### more layers dpb small