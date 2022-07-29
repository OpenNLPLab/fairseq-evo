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