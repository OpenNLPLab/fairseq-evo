# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn
from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (FairseqIncrementalDecoder, FairseqLanguageModel,
                            register_model, register_model_architecture)

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional

import torch
from fairseq.models.transformer import (DEFAULT_MIN_PARAMS_TO_WRAP, Embedding,
                                        TransformerDecoder)
from fairseq.models.transformer_lm import (DEFAULT_MAX_TARGET_POSITIONS,
                                           TransformerLanguageModel,
                                           TransformerLanguageModelConfig,
                                           base_lm_architecture,
                                           transformer_lm_big)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II

from ..xformer import TNOFFNDecoder


@register_model("tno_ffn_lm", dataclass=TransformerLanguageModelConfig)
class TNOFFNLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TNOFFNLanguageModel, self).__init__(decoder)

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

        decoder = TNOFFNDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

########## forward1
##### baseline
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1(args):
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
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head(args):
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
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1(args):
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
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head(args):
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
    args.expand_ratio = 4 / 3
##### baseline

##### gelu
@register_model_architecture("tno_ffn_lm", "tno_ffn_gelu_simplermsnorm_toep_use_exp_1")
def tno_ffn_gelu_simplermsnorm_toep_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_gelu_simplermsnorm_toep_use_exp_1_one_head")
def tno_ffn_gelu_simplermsnorm_toep_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
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
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_gelu_simplermsnorm_toep_no_use_exp_1")
def tno_ffn_gelu_simplermsnorm_toep_no_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_gelu_simplermsnorm_toep_no_use_exp_1_one_head")
def tno_ffn_gelu_simplermsnorm_toep_no_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
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
    args.expand_ratio = 4 / 3
##### gelu

##### norm test
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_layernorm_toep_use_exp_1")
def tno_ffn_silu_layernorm_toep_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "layernorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_layernorm_toep_use_exp_1_one_head")
def tno_ffn_silu_layernorm_toep_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "layernorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_layernorm_toep_no_use_exp_1")
def tno_ffn_silu_layernorm_toep_no_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "layernorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_layernorm_toep_no_use_exp_1_one_head")
def tno_ffn_silu_layernorm_toep_no_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "layernorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
##### norm test

##### rate test
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_2_3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_2_3(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_3(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_2_3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_2_3(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_3(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3

@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_5")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_5(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_5")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_5(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_5")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_5(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_5")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_5(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
##### rate test
########## forward1

########## forward2
##### baseline
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_forward2")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_forward2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 2
    args.max_l = 512
    # model
    args.expand_ratio = 2
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_forward2")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_forward2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 2
    args.max_l = 512
    # model
    args.expand_ratio = 2
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_forward2")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_forward2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 2
    args.max_l = 512
    # model
    args.expand_ratio = 2
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_forward2")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_forward2(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 2
    args.max_l = 512
    # model
    args.expand_ratio = 2
##### baseline
########## forward2

########## se
##### baseline
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_se_16")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_se_16(args):
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
    args.expand_ratio = 4 / 3
    # se
    args.use_se = True
    args.se_ratio = 16
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_se_16")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_se_16(args):
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
    args.expand_ratio = 4 / 3
    # se
    args.use_se = True
    args.se_ratio = 16
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_se_16")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_se_16(args):
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
    args.expand_ratio = 4 / 3
    # se
    args.use_se = True
    args.se_ratio = 16
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_se_16")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_se_16(args):
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
    args.expand_ratio = 4 / 3
    # se
    args.use_se = True
    args.se_ratio = 16
##### baseline

##### ratio test
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_se_16_rate_2_3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_se_16_rate_2_3(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # se
    args.use_se = True
    args.se_ratio = 16
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_se_16_rate_2_3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_se_16_rate_2_3(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # se
    args.use_se = True
    args.se_ratio = 16
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_se_16_rate_2_3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_se_16_rate_2_3(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # se
    args.use_se = True
    args.se_ratio = 16
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_se_16_rate_2_3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_se_16_rate_2_3(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # se
    args.use_se = True
    args.se_ratio = 16
##### ratio test
########## se

########## forward3
##### rate test
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_forward3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_forward3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_forward3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_forward3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 4 / 3

@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_2_3_forward3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_2_3_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_3_forward3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_3_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_2_3_forward3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_2_3_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_3_forward3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_3_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3

@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_5_forward3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_5_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_5_forward3")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_5_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_5_forward3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_5_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_5_forward3")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_5_forward3(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 3
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
##### rate test
########## forward3

########## forward4
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_forward4")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_forward4")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_forward4")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_forward4")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_forward4(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
########## forward4

########## dpb
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_forward4_dpb")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_forward4_dpb(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_forward4_dpb")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_forward4_dpb(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_forward4_dpb")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_forward4_dpb(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_forward4_dpb")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_forward4_dpb(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 4
    args.max_l = 512
    # model
    args.expand_ratio = 3
    args.shrink_ratio = 1
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_dpb")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_dpb(args):
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
    args.expand_ratio = 4 / 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_dpb")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_dpb(args):
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
    args.expand_ratio = 4 / 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_dpb")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_dpb(args):
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
    args.expand_ratio = 4 / 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_dpb")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_dpb(args):
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
    args.expand_ratio = 4 / 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim

@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_2_3_dpb")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_2_3_dpb(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_3_dpb")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_3_dpb(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_2_3_dpb")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_2_3_dpb(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_3_dpb")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_3_dpb(args):
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
    args.expand_ratio = 2
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim

@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_5_dpb")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_5_dpb(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_5_dpb")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_5_dpb(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_5_dpb")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_5_dpb(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_5_dpb")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_5_dpb(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
########## dpb

########## dpb no norm
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_forward4_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_forward4_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_forward4_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_forward4_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_forward4_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_forward4_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_forward4_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_forward4_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = args.decoder_embed_dim
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_dpb_no_norm(args):
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
    args.expand_ratio = 4 / 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_dpb_no_norm(args):
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
    args.expand_ratio = 4 / 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_dpb_no_norm(args):
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
    args.expand_ratio = 4 / 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_dpb_no_norm(args):
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
    args.expand_ratio = 4 / 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim

@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_2_3_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_2_3_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_3_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_2_3_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_2_3_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_2_3_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_3_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_2_3_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = args.decoder_embed_dim * 3
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim

@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_5_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_rate_3_1_5_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_5_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_use_exp_1_one_head_rate_3_1_5_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_5_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_rate_3_1_5_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
    
@register_model_architecture("tno_ffn_lm", "tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_5_dpb_no_norm")
def tno_ffn_silu_simplermsnorm_toep_no_use_exp_1_one_head_rate_3_1_5_dpb_no_norm(args):
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
    args.decoder_ffn_embed_dim = int(args.decoder_embed_dim * 1.5)
    # dpb
    args.use_dynamic = True
    args.dpb_embedding = args.decoder_embed_dim
########## dpb no norm
