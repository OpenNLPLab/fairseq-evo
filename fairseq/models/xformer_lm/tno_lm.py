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

from ..xformer import TNODecoder

@register_model("tno_lm", dataclass=TransformerLanguageModelConfig)
class TNOLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TNOLanguageModel, self).__init__(decoder)

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

        decoder = TNODecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

########## small ratio
##### baseline
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1")
def tno_silu_simplermsnorm_toep_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1")
def tno_silu_simplermsnorm_toep_no_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
##### baseline

##### decay
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_decay")
def tno_silu_simplermsnorm_toep_use_exp_1_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_decay")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_decay")
def tno_silu_simplermsnorm_toep_no_use_exp_1_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_decay")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
##### decay

##### act test
@register_model_architecture("tno_lm", "tno_gelu_simplermsnorm_toep_use_exp_1")
def tno_gelu_simplermsnorm_toep_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_gelu_simplermsnorm_toep_use_exp_1_one_head")
def tno_gelu_simplermsnorm_toep_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_gelu_simplermsnorm_toep_no_use_exp_1")
def tno_gelu_simplermsnorm_toep_no_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_gelu_simplermsnorm_toep_no_use_exp_1_one_head")
def tno_gelu_simplermsnorm_toep_no_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "gelu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
##### act test

##### resi param
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_resi_para")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head_resi_para(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    args.resi_param  = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_resi_para")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_resi_para(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    args.resi_param  = True
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
##### resi param
########## small ratio

########## big ratio
##### baseline
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_ratio4")
def tno_silu_simplermsnorm_toep_use_exp_1_ratio4(args):
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
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_ratio4")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head_ratio4(args):
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
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_ratio4")
def tno_silu_simplermsnorm_toep_no_use_exp_1_ratio4(args):
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
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_ratio4")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_ratio4(args):
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
    args.expand_ratio = 4
##### baseline

##### decay
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_decay_ratio4")
def tno_silu_simplermsnorm_toep_use_exp_1_decay_ratio4(args):
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
    args.use_decay = True
    # model
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_decay_ratio4")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head_decay_ratio4(args):
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
    args.use_decay = True
    # model
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_decay_ratio4")
def tno_silu_simplermsnorm_toep_no_use_exp_1_decay_ratio4(args):
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
    args.use_decay = True
    # model
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_decay_ratio4")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_decay_ratio4(args):
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
    args.use_decay = True
    # model
    args.expand_ratio = 4
##### decay
########## big ratio

########## no norm
########## small ratio
##### baseline
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_no_norm")
def tno_silu_simplermsnorm_toep_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_no_norm")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_no_norm")
def tno_silu_simplermsnorm_toep_no_use_exp_1(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_no_norm")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
##### baseline

##### decay
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_decay_no_norm")
def tno_silu_simplermsnorm_toep_use_exp_1_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_decay_no_norm")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = True
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_decay_no_norm")
def tno_silu_simplermsnorm_toep_no_use_exp_1_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_decay_no_norm")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_decay(args):
    base_lm_architecture(args)
    args.act_fun = "silu"
    args.causal = True
    args.decoder_layers = args.decoder_layers * 2
    args.decoder_attention_heads = 1
    # norm
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    # Toeplizt
    args.use_exp = False
    args.toep_type = 1
    args.max_l = 512
    args.use_decay = True
##### decay
########## small ratio

########## big ratio
##### baseline
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_ratio4_no_norm")
def tno_silu_simplermsnorm_toep_use_exp_1_ratio4(args):
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
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_ratio4_no_norm")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head_ratio4(args):
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
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_ratio4_no_norm")
def tno_silu_simplermsnorm_toep_no_use_exp_1_ratio4(args):
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
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_ratio4_no_norm")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_ratio4(args):
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
    args.expand_ratio = 4
##### baseline

##### decay
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_decay_ratio4_no_norm")
def tno_silu_simplermsnorm_toep_use_exp_1_decay_ratio4(args):
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
    # model
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_decay_ratio4_no_norm")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head_decay_ratio4(args):
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
    args.use_decay = True
    # model
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_decay_ratio4_no_norm")
def tno_silu_simplermsnorm_toep_no_use_exp_1_decay_ratio4(args):
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
    # model
    args.expand_ratio = 4
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_decay_ratio4_no_norm")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_decay_ratio4(args):
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
    args.use_decay = True
    # model
    args.expand_ratio = 4
##### decay
########## big ratio
########## no norm

########## forward4
##### baseline
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_forward4")
def tno_silu_simplermsnorm_toep_use_exp_1_forward4(args):
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
    args.expand_ratio = 4
    args.shrink_ratio = 1
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_one_head_forward4")
def tno_silu_simplermsnorm_toep_use_exp_1_one_head_forward4(args):
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
    args.expand_ratio = 4
    args.shrink_ratio = 1
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_forward4")
def tno_silu_simplermsnorm_toep_no_use_exp_1_forward4(args):
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
    args.expand_ratio = 4
    args.shrink_ratio = 1
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_forward4")
def tno_silu_simplermsnorm_toep_no_use_exp_1_one_head_forward4(args):
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
    args.expand_ratio = 4
    args.shrink_ratio = 1
##### baseline
########## forward4

##### 2
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_rate_4_layer_6_dpb_v4_no_norm_no_pos_forward1_large")
def tno_silu_simplermsnorm_toep_use_exp_1_rate_4_layer_6_dpb_v4_no_norm_no_pos_forward1_large(args):
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
    args.expand_ratio = 4
    args.decoder_layers = 6
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_rate_4_layer_6_dpb_v4_no_norm_no_pos_forward1_large_one_head")
def tno_silu_simplermsnorm_toep_use_exp_1_rate_4_layer_6_dpb_v4_no_norm_no_pos_forward1_large_one_head(args):
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
    args.expand_ratio = 4
    args.decoder_layers = 6
    args.decoder_attention_heads = 1
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_rate_4_layer_6_dpb_v4_no_norm_no_pos_forward1_large")
def tno_silu_simplermsnorm_toep_no_use_exp_1_rate_4_layer_6_dpb_v4_no_norm_no_pos_forward1_large(args):
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
    args.expand_ratio = 4
    args.decoder_layers = 6
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_rate_4_layer_6_dpb_v4_no_norm_no_pos_forward1_large_one_head")
def tno_silu_simplermsnorm_toep_no_use_exp_1_rate_4_layer_6_dpb_v4_no_norm_no_pos_forward1_large_one_head(args):
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
    args.expand_ratio = 4
    args.decoder_attention_heads = 1
    args.decoder_layers = 6
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### 2

##### 3 8
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_rate_3_layer_8_dpb_v4_no_norm_no_pos_forward1_large")
def tno_silu_simplermsnorm_toep_use_exp_1_rate_3_layer_8_dpb_v4_no_norm_no_pos_forward1_large(args):
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
    args.decoder_layers = 9
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_use_exp_1_rate_3_layer_8_dpb_v4_no_norm_no_pos_forward1_large_one_head")
def tno_silu_simplermsnorm_toep_use_exp_1_rate_3_layer_8_dpb_v4_no_norm_no_pos_forward1_large_one_head(args):
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
    args.decoder_layers = 8
    args.decoder_attention_heads = 1
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True

@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_rate_3_layer_8_dpb_v4_no_norm_no_pos_forward1_large")
def tno_silu_simplermsnorm_toep_no_use_exp_1_rate_3_layer_8_dpb_v4_no_norm_no_pos_forward1_large(args):
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
    args.decoder_layers = 8
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
    
@register_model_architecture("tno_lm", "tno_silu_simplermsnorm_toep_no_use_exp_1_rate_3_layer_8_dpb_v4_no_norm_no_pos_forward1_large_one_head")
def tno_silu_simplermsnorm_toep_no_use_exp_1_rate_3_layer_8_dpb_v4_no_norm_no_pos_forward1_large_one_head(args):
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
    args.decoder_layers = 8
    # dpb
    args.dynamic_type = 4
    args.dpb_type = 4
    args.dpb_embedding = args.decoder_embed_dim // 4
    # pos
    args.no_token_positional_embeddings = True
##### 3 8