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

from ..xformer import TransformerCosDecoder

@register_model("transformer_cos_lm", dataclass=TransformerLanguageModelConfig)
class TransformerCosLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerCosLanguageModel, self).__init__(decoder)

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

        decoder = TransformerCosDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

##### causal lm test
# base
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base")
def transformer_cos_lm_base(args):
    base_lm_architecture(args)
    args.energy_scale = 10
    args.matrix_scale = 1.0
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_1")
def transformer_cos_lm_base_1(args):
    base_lm_architecture(args)
    args.energy_scale = 1
    args.matrix_scale = 1.0
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_01")
def transformer_cos_lm_base_01(args):
    base_lm_architecture(args)
    args.energy_scale = 0.1
    args.matrix_scale = 1.0
# base

@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_nope")
def transformer_cos_lm_base_nope(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.energy_scale = 10
    args.matrix_scale = 1.0
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_nope_random")
def transformer_cos_lm_base_nope_random(args):
    base_lm_architecture(args)
    args.use_toep = True
    args.toep_type = -1
    args.no_token_positional_embeddings = True
    args.energy_scale = 10
    args.matrix_scale = 1.0
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_nope_random_toep")
def transformer_cos_lm_base_nope_random_toep(args):
    base_lm_architecture(args)
    args.use_toep = True
    args.toep_type = 1
    args.no_token_positional_embeddings = True
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_nope_incre_toep")
def transformer_cos_lm_base_nope_incre_toep(args):
    base_lm_architecture(args)
    args.use_toep = True
    args.toep_type = 2
    args.no_token_positional_embeddings = True
    args.energy_scale = 10
    args.matrix_scale = 1.0
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_nope_decre_toep")
def transformer_cos_lm_base_nope_decre_toep(args):
    base_lm_architecture(args)
    args.use_toep = True
    args.toep_type = 3
    args.no_token_positional_embeddings = True

@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_nope_incre_toep_normalize")
def transformer_cos_lm_base_nope_incre_toep_normalize(args):
    base_lm_architecture(args)
    args.use_toep = True
    args.toep_type = 4
    args.no_token_positional_embeddings = True
    args.energy_scale = 10
    args.matrix_scale = 1.0
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_nope_decre_toep_normalize")
def transformer_cos_lm_base_nope_decre_toep_normalize(args):
    base_lm_architecture(args)
    args.use_toep = True
    args.toep_type = 5
    args.no_token_positional_embeddings = True
    args.energy_scale = 10
    args.matrix_scale = 1.0
##### causal lm test

##### post norm
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_postnorm_10")
def transformer_cos_lm_base_postnorm_10(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = False
    args.energy_scale = 10
    args.matrix_scale = 1.0
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_postnorm_1")
def transformer_cos_lm_base_postnorm_1(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = False
    args.energy_scale = 1
    args.matrix_scale = 1.0
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_base_postnorm_01")
def transformer_cos_lm_base_postnorm_01(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = False
    args.energy_scale = 0.1
    args.matrix_scale = 1.0
##### post norm

##### matrix scale test
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_nope")
def transformer_cos_lm_nope(args):
    base_lm_architecture(args)
    args.energy_scale = 0.1
    args.no_token_positional_embeddings = True

@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_nope_random_01_01")
def transformer_cos_lm_nope_random_01_01(args):
    base_lm_architecture(args)
    args.energy_scale = 0.1
    args.matrix_scale = 0.1
    args.use_toep = True
    args.toep_type = -1
    args.no_token_positional_embeddings = True
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_nope_random_01_1")
def transformer_cos_lm_nope_random_01_1(args):
    base_lm_architecture(args)
    args.energy_scale = 0.1
    args.matrix_scale = 1.0
    args.use_toep = True
    args.toep_type = -1
    args.no_token_positional_embeddings = True
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_nope_random_01_10")
def transformer_cos_lm_nope_random_01_10(args):
    base_lm_architecture(args)
    args.energy_scale = 0.1
    args.matrix_scale = 10
    args.use_toep = True
    args.toep_type = -1
    args.no_token_positional_embeddings = True

@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_nope_random_toep_01_01")
def transformer_cos_lm_nope_random_toep_01_01(args):
    base_lm_architecture(args)
    args.energy_scale = 0.1
    args.matrix_scale = 0.1
    args.use_toep = True
    args.toep_type = 1
    args.no_token_positional_embeddings = True
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_nope_random_toep_01_1")
def transformer_cos_lm_nope_random_toep_01_1(args):
    base_lm_architecture(args)
    args.energy_scale = 0.1
    args.matrix_scale = 1.0
    args.use_toep = True
    args.toep_type = 1
    args.no_token_positional_embeddings = True
    
@register_model_architecture("transformer_cos_lm", "transformer_cos_lm_nope_random_toep_01_10")
def transformer_cos_lm_nope_random_toep_01_10(args):
    base_lm_architecture(args)
    args.energy_scale = 0.1
    args.matrix_scale = 10
    args.use_toep = True
    args.toep_type = 1
    args.no_token_positional_embeddings = True
##### matrix scale test