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
from omegaconf import II

from fairseq.models.transformer import (DEFAULT_MIN_PARAMS_TO_WRAP, Embedding,
                                        TransformerDecoder)
from fairseq.models.transformer_lm import (DEFAULT_MAX_TARGET_POSITIONS,
                                           TransformerLanguageModel,
                                           TransformerLanguageModelConfig,
                                           base_lm_architecture,
                                           transformer_lm_big)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder

from ..xformer import TransformerSpeDecoder


@register_model("transformer_spe_lm", dataclass=TransformerLanguageModelConfig)
class TransformerSpeLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerSpeLanguageModel, self).__init__(decoder)

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

        decoder = TransformerSpeDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_sincos_m1")
def transformer_lm_spe_sincos_m1(args):
    base_lm_architecture(args)
    args.max_seq = 512
    args.method = 1
    
@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_learned_m1")
def transformer_lm_spe_learned_m1(args):
    base_lm_architecture(args)
    args.decoder_learned_pos = True
    args.max_seq = 512
    args.method = 1
    
@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_sincos_m1_penorm")
def transformer_lm_spe_sincos_m1_penorm(args):
    base_lm_architecture(args)
    args.max_seq = 512
    args.method = 1
    args.use_penorm = True
    
@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_learned_m1_penorm")
def transformer_lm_spe_learned_m1_penorm(args):
    base_lm_architecture(args)
    args.decoder_learned_pos = True
    args.max_seq = 512
    args.method = 1
    args.use_penorm = True
    
@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_sincos_m1_penorm_token_norm")
def transformer_lm_spe_sincos_m1_penorm_token_norm(args):
    base_lm_architecture(args)
    args.max_seq = 512
    args.method = 1
    args.use_penorm = True
    args.use_token_norm = True
    
@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_learned_m1_penorm_token_norm")
def transformer_lm_spe_learned_m1_penorm_token_norm(args):
    base_lm_architecture(args)
    args.decoder_learned_pos = True
    args.max_seq = 512
    args.method = 1
    args.use_penorm = True
    args.use_token_norm = True

@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_sincos_m2")
def transformer_lm_spe_sincos_m2(args):
    base_lm_architecture(args)
    args.max_seq = 512
    args.method = 2
    
@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_learned_m2")
def transformer_lm_spe_learned_m2(args):
    base_lm_architecture(args)
    args.decoder_learned_pos = True
    args.max_seq = 512
    args.method = 2
    
@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_sincos_m3")
def transformer_lm_spe_sincos_m3(args):
    base_lm_architecture(args)
    args.max_seq = 512
    args.method = 3
    
@register_model_architecture("transformer_spe_lm", "transformer_lm_spe_learned_m3")
def transformer_lm_spe_learned_m3(args):
    base_lm_architecture(args)
    args.decoder_learned_pos = True
    args.max_seq = 512
    args.method = 3