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

from ..xformer import TransformerRpeDecoder


@register_model("transformer_rpe_lm", dataclass=TransformerLanguageModelConfig)
class TransformerRpeLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerRpeLanguageModel, self).__init__(decoder)

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

        decoder = TransformerRpeDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe1")
def transformer_lm_base_rpe1(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 1
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe2")
def transformer_lm_base_rpe2(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 2
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe3")
def transformer_lm_base_rpe3(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 3
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe4")
def transformer_lm_base_rpe4(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 4
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe5")
def transformer_lm_base_rpe5(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 5
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe6")
def transformer_lm_base_rpe6(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 6
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe7")
def transformer_lm_base_rpe7(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 7
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe8")
def transformer_lm_base_rpe8(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 8
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe9")
def transformer_lm_base_rpe9(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 9
    
@register_model_architecture("transformer_rpe_lm", "transformer_lm_base_rpe10")
def transformer_lm_base_rpe10(args):
    base_lm_architecture(args)
    args.no_token_positional_embeddings = True
    args.rpe_type = 10