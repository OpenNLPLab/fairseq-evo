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

from ..xformer import TnnDecoder


@register_model("tnn_lm", dataclass=TransformerLanguageModelConfig)
class TnnLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TnnLanguageModel, self).__init__(decoder)

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

        decoder = TnnDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

@register_model_architecture("tnn_lm", "tnn_no_decay")
def tnn_no_decay(args):
    base_lm_architecture(args)
    # model
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.act_fun = "silu"
    args.causal = True
    args.expand_ratio = 3
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    args.use_decay = False
    # rpe
    args.rpe_embedding = 64
    args.rpe_layers = 6
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim

@register_model_architecture("tnn_lm", "tnn_decay_99")
def tnn_decay_99(args):
    base_lm_architecture(args)
    # model
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.act_fun = "silu"
    args.causal = True
    args.expand_ratio = 3
    args.use_norm = False
    args.norm_type = "simplermsnorm"
    args.use_decay = True
    args.gamma = 0.99
    # rpe
    args.rpe_embedding = 64
    args.rpe_layers = 6
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
