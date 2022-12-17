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

from ..xformer import TransnormerV2Decoder


@register_model("transnormer_v2_lm", dataclass=TransformerLanguageModelConfig)
class TransnormerV2LanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransnormerV2LanguageModel, self).__init__(decoder)

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

        decoder = TransnormerV2Decoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

##### v1
@register_model_architecture("transnormer_v2_lm", "transnormer_v2_lm_t1_prenorm")
def transnormer_v2_lm_t1_prenorm(args):
    base_lm_architecture(args)
    # add
    args.chunk_size = 64
    args.decoder_layers = 3 * args.decoder_layers // 2
    n = args.decoder_layers
    m = n // 2
    args.decoder_attention_types = [2 for _ in range(m)] + [1 for _ in range(n - m)]
    args.norm_type = "layernorm"
    args.final_layernorm = "layernorm"
    args.causal = True
    args.local_act_fun = "relu"
    args.use_softmax = False
    args.linear_act_fun = "elu"
    args.uv_act_fun = "swish"
    args.hidden_dim = 2 * args.decoder_embed_dim

@register_model_architecture("transnormer_v2_lm", "transnormer_v2_lm_t2_prenorm")
def transnormer_v2_lm_t2_prenorm(args):
    base_lm_architecture(args)
    # add
    args.chunk_size = 64
    args.decoder_layers = 3 * args.decoder_layers // 2
    n = args.decoder_layers
    m = n // 2
    args.decoder_attention_types = [2 for _ in range(m)] + [1 for _ in range(n - m)]
    args.norm_type = "layernorm"
    args.final_layernorm = "layernorm"
    args.causal = True
    args.local_act_fun = "relu"
    args.use_softmax = True
    args.linear_act_fun = "1+elu"
    args.uv_act_fun = "swish"
    args.hidden_dim = 2 * args.decoder_embed_dim

@register_model_architecture("transnormer_v2_lm", "transnormer_v2_lm_t1_postnorm")
def transnormer_v2_lm_t1_postnorm(args):
    base_lm_architecture(args)
    # add
    args.chunk_size = 64
    args.decoder_layers = 3 * args.decoder_layers // 2
    n = args.decoder_layers
    m = n // 2
    args.decoder_attention_types = [2 for _ in range(m)] + [1 for _ in range(n - m)]
    args.norm_type = "layernorm"
    args.final_layernorm = "layernorm"
    args.causal = True
    args.local_act_fun = "relu"
    args.use_softmax = False
    args.linear_act_fun = "elu"
    args.uv_act_fun = "swish"
    args.decoder_normalize_before = False
    args.hidden_dim = 2 * args.decoder_embed_dim

@register_model_architecture("transnormer_v2_lm", "transnormer_v2_lm_t2_postnorm")
def transnormer_v2_lm_t2_postnorm(args):
    base_lm_architecture(args)
    # add
    args.chunk_size = 64
    args.decoder_layers = 3 * args.decoder_layers // 2
    n = args.decoder_layers
    m = n // 2
    args.decoder_attention_types = [2 for _ in range(m)] + [1 for _ in range(n - m)]
    args.norm_type = "layernorm"
    args.final_layernorm = "layernorm"
    args.causal = True
    args.local_act_fun = "relu"
    args.use_softmax = True
    args.linear_act_fun = "1+elu"
    args.uv_act_fun = "swish"
    args.decoder_normalize_before = False
    args.hidden_dim = 2 * args.decoder_embed_dim
