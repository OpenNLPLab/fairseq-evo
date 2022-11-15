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

from ..xformer import TnnV2Decoder


@register_model("tnn_v2_lm", dataclass=TransformerLanguageModelConfig)
class TnnV2LanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TnnV2LanguageModel, self).__init__(decoder)

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

        decoder = TnnV2Decoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

@register_model_architecture("tnn_v2_lm", "tnn_v2_no_decay")
def tnn_v2_no_decay(args):
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

##### norm test
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_post")
def tnn_v2_decay_99_post(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = False
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
    
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre")
def tnn_v2_decay_99_pre(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = True
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
    
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_post_usenorm")
def tnn_v2_decay_99_post(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = False
    # model
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.act_fun = "silu"
    args.causal = True
    args.expand_ratio = 3
    args.use_norm = True
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
    
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_usenorm")
def tnn_v2_decay_99_pre_usenorm(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = True
    # model
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.act_fun = "silu"
    args.causal = True
    args.expand_ratio = 3
    args.use_norm = True
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
##### norm test

##### layer test
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_l6")
def tnn_v2_decay_99_pre_l6(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = True
    # model
    args.decoder_attention_heads = 1
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
##### layer test

##### glu test
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_gluv2")
def tnn_v2_decay_99_pre_gluv2(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = True
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
    args.glu_type = 2
    args.glu_act = "silu"
    args.glu_dim = (3 * args.decoder_embed_dim) // 2
##### glu test

##### v3 test
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_glu_v2_gtu_v3")
def tnn_v2_decay_99_pre_glu_v2_gtu_v3(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = True
    # model
    args.decoder_attention_heads = 1
    args.decoder_layers = 7
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.gtu_type = 3
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
    args.glu_type = 2
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
    
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_glu_v2_gtu_v3_l6")
def tnn_v2_decay_99_pre_glu_v2_gtu_v3_l6(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = True
    # model
    args.decoder_attention_heads = 1
    args.decoder_layers = 6
    # pos
    args.no_token_positional_embeddings = True
    # gtu
    args.gtu_type = 3
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
    args.glu_type = 2
    args.glu_act = "silu"
    args.glu_dim = 3 * args.decoder_embed_dim
##### v3 test

##### large
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_ada")
def tnn_v2_decay_99_pre_ada(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    # gtu
    args.decoder_normalize_before = True
    # model
    args.decoder_attention_heads = 1
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
    
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_ada_128_base")
def tnn_v2_decay_99_pre_ada_128_base(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.decoder_embed_dim = 512
    args.decoder_input_dim = args.decoder_embed_dim
    args.decoder_output_dim = args.decoder_embed_dim
    # gtu
    args.decoder_normalize_before = True
    # model
    args.decoder_attention_heads = 1
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
    args.rpe_embedding = 128
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
    
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_ada_128")
def tnn_v2_decay_99_pre_ada_128(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    # gtu
    args.decoder_normalize_before = True
    # model
    args.decoder_attention_heads = 1
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
    args.rpe_embedding = 128
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
##### large

##### test
@register_model_architecture("tnn_v2_lm", "tnn_v2_decay_99_pre_l1")
def tnn_v2_decay_99_pre_l1(args):
    base_lm_architecture(args)
    args.decoder_normalize_before = True
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
    args.rpe_layers = 1
    args.residual = False
    # glu
    args.glu_act = "silu"
    args.glu_dim = args.decoder_embed_dim
##### test
