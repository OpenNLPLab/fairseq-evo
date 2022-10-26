# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

from ..xformer import GauMixDecoder, GauDecoder

@register_model("gau_mix_lm", dataclass=TransformerLanguageModelConfig)
class GauLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(GauLanguageModel, self).__init__(decoder)

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

        decoder = GauMixDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

########## new version
##### pure block
@register_model_architecture("gau_mix_lm", "gau_mix_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk32")
def gau_mix_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk32(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 32
    args.forward_type = "chunk2"

@register_model_architecture("gau_mix_lm", "gau_mix_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk64")
def gau_mix_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk64(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 64
    args.forward_type = "chunk2"

@register_model_architecture("gau_mix_lm", "gau_mix_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk128")
def gau_mix_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk128(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 128
    args.forward_type = "chunk2"

@register_model_architecture("gau_mix_lm", "gau_mix_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk32")
def gau_mix_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk32(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    args.attention_use_layer_norm = True
    args.norm_act = "relu"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 32
    args.forward_type = "chunk2"

@register_model_architecture("gau_mix_lm", "gau_mix_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk64")
def gau_mix_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk64(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    args.attention_use_layer_norm = True
    args.norm_act = "relu"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 64
    args.forward_type = "chunk2"

@register_model_architecture("gau_mix_lm", "gau_mix_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk128")
def gau_mix_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk128(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    args.attention_use_layer_norm = True
    args.norm_act = "relu"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 128
    args.forward_type = "chunk2"
##### pure block

##### mix
@register_model_architecture("gau_mix_lm", "gau_mix_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_one_head_chunk64")
def gau_mix_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_one_head_chunk64(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    d = args.decoder_layers // 2
    args.decoder_attention_types = [4 for _ in range(d)] + [-4 for _ in range(d)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 64
    args.forward_type = "chunk2"
##### mix
########## new version

########## old version
##### pure block
@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk32")
def gau_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk32(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 32
    args.forward_type = "chunk1"

@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk64")
def gau_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk64(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 64
    args.forward_type = "chunk1"

@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk128")
def gau_lm_v4_simplermsnorm_softmax_urpe_1d3_one_head_chunk128(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 128
    args.forward_type = "chunk1"

@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk32")
def gau_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk32(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    args.attention_use_layer_norm = True
    args.norm_act = "relu"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 32
    args.forward_type = "chunk1"

@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk64")
def gau_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk64(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    args.attention_use_layer_norm = True
    args.norm_act = "relu"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 64
    args.forward_type = "chunk1"

@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk128")
def gau_lm_v4_simplermsnorm_relu_urpe_1d3_one_head_chunk128(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [4 for _ in range(args.decoder_layers)]
    args.decoder_attention_heads = 1
    args.attention_use_layer_norm = True
    args.norm_act = "relu"
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 128
    args.forward_type = "chunk1"
##### pure block

##### mix block
@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_one_head_chunk64")
def gau_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_one_head_chunk64(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    d = args.decoder_layers // 2
    args.decoder_attention_types = [4 for _ in range(d)] + [-4 for _ in range(d)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 64
    args.forward_type = "chunk1"

@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_chunk64")
def gau_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_chunk64(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    d = args.decoder_layers // 2
    args.decoder_attention_types = [4 for _ in range(d)] + [-4 for _ in range(d)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 64
    args.forward_type = "chunk1"

@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_one_head_chunk128")
def gau_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_one_head_chunk128(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    d = args.decoder_layers // 2
    args.decoder_attention_types = [4 for _ in range(d)] + [-4 for _ in range(d)]
    args.decoder_attention_heads = 1
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 128
    args.forward_type = "chunk1"
    
@register_model_architecture("gau_lm", "gau_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_chunk128")
def gau_lm_v4_simplermsnorm_softmax_linear_silu_urpe_1d3_chunk128(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    d = args.decoder_layers // 2
    args.decoder_attention_types = [4 for _ in range(d)] + [-4 for _ in range(d)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ##### block
    args.chunk_size = 128
    args.forward_type = "chunk1"
##### mix block
########## old version