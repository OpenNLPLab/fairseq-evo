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

from ..xformer import GauDecoder

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
    args.forward_type = "chunk"

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
    args.forward_type = "chunk"

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
    args.forward_type = "chunk"
##### pure block