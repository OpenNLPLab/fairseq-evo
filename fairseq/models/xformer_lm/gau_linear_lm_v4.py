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


########## v4
##### base
@register_model_architecture("gau_lm", "gau_linear_lm_v4_denorm_silu")
def gau_linear_lm_v4_denorm_silu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_denorm_silu_single_head")
def gau_linear_lm_v4_denorm_silu_single_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_heads = 1
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_denorm_relu")
def gau_linear_lm_v4_denorm_silu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.act_fun = "relu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_denorm_relu_single_head")
def gau_linear_lm_v4_denorm_silu_single_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = False
    args.act_fun = "relu"
    args.causal = True
    args.decoder_attention_heads = 1
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
##### base

##### head test
@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_1+elu_single_head")
def gau_linear_lm_v4_simplermsnorm_1_elu_single_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.causal = True
    args.decoder_attention_heads = 1
    args.act_fun = "1+elu"
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_elu_single_head")
def gau_linear_lm_v4_simplermsnorm_elu_single_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.decoder_attention_heads = 1
    args.act_fun = "elu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_silu_single_head")
def gau_linear_lm_v4_simplermsnorm_silu_single_head(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.decoder_attention_heads = 1
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
##### head test

##### norm test
@register_model_architecture("gau_lm", "gau_linear_lm_v4_layernorm_1+elu")
def gau_linear_lm_v4_layernorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "layernorm"
    args.causal = True
    args.act_fun = "1+elu"
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_rmsnorm_1+elu")
def gau_linear_lm_v4_rmsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "rmsnorm"
    args.causal = True
    args.act_fun = "1+elu"
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_gatedrmsnorm_1+elu")
def gau_linear_lm_v4_gatedrmsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "gatedrmsnorm"
    args.causal = True
    args.act_fun = "1+elu"
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_1+elu")
def gau_linear_lm_v4_simplermsnorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.causal = True
    args.act_fun = "1+elu"
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_scalenorm_1+elu")
def gau_linear_lm_v4_scalenorm_1_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "scalenorm"
    args.causal = True
    args.act_fun = "1+elu"
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
##### norm test

##### act test
@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_silu")
def gau_linear_lm_v4_simplermsnorm_silu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_elu")
def gau_linear_lm_v4_simplermsnorm_elu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "elu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_relu")
def gau_linear_lm_v4_simplermsnorm_relu(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "relu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_relu2")
def gau_linear_lm_v4_simplermsnorm_relu2(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "relu2"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_sigmoid")
def gau_linear_lm_v4_simplermsnorm_sigmoid(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.act_fun = "sigmoid"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
##### act test

##### urpe test
@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_1+elu_single_head_urpe_1d3")
def gau_linear_lm_v4_simplermsnorm_1_elu_single_head_urpe_1d3(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.causal = True
    args.decoder_attention_heads = 1
    args.act_fun = "1+elu"
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_1+elu_single_head_urpe_1")
def gau_linear_lm_v4_simplermsnorm_1_elu_single_head_urpe_1(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.causal = True
    args.decoder_attention_heads = 1
    args.act_fun = "1+elu"
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_elu_single_head_urpe_1d3")
def gau_linear_lm_v4_simplermsnorm_elu_single_head_urpe_1d3(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.decoder_attention_heads = 1
    args.act_fun = "elu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_elu_single_head_urpe_1")
def gau_linear_lm_v4_simplermsnorm_elu_single_head_urpe_1(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.decoder_attention_heads = 1
    args.act_fun = "elu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_silu_single_head_urpe_1d3")
def gau_linear_lm_v4_simplermsnorm_silu_single_head_urpe_1d3(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.decoder_attention_heads = 1
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("gau_lm", "gau_linear_lm_v4_simplermsnorm_silu_single_head_urpe_1")
def gau_linear_lm_v4_simplermsnorm_silu_single_head_urpe_1(args):
    base_lm_architecture(args)
    args.has_out = True
    args.decoder_layers = 2 * args.decoder_layers
    args.attention_use_layer_norm = True
    args.norm_type = "simplermsnorm"
    args.decoder_attention_heads = 1
    args.act_fun = "silu"
    args.causal = True
    args.decoder_attention_types = [-4 for _ in range(args.decoder_layers)]
    ##### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
##### urpe test
########## v4

