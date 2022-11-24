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

from ..xformer import WeightLinearDecoder


@register_model("weight_linear_lm", dataclass=TransformerLanguageModelConfig)
class WeightLinearLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(WeightLinearLanguageModel, self).__init__(decoder)

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

        decoder = WeightLinearDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

########## relu
########## no norm
##### baseline
@register_model_architecture("weight_linear_lm", "weight_linear_relu")
def weight_linear_relu(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "relu"
    args.weight_type = -1
##### baseline

@register_model_architecture("weight_linear_lm", "weight_linear_relu_cos")
def weight_linear_relu_cos(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "relu"
    args.weight_type = 1
    
@register_model_architecture("weight_linear_lm", "weight_linear_relu_quad")
def weight_linear_relu_quad(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "relu"
    args.weight_type = 2
    
@register_model_architecture("weight_linear_lm", "weight_linear_relu_quad_sigmoid")
def weight_linear_relu_quad_sigmoid(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "relu"
    args.weight_type = 2
    args.use_sigmoid = True
########## no norm

########## norm
##### baseline
@register_model_architecture("weight_linear_lm", "weight_linear_relu_norm")
def weight_linear_relu_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "relu"
    args.weight_type = -1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
##### baseline

@register_model_architecture("weight_linear_lm", "weight_linear_relu_cos_norm")
def weight_linear_relu_cos_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "relu"
    args.weight_type = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    
@register_model_architecture("weight_linear_lm", "weight_linear_relu_quad_norm")
def weight_linear_relu_quad_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "relu"
    args.weight_type = 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    
@register_model_architecture("weight_linear_lm", "weight_linear_relu_quad_norm_sigmoid")
def weight_linear_relu_quad_norm_sigmoid(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "relu"
    args.weight_type = 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    args.use_sigmoid = True
########## norm
########## relu

########## 1+elu
########## no norm
##### baseline
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu")
def weight_linear_1_elu(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = -1
##### baseline

@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_cos")
def weight_linear_1_elu_cos(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 1
    
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_quad")
def weight_linear_1_elu_quad(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 2
    
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_quad_sigmoid")
def weight_linear_1_elu_quad_sigmoid(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 2
    args.use_sigmoid = True
########## no norm

########## norm
##### baseline
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_norm")
def weight_linear_1_elu_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = -1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
##### baseline

@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_cos_norm")
def weight_linear_1_elu_cos_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 1
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_quad_norm")
def weight_linear_1_elu_quad_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_quad_norm_sigmoid")
def weight_linear_1_elu_quad_norm_sigmoid(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 2
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
    args.use_sigmoid = True
########## norm
########## 1+elu

########## 1+elu
##### weight 3
##### no norm
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_laplace_legendre_no_norm")
def weight_linear_1_elu_laplace_legendre_no_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 3
    
##### with norm
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_laplace_legendre_with_norm")
def weight_linear_1_elu_laplace_legendre_with_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 3
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
##### weight 3

##### weight 4
##### no norm
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_laplace_fft_no_norm")
def weight_linear_1_elu_laplace_fft_no_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 4
    
##### with norm
@register_model_architecture("weight_linear_lm", "weight_linear_1+elu_laplace_fft_with_norm")
def weight_linear_1_elu_laplace_fft_with_norm(args):
    base_lm_architecture(args)
    args.causal = True
    args.act_fun = "1+elu"
    args.weight_type = 4
    # norm
    args.use_norm = True
    args.norm_type = "simplermsnorm"
##### weight 4
########## 1+elu

########## cos(q - k)
##### with denom
@register_model_architecture("weight_linear_lm", "cos_qk_denom")
def cos_qk_denom(args):
    base_lm_architecture(args)
    args.causal = True
    args.weight_type = 5
    # norm
    args.use_norm = False
    args.cos_prenorm = False
    args.cos_postnorm = False
    
@register_model_architecture("weight_linear_lm", "cos_normqk_denom")
def cos_normqk_denom(args):
    base_lm_architecture(args)
    args.causal = True
    args.weight_type = 5
    # norm
    args.use_norm = False
    args.cos_prenorm = True
    args.cos_postnorm = False
##### with denom

##### without denom
@register_model_architecture("weight_linear_lm", "cos_qk_nodenom")
def cos_qk_nodenom(args):
    base_lm_architecture(args)
    args.causal = True
    args.weight_type = 5
    # norm
    args.use_norm = True
    args.cos_prenorm = False
    args.cos_postnorm = True
    
@register_model_architecture("weight_linear_lm", "cos_normqk_nodenom")
def cos_normqk_nodenom(args):
    base_lm_architecture(args)
    args.causal = True
    args.weight_type = 5
    # norm
    args.use_norm = True
    args.cos_prenorm = True
    args.cos_postnorm = True
    
@register_model_architecture("weight_linear_lm", "cos_normqk_nodenom_onehead")
def cos_normqk_nodenom_onehead(args):
    base_lm_architecture(args)
    args.causal = True
    args.weight_type = 5
    # norm
    args.use_norm = True
    args.cos_prenorm = True
    args.cos_postnorm = True
    args.decoder_attention_heads = 1
##### without denom

##### without denom and without postnorm
@register_model_architecture("weight_linear_lm", "cos_qk_nodenom_nopost")
def cos_qk_nodenom_nopost(args):
    base_lm_architecture(args)
    args.causal = True
    args.weight_type = 5
    # norm
    # no denorm
    args.use_norm = True
    args.cos_prenorm = False
    # no postnorm
    args.cos_postnorm = False
    
@register_model_architecture("weight_linear_lm", "cos_normqk_nodenom_nopost")
def cos_normqk_nodenom_nopost(args):
    base_lm_architecture(args)
    args.causal = True
    args.weight_type = 5
    # norm
    # no denorm
    args.use_norm = True
    args.cos_prenorm = True
    # no postnorm
    args.cos_postnorm = False
##### without denom and without postnorm
########## cos(q - k)
