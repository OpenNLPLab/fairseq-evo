# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (FairseqEncoder, FairseqEncoderModel,
                            register_model, register_model_architecture)
from fairseq.models.roberta import (RobertaEncoder, RobertaModel,
                                    base_architecture)
from fairseq.models.transformer import (DEFAULT_MIN_PARAMS_TO_WRAP,
                                        TransformerEncoder)
from fairseq.models.xformer import LinearKernelAttentionEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from numpy import False_


class RobertaLinearKernelEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = LinearKernelAttentionEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_linear_kernel")
class RobertaLinearKernelModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaLinearKernelEncoder(args, task.source_dictionary)
        return cls(args, encoder)

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu")
def roberta_base_architecture_1_elu(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "1+elu"

@register_model_architecture("roberta_linear_kernel", "roberta_elu")
def roberta_base_elu_architecture_elu(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "elu"

########## urpe
##### 单位阵
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1_1")
def roberta_base_architecture_1_elu_1_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1b_1")
def roberta_base_architecture_1_elu_1b_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1c_1")
def roberta_base_architecture_1_elu_1c_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1d_1")
def roberta_base_architecture_1_elu_1d_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_2_1")
def roberta_base_architecture_1_elu_2_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_3_1")
def roberta_base_architecture_1_elu_3_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1
##### 单位阵

##### rope
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_rope")
def roberta_base_architecture_1_elu_rope(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "1+elu"
    args.use_rope = True
##### rope

##### Odd Even
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1_5")
def roberta_base_architecture_1_elu_1_5(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1d_5")
def roberta_base_architecture_1_elu_1d_5(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_2_5")
def roberta_base_architecture_1_elu_2_5(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_3_5")
def roberta_base_architecture_1_elu_3_5(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5
##### Odd Even

##### DCT
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1_2")
def roberta_base_architecture_1_elu_1_2(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1b_2")
def roberta_base_architecture_1_elu_1b_2(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1c_2")
def roberta_base_architecture_1_elu_1c_2(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "c"

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1d_2")
def roberta_base_architecture_1_elu_1d_2(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_learned = True

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_2_2")
def roberta_base_architecture_1_elu_2_2(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_3_2")
def roberta_base_architecture_1_elu_3_2(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 2
##### DCT

##### Householder
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1_3")
def roberta_base_architecture_1_elu_1_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1b_3")
def roberta_base_architecture_1_elu_1b_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_type = "b"

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1c_3")
def roberta_base_architecture_1_elu_1c_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_type = "c"

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1d_3")
def roberta_base_architecture_1_elu_1d_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_2_3")
def roberta_base_architecture_1_elu_2_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_3_3")
def roberta_base_architecture_1_elu_3_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3
##### Householder

##### Householder learned
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1_3a")
def roberta_base_architecture_1_elu_1_3a(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1d_3a")
def roberta_base_architecture_1_elu_1d_3a(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
##### Householder learned

##### Fourier
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_4_4")
def roberta_base_architecture_1_elu_4_4(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_4d_4")
def roberta_base_architecture_1_elu_4d_4(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
##### Fourier

##### abl
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_spe")
def roberta_base_architecture_1_elu_spe(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "1+elu"
    args.use_spe = True

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_per")
def roberta_base_architecture_1_elu_per(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "1+elu"
    args.use_spe = False
    args.use_permutate = True
##### abl

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_one_head")
def roberta_base_architecture_1_elu_one_head(args):
    base_architecture(args)
    ##### add
    args.encoder_attention_heads = 1

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1_1_no_abs")
def roberta_base_architecture_1_elu_1_1_no_abs(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    ##### add
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_1d_3_no_abs")
def roberta_base_architecture_1_elu_1d_3_no_abs(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    ##### add
    args.no_token_positional_embeddings = True

##### rebuttal experiments
@register_model_architecture("roberta_linear_kernel", "roberta_relu")
def roberta_base_architecture_relu(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "relu"

@register_model_architecture("roberta_linear_kernel", "roberta_relu_1d_1")
def roberta_base_architecture_relu_1d_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "relu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True
##### rebuttal experiments

########## rebuttal
@register_model_architecture("roberta_linear_kernel", "roberta_1+elu_nope")
def roberta_base_architecture_1_elu_nope(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "1+elu"
    args.no_token_positional_embeddings = True
########## rebuttal
