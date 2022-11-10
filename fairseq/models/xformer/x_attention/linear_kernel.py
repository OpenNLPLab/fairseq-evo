# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (FairseqEncoder, FairseqEncoderDecoderModel,
                            FairseqIncrementalDecoder, register_model,
                            register_model_architecture)
from fairseq.models.transformer import (DEFAULT_MAX_SOURCE_POSITIONS,
                                        DEFAULT_MAX_TARGET_POSITIONS,
                                        DEFAULT_MIN_PARAMS_TO_WRAP,
                                        TransformerDecoder, TransformerEncoder,
                                        TransformerModel, base_architecture)
from fairseq.modules import (AdaptiveInput, CharacterTokenEmbedder,
                             LinearKernelAttentionDecoderLayer,
                             LinearKernelAttentionEncoderLayer)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from omegaconf import II
from torch import Tensor


class LinearKernelAttentionEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

    def build_encoder_layer(self, args):
        layer = LinearKernelAttentionEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

class LinearKernelAttentionDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = LinearKernelAttentionDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def build_encoder_layer(self, args):
        layer = LinearKernelAttentionEncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

@register_model("encoder_linear")
class LinearKernelModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return LinearKernelAttentionEncoder(args, src_dict, embed_tokens)

##### no urpe
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de")
def transformer_1_elu_wmt_en_de(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "1+elu"
##### no urpe

##### urpe
##### Identity
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1_1")
def transformer_1_elu_wmt_en_de_1_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1d_1")
def transformer_1_elu_wmt_en_de_1d_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_2_1")
def transformer_1_elu_wmt_en_de_2_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_3_1")
def transformer_1_elu_wmt_en_de_3_1(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1
##### Identity

##### Householder
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1_3")
def transformer_1_elu_wmt_en_de_1_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1d_3")
def transformer_1_elu_wmt_en_de_1d_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_2_3")
def transformer_1_elu_wmt_en_de_2_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_3_3")
def transformer_1_elu_wmt_en_de_3_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1d_3a")
def transformer_1_elu_wmt_en_de_1d_3a(args):
    base_architecture(args)
    ##### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
##### Householder

##### Odd Even
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1_5")
def transformer_1_elu_wmt_en_de_1_5(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1d_5")
def transformer_1_elu_wmt_en_de_1d_5(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_2_5")
def transformer_1_elu_wmt_en_de_2_5(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_3_5")
def transformer_1_elu_wmt_en_de_3_5(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5
##### Odd Even

##### Fourier
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_4_4")
def transformer_1_elu_wmt_en_de_4_4(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_4d_4")
def transformer_1_elu_wmt_en_de_4d_4(args):
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
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_spe")
def transformer_1_elu_wmt_en_de_spe(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.use_spe = True

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_per")
def transformer_1_elu_wmt_en_de_per(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.use_spe = False
    args.use_permutate = True
##### abl

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_3_3_no_abs")
def transformer_1_elu_wmt_en_de_3_3_no_abs(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3
    ##### add
    args.no_encoder_token_positional_embeddings = True

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1_1_no_abs")
def transformer_1_elu_wmt_en_de_1_1_no_abs(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    ##### add
    args.no_encoder_token_positional_embeddings = True

@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1d_3_no_abs")
def transformer_1_elu_wmt_en_de_1d_3_no_abs(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    ##### add
    args.no_encoder_token_positional_embeddings = True
##### urpe

########## rebuttal
@register_model_architecture("encoder_linear", "relu_wmt_en_de")
def transformer_1_elu_wmt_en_de(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "relu"

@register_model_architecture("encoder_linear", "relu_wmt_en_de_3_3")
def transformer_relu_wmt_en_de_3_3(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "relu"
    args.core_matrix = 3
    args.p_matrix = 3
    
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_nope")
def transformer_1_elu_wmt_en_de_nope(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "1+elu"
    args.no_token_positional_embeddings = True
    
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_1d_3_nope")
def transformer_1_elu_wmt_en_de_1d_3_nope(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    args.no_token_positional_embeddings = True
    
@register_model_architecture("encoder_linear", "1+elu_wmt_en_de_krpe")
def transformer_1_elu_wmt_en_de_krpe(args):
    base_architecture(args)
    ##### add
    args.causal = False
    args.use_urpe = False
    args.kernel_type = "1+elu"
    args.use_krpe = True
    args.max_seq_len = 512
########## rebuttal
