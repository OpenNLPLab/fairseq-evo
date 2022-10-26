# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .adaptive_input import AdaptiveInput
from .adaptive_softmax import AdaptiveSoftmax
from .base_layer import BaseLayer
from .beamable_mm import BeamableMM
from .character_token_embedder import CharacterTokenEmbedder
from .conv_tbc import ConvTBC
from .cross_entropy import cross_entropy
from .downsampled_multihead_attention import DownsampledMultiHeadAttention
from .dynamic_convolution import DynamicConv, DynamicConv1dTBC
from .dynamic_crf_layer import DynamicCRF
from .fairseq_dropout import FairseqDropout
from .fp32_group_norm import Fp32GroupNorm
from .gelu import gelu, gelu_accurate
from .grad_multiply import GradMultiply
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .kmeans_vector_quantizer import KmeansVectorQuantizer
from .layer_drop import LayerDropModuleList
from .layer_norm import Fp32LayerNorm, LayerNorm
from .learned_positional_embedding import LearnedPositionalEmbedding
from .lightweight_convolution import LightweightConv, LightweightConv1dTBC
from .linearized_convolution import LinearizedConvolution
from .multihead_attention import MultiheadAttention
from .positional_embedding import PositionalEmbedding
from .same_pad import SamePad
from .scalar_bias import ScalarBias
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .transformer_sentence_encoder_layer import TransformerSentenceEncoderLayer
from .transformer_sentence_encoder import TransformerSentenceEncoder
from .transpose_last import TransposeLast
from .unfold import unfold1d
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer
from .vggblock import VGGBlock

##########
from .helpers import logging_info, print_params, print_config
##########

########## positional encoding
from .positional_encoding import rope
from .positional_encoding import RpeVanilla
from .positional_encoding import SineSPE, ConvSPE, SPEFilter
from .positional_encoding import T5RPE
from .positional_encoding import Urpe
from .positional_encoding import UrpeV2
from .positional_encoding import Toeplizt
from .positional_encoding import ToepliztV2
from .positional_encoding import ToepliztV3
from .positional_encoding import ToepliztV4
from .positional_encoding import ToepliztMultihead
from .positional_encoding import DynamicPosBias
from .positional_encoding import DynamicPosBiasV2
from .positional_encoding import DynamicPosBiasV3
from .positional_encoding import DynamicPosBiasV4
from .positional_encoding import DynamicToepliztMultihead
from .positional_encoding import DynamicToepliztMultiheadV2
from .positional_encoding import DynamicToepliztMultiheadV3
from .positional_encoding import DynamicToepliztMultiheadV4
from .positional_encoding import NonDynamicToepliztMultihead
########## positional encoding

########## norm
from .norm import SimpleRMSNorm, RMSNorm, GatedRMSNorm, ScaleNorm, OffsetScale
########## norm

########## token shift
from .token_shift import TokenShift
from .token_shift import ConvMix
########## token shift

########## ffn
from .ffn import GLU
########## ffn

########## others
from .others import SEBlock
########## others

########## token mixer test
##### double fusion
from .token_mixer_test import DoubleFusion
from .token_mixer_test import DoubleFusionEncoderLayer, DoubleFusionDecoderLayer
from .token_mixer_test import DoubleFusionV2
from .token_mixer_test import DoubleFusionV2EncoderLayer, DoubleFusionV2DecoderLayer
from .token_mixer_test import DoubleFusionV3
from .token_mixer_test import DoubleFusionV3EncoderLayer, DoubleFusionV3DecoderLayer
from .token_mixer_test import DoubleFusionQuad
from .token_mixer_test import DoubleFusionQuadEncoderLayer, DoubleFusionQuadDecoderLayer
##### double fusion
##### gau
from .token_mixer_test import GauQuad
from .token_mixer_test import GauQuadV2
from .token_mixer_test import GauQuadV3
from .token_mixer_test import GauEncoderLayer, GauDecoderLayer
##### gau
########## token mixer test

########## xattention
##### linearkernel
from .x_attention import LinearKernelAttention
from .x_attention import LinearKernelAttentionEncoderLayer, LinearKernelAttentionDecoderLayer
##### performer
from .x_attention import PerformerAttention
from .x_attention import PerformerEncoderLayer, PerformerDecoderLayer
##### flash
from .x_attention import FlashQuadAttention
from .x_attention import FlashQuadEncoderLayer, FlashQuadDecoderLayer
from .x_attention import FlashLinearAttention
from .x_attention import FlashLinearEncoderLayer, FlashLinearDecoderLayer
##### mha plus
from .x_attention import MultiheadAttentionPlus
from .x_attention import TransformerEncoderLayerPlus, TransformerDecoderLayerPlus
##### long short attention
from .x_attention import LSCausalAttention
from .x_attention import LSNonCausalAttention
from .x_attention import LSAttentionEncoderLayer
from .x_attention import TransformerLSModel
##### ReLA
from .x_attention import ReLAttention
from .x_attention import ReLAEncoderLayer, ReLADecoderLayer
##### cosformer
from .x_attention import CosformerAttention
from .x_attention import CosformerEncoderLayer, CosformerDecoderLayer
##### norm mix attention
from .x_attention import NormMixAttention
from .x_attention import NormMixAttentionDecoderLayer, NormMixAttentionEncoderLayer
##### norm local/linear attention
from .x_attention import NormLinearAttention
from .x_attention import NormLocalAttention
from .x_attention import NormAttentionDecoderLayer, NormAttentionEncoderLayer
##### linear combination
from .x_attention import LinearCombinationEncoderLayer
##### sparse attetntioin
from .x_attention import SparseMultiheadAttention
from .x_attention import SparseTransformerEncoderLayer, SparseTransformerDecoderLayer
##### sparse attetntioin
##### toeplitz attention
from .x_attention import ToeplitzAttention
from .x_attention import ToeplitzAttentionEncoderLayer, ToeplitzAttentionDecoderLayer
##### toeplitz attention
##### weight linear
from .x_attention import WeightLinearAttention
from .x_attention import WeightLinearEncoderLayer, WeightLinearDecoderLayer
##### weight linear
##### tno
from .x_attention import TNO
from .x_attention import TNOEncoderLayer, TNODecoderLayer
from .x_attention import TNOFFNEncoderLayer, TNOFFNDecoderLayer
from .x_attention import TNOGLUEncoderLayer, TNOGLUDecoderLayer
##### tno
########## xattention

########## state space
##### GSS
from .state_space import GSS
from .state_space import GSSEncoderLayer, GSSDecoderLayer
##### GSS
##### DSS
from .state_space import DSS
from .state_space import DSSEncoderLayer, DSSDecoderLayer
##### DSS
##### S4
from .state_space import S4EncoderLayer, S4DecoderLayer
##### S4
########## state space

########## fourier
##### fnet
from .fourier import FNetEncoderLayer, FNetDecoderLayer
##### fnet
##### afno
from .fourier import AFNOEncoderLayer, AFNODecoderLayer
##### afno
##### gfn
from .fourier import GlobalFilterEncoderLayer, GlobalFilterDecoderLayer
##### gfn
########## fourier

########## mlp
##### gmlp
from .mlp import GMLPEncoderLayer, GMLPDecoderLayer
##### gmlp
##### Synthesizer
from .mlp import Synthesizer
from .mlp import SynthesizerEncoderLayer, SynthesizerDecoderLayer
##### Synthesizer
########## mlp

##### cos attention
from .x_attention import MultiheadCosAttention
from .x_attention import TransformerCosEncoderLayer, TransformerCosDecoderLayer
##### cos attention

########## sparse
# from .sparse_multihead_attention import SparseMultiheadAttention
# from .sparse_transformer_layer import SparseTransformerEncoderLayer, SparseTransformerDecoderLayer
########## sparse relu


__all__ = [
    "AdaptiveInput",
    "AdaptiveSoftmax",
    "BaseLayer",
    "BeamableMM",
    "CharacterTokenEmbedder",
    "ConvTBC",
    "cross_entropy",
    "DownsampledMultiHeadAttention",
    "DynamicConv1dTBC",
    "DynamicConv",
    "DynamicCRF",
    "FairseqDropout",
    "Fp32GroupNorm",
    "Fp32LayerNorm",
    "gelu",
    "gelu_accurate",
    "GradMultiply",
    "GumbelVectorQuantizer",
    "KmeansVectorQuantizer",
    "LayerDropModuleList",
    "LayerNorm",
    "LearnedPositionalEmbedding",
    "LightweightConv1dTBC",
    "LightweightConv",
    "LinearizedConvolution",
    "MultiheadAttention",
    "PositionalEmbedding",
    "SamePad",
    "ScalarBias",
    "SinusoidalPositionalEmbedding",
    "TransformerSentenceEncoderLayer",
    "TransformerSentenceEncoder",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "TransposeLast",
    "VGGBlock",
    "unfold1d",
]
