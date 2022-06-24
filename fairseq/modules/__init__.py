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

########## positional encoding
from .positional_encoding import rope
from .positional_encoding import RpeVanilla
from .positional_encoding import SineSPE, ConvSPE, SPEFilter
from .positional_encoding import T5RPE
from .positional_encoding import Urpe
from .positional_encoding import UrpeV2
########## positional encoding

########## norm
from .norm import SimpleRMSNorm, RMSNorm, GatedRMSNorm
########## norm

########## ffn
from .ffn import GLU
########## ffn

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
########## xattention

# sparse
from .sparse_multihead_attention import SparseMultiheadAttention
from .sparse_transformer_layer import SparseTransformerEncoderLayer, SparseTransformerDecoderLayer
# sparse relu
from .multihead_sparse_relu_attention import MultiheadSparseReluAttention
from .transformer_sparse_relu_layer import (
    TransformerSparseReluEncoderLayer,
    TransformerSparseReluDecoderLayer,
)

# from .multihead_cos_attention import MultiheadCosAttention
# from .transformer_cos_layer import (
#     TransformerCosEncoderLayer,
#     TransformerCosDecoderLayer,
# )
# cosformer
from .multihead_cosformer_attention import MultiheadCosformerAttention
from .transformer_cosformer_layer import (
    CosformerEncoderLayer,
    CosformerDecoderLayer,
)


# debug
from .multihead_cosformer_attention_ import MultiheadCosformerAttention_
from .transformer_cosformer_layer_ import (
    CosformerEncoderLayer_,
    CosformerDecoderLayer_,
)

# glu
# from .glu import GLU

# norm rfa
#from .multihead_rfa_causal_attention import MultiheadRfaCausalAttentionNorm
#from .transformer_rfa_layer import TransformerRfaNormDecoderLayer
# debug
#from .multihead_rfa_causal_attention import MultiheadRfaCausalAttentionDebug
#from .transformer_rfa_layer import TransformerRfaDecoderDebugLayer
# performer
# from .multihead_performer_attention import PerformerAttention
# from .performer_layer import PerformerDecoderLayer, PerformerEncoderLayer
# transformer merge
from .multihead_merge_attention import MultiheadMergeAttention
from .transformer_merge_layer import TransformerMergeDecoderLayer, TransformerMergeEncoderLayer
# simple attention
from .multihead_simple_attention import MultiheadSimpleAttention



# norm mix
from .norm_mix_attention import NormMixAttention
from .norm_mix_layer import NormMixAttentionDecoderLayer, NormMixAttentionEncoderLayer


# norm
from .norm_linear_attention import NormLinearAttention
from .norm_local_attention import NormLocalAttention
from .norm_attention_layer import NormAttentionDecoderLayer, NormAttentionEncoderLayer


# another version
# from .ls_attention import LongShortAttention
# long short attention
# from .ls_attention_causal import LSCausalAttention
# from .ls_attention_non_causal import LSAttentionNonCausal
# from .ls_attention_layer import LSAttentionEncoderLayer, LSAttentionDecoderLayer
# from .ls_attention_causal_model import TransformerLSModel

from .linear_vanilla_attention_layer import LinearVanillaEncoderLayer

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
