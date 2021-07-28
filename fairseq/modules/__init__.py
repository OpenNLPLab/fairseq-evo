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


# rfa
from .multihead_rfa_attention import MultiheadRfaAttention
from .multihead_rfa_causal_attention import (
    # rfa
    MultiheadRfaCausalAttention,
    # rfa debug
    MultiheadRfaCausalAttentionDebug,
)
from .transformer_rfa_layer import (
    # rfa
    TransformerRfaEncoderLayer,
    TransformerRfaDecoderLayer,
    # rfa debug
    TransformerRfaDecoderDebugLayer,
)
# sparse
from .sparse_multihead_attention import SparseMultiheadAttention
from .sparse_transformer_layer import SparseTransformerEncoderLayer, SparseTransformerDecoderLayer
# linear
from .linear_transformer_attention import MultiheadLinearAttention
from .linear_transformer_layer import LinearTransformerEncoderLayer, LinearTransformerDecoderLayer
# reformer
from .reformer_attention import ReformerAttention_
# lsh
from .lsh_attention import LSHAttention
from .reformer_layer import ReformerEncoderLayer, ReformerDecoderLayer
# Longformer
from .multihead_longformer_attention import LongformerSelfAttention
from .transformer_longformer_layer import (
    TransformerLongformerEncoderLayer, 
    TransformerLongformerDecoderLayer,
    )


# norm rfa
#from .multihead_rfa_causal_attention import MultiheadRfaCausalAttentionNorm
#from .transformer_rfa_layer import TransformerRfaNormDecoderLayer
# debug
#from .multihead_rfa_causal_attention import MultiheadRfaCausalAttentionDebug
#from .transformer_rfa_layer import TransformerRfaDecoderDebugLayer
# performer
from .multihead_performer_attention import MultiheadPerformerAttention
from .performer_layer import PerformerDecoderLayer, PerformerEncoderLayer

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
    # rfa
    "MultiheadRfaCausalAttention",
    "MultiheadRfaAttention",
    "TransformerRfaEncoderLayer", 
    "TransformerRfaDecoderLayer",
    # debug
    "MultiheadRfaCausalAttentionDebug",
    "TransformerRfaDecoderDebugLayer",
    # performer
    "MultiheadPerformerAttention",
    "PerformerEncoderLayer",
    "PerformerDecoderLayer",
    # sparse attention
    "SparseMultiheadAttention",
    "SparseTransformerEncoderLayer",
    "SparseTransformerDecoderLayer",
    # linear attention
    "MultiheadLinearAttention",
    "LinearTransformerEncoderLayer", 
    "LinearTransformerDecoderLayer",
    # reformer
    "ReformerAttention_",
    "ReformerEncoderLayer", 
    "ReformerDecoderLayer",
    "LSHAttention",
     # longformer
    'LongformerSelfAttention',
    'TransformerLongformerEncoderLayer'
    'TransformerLongformerDecoderLayer'
]
