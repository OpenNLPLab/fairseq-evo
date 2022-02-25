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
# from .multihead_rfa_attention import MultiheadRfaAttention
# from .multihead_rfa_causal_attention import (
#     # rfa
#     MultiheadRfaCausalAttention,
#     # rfa debug
#     MultiheadRfaCausalAttentionDebug,
# )
# from .transformer_rfa_layer import (
#     # rfa
#     TransformerRfaEncoderLayer,
#     TransformerRfaDecoderLayer,
#     # rfa debug
#     TransformerRfaDecoderDebugLayer,
# )
# sparse
from .sparse_multihead_attention import SparseMultiheadAttention
from .sparse_transformer_layer import SparseTransformerEncoderLayer, SparseTransformerDecoderLayer
# # linear
# from .linear_transformer_attention import MultiheadLinearAttention
# from .linear_transformer_layer import LinearTransformerEncoderLayer, LinearTransformerDecoderLayer
# # reformer
# from .reformer_attention import ReformerAttention_
# # lsh
# from .lsh_attention import LSHAttention
# from .reformer_layer import ReformerEncoderLayer, ReformerDecoderLayer
# # Longformer
# from .multihead_longformer_attention import LongformerSelfAttention
# from .transformer_longformer_layer import (
#     TransformerLongformerEncoderLayer, 
#     TransformerLongformerDecoderLayer,
# )
# taylor
from .multihead_taylor_attention import MultiheadTaylorAttention
from .transformer_taylor_layer import (
    TransformerTaylorDecoderLayer,
    TransformerTaylorEncoderLayer,
)
# sparse relu
from .multihead_sparse_relu_attention import MultiheadSparseReluAttention
from .transformer_sparse_relu_layer import (
    TransformerSparseReluEncoderLayer,
    TransformerSparseReluDecoderLayer,
)

# multi splu
from .multihead_splu_attention import MultiheadSpluAttention
from .transformer_splu_layer import (
    TransformerSpluEncoderLayer,
    TransformerSpluDecoderLayer,
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

# norm rfa
#from .multihead_rfa_causal_attention import MultiheadRfaCausalAttentionNorm
#from .transformer_rfa_layer import TransformerRfaNormDecoderLayer
# debug
#from .multihead_rfa_causal_attention import MultiheadRfaCausalAttentionDebug
#from .transformer_rfa_layer import TransformerRfaDecoderDebugLayer
# performer
# from .multihead_performer_attention import MultiheadPerformerAttention
# from .performer_layer import PerformerDecoderLayer, PerformerEncoderLayer
# transformer merge
from .multihead_merge_attention import MultiheadMergeAttention
from .transformer_merge_layer import TransformerMergeDecoderLayer, TransformerMergeEncoderLayer
# simple attention
from .multihead_simple_attention import MultiheadSimpleAttention
from .transformer_simple_layer import TransformerSimpleEncoderLayer, TransformerSimpleDecoderLayer
# attention with head weight
from .multihead_attention_ import MultiheadAttention_
from .transformer_layer_ import TransformerDecoderLayer_, TransformerEncoderLayer_
# simformer
from .simformer_layer import SimformerDecoderLayer, SimformerEncoderLayer, FFN
# pcc
from .pcc import PccModule
from .pcc_layer import PccEncoderLayer, PccDecoderLayer
# weight
from .multihead_weight_attention import MultiheadWeightAttention
from .transformer_weight_layer import WeightFormerEncoderLayer, WeightFormerDecoderLayer
# weight with diff head
from .multihead_weight_attention_diff import MultiheadWeightAttention_diff
from .transformer_weight_layer_diff import WeightFormerEncoderLayer_diff, WeightFormerDecoderLayer_diff
# GAU
from .flash_attention import FlashAttention
from .flash_layer import GAUEncoderLayer

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
    'TransformerLongformerEncoderLayer',
    'TransformerLongformerDecoderLayer',
    # merge attention
    "MultiheadMergeAttention",
    "TransformerMergeDecoderLayer",
    "TransformerMergeEncoderLayer",
    # simple attention'
    "MultiheadSimpleAttention",
    "TransformerSimpleEncoderLayer", 
    "TransformerSimpleDecoderLayer",
    # attention with weight
    "MultiheadAttention_",
    "TransformerDecoderLayer_", 
    "TransformerEncoderLayer_",
    # simformer
    "SimformerDecoderLayer", 
    "SimformerEncoderLayer",
    "FFN",
    # cos
    "MultiheadCosAttention",
    "TransformerCosEncoderLayer",
    "TransformerCosDecoderLayer",
    # pcc
    "PccModule",
    "PccEncoderLayer", 
    "PccDecoderLayer",
    # weight
    "MultiheadWeightAttention",
    "WeightFormerEncoderLayer", 
    "WeightFormerDecoderLayer",
    "FlashAttention",
    "GAUEncoderLayer",
]
