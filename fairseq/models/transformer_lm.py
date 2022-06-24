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
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, TransformerDecoder, TransformerLongformerDecoder, 
    # # rfa
    # TransformerRfaDecoder, 
    # performer
    PerformerDecoder, 
    # # debug
    # TransformerRfaDebugDecoder,
    # # sparse transformer
    # SparseTransformerDecoder,
    # # linear transformer
    # LinearTransformerDecoder,
    # # reformer
    # ReformerDecoder,
    # # longformer
    # TransformerLongformerDecoder,
    # transformer merge
    TransformerMergeDecoder,
    # transformer simple
    # head
    TransformerDecoderPlus,
    # splu
    TransformerSparseReluDecoder,
    # cosformer
    CosformerDecoder,
    CosformerSoftmaxDecoder,
    CosformerDecoder_,
    # Mem
    # MemDecoder,
    # MemGauDecoder,
    # rela
    ReLADecoder,
    # Flash
    FlashQuadDecoder,
    FlashLinearDecoder,
    # gmu 
    # GmuDecoder,
    # linear kernel with urpe
    LinearKernelAttentionDecoder,
    # norm attention
    NormAttentionDecoder,
    # norm mix attention
    NormMixAttentionDecoder,
    # ls attention
    # LSAttentionDecoder,
)
from fairseq.modules import TransformerLSModel

# simple transformer
# from fairseq.models.simformer import SimformerEncoder, SimformerDecoder  

from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II
from typing import Dict, List, Optional
import torch


DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class TransformerLanguageModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    character_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, uses character embedding convolutions to produce token embeddings"
        },
    )
    character_filters: str = field(
        default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
        metadata={"help": "size of character embeddings"},
    )
    character_embedding_dim: int = field(
        default=4, metadata={"help": "size of character embeddings"}
    )
    char_embedder_highway_layers: int = field(
        default=2,
        metadata={"help": "number of highway layers for character token embeddder"},
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_input_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    adaptive_input_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive input cutoff points."},
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        }
    )
    # config for "BASE Layers: Simplifying Training of Large, Sparse Models"
    base_layers: Optional[int] = field(
        default=0, metadata={"help": "number of BASE layers in total"}
    )
    base_sublayers: Optional[int] = field(
        default=1, metadata={"help": "number of sublayers in each BASE layer"}
    )
    base_shuffle: Optional[int] = field(
        default=1, metadata={"help": "shuffle tokens between workers before computing assignment"}
    )
    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")
    # add max_l
    # max_l: Optional[int] = field(
    #     default=512, metadata={"help": "max_l"}
    # )


@register_model("transformer_lm", dataclass=TransformerLanguageModelConfig)
class TransformerLanguageModel(FairseqLanguageModel):
    @classmethod
    def hub_models(cls):
        def moses_fastbpe(path):
            return {"path": path, "tokenizer": "moses", "bpe": "fastbpe"}

        def spm(path):
            return {"path": path, "tokenizer": "space", "bpe": "sentencepiece"}

        return {
            "transformer_lm.gbw.adaptive_huge": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2",
            "transformer_lm.wiki103.adaptive": "https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2",
            "transformer_lm.wmt19.en": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2"
            ),
            "transformer_lm.wmt19.de": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2"
            ),
            "transformer_lm.wmt19.ru": moses_fastbpe(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2"
            ),
            "transformer_lm.wmt20.en": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.en.tar.gz"
            ),
            "transformer_lm.wmt20.ta": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.ta.tar.gz"
            ),
            "transformer_lm.wmt20.iu.news": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.iu.news.tar.gz"
            ),
            "transformer_lm.wmt20.iu.nh": spm(
                "https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt20.iu.nh.tar.gz"
            ),
        }

    def __init__(self, decoder):
        super().__init__(decoder)

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

        decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens

# add for rfa
@register_model("transformer_rfa_lm", dataclass=TransformerLanguageModelConfig)
class TransformerRfaLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerRfaLanguageModel, self).__init__(decoder)

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

        decoder = TransformerRfaDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# rfa debug
@register_model("transformer_rfa_debug_lm", dataclass=TransformerLanguageModelConfig)
class TransformerRfaDebugLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerRfaDebugLanguageModel, self).__init__(decoder)

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

        decoder = TransformerRfaDebugDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for performer
@register_model("performer_lm", dataclass=TransformerLanguageModelConfig)
class PerformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(PerformerLanguageModel, self).__init__(decoder)

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

        decoder = PerformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for sparse transformer
@register_model("sparse_transformer_lm", dataclass=TransformerLanguageModelConfig)
class SparseTransformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(SparseTransformerLanguageModel, self).__init__(decoder)

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

        decoder = SparseTransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for linear transformer
@register_model("linear_transformer_lm", dataclass=TransformerLanguageModelConfig)
class LinearTransformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(LinearTransformerLanguageModel, self).__init__(decoder)

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

        decoder = LinearTransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for reformer
@register_model("reformer_lm", dataclass=TransformerLanguageModelConfig)
class ReformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(ReformerLanguageModel, self).__init__(decoder)

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

        decoder = ReformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for transformer merge
@register_model("transformer_merge_lm", dataclass=TransformerLanguageModelConfig)
class TransformerMergeLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerMergeLanguageModel, self).__init__(decoder)

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

        decoder = TransformerMergeDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for head
@register_model("transformer_head_lm", dataclass=TransformerLanguageModelConfig)
class TransformerHeadLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerHeadLanguageModel, self).__init__(decoder)

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

        decoder = TransformerDecoderPlus(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)


def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.base_layers = getattr(args, "base_layers", 0)
    args.base_sublayers = getattr(args, "base_sublayers", 1)
    args.base_shuffle = getattr(args, "base_shuffle", False)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True

# add for longformer
@register_model("transformer_longformer_lm", dataclass=TransformerLanguageModelConfig)
class TransformerLongformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerLongformerLanguageModel, self).__init__(decoder)
 
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
 
        decoder = TransformerLongformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

@register_model("cosformer_lm", dataclass=TransformerLanguageModelConfig)
class CosformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(CosformerLanguageModel, self).__init__(decoder)

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

        decoder = CosformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# cosformer_
@register_model("cosformer_lm_", dataclass=TransformerLanguageModelConfig)
class CosformerLanguageModel_(TransformerLanguageModel):
    def __init__(self, decoder):
        super(CosformerLanguageModel_, self).__init__(decoder)

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

        decoder = CosformerDecoder_(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# cosformer + softmax
@register_model("cosformer_softmax_lm", dataclass=TransformerLanguageModelConfig)
class CosformerSoftmaxLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(CosformerSoftmaxLanguageModel, self).__init__(decoder)

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

        decoder = CosformerSoftmaxDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)


@register_model("rela_lm", dataclass=TransformerLanguageModelConfig)
class ReLALanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(ReLALanguageModel, self).__init__(decoder)

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

        decoder = ReLADecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# flash
@register_model("flash_lm", dataclass=TransformerLanguageModelConfig)
class FlashLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(FlashLanguageModel, self).__init__(decoder)

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

        decoder = FlashQuadDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# flash linear 
@register_model("flash_linear_lm", dataclass=TransformerLanguageModelConfig)
class FlashLinearLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(FlashLinearLanguageModel, self).__init__(decoder)

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

        decoder = FlashLinearDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

################################################################
# GmuDecoder
@register_model("gmu_lm", dataclass=TransformerLanguageModelConfig)
class GmuLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(GmuLanguageModel, self).__init__(decoder)

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

        decoder = GmuDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

################################################################
# LinearKernelAttentionDecoder with urpe
@register_model("linear_urpe_lm", dataclass=TransformerLanguageModelConfig)
class LKOrLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(LKOrLanguageModel, self).__init__(decoder)

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

        decoder = LinearKernelAttentionDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

################################################################
def small_init_weights(module):
    if isinstance(module, (nn.Linear)):
        module.weight.data.normal_(mean=0.0, std=0.02)        

    if isinstance(module, (nn.Embedding)):
        # nn.init.uniform_(module.weight, a=-1e-4, b=1e-4) # SmallInit(Emb)
        print("Embdding norm before")
        print(torch.norm(module.weight.data))
        module.weight.data.normal_(mean=0.0, std=1e-5)
        print("Embdding norm after")
        print(torch.norm(module.weight.data))

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    
# NormAttentionDecoder
@register_model("norm_attention_lm", dataclass=TransformerLanguageModelConfig)
class NormAttentionLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(NormAttentionLanguageModel, self).__init__(decoder)

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

        init_method = getattr(args, 'init_method', "default")
        print(f"init_method {init_method}")
        if init_method != "default":
            print("small init")
            embed_tokens.apply(small_init_weights)

        decoder = NormAttentionDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)



# NormMixAttentionDecoder
@register_model("norm_mix_attention_lm", dataclass=TransformerLanguageModelConfig)
class NormMixAttentionLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(NormMixAttentionLanguageModel, self).__init__(decoder)

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

        decoder = NormMixAttentionDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# LSAttentionDecoder
@dataclass
class TransformerLSConfig(FairseqDataclass):
    # defaults come from https://github.com/facebookresearch/adaptive-span/blob/master/experiments/enwik8_small.sh
    vocab_size: int = 50
    d_model: int = 256
    n_head: int = 4
    d_inner: int = 1024
    n_layer: int = 8
    dropout: float = 0.0
    emb_dropout: float = 0.0
    chunk_rank: int = 1
    chunk_size: int = 32
    mem_len: int = 4096
    window_len: int = 256
    grad_chk: bool = False
    pre_ln: bool = False
    use_gelu: bool = False
    use_bias: bool = False
    clamp_len: int = -1
    cpos_clamp_len: int = -1
    probing: bool = False


# @register_model("transformer-ls", dataclass=TransformerLSConfig)
@register_model("transformer-ls", dataclass=TransformerLanguageModelConfig)
class TransformerLS(FairseqLanguageModel):
    @classmethod
    def build_model(cls, args, task):
        print(args)
        return cls(TransformerLSDecoder(args, task))

    def get_aux_loss(self):
        return self.decoder.get_aux_loss()

    def get_current_max_span(self):
        return self.decoder.get_current_max_span()

    def get_current_avg_span(self):
        return self.decoder.get_current_avg_span()


class TransformerLSDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, task):

        super().__init__(task.target_dictionary)

        config = TransformerLSConfig(
            vocab_size=len(task.target_dictionary),
            d_model=args.d_model,
            n_head=args.n_head,
            d_inner=args.d_inner,
            n_layer=args.n_layer,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
            mem_len=args.mem_len,
            chunk_rank=args.chunk_rank,
            chunk_size=args.chunk_size,
            window_len=args.window_len,
            grad_chk=args.grad_chk,
            pre_ln=args.pre_ln,
            use_gelu=args.use_gelu,
            use_bias=args.use_bias,
            clamp_len=args.clamp_len,
            cpos_clamp_len=args.cpos_clamp_len,
            probing=args.probing,
        )
        self.config = config
        logger.info(config)
        del config.__dict__['_name']
        self.model = TransformerLSModel(**config.__dict__)
        self.cache_size = args.mem_len

        self._mems = None

    def forward(
        self,
        src_tokens,
        incremental_state: Optional[Dict[str, List[torch.Tensor]]] = None,
        encoder_out=None,
        src_lengths=None,
    ):
        bsz = src_tokens.size(0)
        if incremental_state is not None:  # used during inference
            mems = self.get_incremental_state("mems")
            src_tokens = src_tokens[:, -1:]  # only keep the most recent token
        else:
            mems = self._mems

        if mems is None:
            # first time init
            mems = self.init_hid_cache(bsz)
        output = self.model(x=src_tokens, h_cache=mems,)
        if incremental_state is not None:
            self.set_incremental_state(incremental_state, "mems", output[1])
        else:
            self._mems = output[1]
        return (output[0],)

    def init_hid_cache(self, batch_sz):
        hid = []
        for layer in self.model.layers:
            param = next(self.model.parameters())
            h = torch.zeros(
                batch_sz,
                self.cache_size,
                self.config.d_model,
                dtype=param.dtype,
                device=param.device,
            )
            hid.append(h)
        return hid

    def get_aux_loss(self):
        return self.model.get_aux_loss()

    def get_current_max_span(self):
        return self.model.get_current_max_span()

    def get_current_avg_span(self):
        return self.model.get_current_avg_span()

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        new_order: torch.Tensor,
    ):
        """Reorder incremental state.
        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        raise NotImplementedError("This is required for generation/beam search")
        # mems = self.get_incremental_state(incremental_state, "mems")
        # if mems is not None:
        #     new_mems = [mems_i.index_select(1, new_order) for mems_i in mems]
        #     self.set_incremental_state(incremental_state, "mems", new_mems)

### longshort attention
@register_model("longshort_lm", dataclass=TransformerLanguageModelConfig)
class LongShortLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(LongShortLanguageModel, self).__init__(decoder)

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

        decoder = LSAttentionDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)
### longshort attention

# longformer
@register_model_architecture("transformer_longformer_lm", "transformer_lm_longformer_wiki103")
def transformer_lm_lomgformer_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    args.attention_window =  getattr(args, "attention_window", [64]*20)
    args.attention_dilation =  getattr(args, "attention_dilation", [1]*20)
    args.autoregressive =  getattr(args, "autoregressive", False)
    args.attention_mode =  getattr(args, "attention_mode", 'sliding_chunks_no_overlap')  # ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']
    transformer_lm_big(args)

# add for rfa test
@register_model_architecture("transformer_lm", "transformer_lm_small")
def transformer_lm_small(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)

@register_model_architecture("transformer_lm", "transformer_lm_big")
def transformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_wiki103")
@register_model_architecture("transformer_lm", "transformer_lm_baevski_wiki103")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lm", "transformer_lm_gbw")
@register_model_architecture("transformer_lm", "transformer_lm_baevski_gbw")
def transformer_lm_baevski_gbw(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    transformer_lm_big(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt")
def transformer_lm_gpt(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_small")
def transformer_lm_gpt2_small(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_tiny")
def transformer_lm_gpt2_tiny(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 64)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 1)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_medium")
def transformer_lm_gpt2_medium(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1280)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 5120)
    args.decoder_layers = getattr(args, "decoder_layers", 36)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 20)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_big")
def transformer_lm_gpt2_big(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1600)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6400)
    args.decoder_layers = getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 25)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


def base_gpt3_architecture(args):
    args.decoder_input_dim = args.decoder_embed_dim
    args.decoder_output_dim = args.decoder_embed_dim
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 4)
    # GPT-3 used learned positional embeddings, rather than sinusoidal
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.share_decoder_input_output_embed = True
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_small")
def transformer_lm_gpt3_small(args):
    # 125M params
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_medium")
def transformer_lm_gpt3_medium(args):
    # 350M params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_large")
def transformer_lm_gpt3_large(args):
    # 760M params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_xl")
def transformer_lm_gpt3_xl(args):
    # 1.3B params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_2_7")
def transformer_lm_gpt3_2_7(args):
    # 2.7B params
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2560)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_6_7")
def transformer_lm_gpt3_6_7(args):
    # 6.7B params
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_13")
def transformer_lm_gpt3_13(args):
    # 13B params
    args.decoder_layers = getattr(args, "decoder_layers", 40)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 5120)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 40)
    base_gpt3_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt3_175")
def transformer_lm_gpt3_175(args):
    # 175B params
    args.decoder_layers = getattr(args, "decoder_layers", 96)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 12288)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 96)
    base_gpt3_architecture(args)

# single head
@register_model_architecture("transformer_lm", "transformer_lm_wiki103_single_head")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.decoder_attention_heads = 1

# add for baseline
@register_model_architecture("transformer_lm", "transformer_lm_small_wiki103")
def transformer_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

# rfa
@register_model_architecture("transformer_rfa_lm", "transformer_lm_rfa_wiki103")
def transformer_lm_rfa_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    args.proj_dim = getattr(args, "proj_dim", 64)
    args.tau = getattr(args, "tau", 1.0)
    args.reparam_proj = getattr(args, "reparam_proj", True)
    args.cuda_causal_rfa = getattr(args, "cuda_causal_rfa", False)
    transformer_lm_big(args)

@register_model_architecture("transformer_rfa_lm", "transformer_lm_rfa_small_wiki103")
def transformer_lm_rfa_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    args.proj_dim = getattr(args, "proj_dim", 64)
    args.tau = getattr(args, "tau", 1.0)
    args.reparam_proj = getattr(args, "reparam_proj", True)
    args.cuda_causal_rfa = getattr(args, "cuda_causal_rfa", False)
    transformer_lm_big(args)

# rfa debug
@register_model_architecture("transformer_rfa_debug_lm", "transformer_lm_rfa_debug_small_wiki103")
def transformer_lm_rfa_debug_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    args.proj_dim = getattr(args, "proj_dim", 64)
    args.tau = getattr(args, "tau", 1.0)
    args.reparam_proj = getattr(args, "reparam_proj", True)
    args.cuda_causal_rfa = getattr(args, "cuda_causal_rfa", True)
    # args.cuda_causal_rfa = getattr(args, "cuda_causal_rfa", False)
    args.sample_num = getattr(args, "sample_num", 200)
    # transformer_lm_big(args)
    transformer_lm_small(args)

@register_model_architecture("transformer_rfa_debug_lm", "transformer_lm_rfa_debug_wiki103")
def transformer_lm_rfa_debug_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    args.proj_dim = getattr(args, "proj_dim", 64)
    args.tau = getattr(args, "tau", 1.0)
    args.reparam_proj = getattr(args, "reparam_proj", True)
    args.cuda_causal_rfa = getattr(args, "cuda_causal_rfa", False)
    args.sample_num = getattr(args, "sample_num", 200)
    transformer_lm_big(args)

# performer
@register_model_architecture("performer_lm", "performer_lm_small_wiki103")
def performer_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    # add
    args.causal = getattr(args, "causal", True)
    args.local_heads = getattr(args, "local_heads", 0)
    args.local_window_size = getattr(args, "local_window_size", 256)

# performer
@register_model_architecture("performer_lm", "performer_lm_wiki103")
def performer_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    # add
    args.causal = getattr(args, "causal", True)
    args.local_heads = getattr(args, "local_heads", 0)
    args.local_window_size = getattr(args, "local_window_size", 256)

# sparse transformer
@register_model_architecture("sparse_transformer_lm", "sparse_transformer_lm_wiki103")
def sparse_transformer_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    transformer_lm_big(args)

@register_model_architecture("sparse_transformer_lm", "sparse_transformer_lm_small_wiki103")
def sparse_transformer_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    args.proj_dim = getattr(args, "proj_dim", 40)
    args.tau = getattr(args, "tau", 1.0)
    args.reparam_proj = getattr(args, "reparam_proj", True)
    args.cuda_causal_rfa = getattr(args, "cuda_causal_rfa", False)
    transformer_lm_big(args)

# linear transformer
@register_model_architecture("linear_transformer_lm", "linear_transformer_lm_small_wiki103")
def linear_transformer_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    transformer_lm_big(args)

@register_model_architecture("linear_transformer_lm", "linear_transformer_lm_wiki103")
def linear_transformer_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    # add
    transformer_lm_big(args)

# reformer
@register_model_architecture("reformer_lm", "reformer_lm_small_wiki103")
def reformer_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    # add
    args.causal = getattr(args, "causal", True)
    args.bucket_size = getattr(args, "bucket_size", 64)
    args.n_hashes = getattr(args, "n_hashes", 8)
    args.attn_chunks = getattr(args, "attn_chunks", 1)

@register_model_architecture("reformer_lm", "reformer_lm_wiki103")
def reformer_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    # add
    args.causal = getattr(args, "causal", True)
    args.bucket_size = getattr(args, "bucket_size", 64)
    args.n_hashes = getattr(args, "n_hashes", 8)
    args.attn_chunks = getattr(args, "attn_chunks", 1)

# transformer merge
@register_model_architecture("transformer_merge_lm", "transformer_merge_lm_small_wiki103")
def transformer_merge_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

@register_model_architecture("transformer_merge_lm", "transformer_merge_lm_wiki103")
def transformer_merge_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

# head
@register_model_architecture("transformer_head_lm", "transformer_head_lm_small_wiki103")
def transformer_head_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

@register_model_architecture("transformer_head_lm", "transformer_head_lm_wiki103")
def transformer_head_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

# head
@register_model_architecture("transformer_head_lm", "transformer_linear_relu_multi_wiki103")
def transformer_head_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.do_scale = getattr(args, "do_scale", False)
    args.use_linear = getattr(args, "use_linear", True)

# relu + multi + weight
@register_model_architecture("transformer_head_lm", "transformer_linear_relu_multi_weight_wiki103")
def transformer_head_lm_small_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.do_scale = getattr(args, "do_scale", False)
    args.use_linear = getattr(args, "use_linear", True)
    args.alpha_beta = getattr(args, "alpha_beta", True)
    args.max_l = getattr(args, "max_l", 3072)
    args.has_out = getattr(args, "has_out", True)



@register_model_architecture("transformer_lm", "transformer_lm_single_wiki103")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 1)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)


# cosformer512
@register_model_architecture("cosformer_lm", "cosformer_lm_wiki103")
def simformer_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_attention_heads = 16
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2200)
    args.causal = True

# cosformer512
@register_model_architecture("cosformer_lm", "cosformer_resi_lm_wiki103")
def simformer_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 1)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2200)
    # args.causal = True
    args.resi = True

# cosformer + softmax
@register_model_architecture("cosformer_softmax_lm", "cosformer_softmax_lm_wiki103")
def simformer_lm_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    # args.cosformer_layers = getattr(args, "decoder_layers", 15)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_attention_heads = 1
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2200)
    args.cosformer_layers = getattr(args, "cosformer_layers", 8)
    args.causal = True

@register_model_architecture("cosformer_lm", "cosformer_lm_big")
def cosformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_big_16")
def cosformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    # args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_embed_dim = 128
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_ffn_embed_dim = 128
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    base_lm_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True
    args.has_out = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_big_64")
def cosformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    # args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    base_lm_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True
    args.has_out = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_big_32")
def cosformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    # args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    base_lm_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True
    args.has_out = True

# speed test
@register_model_architecture("cosformer_lm", "cosformer_lm_wiki")
def cosformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_wiki8_")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_wiki8_witho")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_wiki16_")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_wiki32_")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_wiki1_")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 1)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_wiki2_")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_wiki4_")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True


## speed test
@register_model_architecture("transformer_lm", "transformer_lm_big_test")
def transformer_lm_big_test(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_embed_dim = 128
    args.decoder_embed_dim = 64
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_ffn_embed_dim = 512
    args.decoder_ffn_embed_dim = 256
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)

@register_model_architecture("cosformer_lm_", "cosformer_lm_big_16_test")
def cosformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    # args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_embed_dim = 128
    args.decoder_embed_dim = 64
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_ffn_embed_dim = 512
    args.decoder_ffn_embed_dim = 256
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    # args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    base_lm_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 2048)
    args.causal = True
    args.has_out = True

@register_model_architecture("cosformer_lm_", "cosformer_lm_gpt2_big")
def cosformer_lm_gpt2_big(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1600)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6400)
    args.decoder_layers = getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 25)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 1024)
    args.causal = True
    args.has_out = True



### rela
@register_model_architecture("rela_lm", "rela_wiki_ada_v1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

@register_model_architecture("rela_lm", "rela_wiki_ada_relu2")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    ## add
    args.act_fun = "relu2"

@register_model_architecture("rela_lm", "rela_wiki_ada_1+elu")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    ## add
    args.act_fun = "1+elu"

@register_model_architecture("rela_lm", "rela_wiki_ada_2+elu")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    ## add
    args.act_fun = "2+elu"

@register_model_architecture("rela_lm", "rela_wiki_ada_1+relu")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    ## add
    args.act_fun = "1+relu"

@register_model_architecture("rela_lm", "rela_wiki_ada_leak")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    ## add
    args.act_fun = "leak"

@register_model_architecture("rela_lm", "rela_wiki_ada_elu")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    ## add
    args.act_fun = "elu"

@register_model_architecture("rela_lm", "rela_wiki_ada_noact")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    ## add
    args.act_fun = "noact"

### rela

# flash
@register_model_architecture("flash_lm", "flash_wiki_ada_v1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.decoder_layers = 32
    args.s = 128
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2

# flash linear linear
@register_model_architecture("flash_linear_lm", "flash_linear_wiki_ada_v1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.decoder_layers = 32
    args.s = 128
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2
    args.chunk_size = 64


### base model
@register_model_architecture("transformer_head_lm", "transformer_lm_cos")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)

@register_model_architecture("transformer_head_lm", "transformer_lm_cos_type2")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = 2

@register_model_architecture("transformer_head_lm", "transformer_lm_rope")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_rope = True

### 单位阵
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1b_1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1d_1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_2_1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_3_1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 1
### 单位阵

### Odd Even
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_5")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5
### Odd Even


### DCT
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_2")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1b_2")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_2_2")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_3_2")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 2
### DCT

### Householder
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_3")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_2_3")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_3_3")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 3
### Householder

### Householder learned
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_3a")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True
### Householder learned

### base model

### linear urpe
@register_model_architecture("linear_urpe_lm", "1+elu_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = False
    args.kernel_type = "1+elu"

### 单位阵
@register_model_architecture("linear_urpe_lm", "1+elu_1_1_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("linear_urpe_lm", "1+elu_1b_1_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("linear_urpe_lm", "1+elu_1d_1_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_1_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("linear_urpe_lm", "1+elu_3_1_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1

### 单位阵

### Rope
@register_model_architecture("linear_urpe_lm", "1+elu_rope_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = False
    args.use_rope = True
    args.kernel_type = "1+elu"

@register_model_architecture("linear_urpe_lm", "relu_rope_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = False
    args.use_rope = True
    args.kernel_type = "relu"
### Rope

### Odd Even
@register_model_architecture("linear_urpe_lm", "1+elu_1_5_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("linear_urpe_lm", "relu_1_5_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "relu"
    args.core_matrix = 1
    args.p_matrix = 5
### Odd Even

### DCT
@register_model_architecture("linear_urpe_lm", "1+elu_1_2_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("linear_urpe_lm", "1+elu_1b_2_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("linear_urpe_lm", "1+elu_2_2_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("linear_urpe_lm", "1+elu_3_2_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 2
### DCT

### Householder
@register_model_architecture("linear_urpe_lm", "1+elu_1_3_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_2_3_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_3_3_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3
### Householder

### Householder learned
@register_model_architecture("linear_urpe_lm", "1+elu_1_3a_wiki")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = True
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.decoder_attention_heads = 1
    ### add
    args.causal = True
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True
### Householder learned



######################################
# norm attention
# linear: attention_type = 1
# local: attention_type = 2
# local, ... , local, linear, ... ,linear
@register_model_architecture("norm_attention_lm", "norm_attention_lm_type1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    # add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_heads = 1
    args.decoder_use_urpe = False
    args.decoder_chunk_size = 32
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]


######################################
# norm mix attention
@register_model_architecture("norm_mix_attention_lm", "norm_mix_attention_lm_type_1")
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    transformer_lm_big(args)
    # add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_attention_heads = 1
    args.decoder_use_urpe = False
    args.decoder_chunk_size = 32

######################### small model
### linear urpe
# @register_model_architecture("linear_urpe_lm", "1+elu_wiki_base")
# def transformer_lm_baevski_wiki103(args):
#     base_lm_architecture(args)
#     args.use_relu = getattr(args, "use_relu", True)
#     args.max_l = getattr(args, "max_l", 512)
#     args.causal = True
#     args.has_out = True
#     args.encoder_attention_heads = 1
#     args.encoder_normalize_before = True
#     args.use_gelu = True
#     args.decoder_attention_heads = 1
#     ### add
#     args.causal = True
#     args.use_urpe = False
#     args.kernel_type = "1+elu"

# ### 单位阵
# @register_model_architecture("linear_urpe_lm", "1+elu_1_1_wiki_base")
# def transformer_lm_baevski_wiki103(args):
#     base_lm_architecture(args)
#     args.max_l = getattr(args, "max_l", 512)
#     args.causal = True
#     # args.has_out = True
#     # args.encoder_attention_heads = 1
#     # args.encoder_normalize_before = True
#     # args.use_gelu = True
#     # args.decoder_attention_heads = 1
#     ### add
#     args.causal = True
#     args.use_urpe = True
#     args.kernel_type = "1+elu"
#     args.core_matrix = 1
#     args.p_matrix = 1
# ### 单位阵

@register_model_architecture("linear_urpe_lm", "1+elu_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.causal = True
    args.use_urpe = False
    args.kernel_type = "1+elu"

###### Identity
@register_model_architecture("linear_urpe_lm", "1+elu_1_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("linear_urpe_lm", "1+elu_1b_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("linear_urpe_lm", "1+elu_1c_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("linear_urpe_lm", "1+elu_1d_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("linear_urpe_lm", "1+elu_3_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1

###### Identity


###### DCT
@register_model_architecture("linear_urpe_lm", "1+elu_1_2_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("linear_urpe_lm", "1+elu_1d_2_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_2_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("linear_urpe_lm", "1+elu_3_2_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 2

###### DCT


###### Householder
@register_model_architecture("linear_urpe_lm", "1+elu_1_3_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_1d_3_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_3_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_3_3_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3

@register_model_architecture("linear_urpe_lm", "1+elu_1d_3a_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
###### Householder

###### Fourier
@register_model_architecture("linear_urpe_lm", "1+elu_4_4_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("linear_urpe_lm", "1+elu_4d_4_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
###### Fourier

###### Odd Even
@register_model_architecture("linear_urpe_lm", "1+elu_1_5_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("linear_urpe_lm", "1+elu_1d_5_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("linear_urpe_lm", "1+elu_2_5_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("linear_urpe_lm", "1+elu_3_5_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5

###### Odd Even

###### abl
@register_model_architecture("linear_urpe_lm", "1+elu_spe_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = False
    args.use_spe = True

@register_model_architecture("linear_urpe_lm", "1+elu_per_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = False
    args.use_spe = False
    args.use_permutate = True
###### abl

################################ transformer urpe
@register_model_architecture("transformer_lm", "transformer_lm_base")
def transformer_lm_big(args):
    base_lm_architecture(args)

### 单位阵
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1b_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1c_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1d_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_2_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_3_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 1
### 单位阵

###### Householder
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_3_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1d_3_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_2_3_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_3_3_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1d_3a_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
###### Householder

###### Fourier
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_4_4_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_4d_4_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
###### Fourier

###### Odd Even
@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1d_5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_2_5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_3_5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 5

###### Odd Even

###### abl
@register_model_architecture("transformer_head_lm", "transformer_lm_spe_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = True

@register_model_architecture("transformer_head_lm", "transformer_lm_per_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_permutate = True
    args.use_urpe = False
    args.use_spe = False

@register_model_architecture("transformer_head_lm", "transformer_lm_t5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.causal = True
    args.use_t5 = True

@register_model_architecture("transformer_head_lm", "transformer_lm_rpe_vanilla_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.causal = True
    args.use_t5 = False
    args.use_rpe_vanilla = True
###### abl

###### only rel
@register_model_architecture("linear_urpe_lm", "1+elu_1d_3_wiki_base_no_abs")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1d_3_base_no_abs")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("transformer_head_lm", "transformer_lm_urpe_1_1_base_no_abs")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("linear_urpe_lm", "1+elu_1_1_wiki_base_no_abs")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    # add
    args.no_token_positional_embeddings = True
###### only rel

################### norm attention(local + linear)
@register_model_architecture("norm_attention_lm", "norm_ln_glu_lm_base")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("norm_attention_lm", "norm_ln_glu_small_lm_base")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2

@register_model_architecture("norm_attention_lm", "norm_ln_ffn_lm_base")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

################### norm attention + glu act
@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_elu")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.fina_act = "elu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_elu")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.fina_act = "elu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
################### norm attention + glu act

################### norm attention + urpe
@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_urpe_1d3")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_urpe_1d3")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_ln_rms_urpe_1d3")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_ln_rms_urpe_1d3")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_layernorm_glu_lm_base_urpe_1d3")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True

@register_model_architecture("norm_attention_lm", "norm_all_layernorm_glu_small_lm_base_urpe_1d3")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
################### norm attention + urpe

################### norm attention + urpe + dropout
@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_urpe_1d3_dropout02")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    args.glu_dropout = 0.2

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_urpe_1d3_dropout02")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    args.glu_dropout = 0.2
################### norm attention + urpe + dropout

################### norm attention + urpe + no_abs
@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_lm_base_ln_rms_urpe_1d3_no_abs")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ###### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("norm_attention_lm", "norm_all_rms_glu_small_lm_base_ln_rms_urpe_1d3_no_abs")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ###### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("norm_attention_lm", "norm_all_layernorm_glu_lm_base_urpe_1d3_no_abs")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ###### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("norm_attention_lm", "norm_all_layernorm_glu_small_lm_base_urpe_1d3_no_abs")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    ###### no abs
    args.no_token_positional_embeddings = True
################### norm attention + urpe + no_abs

################### norm attention + urpe + pure rms norm
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_small_init")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_small_init")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_no_abs_small_init")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
    ###### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_no_abs_small_init")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
    ###### no abs
    args.no_token_positional_embeddings = True
################### norm attention + urpe + pure rms norm

################### norm attention + urpe + pure rms norm + geglu
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_geglu")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_geglu")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_small_init_geglu")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_small_init_geglu")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
################### norm attention + urpe + pure rms norm + geglu

################### pure rms norm + urpe + weight
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_laplace")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 1

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_laplace")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 1

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_gaussian")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 2

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_gaussian")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 2

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_final_dropout")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    # final dropout
    args.use_final_dropout = True
    args.final_dropout = 0.1

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_final_dropout")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    # final dropout
    args.use_final_dropout = True
    args.final_dropout = 0.1
################### pure rms norm + urpe + weight

### relu2
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_relu2")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "relu2"
    args.local_act_fun = "relu2"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_relu2")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "relu2"
    args.local_act_fun = "relu2"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
### relu2

### linear_chunk
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_32")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 32
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_32")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 32
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_16")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 16
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("norm_attention_lm", "norm_small_glu_lm_base_pure_rms_urpe_1d3_linear_chunk_16")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.decoder_chunk_size = 16
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
### linear_chunk

### speed test
# 删除mask, 不影响速度
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_abl")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.decoder_causal = False
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_chunk")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers)]
    ### glu
    args.use_glu = False
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_linear")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [1 for _ in range(args.decoder_layers)]
    ### glu
    args.use_glu = False
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
### speed test
################### mix attention

############# ls_attention_lm
# @register_model_architecture("ls_attention_lm", "ls_attention_lm")
# def transformer_lm_big(args):
#     base_lm_architecture(args)
#     args.chunk_size = 16
#     args.chunk_rank = 1
#     args.window_len = 512

@register_model_architecture("transformer-ls", "ls_attention_lm")
def transformer_lm_big(args):
    base_lm_architecture(args)
    args.d_model = args.decoder_embed_dim
    args.n_head = args.decoder_attention_heads
    args.d_inner = args.decoder_ffn_embed_dim
    args.n_layer = args.decoder_layers
    args.dropout = args.dropout
    args.emb_dropout = 0.0
    args.chunk_rank = 1
    args.chunk_size = 32
    # args.mem_len = 4096
    # args.window_len = 512
    args.mem_len = 512
    args.window_len = 64
    args.grad_chk = False
    args.pre_ln = False
    args.use_gelu = False
    args.use_bias = True
    args.clamp_len = -1
    args.cpos_clamp_len = -1
    args.probing = False
############# ls_attention_lm

############# longshort_lm
@register_model_architecture("longshort_lm", "longshort_base_lm")
def transformer_lm_big(args):
    base_lm_architecture(args)
    args.causal = True
    args.window_size = 64
    args.segment_size = 16
    args.r = 1
############# longshort_lm

############# performer_lm
@register_model_architecture("performer_lm", "performer_lm_wiki_base")
def transformer_lm_big(args):
    base_lm_architecture(args)
    args.approx_attn_dim = 64
    args.causal = True
############# performer_lm

############# flash_lm
@register_model_architecture("flash_lm", "flash_wiki")
def transformer_lm_flash(args):
    transformer_lm_big(args)
    args.decoder_layers = 10
    args.s = 128
    # args.s = 512
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2
    args.decoder_attention_types = []

@register_model_architecture("flash_linear_lm", "flash_linear_wiki")
def transformer_lm_flash_linear(args):
    transformer_lm_big(args)
    args.decoder_layers = 10
    args.s = 128
    # args.s = 512
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2
    args.chunk_size = 64
    args.decoder_attention_types = []

@register_model_architecture("flash_lm", "flash_wiki_one_head")
def transformer_lm_flash(args):
    transformer_lm_big(args)
    args.decoder_layers = 10
    args.s = 128
    # args.s = 512
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2
    args.decoder_attention_types = []
    args.decoder_attention_heads = 1

@register_model_architecture("flash_linear_lm", "flash_linear_wiki_one_head")
def transformer_lm_flash_linear(args):
    transformer_lm_big(args)
    args.decoder_layers = 10
    args.s = 128
    # args.s = 512
    args.norm_type = "scale_norm"
    args.eps = 1e-5
    args.max_position_embeddings = 512
    args.expansion_factor = 2
    args.chunk_size = 64
    args.decoder_attention_types = []
    args.decoder_attention_heads = 1
############# flash_lm

############# softmax + 1 + elu
@register_model_architecture("norm_attention_lm", "norm_glu_lm_base_pure_rms_urpe_1d3_softmax_1+elu")
def transformer_lm_big(args):
    base_lm_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.decoder_use_urpe = False
    args.group_type = "chunk"
    args.decoder_chunk_size = 64
    args.decoder_attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.decoder_use_urpe = True
    args.decoder_core_matrix = 1
    args.decoder_p_matrix = 3
    args.decoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    ###### softmax
    args.use_softmax = True
############# softmax + 1 + elu