# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Optional

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding, TransformerDecoder, TransformerLongformerDecoder, 
    # # rfa
    # TransformerRfaDecoder, 
    # # performer
    # PerformerDecoder, 
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
    TransformerSimpleDecoder,
    # head
    TransformerHeadDecoder,
    # taylor
    TransformerTaylorDecoder,
    # splu
    TransformerSparseReluDecoder,
    # cosformer
    CosformerDecoder,
    CosformerSoftmaxDecoder,
    CosformerDecoder_,
    # Mem
    MemDecoder,
    MemGauDecoder,
    # rela
    ReLADecoder,
    # Flash
    FlashDecoder,
    FlashLinearDecoder,
    # gmu 
    GmuDecoder,
    # linear kernel with orpe
    LinearKernelAttentionDecoder,
    # norm attention
    NormAttentionDecoder,
    # norm mix attention
    NormMixAttentionDecoder,
)
# simple transformer
from fairseq.models.simformer import SimformerEncoder, SimformerDecoder  

from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II


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

# add for transformer splu
@register_model("transformer_splu_lm", dataclass=TransformerLanguageModelConfig)
class TransformerSpluLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerSpluLanguageModel, self).__init__(decoder)

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

        decoder = TransformerSparseReluDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for transformer simple
@register_model("transformer_simple_lm", dataclass=TransformerLanguageModelConfig)
class TransformerSimpleLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerSimpleLanguageModel, self).__init__(decoder)

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

        decoder = TransformerSimpleDecoder(
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

        decoder = TransformerHeadDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for simformer
@register_model("simformer_lm", dataclass=TransformerLanguageModelConfig)
class SimformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(SimformerLanguageModel, self).__init__(decoder)

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

        decoder = SimformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

# add for transformer taylor
@register_model("transformer_taylor_lm", dataclass=TransformerLanguageModelConfig)
class TransformerTaylorLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(TransformerTaylorLanguageModel, self).__init__(decoder)

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

        decoder = TransformerTaylorDecoder(
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

@register_model("mem_lm", dataclass=TransformerLanguageModelConfig)
class MemLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(MemLanguageModel, self).__init__(decoder)

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

        decoder = MemDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

@register_model("mem_gau_lm", dataclass=TransformerLanguageModelConfig)
class MemGauLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super(MemGauLanguageModel, self).__init__(decoder)

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

        decoder = MemGauDecoder(
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

        decoder = FlashDecoder(
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
# LinearKernelAttentionDecoder with orpe
@register_model("linear_orpe_lm", dataclass=TransformerLanguageModelConfig)
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

# transformer simple
@register_model_architecture("transformer_simple_lm", "transformer_simple_lm_small_wiki103")
def transformer_simple_lm_small_wiki103(args):
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

@register_model_architecture("transformer_simple_lm", "transformer_simple_lm_wiki103")
def transformer_simple_lm_wiki103(args):
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

# simformer
@register_model_architecture("simformer_lm", "simformer_lm_small_wiki103")
def simformer_lm_small_wiki103(args):
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

@register_model_architecture("simformer_lm", "simformer_lm_wiki103")
def simformer_lm_wiki103(args):
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

# simformer
@register_model_architecture("transformer_taylor_lm", "transformer_taylor_lm_small_wiki103")
def simformer_lm_small_wiki103(args):
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

@register_model_architecture("transformer_taylor_lm", "transformer_taylor_lm_wiki103")
def simformer_lm_wiki103(args):
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

@register_model_architecture("transformer_taylor_lm", "transformer_linear_abs_lm_wiki103")
def simformer_lm_wiki103(args):
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
    args.norm_taylor = getattr(args, "norm_taylor", False)
    transformer_lm_big(args)

# linear relu
@register_model_architecture("transformer_taylor_lm", "transformer_linear_relu_lm_wiki103")
def simformer_lm_wiki103(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    transformer_lm_big(args)

# linear relu weight
@register_model_architecture("transformer_taylor_lm", "transformer_linear_relu_weight_lm_wiki103")
def simformer_lm_wiki103(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.alpha_beta = getattr(args, "alpha_beta", True)
    # change
    # args.max_l = getattr(args, "max_l", 3072)
    args.max_l = getattr(args, "max_l", 512)
    transformer_lm_big(args)

# splu
@register_model_architecture("transformer_splu_lm", "transformer_splu_lm_small_wiki103")
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
    # add
    args.n_groups = getattr(args, "n_groups", 4)
    args.step = getattr(args, "step", 4)
    args.d_global = getattr(args, "d_global", 32)
    args.with_global = getattr(args, "with_global", True)
    transformer_lm_big(args)

@register_model_architecture("transformer_splu_lm", "transformer_splu_lm_wiki103")
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
    # add
    args.n_groups = getattr(args, "n_groups", 4)
    args.step = getattr(args, "step", 4)
    args.d_global = getattr(args, "d_global", 32)
    args.with_global = getattr(args, "with_global", True)
    args.num = getattr(args, "num", 2)
    transformer_lm_big(args)

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

# linear
@register_model_architecture("transformer_taylor_lm", "transformer_linear_lm_wiki103")
def simformer_lm_wiki103(args):
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
    args.use_linear = getattr(args, "use_linear", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.do_scale = getattr(args, "do_scale", False)
    # change
    transformer_lm_big(args)

# cosformer512
@register_model_architecture("transformer_taylor_lm", "cosformer1024_lm_wiki103")
def simformer_lm_wiki103(args):
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
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.alpha_beta = getattr(args, "alpha_beta", True)
    # change
    # args.max_l = getattr(args, "max_l", 3072)
    args.max_l = getattr(args, "max_l", 1024)
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


@register_model_architecture("mem_lm", "mem_wiki_ada")
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
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True

@register_model_architecture("mem_lm", "mem_wiki_ada_single_head_has_out")
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

@register_model_architecture("mem_gau_lm", "mem_gau_wiki_ada_single_head_has_out")
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
    # args.decoder_layers = 36
    # args.decoder_layers = 40
    args.decoder_layers = 32

@register_model_architecture("mem_gau_lm", "mem_gau_wiki_ada_single_head_has_out_v2")
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
    d = int(2 ** 0.5 * args.decoder_embed_dim)
    args.decoder_embed_dim = d
    args.decoder_output_dim = d
    args.decoder_input_dim = d

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

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_gelu_init")
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
    args.act_fun = "gelu"
    args.init_type = "gelu"
    args.norm_type = "layernorm"

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_gelu_init_outnogelu")
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
    args.act_fun = "gelu"
    args.init_type = "gelu"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_gelu_init_rms_norm")
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
    args.act_fun = "gelu"
    args.init_type = "gelu"
    args.norm_type = "rmsnorm"


@register_model_architecture("mem_lm", "mem_wiki_ada_single_head_has_out_dropout")
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
    args.seq_dropout = True
    args.seq_p = 0.3


# gmu
@register_model_architecture("gmu_lm", "gmu_wiki_ada_v1")
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
    args.eps = 1e-5
    args.expansion_factor = 2
    args.norm_type = "rms_norm"
    args.act_fun = "silu"

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"

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
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1_1")
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
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1b_1")
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
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1d_1")
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
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_2_1")
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
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_3_1")
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
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 1
### 单位阵

### Odd Even
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1_5")
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
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 5
### Odd Even


### DCT
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1_2")
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
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1b_2")
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
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_2_2")
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
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_3_2")
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
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 2
### DCT

### Householder
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1_3")
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
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_2_3")
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
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_3_3")
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
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 3
### Householder

### Householder learned
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1_3a")
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
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True
### Householder learned

### base model


@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_rms_norm")
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
    args.act_fun = "elu"
    args.norm_type = "rmsnorm"

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_4head")
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
    args.decoder_attention_heads = 4
    ### add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_leak_out_no_act")
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
    args.act_fun = "leak"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_leak_out_no_act_0.01")
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
    args.act_fun = "leak"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.negative_slope = 0.01

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_lambda0")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.lambda_ = 0

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_lambda05")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.lambda_ = 0.5

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_usek")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.mem_use_q = False

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_sigmoid")
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
    args.act_fun = "sigmoid"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_exp")
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
    args.act_fun = "exp"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_postnorm")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.decoder_normalize_before = False

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_rope")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_rope_use_v")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True
    args.use_v = True

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_rope_use_v_multi_head")
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
    ### add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True
    args.use_v = True

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_rope_c")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True
    args.rope_type = "c"

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_rope_no_abs_pos")
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
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True
    args.no_token_positional_embeddings = True

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_rope_multi_head")
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
    # args.decoder_attention_heads = 1
    ### add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_1+elu_out_no_act_rope")
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
    args.act_fun = "1+elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_relu_out_no_act_rope")
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
    args.act_fun = "relu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

@register_model_architecture("mem_lm", "mem_wiki_ada_has_out_elu_out_no_act_gatednorm")
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
    args.act_fun = "elu"
    args.norm_type = "gatedrmsnorm"
    args.out_use_act = False

### linear orpe
@register_model_architecture("linear_orpe_lm", "1+elu_wiki")
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
    args.use_orpe = False
    args.kernel_type = "1+elu"

### 单位阵
@register_model_architecture("linear_orpe_lm", "1+elu_1_1_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("linear_orpe_lm", "1+elu_1b_1_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("linear_orpe_lm", "1+elu_1d_1_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("linear_orpe_lm", "1+elu_2_1_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("linear_orpe_lm", "1+elu_3_1_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1

### 单位阵

### Rope
@register_model_architecture("linear_orpe_lm", "1+elu_rope_wiki")
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
    args.use_orpe = False
    args.use_rope = True
    args.kernel_type = "1+elu"

@register_model_architecture("linear_orpe_lm", "relu_rope_wiki")
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
    args.use_orpe = False
    args.use_rope = True
    args.kernel_type = "relu"
### Rope

### Odd Even
@register_model_architecture("linear_orpe_lm", "1+elu_1_5_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("linear_orpe_lm", "relu_1_5_wiki")
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
    args.use_orpe = True
    args.kernel_type = "relu"
    args.core_matrix = 1
    args.p_matrix = 5
### Odd Even

### DCT
@register_model_architecture("linear_orpe_lm", "1+elu_1_2_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("linear_orpe_lm", "1+elu_1b_2_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("linear_orpe_lm", "1+elu_2_2_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("linear_orpe_lm", "1+elu_3_2_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 2
### DCT

### Householder
@register_model_architecture("linear_orpe_lm", "1+elu_1_3_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("linear_orpe_lm", "1+elu_2_3_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("linear_orpe_lm", "1+elu_3_3_wiki")
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
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3
### Householder

### Householder learned
@register_model_architecture("linear_orpe_lm", "1+elu_1_3a_wiki")
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
    args.use_orpe = True
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
    args.decoder_use_orpe = False
    args.decoder_chunk_size = 32
    args.attention_types = [2 for _ in range(args.decoder_layers // 2)] + [1 for _ in range(args.decoder_layers // 2)]


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
    args.decoder_use_orpe = False
    args.decoder_chunk_size = 32

######################### small model
### linear orpe
# @register_model_architecture("linear_orpe_lm", "1+elu_wiki_base")
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
#     args.use_orpe = False
#     args.kernel_type = "1+elu"

# ### 单位阵
# @register_model_architecture("linear_orpe_lm", "1+elu_1_1_wiki_base")
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
#     args.use_orpe = True
#     args.kernel_type = "1+elu"
#     args.core_matrix = 1
#     args.p_matrix = 1
# ### 单位阵

@register_model_architecture("linear_orpe_lm", "1+elu_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.causal = True
    args.use_orpe = False
    args.kernel_type = "1+elu"

###### Identity
@register_model_architecture("linear_orpe_lm", "1+elu_1_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("linear_orpe_lm", "1+elu_1b_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("linear_orpe_lm", "1+elu_1c_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("linear_orpe_lm", "1+elu_1d_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("linear_orpe_lm", "1+elu_2_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("linear_orpe_lm", "1+elu_3_1_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1

###### Identity


###### DCT
@register_model_architecture("linear_orpe_lm", "1+elu_1_2_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("linear_orpe_lm", "1+elu_1d_2_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_learned = True

@register_model_architecture("linear_orpe_lm", "1+elu_2_2_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("linear_orpe_lm", "1+elu_3_2_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 2

###### DCT


###### Householder
@register_model_architecture("linear_orpe_lm", "1+elu_1_3_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("linear_orpe_lm", "1+elu_1d_3_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("linear_orpe_lm", "1+elu_2_3_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("linear_orpe_lm", "1+elu_3_3_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3

@register_model_architecture("linear_orpe_lm", "1+elu_1d_3a_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
###### Householder

###### Fourier
@register_model_architecture("linear_orpe_lm", "1+elu_4_4_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("linear_orpe_lm", "1+elu_4d_4_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
###### Fourier

###### Odd Even
@register_model_architecture("linear_orpe_lm", "1+elu_1_5_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("linear_orpe_lm", "1+elu_1d_5_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("linear_orpe_lm", "1+elu_2_5_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("linear_orpe_lm", "1+elu_3_5_wiki_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.causal = True
    ### add
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5

###### Odd Even

################################ transformer orpe
@register_model_architecture("transformer_lm", "transformer_lm_base")
def transformer_lm_big(args):
    base_lm_architecture(args)

### 单位阵
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1b_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1c_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1d_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_2_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_3_1_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 1
### 单位阵

###### Householder
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1_3_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1d_3_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_2_3_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_3_3_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 3

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1d_3a_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
###### Householder

###### Fourier
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_4_4_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_4d_4_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True
###### Fourier

###### Odd Even
@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1_5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_1d_5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_2_5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("transformer_head_lm", "transformer_lm_orpe_3_5_base")
def transformer_lm_baevski_wiki103(args):
    base_lm_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 5

###### Odd Even
