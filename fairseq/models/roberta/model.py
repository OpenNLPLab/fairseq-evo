# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging
from numpy import False_

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder, TransformerSparseReluEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import RobertaHubInterface

# reformer
from fairseq.models.transformer import ReformerEncoder, TransformerMergeEncoder
# simple
from fairseq.models.transformer import TransformerSimpleEncoder
# head
from fairseq.models.transformer import TransformerHeadEncoder
# simformer
from fairseq.models.simformer import SimformerEncoder
# mix
from fairseq.models.simformer import TransformerMixEncoder
# taylor
from fairseq.models.transformer import TransformerTaylorEncoder
# sparse relu
from fairseq.models.transformer import TransformerSparseReluEncoderLayer
# splu
from fairseq.models.transformer import TransformerSpluEncoder
# cosformer
from fairseq.models.transformer import CosformerEncoder
# head
from fairseq.models.transformer import TransformerHeadEncoder
# cosformer
from fairseq.models.transformer import CosformerEncoder_
# pcc
from fairseq.models.transformer import PccEncoder
# cos
# from fairseq.models.transformer import TransformerCosEncoder
# weight
from fairseq.models.transformer import WeightFormerEncoder
# weight diff head
from fairseq.models.transformer import WeightFormerEncoder_diff
# GAU
from fairseq.models.transformer import FlashEncoder
# Flash Linear
from fairseq.models.transformer import FlashLinearEncoder
# mem
from fairseq.models.transformer import MemEncoder
# memgau
from fairseq.models.transformer import MemGauEncoder
# ReLA
from fairseq.models.transformer import ReLAEncoder
# Gmu
from fairseq.models.transformer import GmuEncoder
# linear kernel with orpe
from fairseq.models.transformer import LinearKernelAttentionEncoder
# norm Attention
from fairseq.models.transformer import NormAttentionEncoder
# norm mix attention
from fairseq.models.transformer import NormMixAttentionEncoder

logger = logging.getLogger(__name__)


@register_model("roberta")
class RobertaModel(FairseqEncoderModel):
    @classmethod
    def hub_models(cls):
        return {
            "roberta.base": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            "roberta.large": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz",
            "roberta.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz",
            "roberta.large.wsc": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz",
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            "--min-params-to-wrap",
            type=int,
            metavar="D",
            default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )

        logger.info(x["args"])
        return RobertaHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

        self.lm_head = self.build_lm_head(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
        )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

# reformer
class RobertaLSHEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = ReformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_lsh")
class RobertaLSHModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaLSHEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# merge
class RobertaMergeEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerMergeEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_merge")
class RobertaMergeModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaMergeEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# simple
class RobertaSimpleEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerSimpleEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_simple")
class RobertaSimpleModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaSimpleEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# head
class RobertaHeadEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerHeadEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_head")
class RobertaHeadModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaHeadEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# simformer
class RobertaSimformerEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = SimformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_simformer")
class RobertaSimformerModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaSimformerEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# mix
class RobertaMixEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerMixEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_mix")
class RobertaMixModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaMixEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# taylor
class RobertaTaylorEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerTaylorEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_taylor")
class RobertaTaylorModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaTaylorEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    # # for fine tune
    # def forward(
    #     self,
    #     src_tokens,
    #     features_only=False,
    #     return_all_hiddens=False,
    #     classification_head_name=None,
    #     **kwargs,
    # ):
    #     if classification_head_name is not None:
    #         features_only = True

    #     with torch.no_grad():
    #         x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

    #     if classification_head_name is not None:
    #         x = self.classification_heads[classification_head_name](x)
    #     return x, extra

# sparse relu
class RobertaSparseReluEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerSparseReluEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_sparse_relu")
class RobertaSparseReluModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaSparseReluEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# multi splu
class RobertaSpluEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerSpluEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_multi_sparse_relu")
class RobertaSpluModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaSpluEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# sparse relu
class RobertaCosEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerCosEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_cos")
class RobertaCosModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaCosEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# cosformer
class RobertaCosformerEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = CosformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_cosformer")
class RobertaCosformerModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaCosformerEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# cosformer
class RobertaCosformerEncoder_(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = CosformerEncoder_(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_cosformer_")
class RobertaCosformerModel_(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaCosformerEncoder_(args, task.source_dictionary)
        return cls(args, encoder)

# normalize, head
class RobertaNormalizeEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerHeadEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_normalize")
class RobertaNormalizeModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaNormalizeEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# pcc
class RobertaPccEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = PccEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_pcc")
class RobertaPccModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaPccEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# weight
class RobertaWeightEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = WeightFormerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_weight")
class RobertaWeightModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaWeightEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# weight diff
class RobertaWeightEncoder_diff(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = WeightFormerEncoder_diff(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_weight_diff")
class RobertaWeightModel_diff(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaWeightEncoder_diff(args, task.source_dictionary)
        return cls(args, encoder)

### Flash
class RobertaFlashEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = FlashEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_flash")
class RobertaFlashModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaFlashEncoder(args, task.source_dictionary)
        return cls(args, encoder)

# Flash Linear
class RobertaFlashLinearEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = FlashLinearEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_flash_linear")
class RobertaFlashLinearModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaFlashLinearEncoder(args, task.source_dictionary)
        return cls(args, encoder)
#### 

class RobertaMemEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = MemEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_mem")
class RobertaMemModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaMemEncoder(args, task.source_dictionary)
        return cls(args, encoder)

class RobertaMemGauEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = MemGauEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_mem_gau")
class RobertaMemModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaMemGauEncoder(args, task.source_dictionary)
        return cls(args, encoder)

class RobertaReLAEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = ReLAEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_rela")
class RobertaReLAModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaReLAEncoder(args, task.source_dictionary)
        return cls(args, encoder)

############# Gmu
class RobertaGmuEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = GmuEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_gmu")
class RobertaGmuModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaGmuEncoder(args, task.source_dictionary)
        return cls(args, encoder)

############# Linear Orpe
class RobertaLinearOrpeEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = LinearKernelAttentionEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_linear_orpe")
class RobertaLinearOrpeModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaLinearOrpeEncoder(args, task.source_dictionary)
        return cls(args, encoder)

############# NormAttentionEncoder
class RobertaNormEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = NormAttentionEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_norm_attention")
class RobertaNormOrpeModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaNormEncoder(args, task.source_dictionary)
        return cls(args, encoder)

############# NormMixAttentionEncoder
class RobertaNormMixEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = NormMixAttentionEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_norm_mix_attention")
class RobertaNormMixModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaNormMixEncoder(args, task.source_dictionary)
        return cls(args, encoder)

@register_model_architecture("roberta", "roberta")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )



@register_model_architecture("roberta", "roberta_prenorm")
def roberta_prenorm_architecture(args):
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    base_architecture(args)


@register_model_architecture("roberta", "roberta_base")
def roberta_base_architecture(args):
    base_architecture(args)

@register_model_architecture("roberta", "roberta_base_single_head")
def roberta_base_architecture(args):
    base_architecture(args)
    args.encoder_attention_heads = 1

@register_model_architecture("roberta", "roberta_large")
def roberta_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("roberta", "xlm")
def xlm_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1280 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)

# reformer
@register_model_architecture("roberta_lsh", "roberta_lsh_base")
def roberta_lsh_base_architecture(args):
    # add
    args.causal = getattr(args, "causal", False)
    args.bucket_size = getattr(args, "bucket_size", 128)
    args.n_hashes = getattr(args, "n_hashes", 8)
    args.attn_chunks = getattr(args, "attn_chunks", 1)
    base_architecture(args)

# merge
@register_model_architecture("roberta_merge", "roberta_merge_base")
def roberta_merge_architecture(args):
    args.dim_scale = getattr(args, "dim_scale", -1)
    base_architecture(args)

# merge right
@register_model_architecture("roberta_merge", "roberta_merge_right_base")
def roberta_merge_architecture(args):
    args.dim_scale = getattr(args, "dim_scale", -1)
    args.has_right_weight = getattr(args, "has_right_weight", True)
    args.do_softmax = getattr(args, "do_softmax", False)
    base_architecture(args)

# merge right softmax
@register_model_architecture("roberta_merge", "roberta_merge_right_soft_base")
def roberta_merge_architecture(args):
    args.dim_scale = getattr(args, "dim_scale", -1)
    args.has_right_weight = getattr(args, "has_right_weight", True)
    args.do_softmax = getattr(args, "do_softmax", True)
    base_architecture(args)

@register_model_architecture("roberta_merge", "roberta_merge_4d_base")
def roberta_merge_architecture(args):
    args.dim_scale = getattr(args, "dim_scale", 4)
    base_architecture(args)

# simple
@register_model_architecture("roberta_simple", "roberta_simple_base")
def roberta_simple_architecture(args):
    base_architecture(args)

# head
@register_model_architecture("roberta_head", "roberta_head_base")
def roberta_head_architecture(args):
    base_architecture(args)

# ada qk
@register_model_architecture("roberta_head", "roberta_ada_base")
def roberta_head_architecture(args):
    base_architecture(args)
    args.is_ada_q = getattr(args, "is_ada_q", True)
    args.is_ada_k = getattr(args, "is_ada_k", True)
    args.do_scale = getattr(args, "do_scale", True),
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.lambda_ = getattr(args, "lambda_", 0.99)
    args.use_q = getattr(args, "use_q", True)
    args.use_k = getattr(args, "use_k", True)
    args.has_out = getattr(args, "has_out", True)

# ada q
@register_model_architecture("roberta_head", "roberta_ada_q_base")
def roberta_head_architecture(args):
    base_architecture(args)
    args.is_ada_q = getattr(args, "is_ada_q", True)
    args.is_ada_k = getattr(args, "is_ada_k", False)
    args.do_scale = getattr(args, "do_scale", True),
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.lambda_ = getattr(args, "lambda_", 0.99)
    args.use_q = getattr(args, "use_q", True)
    args.use_k = getattr(args, "use_k", False)
    args.has_out = getattr(args, "has_out", True)

# multi relu weight 
@register_model_architecture("roberta_head", "roberta_multi_relu_weight_base")
def roberta_head_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.do_scale = getattr(args, "do_scale", False)
    args.use_linear = getattr(args, "use_linear", True)
    args.alpha_beta = getattr(args, "alpha_beta", True)
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = getattr(args, "has_out", True)

# simformer
@register_model_architecture("roberta_simformer", "roberta_simformer_base")
def roberta_simformer_architecture(args):
    base_architecture(args)

# mix
@register_model_architecture("roberta_mix", "roberta_mix_base")
def roberta_mix_architecture(args):
    base_architecture(args)

# dp_bf
@register_model_architecture("roberta_merge", "roberta_dp_bf")
def roberta_mix_architecture(args):
    # args.is_base = getattr(args, "is_base", True)
    # args.is_ada_q = getattr(args, "is_ada_q", False)
    # args.is_ada_k = getattr(args, "is_ada_k", False)
    # args.lambda_ = getattr(args, "lambda_", 0.99)
    # args.up_fq = getattr(args, "up_fq", 16)
    args.dropout_before = getattr(args, "dropout_before", True)
    base_architecture(args)

# ada q
@register_model_architecture("roberta_merge", "roberta_ada_q")
def roberta_mix_architecture(args):
    args.is_ada_q = getattr(args, "is_ada_q", True)
    args.is_ada_k = getattr(args, "is_ada_k", False)
    args.lambda_ = getattr(args, "lambda_", 0.99)
    args.use_q = getattr(args, "use_q", True),
    args.use_k = getattr(args, "use_k", False),
    base_architecture(args)

# with o
@register_model_architecture("roberta_merge", "roberta_with_o")
def roberta_mix_architecture(args):
    args.has_out = getattr(args, "has_out", True)
    base_architecture(args)

# taylor
@register_model_architecture("roberta_taylor", "roberta_taylor_base")
def roberta_taylor_architecture(args):
    base_architecture(args)

# taylor low
@register_model_architecture("roberta_taylor", "roberta_taylor_low_base")
def roberta_taylor_architecture(args):
    args.low_d = getattr(args, "low_d", True)
    base_architecture(args)

# taylor out
@register_model_architecture("roberta_taylor", "roberta_taylor_out_base")
def roberta_taylor_architecture(args):
    args.has_out = getattr(args, "has_out", True)
    base_architecture(args)

# taylor no scale
@register_model_architecture("roberta_taylor", "roberta_taylor_no_scale_base")
def roberta_taylor_architecture(args):
    args.do_scale = getattr(args, "do_scale", False)
    base_architecture(args)

# taylor no scale ada
@register_model_architecture("roberta_taylor", "roberta_taylor_no_scale_ada_base")
def roberta_taylor_architecture(args):
    args.is_ada_q = getattr(args, "is_ada_q", True)
    args.use_q = getattr(args, "use_q", True)
    args.do_scale = getattr(args, "do_scale", False)
    base_architecture(args)

# taylor scale ada
@register_model_architecture("roberta_taylor", "roberta_taylor_scale_ada_base")
def roberta_taylor_architecture(args):
    args.is_ada_q = getattr(args, "is_ada_q", True)
    args.use_q = getattr(args, "use_q", True)
    base_architecture(args)

# linear
@register_model_architecture("roberta_taylor", "roberta_taylor_linear_base")
def roberta_taylor_architecture(args):
    args.norm_taylor = getattr(args, "norm_taylor", False)
    base_architecture(args)

# linear relu
@register_model_architecture("roberta_taylor", "roberta_linear_relu_base")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    base_architecture(args)

# linear relu alpha beta
@register_model_architecture("roberta_taylor", "roberta_linear_relu_weight_base")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.alpha_beta = getattr(args, "alpha_beta", True)
    base_architecture(args)


# linear relu pos lear
@register_model_architecture("roberta_taylor", "roberta_linear_relu_no_lear_pos_base")
def roberta_taylor_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.encoder_learned_pos = False

# linear relu right
@register_model_architecture("roberta_taylor", "roberta_linear_relu_right_base")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.has_right_weight = getattr(args, "has_right_weight", True)
    args.do_softmax = getattr(args, "do_softmax", False)
    base_architecture(args)

# linear relu right
@register_model_architecture("roberta_taylor", "roberta_linear_relu_right_sf_base")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.has_right_weight = getattr(args, "has_right_weight", True)
    args.do_softmax = getattr(args, "do_softmax", True)
    base_architecture(args)

# linear relu right not share sf
@register_model_architecture("roberta_taylor", "roberta_linear_relu_right_not_share_sf_base")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.has_right_weight = getattr(args, "has_right_weight", False)
    args.do_softmax = getattr(args, "do_softmax", True)
    args.has_right_weight_not_share = getattr(args, "has_right_weight_not_share", True)
    base_architecture(args)


# linear relu res
@register_model_architecture("roberta_taylor", "roberta_linear_relu_res_base")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.has_res = getattr(args, "has_res", True)
    base_architecture(args)


@register_model_architecture("roberta_taylor", "roberta_linear_relu_sparse_base")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.sparse = getattr(args, "sparse", True),
    args.d1 = getattr(args, "d1", 32)
    # args.d1 = getattr(args, "d1", 1)
    args.d2 = getattr(args, "d2", 8)
    # args.d2 = getattr(args, "d2", 1)
    base_architecture(args)
    # args.encoder_layers = getattr(args, "encoder_layers", 1)

@register_model_architecture("roberta_taylor", "roberta_linear_relu_large")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    base_architecture(args)

# linear elu
@register_model_architecture("roberta_taylor", "roberta_linear_elu_base")
def roberta_taylor_architecture(args):
    args.use_elu = getattr(args, "use_elu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    base_architecture(args)

# linear
@register_model_architecture("roberta_taylor", "roberta_linear_elu_p1_base")
def roberta_taylor_architecture(args):
    args.use_linear = getattr(args, "use_linear", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    base_architecture(args)

# linear no scale
@register_model_architecture("roberta_taylor", "roberta_linear_elu_p1_no_scale_base")
def roberta_taylor_architecture(args):
    args.do_scale = getattr(args, "do_scale", False)
    args.use_linear = getattr(args, "use_linear", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    base_architecture(args)

# linear leak
@register_model_architecture("roberta_taylor", "roberta_linear_leak_base")
def roberta_taylor_architecture(args):
    args.use_leak = getattr(args, "use_leak", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    base_architecture(args)

# linear square
@register_model_architecture("roberta_taylor", "roberta_linear_square_base")
def roberta_taylor_architecture(args):
    args.use_square = getattr(args, "use_square", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    base_architecture(args)

# linear square
@register_model_architecture("roberta_taylor", "roberta_sigmoid_base")
def roberta_taylor_architecture(args):
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.use_sigmoid = getattr(args, "use_sigmoid", True)
    args.do_scale = getattr(args, "do_scale", False)
    base_architecture(args)

# linear leak
@register_model_architecture("roberta_taylor", "roberta_leak_l2_base")
def roberta_taylor_architecture(args):
    args.use_leak = getattr(args, "use_leak", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.use_l2 = getattr(args, "use_l2", True)
    base_architecture(args)

# relu high
@register_model_architecture("roberta_taylor", "roberta_linear_relu_high_base")
def roberta_taylor_architecture(args):
    args.use_relu = getattr(args, "use_relu", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.dim_scale = getattr(args, "dim_scale", 4)
    base_architecture(args)

# sparse relu
@register_model_architecture("roberta_sparse_relu", "roberta_splu_base")
def roberta_taylor_architecture(args):
    args.n_groups = getattr(args, "n_groups", 4)
    args.step = getattr(args, "step", 4)
    args.num = getattr(args, "num", 2)
    base_architecture(args)

# sparse relu global
@register_model_architecture("roberta_sparse_relu", "roberta_splu_global_base")
def roberta_taylor_architecture(args):
    args.n_groups = getattr(args, "n_groups", 4)
    args.step = getattr(args, "step", 4)
    args.d_global = getattr(args, "d_global", 32)
    args.with_global = getattr(args, "with_global", True)
    args.num = getattr(args, "num", 2)
    base_architecture(args)

# multi splu
@register_model_architecture("roberta_multi_sparse_relu", "roberta_multi_splu_base")
def roberta_taylor_architecture(args):
    args.n_groups = getattr(args, "n_groups", 4)
    args.step = getattr(args, "step", 4)
    args.d_global = getattr(args, "d_global", 32)
    args.num = getattr(args, "num", 2)
    args.max_n = getattr(args, "max_n", 512)
    base_architecture(args)

# cos
@register_model_architecture("roberta_cos", "roberta_cos_base")
def roberta_taylor_architecture(args):
    base_architecture(args)

# cosformer
@register_model_architecture("roberta_cosformer", "roberta_cosformer_base")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 4400)
    args.causal = False

# cosformer
@register_model_architecture("roberta_cosformer_", "roberta_cosformer_base_")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.drop_out = getattr(args, "drop_out", True)
    args.p = getattr(args, "p", 0.3)

@register_model_architecture("roberta_cosformer_", "roberta_cosformer_base_wo")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.drop_out = getattr(args, "drop_out", True)
    args.p = getattr(args, "p", 0.3)
    args.has_out = True

@register_model_architecture("roberta_cosformer_", "roberta_cosformer_base_high")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.drop_out = getattr(args, "drop_out", True)
    args.p = getattr(args, "p", 0.7)

# roberta_pcc
@register_model_architecture("roberta_pcc", "roberta_pcc_model")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.causal = False

#### multi
# leaky
@register_model_architecture("roberta_head", "roberta_multi_leaky_base")
def roberta_taylor_architecture(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.2)
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.use_leak = getattr(args, "use_leak", True)
    args.norm_taylor = getattr(args, "norm_taylor", False)
    args.do_scale = getattr(args, "do_scale", False)
    args.use_linear = getattr(args, "use_linear", True)
    base_architecture(args)

# normalize
@register_model_architecture("roberta_normalize", "roberta_normalize_base")
def roberta_cosformer_architecture(args):
    base_architecture(args)

# weight former
@register_model_architecture("roberta_weight", "roberta_weight1")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = False
    args.encoder_attention_heads = 1

@register_model_architecture("roberta_weight", "roberta_weight1_prenorm")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    print(f"pre_norm {args.encoder_normalize_before}")

@register_model_architecture("roberta_weight", "roberta_weight1_wol")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = False
    args.encoder_attention_heads = 1
    args.use_layernorm = False
    print(f"here {args.use_layernorm}")

# bound
@register_model_architecture("roberta_weight", "roberta_weight1_bound")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = False
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = False
    args.encoder_attention_heads = 1
    args.use_bound = True

@register_model_architecture("roberta_weight", "roberta_weight1_sqrt_bound")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = False
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = False
    args.encoder_attention_heads = 1
    args.use_bound = True

@register_model_architecture("roberta_weight", "roberta_weight1_layer_norm")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = True
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = False
    args.encoder_attention_heads = 1
    args.use_layer_norm = True

@register_model_architecture("roberta_weight", "roberta_weight0_layer_norm")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = True
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = 0
    args.has_out = True
    args.encoder_attention_heads = 12
    args.use_layer_norm = True

@register_model_architecture("roberta_weight", "roberta_weight0_layer_norm_seq_drop")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = True
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = 0
    args.has_out = True
    args.encoder_attention_heads = 12
    args.qk_layer_norm = True
    args.seq_dropout = True
    args.seq_p = 0.3

@register_model_architecture("roberta_weight", "roberta_weight0_seq_drop")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = True
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = 0
    args.has_out = True
    args.encoder_attention_heads = 12
    args.seq_dropout = True
    args.seq_p = 0.3

@register_model_architecture("roberta_weight", "roberta_weight0_qk_layer_norm_multi")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = False
    args.use_bound = True
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = 0
    args.has_out = True
    args.encoder_attention_heads = 12
    args.qk_layer_norm = True

# dropout
@register_model_architecture("roberta_weight", "roberta_weight1_dropout")
def roberta_weight1_wol_architecture(args):
    base_architecture(args)
    args.use_relu = True
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = False
    args.use_dropout = True
    args.p = 0.5
    args.encoder_attention_heads = 1

# v激活
@register_model_architecture("roberta_weight", "roberta_weight1_actv_v1")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = False
    args.encoder_attention_heads = 1
    args.v_act = True

@register_model_architecture("roberta_weight", "roberta_weight1_actv_v2")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 1)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.v_act = True

@register_model_architecture("roberta_weight", "roberta_weight2")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 2)
    args.has_out = False
    args.encoder_attention_heads = 1

@register_model_architecture("roberta_weight", "roberta_weight2_0_5")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 2)
    args.has_out = False
    args.encoder_attention_heads = 1
    args.c = 0.5

@register_model_architecture("roberta_weight", "roberta_weight3")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 3)
    args.has_out = False
    args.encoder_attention_heads = 1

@register_model_architecture("roberta_weight", "roberta_weight4")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 4)
    args.has_out = False
    args.encoder_attention_heads = 1

@register_model_architecture("roberta_weight_diff", "roberta_weight3_diff")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.weight_type = getattr(args, "weight_type", 3)
    args.has_out = False
    args.all_heads = [128, 96, 64, 48, 24, 16, 12, 8, 4, 3, 2, 1]
    # args.encoder_attention_heads = 1
    # args.encoder_layers = 2
    # args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)

@register_model_architecture("roberta_flash", "roberta_flash_v1")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_layers = 24

@register_model_architecture("roberta_flash_linear", "roberta_flash_linear_v1")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_layers = 24
    args.chunk_size = 64

@register_model_architecture("roberta_mem", "roberta_mem_v1")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True

@register_model_architecture("roberta_mem", "roberta_mem_v2")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True

@register_model_architecture("roberta_mem", "roberta_mem_v3")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True

@register_model_architecture("roberta_mem", "roberta_mem_v4")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.encoder_layers = 24
    args.use_forward = False

@register_model_architecture("roberta_mem", "roberta_mem_v5")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.use_forward = False
    args.encoder_ffn_embed_dim = 1536
    args.use_anotherforward = True
    
@register_model_architecture("roberta_mem", "roberta_mem_v6")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.use_forward = False
    args.encoder_ffn_embed_dim = 1536
    args.use_anotherforward = True
    args.encoder_layers = 17

@register_model_architecture("roberta_mem", "roberta_mem_hasout")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True

@register_model_architecture("roberta_mem", "roberta_mem_gelu_nolayer_norm")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.attention_use_layer_norm = False

@register_model_architecture("roberta_mem", "roberta_mem_gelu_multi_head")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True

@register_model_architecture("roberta_mem", "roberta_mem_no_grad")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    # args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_grad = False
    args.model_update_freq = args.update_freq[0]
    print("-------------------")
    print(args.model_update_freq)
    print("-------------------")

# test
@register_model_architecture("roberta_mem", "roberta_mem_use_q")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_grad = False

@register_model_architecture("roberta_mem", "roberta_mem_v2_test")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True

# 1 / 3参数
@register_model_architecture("roberta_mem_gau", "roberta_mem_gau_v1")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.encoder_layers = 24

@register_model_architecture("roberta_mem_gau", "roberta_mem_gau_v2")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.encoder_embed_dim
    args.encoder_layers = 12
    args.encoder_embed_dim = int(2 ** 0.5 * args.encoder_embed_dim)

# @register_model_architecture("roberta_mem_gau", "roberta_mem_gau_v1")
# def roberta_cosformer_architecture(args):
#     base_architecture(args)
#     args.use_relu = getattr(args, "use_relu", True)
#     args.max_l = getattr(args, "max_l", 512)
#     args.causal = False
#     args.has_out = False
#     args.encoder_attention_heads = 1
#     args.encoder_normalize_before = True
#     args.use_gelu = True
#     args.encoder_layers = 72

@register_model_architecture("roberta_rela", "roberta_rela_v1")
def roberta_rela_architecture(args):
    base_architecture(args)

@register_model_architecture("roberta_rela", "roberta_rela_relu2")
def roberta_rela_architecture(args):
    base_architecture(args)
    args.act_fun = "relu2"

@register_model_architecture("roberta_rela", "roberta_rela_1+elu")
def roberta_rela_architecture(args):
    base_architecture(args)
    args.act_fun = "1+elu"

@register_model_architecture("roberta_rela", "roberta_rela_leak")
def roberta_rela_architecture(args):
    base_architecture(args)
    args.act_fun = "leak"

@register_model_architecture("roberta_rela", "roberta_rela_1+relu")
def roberta_rela_architecture(args):
    base_architecture(args)
    args.act_fun = "1+relu"

@register_model_architecture("roberta_rela", "roberta_rela_2+elu")
def roberta_rela_architecture(args):
    base_architecture(args)
    args.act_fun = "2+elu"

@register_model_architecture("roberta_rela", "roberta_rela_elu")
def roberta_rela_architecture(args):
    base_architecture(args)
    args.act_fun = "elu"

@register_model_architecture("roberta_rela", "roberta_rela_noact")
def roberta_rela_architecture(args):
    base_architecture(args)
    args.act_fun = "noact"

@register_model_architecture("roberta_mem", "roberta_mem_hasout_gelu_init")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    args.act_fun = "gelu"
    args.init_type = "gelu"
    args.norm_type = "layernorm"

@register_model_architecture("roberta_mem", "roberta_mem_hasout_gelu_init_outnogelu")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    # add
    args.act_fun = "gelu"
    args.init_type = "gelu"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_gelu_init_rms_norm")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ####
    args.act_fun = "gelu"
    args.init_type = "gelu"
    args.norm_type = "rmsnorm"

@register_model_architecture("roberta_mem", "roberta_mem_hasout_seqdrop")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    args.seq_dropout = True
    args.seq_p = 0.3

@register_model_architecture("roberta_gmu", "roberta_gmu_v1")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_layers = 24
    args.norm_type = "rms_norm"
    args.act_fun = "silu"

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"

### base model
@register_model_architecture("roberta_head", "roberta_cos")
def roberta_base_architecture(args):
    base_architecture(args)

@register_model_architecture("roberta_head", "roberta_cos_type2")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = 2

@register_model_architecture("roberta_head", "roberta_rope")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_rope = True

### 单位阵
@register_model_architecture("roberta_head", "roberta_orpe_1_1")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("roberta_head", "roberta_orpe_1b_1")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("roberta_head", "roberta_orpe_1c_1")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("roberta_head", "roberta_orpe_1d_1")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True
    

@register_model_architecture("roberta_head", "roberta_orpe_2_1")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("roberta_head", "roberta_orpe_3_1")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 1
### 单位阵

### Odd_Even
@register_model_architecture("roberta_head", "roberta_orpe_1_5")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("roberta_head", "roberta_orpe_1d_5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("roberta_head", "roberta_orpe_2_5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("roberta_head", "roberta_orpe_3_5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5
### Odd_Even

### DCT
@register_model_architecture("roberta_head", "roberta_orpe_1_2")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("roberta_head", "roberta_orpe_1b_2")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("roberta_head", "roberta_orpe_1c_2")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "c"

@register_model_architecture("roberta_head", "roberta_orpe_1d_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_learned = True

@register_model_architecture("roberta_head", "roberta_orpe_2_2")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("roberta_head", "roberta_orpe_3_2")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 2
### DCT

### Householder
@register_model_architecture("roberta_head", "roberta_orpe_1_3")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("roberta_head", "roberta_orpe_1d_3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("roberta_head", "roberta_orpe_2_3")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("roberta_head", "roberta_orpe_3_3")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 3
    args.p_matrix = 3
### Householder

### Householder learned
@register_model_architecture("roberta_head", "roberta_orpe_1_3a")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True

@register_model_architecture("roberta_head", "roberta_orpe_1d_3a")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
### Householder learned

###### Fourier
@register_model_architecture("roberta_head", "roberta_orpe_4_4")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("roberta_head", "roberta_orpe_4d_4")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True

###### Fourier

###### abl
@register_model_architecture("roberta_head", "roberta_spe")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = False
    args.use_spe = True

@register_model_architecture("roberta_head", "roberta_per")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = False
    args.use_spe = False
    args.use_permutate = True
###### abl



### base model

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_rms_norm")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "rmsnorm"

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_6head")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 6
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_lambda0")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.lambda_ = 0

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_lambda05")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.lambda_ = 0.5

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_usek")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.mem_use_q = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_sigmoid")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "sigmoid"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_exp")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "exp"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_actt_postnorm")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.encoder_normalize_before = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_rope")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_rope_prenorm")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_rope_use_v")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True
    args.use_v = True

@register_model_architecture("roberta_mem", "roberta_mem_hasout_leak_out_no_act")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "leak"
    args.norm_type = "layernorm"
    args.out_use_act = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_leak_out_no_act_0.01")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "leak"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.negative_slope = 0.01

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_rope_use_v_multi_head")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True
    args.use_v = True

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_rope_c")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True
    args.rope_type = "c"

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_rope_no_abs_pos")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_rope_multi_head")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

@register_model_architecture("roberta_mem", "roberta_mem_hasout_elu_out_no_act_gatednorm")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "elu"
    args.norm_type = "gatedrmsnorm"
    args.out_use_act = False

@register_model_architecture("roberta_mem", "roberta_mem_hasout_relu_out_no_act_rope")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "relu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

@register_model_architecture("roberta_mem", "roberta_mem_hasout_1+elu_out_no_act_rope")
def roberta_cosformer_architecture(args):
    base_architecture(args)
    args.use_relu = getattr(args, "use_relu", True)
    args.max_l = getattr(args, "max_l", 512)
    args.causal = False
    args.has_out = False
    args.encoder_attention_heads = 1
    # args.encoder_normalize_before = True
    args.use_gelu = True
    args.mem_use_gelu = True
    args.has_out = True
    ## add
    args.act_fun = "1+elu"
    args.norm_type = "layernorm"
    args.out_use_act = False
    args.use_rope = True

### linear orpe
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = False
    args.kernel_type = "1+elu"

### 单位阵
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1b_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "b"

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1c_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_type = "c"

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1d_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    args.theta_learned = True

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_2_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 1

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_3_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 1
### 单位阵

### rope
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_rope")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = False
    args.kernel_type = "1+elu"
    args.use_rope = True
### rope

### Odd Even
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1_5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1d_5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_2_5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 5

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_3_5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5
### Odd Even

### DCT
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1b_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "b"

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1c_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_type = "c"

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1d_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 2
    args.theta_learned = True

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_2_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 2

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_3_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 2
### DCT

### Householder
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1_3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1b_3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_type = "b"

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1c_3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_type = "c"

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1d_3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_2_3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 2
    args.p_matrix = 3

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_3_3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 3
### Householder

### Householder learned
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1_3a")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.householder_learned = True

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1d_3a")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
### Householder learned

###### Fourier
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_4_4")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_4d_4")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4
    args.theta_learned = True

###### Fourier

###### abl
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_spe")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = False
    args.kernel_type = "1+elu"
    args.use_spe = True

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_per")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = False
    args.kernel_type = "1+elu"
    args.use_spe = False
    args.use_permutate = True
###### abl


############# NormAttentionEncoder
# linear: attention_type = 1
# local: attention_type = 2
# local, ... , local, linear, ... ,linear
# 统一格式_2_1_w32_h1
# 1, 2, 3: 1表示linear, 2表示分组, 3表示window, c表示window size, h表示头数
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.encoder_chunk_size = 32
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.encoder_chunk_size = 32
    args.attention_types = [2 for _ in range(args.encoder_layers)] 

######################################### add
################ pure window
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w32_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.left_window = 1
    args.right_window = 1
    args.attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w64_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w32_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w64_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.attention_types = [2 for _ in range(args.encoder_layers)] 

################ pure chunk
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w32_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w64_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w32_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w64_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers)] 

################ mix
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w64_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w64_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w32_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w32_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

######## type 2
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h12_p0.25")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h12_p0.75")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.25")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.25)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.75")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.75)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

##### window size
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w128_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 128
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w256_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 256
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
##### window size
######## type 2

######## Orpe
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w32_h12_13")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ###### orpe
    args.encoder_use_orpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w64_h12_13")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ###### orpe
    args.encoder_use_orpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w128_h12_13")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 128
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ###### orpe
    args.encoder_use_orpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w256_h12_13")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 256
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ###### orpe
    args.encoder_use_orpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True


######## Orpe

######## Pure Linear 百
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_11")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.attention_types = [1 for _ in range(args.encoder_layers)]
######## Pure Linear

######## Linear + Norm 分比测试
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1_2_w64_h12_p0.25")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1_2_w64_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1_2_w64_h12_p0.75")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]

######## act测试
######## 0: elu, 1: relu, 2: silu
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_act21")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'silu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_act22")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'silu'
    args.local_act_fun = 'silu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_softmax")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.use_softmax = True
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

######## act测试

######## dropout
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_drop")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.use_dropout = True
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
######## dropout

######## norm type
##### 0 default
##### 1: simple rms norm
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_10")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_01")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_11")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_11_h12")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_norm_33_h12")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
######## norm type

######## chunk从大变小
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_chunk_stl")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = [16, 16, 32, 32, 64, 64] + [64] * 6
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]


########

######## Linear + Norm


######## Linear + kv act
@register_model_architecture("roberta_norm_attention", "roberta_linear_standard_11")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.attention_types = [1 for _ in range(args.encoder_layers)]
    #### add
    args.encoder_kv_act = "sigmoid"

@register_model_architecture("roberta_norm_attention", "roberta_linear_standard_12")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.attention_types = [1 for _ in range(args.encoder_layers)]
    #### add
    args.encoder_kv_act = "relu"

@register_model_architecture("roberta_norm_attention", "roberta_linear_standard_01")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'identity'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.attention_types = [1 for _ in range(args.encoder_layers)]
    #### add
    args.encoder_kv_act = "sigmoid"

@register_model_architecture("roberta_norm_attention", "roberta_linear_standard_02")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'identity'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.attention_types = [1 for _ in range(args.encoder_layers)]
    #### add
    args.encoder_kv_act = "relu"

######## Linear + kv act

######## GLU
@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_glu_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_normtype_11_glu_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.attention_types = [1 for _ in range(args.encoder_layers)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_normtype_22_glu_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"

### add
@register_model_architecture("roberta_norm_attention", "roberta_glu_rms_layer")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_dropout_rms_layer")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.use_dropout = True
    ### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_moreheads_rms_layer")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_use_orpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
######## GLU

######## Heads
@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_head_12_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_11_head_1_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_22_head_1_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.attention_types = [2 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_head_12_24")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_head_24_24")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
######## Heads

######################################### add

############# NormMixAttentionEncoder, 并联太慢
@register_model_architecture("roberta_norm_mix_attention", "roberta_norm_mix_type_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_orpe = False
    args.encoder_chunk_size = 32

@register_model_architecture("roberta_norm_mix_attention", "roberta_norm_mix_type_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_orpe = False
    args.encoder_chunk_size = 64

###### only rel
@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1d_3_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_head", "roberta_orpe_1d_3_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_linear_orpe", "roberta_1+elu_1_1_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.causal = False
    args.use_orpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 1
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_head", "roberta_orpe_1_1_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    args.weight_type = -1
    args.use_orpe = True
    args.core_matrix = 1
    args.p_matrix = 1
    # add
    args.no_token_positional_embeddings = True


###### only rel