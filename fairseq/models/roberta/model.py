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
# from fairseq.models.transformer import ReformerEncoder, TransformerMergeEncoder
# head
from fairseq.models.transformer import TransformerEncoderPlus
# sparse relu
from fairseq.models.transformer import TransformerSparseReluEncoderLayer
# cosformer
from fairseq.models.transformer import CosformerEncoder
# head
from fairseq.models.transformer import TransformerEncoderPlus
# cosformer
from fairseq.models.transformer import CosformerEncoder_
# pcc
# from fairseq.models.transformer import PccEncoder
# cos
# from fairseq.models.transformer import TransformerCosEncoder
# ReLA
from fairseq.models.transformer import ReLAEncoder
# linear kernel with urpe
from fairseq.models.transformer import LinearKernelAttentionEncoder
# norm Attention
from fairseq.models.transformer import NormAttentionEncoder
# norm mix attention
from fairseq.models.transformer import NormMixAttentionEncoder
# ls attention
# from fairseq.models.transformer import LSAttentionEncoder
# performer
# from fairseq.models.transformer import PerformerEncoder
# normlinear
from fairseq.models.transformer import LinearVanillaAttentionEncoder

logger = logging.getLogger(__name__)

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
        init_method = getattr(args, 'init_method', "default")
        print(f"init_method {init_method}")
        if init_method == "default":
            print("default init")
            self.apply(init_bert_params)
        else:
            print("small init")
            print(init_method)
            self.apply(small_init_weights)

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
        encoder = TransformerEncoderPlus(args, dictionary, embed_tokens)
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


# weight diff

### Flash
# class RobertaFlashEncoder(RobertaEncoder):
#     """RoBERTa encoder."""

#     def __init__(self, args, dictionary):
#         super().__init__(args, dictionary)

#     def build_encoder(self, args, dictionary, embed_tokens):
#         encoder = FlashEncoder(args, dictionary, embed_tokens)
#         encoder.apply(init_bert_params)
#         return encoder

# @register_model("roberta_flash")
# class RobertaFlashModel(RobertaModel):
#     def __init__(self, args, encoder):
#         super().__init__(args, encoder)

#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""

#         # make sure all arguments are present
#         base_architecture(args)

#         if not hasattr(args, "max_positions"):
#             args.max_positions = args.tokens_per_sample

#         encoder = RobertaFlashEncoder(args, task.source_dictionary)
#         return cls(args, encoder)

# Flash Linear

#### 






############# Linear Urpe
# class RobertaLinearUrpeEncoder(RobertaEncoder):
#     """RoBERTa encoder."""

#     def __init__(self, args, dictionary):
#         super().__init__(args, dictionary)

#     def build_encoder(self, args, dictionary, embed_tokens):
#         encoder = LinearKernelAttentionEncoder(args, dictionary, embed_tokens)
#         encoder.apply(init_bert_params)
#         return encoder

# @register_model("roberta_linear_urpe")
# class RobertaLinearUrpeModel(RobertaModel):
#     def __init__(self, args, encoder):
#         super().__init__(args, encoder)

#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""

#         # make sure all arguments are present
#         base_architecture(args)

#         if not hasattr(args, "max_positions"):
#             args.max_positions = args.tokens_per_sample

#         encoder = RobertaLinearUrpeEncoder(args, task.source_dictionary)
#         return cls(args, encoder)

############# NormAttentionEncoder
class RobertaNormEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = NormAttentionEncoder(args, dictionary, embed_tokens)
        # encoder.apply(init_bert_params)
        init_method = getattr(args, 'init_method', "default")
        encoder.apply(init_bert_params)
        # print(f"init_method {init_method}")
        # if init_method == "small_embdding":
        #     encoder.apply(self._init_weights)
        # else:
        #     encoder.apply(init_bert_params)
        return encoder
    
    def _init_weights(self, module):
        print("small init")
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)        

        if isinstance(module, (nn.Embedding)):
            print("here")
            # nn.init.uniform_(module.weight, a=-1e-4, b=1e-4) # SmallInit(Emb)
            print(torch.norm(module.weight.data))
            module.weight.data.normal_(mean=0.0, std=1e-5)
            print(torch.norm(module.weight.data))

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@register_model("roberta_norm_attention")
class RobertaNormUrpeModel(RobertaModel):
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



############# PerformerEncoder
# class RobertaPerformerEncoder(RobertaEncoder):
#     """RoBERTa encoder."""

#     def __init__(self, args, dictionary):
#         super().__init__(args, dictionary)

#     def build_encoder(self, args, dictionary, embed_tokens):
#         encoder = PerformerEncoder(args, dictionary, embed_tokens)
#         encoder.apply(init_bert_params)
#         return encoder

# @register_model("roberta_performer")
# class RobertaPerformerModel(RobertaModel):
#     def __init__(self, args, encoder):
#         super().__init__(args, encoder)

#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""

#         # make sure all arguments are present
#         base_architecture(args)

#         if not hasattr(args, "max_positions"):
#             args.max_positions = args.tokens_per_sample

#         encoder = RobertaPerformerEncoder(args, task.source_dictionary)
#         return cls(args, encoder)
############# PerformerEncoder

############# LinearVanillaEncoder
class RobertaLinearVanillaEncoder(RobertaEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = LinearVanillaAttentionEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

@register_model("roberta_linear_vanilla")
class RobertaLinearVanillaModel(RobertaModel):
    def __init__(self, args, encoder):
        super().__init__(args, encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaLinearVanillaEncoder(args, task.source_dictionary)
        return cls(args, encoder)
############# PerformerEncoder

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



# normalize
@register_model_architecture("roberta_normalize", "roberta_normalize_base")
def roberta_cosformer_architecture(args):
    base_architecture(args)










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
    args.encoder_use_urpe = False
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

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
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w64_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w32_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_3_w64_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w64_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w32_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_2_w64_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)] 

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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w64_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w64_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w32_h1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_3_1_w32_h8")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 8
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]

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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w64_h12_p0.75")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.25")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w32_h12_p0.75")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 128
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_2_1_w256_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 256
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
##### window size
######## type 2

######## Urpe
@register_model_architecture("roberta_norm_attention", "roberta_norm_type_w32_h12_13")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 32
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ###### urpe
    args.encoder_use_urpe = True
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ###### urpe
    args.encoder_use_urpe = True
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 128
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ###### urpe
    args.encoder_use_urpe = True
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 256
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True


######## Urpe

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
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1_2_w64_h12_p0.5")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_1_2_w64_h12_p0.75")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [1 for _ in range(l)] + [2 for _ in range(args.encoder_layers - l)]

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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_type_stand_act22")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'silu'
    args.local_act_fun = 'silu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.use_dropout = True
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = [16, 16, 32, 32, 64, 64] + [64] * 6
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]


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
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
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
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
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
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
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
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]
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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"

# speed test
@register_model_architecture("roberta_norm_attention", "roberta_rms_layer_standard")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_dropout_rms_layer")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
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
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"

#### change multiple
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_rms_layer_small")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2

#### layer norm rms
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_ln_rms")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
#### layer norm rms

#### GLU + ORPE
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_no_urpe")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = False

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_urpe_1d3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_urpe_1d3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
#### GLU + ORPE

#### GLU + FINAL ACT
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.fina_act = "elu"

#### change multiple
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.fina_act = "elu"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.fina_act = "elu"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    args.fina_act = "elu"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_swish")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.fina_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_swish")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.fina_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_swish")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.fina_act = "swish"

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_swish")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    args.fina_act = "swish"
#### GLU + FINAL ACT

#### GLU + ORPE + DROPOUT
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_dropout02")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    args.glu_dropout = 0.2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3_dropout02")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    args.glu_dropout = 0.2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_urpe_1d3_dropout02")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    args.glu_dropout = 0.2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_urpe_1d3_dropout02")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    args.glu_dropout = 0.2
#### GLU + ORPE + DROPOUT

#### GLU + ORPE + NO ABS
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_small_ln_rms_urpe_1d3_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_urpe_1d3_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### no abs
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_layernorm_small_urpe_1d3_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.local_norm_type = "layernorm"
    args.norm_type = "layernorm"
    args.multiple = 2
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### no abs
    args.no_token_positional_embeddings = True
#### GLU + ORPE
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
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_11_head_1_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_22_head_1_1")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 1
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_head_12_24")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]

@register_model_architecture("roberta_norm_attention", "roberta_normtype_21_head_24_24")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = 'elu'
    args.local_act_fun = 'relu'
    args.max_l = getattr(args, 'max_l', 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = 'chunk'
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.5)
    args.encoder_attention_heads_list = [12 for _ in range(l)] + [24 for _ in range(args.encoder_layers - l)]
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
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
    args.encoder_use_urpe = False
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
    args.encoder_use_urpe = False
    args.encoder_chunk_size = 64

###### only rel







###### only rel

###### small init + pure rms norm + urpe
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_small_init")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_small_init")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_small_init_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
    # add
    args.no_token_positional_embeddings = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_small_init_no_abs")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"
    # add
    args.no_token_positional_embeddings = True
###### small init + pure rms norm + urpe

###### pure rms norm + urpe + GEGLU
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_geglu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_geglu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_geglu_small_init")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_geglu_small_init")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "gelu"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    args.init_method = "small_embdding"

###### pure rms norm + urpe + GEGLU

###### pure rms norm + urpe + weight
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_laplace")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 1

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_laplace")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 1

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_gaussian")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_gaussian")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    #### weight
    args.weight_type = 2
###### pure rms norm + urpe + weight

###### final dropout
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_final_dropout")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    # final dropout
    args.use_final_dropout = True
    args.final_dropout = 0.1

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_final_dropout")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
    # final dropout
    args.use_final_dropout = True
    args.final_dropout = 0.1
###### final dropout

###### relu2
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_relu2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "relu2"
    args.local_act_fun = "relu2"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_relu2")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "relu2"
    args.local_act_fun = "relu2"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
###### relu2

###### linear chunk
@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_linear_chunk")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_linear_chunk")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_linear_chunk_32")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_linear_chunk_32")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_pure_rms_urpe_1d3_linear_chunk_16")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 16
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_glu_small_pure_rms_urpe_1d3_linear_chunk_16")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "linear_chunk"
    args.encoder_chunk_size = 16
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.multiple = 2
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    #### pure layernorm 
    args.embdding_layernorm = "simplermsnorm"
    args.final_layernorm = "simplermsnorm"
###### linear chunk

############## GLU




###### performer
# @register_model_architecture("roberta_performer", "roberta_performer")
# def roberta_base_architecture(args):
#     base_architecture(args)
#     args.approx_attn_dim = 64
#     args.causal = False
###### performer


##### for visual
@register_model_architecture("roberta", "roberta_base_one_head")
def roberta_base_architecture(args):
    base_architecture(args)
    args.encoder_attention_heads = 1


##### for visual

##### local global visual
#### GLU + ORPE
@register_model_architecture("roberta_norm_attention", "roberta_ffn_all_rms_layer_ln_rms_urpe_1d3")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_pure_chunk")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_pure_linear")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_linear_chunk")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True

# mix 并联
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_parallel")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 1

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_linear_local")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_local_linear")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 3

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_25_75")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_75_25")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
# mix 并联

# chunk size
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_chunk32")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_chunk128")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 128
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
# chunk size

########### softmax + 1 + elu
@register_model_architecture("roberta_norm_attention", "roberta_ffn_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True


@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_pure_chunk")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_pure_linear")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_linear_chunk")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True

# mix 并联
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_parallel")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 1
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_linear_local")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 2
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_local_linear")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    #### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ### type
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers)]
    args.forward_type = 3
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_25_75")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.25)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_75_25")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    l = int(args.encoder_layers * 0.75)
    args.encoder_attention_types = [2 for _ in range(l)] + [1 for _ in range(args.encoder_layers - l)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True
# mix 并联

##### local global new version
@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_no_urpe_softmax_1+elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = False
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_small")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True
    args.multiple = 2

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_chunk32")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 32
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_glu_all_rms_layer_ln_rms_urpe_1d3_softmax_1+elu_chunk128")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 128
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### urpe
    args.encoder_use_urpe = True
    args.encoder_core_matrix = 1
    args.encoder_p_matrix = 3
    args.encoder_theta_learned = True
    ###### softmax
    args.use_softmax = True
########### softmax + 1 + elu

########### performer / 1 + elu / cosformer
@register_model_architecture("roberta_linear_vanilla", "roberta_performer_vanilla")
def roberta_base_architecture_performer_vanilla(args):
    base_architecture(args)
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [-1 for _ in range(args.encoder_layers // 2)]
    args.approx_attn_dim = 64
    args.causal = False

@register_model_architecture("roberta_linear_vanilla", "roberta_vanilla_performer")
def roberta_base_architecture_vanilla_performer(args):
    base_architecture(args)
    args.encoder_attention_types = [-1 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)] 
    args.approx_attn_dim = 64
    args.causal = False

@register_model_architecture("roberta_linear_vanilla", "roberta_1+elu_vanilla")
def roberta_base_architecture_1elu_vanilla(args):
    base_architecture(args)
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers // 2)] + [-1 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"

@register_model_architecture("roberta_linear_vanilla", "roberta_vanilla_1+elu")
def roberta_base_architecture_vanilla_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [-1 for _ in range(args.encoder_layers // 2)] + [3 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"

@register_model_architecture("roberta_linear_vanilla", "roberta_1+elu_vanilla_no_urpe")
def roberta_base_architecture_1elu_vanilla(args):
    base_architecture(args)
    args.encoder_attention_types = [3 for _ in range(args.encoder_layers // 2)] + [-1 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    args.use_urpe = False

@register_model_architecture("roberta_linear_vanilla", "roberta_vanilla_1+elu_no_urpe")
def roberta_base_architecture_vanilla_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [-1 for _ in range(args.encoder_layers // 2)] + [3 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    args.use_urpe = False

#### local softmax + other linear
@register_model_architecture("roberta_linear_vanilla", "roberta_local_softmax_cosformer")
def roberta_base_architecture_local_cos(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### softmax
    args.use_softmax = True
    ###### cosformer
    args.use_relu = True

@register_model_architecture("roberta_linear_vanilla", "roberta_local_softmax_1+elu")
def roberta_base_architecture_local_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [3 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    ### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_linear_vanilla", "roberta_local_softmax_performer")
def roberta_base_architecture_local_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    ### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### softmax
    args.use_softmax = True

@register_model_architecture("roberta_norm_attention", "roberta_window_softmax_1+elu")
def roberta_base_architecture_window_norm_1elu(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### softmax
    args.use_softmax = True
#### local softmax + other linear

#### local relu + other linear
@register_model_architecture("roberta_linear_vanilla", "roberta_local_relu_cosformer")
def roberta_base_architecture_local_cos(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    ### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
    ###### cosformer
    args.use_relu = True

@register_model_architecture("roberta_linear_vanilla", "roberta_local_relu_1+elu")
def roberta_base_architecture_local_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [3 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    ### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"

@register_model_architecture("roberta_linear_vanilla", "roberta_local_relu_performer")
def roberta_base_architecture_local_1elu(args):
    base_architecture(args)
    args.encoder_attention_types = [5 for _ in range(args.encoder_layers // 2)] + [2 for _ in range(args.encoder_layers // 2)]
    args.kernel_type = "1+elu"
    ### add
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "chunk"
    args.encoder_chunk_size = 64
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"

@register_model_architecture("roberta_norm_attention", "roberta_window_relu_elu")
def roberta_base_architecture_window_relu_elu(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.local_act_fun = "relu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.encoder_attention_heads = 12
    args.encoder_use_urpe = False
    args.group_type = "window"
    args.encoder_chunk_size = 64
    args.left_window = 1
    args.right_window = 1
    args.encoder_attention_types = [2 for _ in range(args.encoder_layers // 2)] + [1 for _ in range(args.encoder_layers // 2)]
    args.local_norm_type = "simplermsnorm"
    args.norm_type = "simplermsnorm"
    ### glu
    args.use_glu = True
    args.glu_act = "swish"
    args.attn_type = "simplermsnorm"
#### local relu + other linear

########### performer / 1 + elu / cosformer

#### norm(1 + elu)
@register_model_architecture("roberta_norm_attention", "roberta_norm_1+elu")
def roberta_base_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "simplermsnorm"
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_layernorm_1+elu")
def roberta_base_architecture_roberta_layernorm_1_elu(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "layernorm"
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_gatedrms_1+elu")
def roberta_base_architecture_roberta_gatedrms_1_elu(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "gatedrmsnorm"
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_norm_elu")
def roberta_base_elu_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "simplermsnorm"
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]

@register_model_architecture("roberta_norm_attention", "roberta_no_norm_1+elu")
def roberta_no_norm_architecture(args):
    base_architecture(args)
    ### add
    args.linear_act_fun = "1+elu"
    args.max_l = getattr(args, "max_l", 512)
    args.has_out = True
    args.norm_type = "simplermsnorm"
    args.attention_use_layer_norm = False
    args.encoder_attention_types = [1 for _ in range(args.encoder_layers)]
#### norm(1 + elu)