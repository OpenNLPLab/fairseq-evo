# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.xformer import LinearKernelModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import BARTHubInterface
from .model import bart_large_architecture

logger = logging.getLogger(__name__)


@register_model("bart_linear_kernel")
class BARTModelLinearKernel(LinearKernelModel):
    __jit_unused_properties__ = ["supported_targets"]

    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
            "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
            "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
            "bart.large.xsum": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz",
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

    @staticmethod
    def add_args(parser):
        super(BARTModelLinearKernel, BARTModelLinearKernel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )

    @property
    def supported_targets(self):
        return {"self"}

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens
        )
        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        eos: int = self.eos
        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(eos), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break
        return x, extra

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        sample_break_mode="eos",
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
            sample_break_mode=sample_break_mode,
            **kwargs,
        )

        return BARTHubInterface(x["args"], x["task"], x["models"][0])

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
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
        self.classification_heads[name] = BARTClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
            ),
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
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

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
            self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                -1, :
            ]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel")
def bart_base_linear_kernel(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)

# Type 1
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_1_3")
def bart_base_linear_kernel_1_3(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # urpe
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3

# Type 2
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_1d_3")
def bart_base_linear_kernel_1d_3(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # urpe
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 3
    args.theta_learned = True
    
# Type 3
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_1d_3a")
def bart_base_linear_kernel_1d_3a(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # urpe
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.theta_learned = True
    args.p_matrix = 3
    args.householder_learned = True
    
# Type 4
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_3_3")
def bart_base_linear_kernel_3_3(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # urpe
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 3
    args.p_matrix = 3
    args.max_positions = 1024

# Type 5
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_1_5")
def bart_base_linear_kernel_1_5(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # urpe
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 5

# Type 6
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_1d_5")
def bart_base_linear_kernel_1d_5(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # urpe
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 1
    args.p_matrix = 5
    args.theta_learned = True

# Type 7
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_3_5")
def bart_base_linear_kernel_3_5(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # urpe
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 3
    args.p_matrix = 5
    
# Type 8
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_4_4")
def bart_base_linear_kernel_4_4(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # urpe
    args.weight_type = -1
    args.use_urpe = True
    args.kernel_type = "1+elu"
    args.core_matrix = 4
    args.p_matrix = 4

# rope
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_1_1")
def bart_base_linear_kernel_1_1(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # rope
    args.weight_type = -1
    args.use_urpe = True
    args.core_matrix = 1
    args.p_matrix = 1

# spe
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_spe")
def bart_base_linear_kernel_spe(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # spe
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = True

# per
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_per")
def bart_base_linear_kernel_per(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # per
    args.max_seq_len = 1024
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.use_permutate = True
    
# t5
@register_model_architecture("bart_linear_kernel", "bart_base_linear_kernel_t5")
def bart_base_linear_kernel_t5(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)
    # t5
    args.weight_type = -1
    args.use_urpe = False
    args.use_spe = False
    args.causal = False
    args.use_t5 = True
