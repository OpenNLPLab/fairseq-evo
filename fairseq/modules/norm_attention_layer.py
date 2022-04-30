# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, TransformerEncoderLayer, TransformerDecoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
# merge attention
from fairseq.modules import NormLocalAttention, NormLinearAttention
from fairseq.modules import GLU

# class NormAttentionEncoderLayer(TransformerEncoderLayer):
#     def __init__(self, args):
#         super().__init__(args)

#     def build_self_attention(self, embed_dim, args):
#         if args.attention_type == 1:
#             print("======================")
#             print("use norm_linear")
#             print("======================")
#             return NormLinearAttention(
#                 embed_dim,
#                 args.encoder_attention_heads,
#                 dropout=args.attention_dropout,
#                 self_attention=True,
#                 q_noise=self.quant_noise,
#                 qn_block_size=self.quant_noise_block_size,
#                 # add
#                 index=args.index,
#                 use_relu=getattr(args, "use_relu", False),
#                 use_elu=getattr(args, "use_elu", False),
#                 use_leak=getattr(args, "use_leak", False),
#                 use_bound=getattr(args, "use_bound", False),
#                 max_l=getattr(args, "max_l", 1024),
#                 has_out=getattr(args, "has_out", False),
#                 weight_type=getattr(args, "weight_type", 1),
#                 c=getattr(args, "c", 1.0),
#                 v_act=getattr(args, "v_act", False),
#                 use_dropout=getattr(args, "use_dropout", False),
#                 p=getattr(args, "p", 0.5),
#                 use_layer_norm=getattr(args, "use_layer_norm", False),
#                 qk_layer_norm=getattr(args, "qk_layer_norm", False),
#                 seq_dropout=getattr(args, "seq_dropout", False),
#                 seq_p=getattr(args, "seq_p", 0.3),
#                 lambda_=getattr(args, "lambda_", 0.001),
#                 use_gelu=getattr(args, "use_gelu", False),
#                 mem_use_gelu=getattr(args, "mem_use_gelu", False),
#                 mem_use_grad=getattr(args, "mem_use_grad", True),
#                 mem_use_q=getattr(args, "mem_use_q", True),
#                 mem_use_k=getattr(args, "mem_use_k", False),
#                 attention_use_layer_norm=getattr(args, "attention_use_layer_norm", True),
#                 model_update_freq=getattr(args, "model_update_freq", 1),
#                 act_fun=getattr(args, "linear_act_fun", "gelu"),
#                 out_use_act=getattr(args, "out_use_act", True),
#                 init_type=getattr(args, "init_type", "default"),
#                 norm_type=getattr(args, "norm_type", "layernorm"),
#                 use_rope=getattr(args, "use_rope", False),
#                 rope_type=getattr(args, "rope_type", "a"),
#                 use_v=getattr(args, "use_v", False),
#                 negative_slope=getattr(args, "negative_slope", 0.1),
#                 # add
#                 causal=getattr(args, "encoder_causal", False),
#                 use_orpe=getattr(args, "encoder_use_orpe", True),
#                 core_matrix=getattr(args, "encoder_core_matrix", 1),
#                 p_matrix=getattr(args, "encoder_p_matrix", 1),
#                 max_positions=getattr(args, "encoder_max_positions", 512),
#                 theta_type=getattr(args, "encoder_theta_type", "a"),
#                 theta_learned=getattr(args, "encoder_theta_learned", False), 
#                 householder_learned=getattr(args, "encoder_householder_learned", False),
#                 kv_act=getattr(args, "encoder_kv_act", "identity")
#             )
#         else:
#             print("======================")
#             print("use norm_local")
#             print("======================")
#             return NormLocalAttention(
#                 embed_dim,
#                 args.encoder_attention_heads,
#                 dropout=args.attention_dropout,
#                 self_attention=True,
#                 q_noise=self.quant_noise,
#                 qn_block_size=self.quant_noise_block_size,
#                 # add
#                 act_fun=getattr(args, "local_act_fun", "relu"),
#                 negative_slope=getattr(args, "negative_slope", 0.1),
#                 # add
#                 causal=getattr(args, "encoder_causal", False),
#                 use_orpe=getattr(args, "encoder_use_orpe", True),
#                 core_matrix=getattr(args, "encoder_core_matrix", 1),
#                 p_matrix=getattr(args, "encoder_p_matrix", 1),
#                 max_positions=getattr(args, "encoder_max_positions", 512),
#                 theta_type=getattr(args, "encoder_theta_type", "a"),
#                 theta_learned=getattr(args, "encoder_theta_learned", False), 
#                 householder_learned=getattr(args, "encoder_householder_learned", False),
#                 # add
#                 chunk_size=getattr(args, "encoder_chunk_size", 32),
#                 left_window=getattr(args, "left_window", 1),
#                 right_window=getattr(args, "right_window", 1),
#                 group_type=getattr(args, "group_type", "chunk"),
#                 use_softmax=getattr(args, "use_softmax", False),
#                 norm_type=getattr(args, "local_norm_type", "gatedrmsnorm")
#             )

class NormAttentionDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if args.attention_type == 1:
            print("======================")
            print("use norm_linear")
            print("======================")
            return NormLinearAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                index=args.index,
                use_relu=getattr(args, "use_relu", False),
                use_elu=getattr(args, "use_elu", False),
                use_leak=getattr(args, "use_leak", False),
                use_bound=getattr(args, "use_bound", False),
                max_l=getattr(args, "max_l", 1024),
                has_out=getattr(args, "has_out", False),
                weight_type=getattr(args, "weight_type", 1),
                c=getattr(args, "c", 1.0),
                v_act=getattr(args, "v_act", False),
                use_dropout=getattr(args, "use_dropout", False),
                p=getattr(args, "p", 0.5),
                use_layer_norm=getattr(args, "use_layer_norm", False),
                qk_layer_norm=getattr(args, "qk_layer_norm", False),
                seq_dropout=getattr(args, "seq_dropout", False),
                seq_p=getattr(args, "seq_p", 0.3),
                lambda_=getattr(args, "lambda_", 0.001),
                use_gelu=getattr(args, "use_gelu", False),
                mem_use_gelu=getattr(args, "mem_use_gelu", False),
                mem_use_grad=getattr(args, "mem_use_grad", True),
                mem_use_q=getattr(args, "mem_use_q", True),
                mem_use_k=getattr(args, "mem_use_k", False),
                attention_use_layer_norm=getattr(args, "attention_use_layer_norm", True),
                model_update_freq=getattr(args, "model_update_freq", 1),
                act_fun=getattr(args, "linear_act_fun", "gelu"),
                out_use_act=getattr(args, "out_use_act", True),
                init_type=getattr(args, "init_type", "default"),
                norm_type=getattr(args, "norm_type", "layernorm"),
                use_rope=getattr(args, "use_rope", False),
                rope_type=getattr(args, "rope_type", "a"),
                use_v=getattr(args, "use_v", False),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "decoder_causal", True),
                use_orpe=getattr(args, "decoder_use_orpe", True),
                core_matrix=getattr(args, "decoder_core_matrix", 1),
                p_matrix=getattr(args, "decoder_p_matrix", 1),
                max_positions=getattr(args, "decoder_max_positions", 512),
                theta_type=getattr(args, "decoder_theta_type", "a"),
                theta_learned=getattr(args, "decoder_theta_learned", False), 
                householder_learned=getattr(args, "decoder_householder_learned", False),
                kv_act=getattr(args, "decoder_kv_act", "identity")
            )
        else:
            print("======================")
            print("use norm_local")
            print("======================")
            return NormLocalAttention(
                embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not getattr(args, "cross_self_attention", False),
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                act_fun=getattr(args, "local_act_fun", "relu"),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "decoder_causal", True),
                use_orpe=getattr(args, "decoder_use_orpe", True),
                core_matrix=getattr(args, "decoder_core_matrix", 1),
                p_matrix=getattr(args, "decoder_p_matrix", 1),
                max_positions=getattr(args, "decoder_max_positions", 512),
                theta_type=getattr(args, "decoder_theta_type", "a"),
                theta_learned=getattr(args, "decoder_theta_learned", False), 
                householder_learned=getattr(args, "decoder_householder_learned", False),
                # add
                chunk_size=getattr(args, "decoder_chunk_size", 32),
                left_window=getattr(args, "left_window", 1),
                right_window=getattr(args, "right_window", 1),
                group_type=getattr(args, "group_type", "chunk"),
                use_softmax=getattr(args, "use_softmax", False),
                norm_type=getattr(args, "local_norm_type", "gatedrmsnorm")
            )

    def build_encoder_attention(self, embed_dim, args):
        if args.attention_type == 1:
            print("======================")
            print("use norm_linear")
            print("======================")
            return NormLinearAttention(
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                # add
                index=args.index,
                use_relu=getattr(args, "use_relu", False),
                use_elu=getattr(args, "use_elu", False),
                use_leak=getattr(args, "use_leak", False),
                use_bound=getattr(args, "use_bound", False),
                max_l=getattr(args, "max_l", 1024),
                has_out=getattr(args, "has_out", False),
                weight_type=getattr(args, "weight_type", 1),
                c=getattr(args, "c", 1.0),
                v_act=getattr(args, "v_act", False),
                use_dropout=getattr(args, "use_dropout", False),
                p=getattr(args, "p", 0.5),
                use_layer_norm=getattr(args, "use_layer_norm", False),
                qk_layer_norm=getattr(args, "qk_layer_norm", False),
                seq_dropout=getattr(args, "seq_dropout", False),
                seq_p=getattr(args, "seq_p", 0.3),
                lambda_=getattr(args, "lambda_", 0.001),
                use_gelu=getattr(args, "use_gelu", False),
                mem_use_gelu=getattr(args, "mem_use_gelu", False),
                mem_use_grad=getattr(args, "mem_use_grad", True),
                mem_use_q=getattr(args, "mem_use_q", True),
                mem_use_k=getattr(args, "mem_use_k", False),
                attention_use_layer_norm=getattr(args, "attention_use_layer_norm", True),
                model_update_freq=getattr(args, "model_update_freq", 1),
                act_fun=getattr(args, "linear_act_fun", "gelu"),
                out_use_act=getattr(args, "out_use_act", True),
                init_type=getattr(args, "init_type", "default"),
                norm_type=getattr(args, "norm_type", "layernorm"),
                use_rope=getattr(args, "use_rope", False),
                rope_type=getattr(args, "rope_type", "a"),
                use_v=getattr(args, "use_v", False),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "encoder_causal", False),
                use_orpe=getattr(args, "encoder_use_orpe", True),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                kv_act=getattr(args, "encoder_kv_act", "identity")
            )
        else:
            print("======================")
            print("use norm_local")
            print("======================")
            return NormLocalAttention(
                embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                act_fun=getattr(args, "local_act_fun", "relu"),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "encoder_causal", False),
                use_orpe=getattr(args, "encoder_use_orpe", True),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                # add
                chunk_size=getattr(args, "encoder_chunk_size", 32),
                left_window=getattr(args, "left_window", 1),
                right_window=getattr(args, "right_window", 1),
                group_type=getattr(args, "group_type", "chunk"),
                use_softmax=getattr(args, "use_softmax", False),
                norm_type=getattr(args, "local_norm_type", "gatedrmsnorm")
            )


class NormAttentionEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.use_glu = getattr(args, "use_glu", False)
        self.glu_act = getattr(args, "glu_act", False)
        print("=============================")
        print(f"self.use_glu {self.use_glu}")
        print(f"self.glu_act {self.glu_act}")
        print("=============================")

        if self.use_glu:
            d1 = self.embed_dim
            d2 = int(8 * d1 / 3)
            self.glu = GLU(d1, d2, self.glu_act)
        else:
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        if args.attention_type == 1:
            print("======================")
            print("use norm_linear")
            print("======================")
            return NormLinearAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                index=args.index,
                use_relu=getattr(args, "use_relu", False),
                use_elu=getattr(args, "use_elu", False),
                use_leak=getattr(args, "use_leak", False),
                use_bound=getattr(args, "use_bound", False),
                max_l=getattr(args, "max_l", 1024),
                has_out=getattr(args, "has_out", False),
                weight_type=getattr(args, "weight_type", 1),
                c=getattr(args, "c", 1.0),
                v_act=getattr(args, "v_act", False),
                use_dropout=getattr(args, "use_dropout", False),
                p=getattr(args, "p", 0.5),
                use_layer_norm=getattr(args, "use_layer_norm", False),
                qk_layer_norm=getattr(args, "qk_layer_norm", False),
                seq_dropout=getattr(args, "seq_dropout", False),
                seq_p=getattr(args, "seq_p", 0.3),
                lambda_=getattr(args, "lambda_", 0.001),
                use_gelu=getattr(args, "use_gelu", False),
                mem_use_gelu=getattr(args, "mem_use_gelu", False),
                mem_use_grad=getattr(args, "mem_use_grad", True),
                mem_use_q=getattr(args, "mem_use_q", True),
                mem_use_k=getattr(args, "mem_use_k", False),
                attention_use_layer_norm=getattr(args, "attention_use_layer_norm", True),
                model_update_freq=getattr(args, "model_update_freq", 1),
                act_fun=getattr(args, "linear_act_fun", "gelu"),
                out_use_act=getattr(args, "out_use_act", True),
                init_type=getattr(args, "init_type", "default"),
                norm_type=getattr(args, "norm_type", "layernorm"),
                use_rope=getattr(args, "use_rope", False),
                rope_type=getattr(args, "rope_type", "a"),
                use_v=getattr(args, "use_v", False),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "encoder_causal", False),
                use_orpe=getattr(args, "encoder_use_orpe", True),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                kv_act=getattr(args, "encoder_kv_act", "identity")
            )
        else:
            print("======================")
            print("use norm_local")
            print("======================")
            return NormLocalAttention(
                embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                q_noise=self.quant_noise,
                qn_block_size=self.quant_noise_block_size,
                # add
                act_fun=getattr(args, "local_act_fun", "relu"),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "encoder_causal", False),
                use_orpe=getattr(args, "encoder_use_orpe", True),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                # add
                chunk_size=getattr(args, "encoder_chunk_size", 32),
                left_window=getattr(args, "left_window", 1),
                right_window=getattr(args, "right_window", 1),
                group_type=getattr(args, "group_type", "chunk"),
                use_softmax=getattr(args, "use_softmax", False),
                norm_type=getattr(args, "local_norm_type", "gatedrmsnorm")
            )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if self.use_glu:
            x = self.glu(x)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x
