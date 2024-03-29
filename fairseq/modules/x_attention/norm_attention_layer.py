# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import (LayerNorm, TransformerDecoderLayer,
                             TransformerEncoderLayer)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

from ..ffn import GLU
from ..helpers import logging_info
from ..norm import GatedRMSNorm, RMSNorm, SimpleRMSNorm
from .norm_linear_attention import NormLinearAttention
from .norm_local_attention import NormLocalAttention
from .norm_mix_attention import NormMixAttention


class NormAttentionDecoderLayer(nn.Module):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before
        self.use_glu = getattr(args, "use_glu", False)
        self.glu_act = getattr(args, "glu_act", False)
        self.fina_act = getattr(args, "fina_act", "None")
        self.glu_dropout = getattr(args, "glu_dropout", 0.0)

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        # self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        attn_type = getattr(args, 'attn_type', 'layernorm')
        logging_info(f"Decoder Norm Type: {attn_type}")
        if attn_type == "simplermsnorm":
            self.self_attn_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
            self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if self.use_glu:
            d1 = self.embed_dim
            p = 8 / 3
            p = getattr(args, "multiple", p)
            d2 = int(p * d1)
            logging_info(f"GLU multiple {p}")
            self.glu = GLU(d1, d2, self.glu_act, self.fina_act, self.glu_dropout)
        else:
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.decoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.decoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )

        # self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        if attn_type == "simplermsnorm":
            self.final_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        if args.attention_type == 1:
            logging_info("======================")
            logging_info("use norm_linear")
            logging_info("======================")
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
                weight_type=getattr(args, "weight_type", -1),
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
                use_urpe=getattr(args, "decoder_use_urpe", False),
                core_matrix=getattr(args, "decoder_core_matrix", 1),
                p_matrix=getattr(args, "decoder_p_matrix", 1),
                max_positions=getattr(args, "decoder_max_positions", 512),
                theta_type=getattr(args, "decoder_theta_type", "a"),
                theta_learned=getattr(args, "decoder_theta_learned", False), 
                householder_learned=getattr(args, "decoder_householder_learned", False),
                kv_act=getattr(args, "decoder_kv_act", "identity"),
                # final dropout
                use_final_dropout=getattr(args, "use_final_dropout", False),
                final_dropout=getattr(args, "final_dropout", 0.0),
                # Toeplizt
                use_toeplizt=getattr(args, "use_toeplizt", False),
                type_num=getattr(args, "type_num", -1),
                # cos
                use_cos=getattr(args, "use_cos", False),
            )
        elif args.attention_type == 2:
            logging_info("======================")
            logging_info("use norm_local")
            logging_info("======================")
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
                use_urpe=getattr(args, "decoder_use_urpe", False),
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
                norm_type=getattr(args, "local_norm_type", "gatedrmsnorm"),
                # weight
                weight_type=getattr(args, "weight_type", -1),
                # final dropout
                use_final_dropout=getattr(args, "use_final_dropout", False),
                final_dropout=getattr(args, "final_dropout", 0.0),
                index=args.index,
            )
        else:
            return NormMixAttention(
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
                linear_act_fun=getattr(args, "linear_act_fun", "gelu"),
                local_act_fun=getattr(args, "local_act_fun", "relu"),
                out_use_act=getattr(args, "out_use_act", True),
                init_type=getattr(args, "init_type", "default"),
                norm_type=getattr(args, "norm_type", "layernorm"),
                use_rope=getattr(args, "use_rope", False),
                rope_type=getattr(args, "rope_type", "a"),
                use_v=getattr(args, "use_v", False),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "decoder_causal", True),
                use_urpe=getattr(args, "decoder_use_urpe", True),
                core_matrix=getattr(args, "decoder_core_matrix", 1),
                p_matrix=getattr(args, "decoder_p_matrix", 1),
                max_positions=getattr(args, "decoder_max_positions", 512),
                theta_type=getattr(args, "decoder_theta_type", "a"),
                theta_learned=getattr(args, "decoder_theta_learned", False), 
                householder_learned=getattr(args, "decoder_householder_learned", False),
                # add
                chunk_size=getattr(args, "chunk_size", 32),
                forward_type=getattr(args, "forward_type", 1),
            )

    def build_encoder_attention(self, embed_dim, args):
        if args.attention_type == 1:
            logging_info("======================")
            logging_info("use norm_linear")
            logging_info("======================")
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
                weight_type=getattr(args, "weight_type", -1),
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
                use_urpe=getattr(args, "encoder_use_urpe", False),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                kv_act=getattr(args, "encoder_kv_act", "identity"),
                # final dropout
                use_final_dropout=getattr(args, "use_final_dropout", False),
                final_dropout=getattr(args, "final_dropout", 0.0),
                # Toeplizt
                use_toeplizt=getattr(args, "use_toeplizt", False),
                type_num=getattr(args, "type_num", -1),
                # cos
                use_cos=getattr(args, "use_cos", False),
            )
        elif args.attention_type == 2:
            logging_info("======================")
            logging_info("use norm_local")
            logging_info("======================")
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
                use_urpe=getattr(args, "encoder_use_urpe", False),
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
                norm_type=getattr(args, "local_norm_type", "gatedrmsnorm"),
                # weight
                weight_type=getattr(args, "weight_type", -1),
                # final dropout
                use_final_dropout=getattr(args, "use_final_dropout", False),
                final_dropout=getattr(args, "final_dropout", 0.0),
                index=args.index,
            )
        else:
            return NormMixAttention(
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
                linear_act_fun=getattr(args, "linear_act_fun", "gelu"),
                local_act_fun=getattr(args, "local_act_fun", "relu"),
                out_use_act=getattr(args, "out_use_act", True),
                init_type=getattr(args, "init_type", "default"),
                norm_type=getattr(args, "norm_type", "layernorm"),
                use_rope=getattr(args, "use_rope", False),
                rope_type=getattr(args, "rope_type", "a"),
                use_v=getattr(args, "use_v", False),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "encoder_causal", False),
                use_urpe=getattr(args, "encoder_use_urpe", True),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                # add
                chunk_size=getattr(args, "chunk_size", 32),
                forward_type=getattr(args, "forward_type", 1),
            )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

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
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

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
        attn_type = getattr(args, 'attn_type', 'layernorm')
        logging_info(f"Encoder Norm Type: {attn_type}")
        if attn_type == "simplermsnorm":
            self.self_attn_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
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
        self.fina_act = getattr(args, "fina_act", "None")
        self.glu_dropout = getattr(args, "glu_dropout", 0.0)
        logging_info("Encoder")

        if self.use_glu:
            d1 = self.embed_dim
            p = 8 / 3
            p = getattr(args, "multiple", p)
            d2 = int(p * d1)
            logging_info(f"GLU multiple {p}")
            self.glu = GLU(d1, d2, self.glu_act, self.fina_act, self.glu_dropout)
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

        if attn_type == "simplermsnorm":
            self.final_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
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
            logging_info("======================")
            logging_info("use norm_linear")
            logging_info("======================")
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
                weight_type=getattr(args, "weight_type", -1),
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
                use_urpe=getattr(args, "encoder_use_urpe", False),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                kv_act=getattr(args, "encoder_kv_act", "identity"),
                # final dropout
                use_final_dropout=getattr(args, "use_final_dropout", False),
                final_dropout=getattr(args, "final_dropout", 0.0),
                # Toeplizt
                use_toeplizt=getattr(args, "use_toeplizt", False),
                type_num=getattr(args, "type_num", -1),
                # cos
                use_cos=getattr(args, "use_cos", False),
            )
        elif args.attention_type == 2:
            logging_info("======================")
            logging_info("use norm_local")
            logging_info("======================")
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
                use_urpe=getattr(args, "encoder_use_urpe", False),
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
                norm_type=getattr(args, "local_norm_type", "gatedrmsnorm"),
                # weight
                weight_type=getattr(args, "weight_type", -1),
                # final dropout
                use_final_dropout=getattr(args, "use_final_dropout", False),
                final_dropout=getattr(args, "final_dropout", 0.0),
                index=args.index,
            )
        else:
            return NormMixAttention(
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
                linear_act_fun=getattr(args, "linear_act_fun", "gelu"),
                local_act_fun=getattr(args, "local_act_fun", "relu"),
                out_use_act=getattr(args, "out_use_act", True),
                init_type=getattr(args, "init_type", "default"),
                norm_type=getattr(args, "norm_type", "layernorm"),
                use_rope=getattr(args, "use_rope", False),
                rope_type=getattr(args, "rope_type", "a"),
                use_v=getattr(args, "use_v", False),
                negative_slope=getattr(args, "negative_slope", 0.1),
                # add
                causal=getattr(args, "encoder_causal", False),
                use_urpe=getattr(args, "encoder_use_urpe", True),
                core_matrix=getattr(args, "encoder_core_matrix", 1),
                p_matrix=getattr(args, "encoder_p_matrix", 1),
                max_positions=getattr(args, "encoder_max_positions", 512),
                theta_type=getattr(args, "encoder_theta_type", "a"),
                theta_learned=getattr(args, "encoder_theta_learned", False), 
                householder_learned=getattr(args, "encoder_householder_learned", False),
                # add
                chunk_size=getattr(args, "chunk_size", 32),
                forward_type=getattr(args, "forward_type", 1),
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
