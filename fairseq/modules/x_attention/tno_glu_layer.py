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
from .tno import TNO
from ..ffn import GLU
from ..norm import SimpleRMSNorm, RMSNorm, GatedRMSNorm

class TNOGLUEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        attn_type = getattr(args, 'norm_type', 'layernorm')
        print(f"Encoder Norm Type: {attn_type}")
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
        self.glu_act = getattr(args, "glu_act", "silu")
        self.fina_act = getattr(args, "fina_act", "None")
        self.glu_dropout = getattr(args, "glu_dropout", 0.0)
        self.glu_dim = getattr(args, "glu_dim", -1)
        if self.glu_dim == -1:
            self.glu_dim = self.embed_dim
        print("=============================")
        print("Encoder")
        print(f"self.glu_act {self.glu_act}")
        print(f"self.fina_act {self.fina_act}")
        print(f"self.glu_dropout {self.glu_dropout}")
        print(f"self.glu_dim {self.glu_dim}")
        print("=============================")

        self.glu = GLU(self.embed_dim, self.glu_dim, self.glu_act, self.fina_act, self.glu_dropout)

        if attn_type == "simplermsnorm":
            self.final_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
            self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_self_attention(self, embed_dim, args):
        return TNO(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            index=args.index,
            act_fun=getattr(args, "act_fun", "silu"),
            causal=getattr(args, "causal", True),
            expand_ratio=getattr(args, "expand_ratio", 2),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            # Toeplizt
            use_exp=getattr(args, "use_exp", False),
            use_neg_exp=getattr(args, "use_neg_exp", False),
            toep_type=getattr(args, "toep_type", 1),
            max_l=getattr(args, "max_l", 512),
            use_decay=getattr(args, "use_decay", False),
            use_multi_decay=getattr(args, "use_multi_decay", False),
            use_dynamic=getattr(args, "use_dynamic", False),
            dpb_embedding=getattr(args, "dpb_embedding", 512),
            use_dynamic_v2=getattr(args, "use_dynamic_v2", False),
            dpb_act=getattr(args, "dpb_act", "relu"),
            dpb_use_pad=getattr(args, "dpb_use_pad", True),
            normalize=getattr(args, "normalize", False),
            use_dynamic_v3=getattr(args, "use_dynamic_v3", False),
            par_type=getattr(args, "par_type", 1),
            # se
            use_se=getattr(args, "use_se", False),
            se_ratio=getattr(args, "se_ratio", 16),
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
        # print("encoder")
        # print("before")
        # print(x.shape)
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
        x = self.glu(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        # print("after")
        # print(x.shape)
        return x

class TNOGLUDecoderLayer(nn.Module):
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
        self.glu_act = getattr(args, "glu_act", "silu")
        self.fina_act = getattr(args, "fina_act", "None")
        self.glu_dropout = getattr(args, "glu_dropout", 0.0)
        self.glu_dim = getattr(args, "glu_dim", -1)
        if self.glu_dim == -1:
            self.glu_dim = self.embed_dim
        print("=============================")
        print("Decoder")
        print(f"self.glu_act {self.glu_act}")
        print(f"self.fina_act {self.fina_act}")
        print(f"self.glu_dropout {self.glu_dropout}")
        print(f"self.glu_dim {self.glu_dim}")
        print("=============================")
        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        # self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        attn_type = getattr(args, 'norm_type', 'layernorm')
        print(f"Decoder Norm Type: {attn_type}")
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

        self.glu = GLU(self.embed_dim, self.glu_dim, self.glu_act, self.fina_act, self.glu_dropout)

        # self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        if attn_type == "simplermsnorm":
            self.final_layer_norm = SimpleRMSNorm(self.embed_dim)
        else:
            self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return TNO(
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
            act_fun=getattr(args, "act_fun", "silu"),
            causal=True,
            expand_ratio=getattr(args, "expand_ratio", 2),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            # Toeplizt
            use_exp=getattr(args, "use_exp", False),
            use_neg_exp=getattr(args, "use_neg_exp", False),
            toep_type=getattr(args, "toep_type", 1),
            max_l=getattr(args, "max_l", 512),
            use_decay=getattr(args, "use_decay", False),
            use_multi_decay=getattr(args, "use_multi_decay", False),
            use_dynamic=getattr(args, "use_dynamic", False),
            dpb_embedding=getattr(args, "dpb_embedding", 512),
            use_dynamic_v2=getattr(args, "use_dynamic_v2", False),
            dpb_act=getattr(args, "dpb_act", "relu"),
            dpb_use_pad=getattr(args, "dpb_use_pad", True),
            normalize=getattr(args, "normalize", False),
            use_dynamic_v3=getattr(args, "use_dynamic_v3", False),
            par_type=getattr(args, "par_type", 1),
            # se
            use_se=getattr(args, "use_se", False),
            se_ratio=getattr(args, "se_ratio", 16),
        )

    def build_encoder_attention(self, embed_dim, args):
        return TNO(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            # add
            index=args.index,
            act_fun=getattr(args, "act_fun", "silu"),
            causal=False,
            expand_ratio=getattr(args, "expand_ratio", 2),
            # norm
            use_norm=getattr(args, "use_norm", False),
            norm_type=getattr(args, "norm_type", "simplermsnorm"),
            # Toeplizt
            use_exp=getattr(args, "use_exp", False),
            use_neg_exp=getattr(args, "use_neg_exp", False),
            toep_type=getattr(args, "toep_type", 1),
            max_l=getattr(args, "max_l", 512),
            use_decay=getattr(args, "use_decay", False),
            use_multi_decay=getattr(args, "use_multi_decay", False),
            use_dynamic=getattr(args, "use_dynamic", False),
            dpb_embedding=getattr(args, "dpb_embedding", 512),
            use_dynamic_v2=getattr(args, "use_dynamic_v2", False),
            dpb_act=getattr(args, "dpb_act", "relu"),
            dpb_use_pad=getattr(args, "dpb_use_pad", True),
            normalize=getattr(args, "normalize", False),
            use_dynamic_v3=getattr(args, "use_dynamic_v3", False),
            par_type=getattr(args, "par_type", 1),
            # se
            use_se=getattr(args, "use_se", False),
            se_ratio=getattr(args, "se_ratio", 16),
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
        #print(x.shape)
        if need_head_weights:
            need_attn = True
        # print("decoder")
        # print("before")
        # print(x.shape, encoder_out.shape)
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        #print(x.shape)

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

        # print("before self")
        # print(x.shape, y.shape)
        #print('inside layer', x.shape)
        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        #print('inside layer', x.shape)
        # import pdb
        # pdb.set_trace()

        x = self.dropout_module(x)
        #print('dropout', x.shape)

        x = self.residual_connection(x, residual)
        #print('residule connection', x.shape)

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

        # print("after")
        # print(x.shape)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.glu(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        # print(x.shape)
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