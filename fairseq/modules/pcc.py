import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.modules import LayerNorm
from torch.nn import BatchNorm1d

# class NormalizeBlock(nn.Module):
#     def __init__(self, batch, embed_dim, t1, t2, eps=1e-6):
#         super().__init__()
#         self.mean = nn.Parameter(torch.zeros(batch, embed_dim), requires_grad=False)
#         self.var = nn.Parameter(torch.zeros(batch, embed_dim), requires_grad=False)
#         print("------------")
#         print(self.mean.shape)
#         print(self.var.shape)
#         print("------------")
#         self.t1 = t1
#         self.t2 = t2
#         self.flag = False
#         self.eps = eps

#     def forward(self, x):
#         # N, L, E
#         n, l, e = x.shape
#         # N, L, E -> N * L, E
#         print(f"x.shape {x.shape}")
#         print(f"x1.shape {x1.shape}")
#         with torch.no_grad():
#             # (b, e)
#             mean = torch.mean(x, axis=1, keepdims=True)
#             var = torch.var(x, axis=1, keepdims=True)
#             if self.flag:
#                 self.mean.data = (1 - self.t1) * self.mean.data + self.t1 * mean
#                 self.var.data = (1 - self.t2) * self.var.data + self.t2 * var
#             else:
#                 self.mean.data = mean
#                 self.var.data = var
#                 self.flag = True
#                 print(self.mean.data.shape)
#                 print(self.var.data.shape)

#         # N, E, L
#         x2 = ((x.transpose(1, 2) - self.mean) / torch.sqrt(self.var + self.eps)).transpose(1, 2)
#         # if self.mean.requires_grad:
#         #     print("mean need grad")
#         # else:
#         #     print("mean don't need grad")

#         # if self.var.requires_grad:
#         #     print("var need grad")
#         # else:
#         #     print("var don't need grad")
#         # print("----------")
#         # print(f"max {torch.max(x2)}")
#         # print(x.shape)
#         # print(x2.shape)
#         return x2

class NormalizeBlock(nn.Module):
    def __init__(self, embed_dim, t1, t2, eps=1e-6):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(embed_dim), requires_grad=False)
        self.var = nn.Parameter(torch.zeros(embed_dim), requires_grad=False)
        # self.mean = nn.Parameter(torch.zeros(embed_dim))
        # self.var = nn.Parameter(torch.zeros(embed_dim))
        print("------------")
        print(self.mean.shape)
        print(self.var.shape)
        if self.mean.requires_grad:
            print("mean need grad")
        else:
            print("mean don't need grad")
        if self.var.requires_grad:
            print("var need grad")
        else:
            print("var don't need grad")
        print("------------")
        self.t1 = t1
        self.t2 = t2
        self.flag = False
        self.eps = eps

    def forward(self, x):
        # N, L, E
        n, l, e = x.shape
        # N, L, E -> N * L, E
        # print(f"x.shape {x.shape}")
        x1 = x.contiguous().view(-1, e)
        # print(f"x1.shape {x1.shape}")
        with torch.no_grad():
            # (e)
            mean = torch.mean(x1, axis=0)
            var = torch.var(x1, axis=0)
            if self.flag:
                self.mean.data = (1 - self.t1) * self.mean.data + self.t1 * mean
                self.var.data = (1 - self.t2) * self.var.data + self.t2 * var
            else:
                self.mean.data = mean
                self.var.data = var
                self.flag = True
                # print(self.mean.data.shape)
                # print(self.var.data.shape)

        # N, E, L
        x2 = ((x - self.mean) / torch.sqrt(self.var + self.eps))
        # if self.mean.requires_grad:
        #     print("mean need grad")
        # else:
        #     print("mean don't need grad")

        # if self.var.requires_grad:
        #     print("var need grad")
        # else:
        #     print("var don't need grad")
        # print("----------")
        # print(f"max {torch.max(x2)}")
        # print(x.shape)
        # print(x2.shape)
        return x2

# cosformer
@with_incremental_state
class PccModule(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        # add
        index=0,
        causal=False,
        has_out=False,
        seq_len=512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.qkv_same_dim = True

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        
        # self.k_proj = quant_noise(
        #     nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        # )
        # self.v_proj = quant_noise(
        #     nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        # )
        self.x_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.has_out = has_out

        # layer norm fail
        # self.x_layer_norm = LayerNorm(embed_dim)
        # self.y_layer_norm = LayerNorm(embed_dim, elementwise_affine=False)
        t1 = 0.995
        t2 = 0.995
        self.y_norm = NormalizeBlock(embed_dim, t1, t2)

        # batch norm
        # self.y_batch_norm = BatchNorm1d(seq_len)

        if self.has_out:
            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )

        # add begin
        self.causal = causal
        self.index = index

        print(num_heads)
        print(self.causal)


        self.reset_parameters()

        self.onnx_trace = False

    def get_yy(self, x):
        # (b, l, e) -> (b, e, l)
        b, l, e = x.shape
        y = F.layer_norm(x.transpose(1, 2), (l,))

        # (b, e, l), (b, l, e) -> (b, e, e)

        return torch.bmm(y, y.transpose(1, 2)) / l

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            # nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.x_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.x_proj.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

    # y相关系数
    # def forward(
    #     self,
    #     x,
    #     y,
    #     key_padding_mask: Optional[Tensor] = None,
    #     incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    #     need_weights: bool = True,
    #     static_kv: bool = False,
    #     attn_mask: Optional[Tensor] = None,
    #     before_softmax: bool = False,
    #     need_head_weights: bool = False,
    # ) -> Tuple[Tensor, Optional[Tensor]]:
    #     """Input shape: Time x Batch x Channel

    #     Args:
    #         key_padding_mask (ByteTensor, optional): mask to exclude
    #             keys that are pads, of shape `(batch, src_len)`, where
    #             padding elements are indicated by 1s.
    #         need_weights (bool, optional): return the attention weights,
    #             averaged over heads (default: False).
    #         attn_mask (ByteTensor, optional): typically used to
    #             implement causal attention, where the mask prevents the
    #             attention from looking forward in time (default: None).
    #         before_softmax (bool, optional): return the raw attention
    #             weights and values before the attention softmax.
    #         need_head_weights (bool, optional): return the attention
    #             weights for each head. Implies *need_weights*. Default:
    #             return the average attention weights over all heads.
    #     """
    #     if need_head_weights:
    #         need_weights = True

    #     '''
    #     - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     '''
    #     tgt_len, bsz, embed_dim = x.size()
    #     src_len = y.size(0)

    #     # L, N, E1 -> N, L, E1
    #     x = self.x_proj(x).transpose(0, 1)
    #     # S, N, E1 -> N, S, E1
    #     y = y.transpose(0, 1)

    #     # to do: self.causal
    #     x_norm = x - torch.mean(x, dim=1, keepdim=True)
    #     y_norm = y - torch.mean(y, dim=1, keepdim=True)
    #     # N, 1, E1
    #     # eps = 1.0
    #     var = torch.sqrt(torch.var(y_norm, dim=1, keepdim=True))# + eps
    #     # N, E1, E1
    #     y_corr = torch.bmm(y_norm.transpose(1, 2), y_norm) / var / var.transpose(1, 2) / (src_len - 1)
    #     if torch.isnan(y_corr).int().sum():
    #         print("y_corr")
    #     # N, L, E1; N, E1, E1 -> N, L, E1
    #     xyty = torch.bmm(x, y_corr)
    #     if torch.isnan(xyty).int().sum():
    #         print("xyty")
    #     # L, N, E1 -> L, N, E1
    #     output = xyty.transpose(0, 1).contiguous()
    #     # L, N, E1 -> L, N, E1
    #     output = self.out_proj(output)
    #     if torch.isnan(output).int().sum():
    #         print("output")

    #     return output, None

    def forward(
        self,
        x,
        y,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        '''
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        '''
        src_len = y.size(0)

        # L, N, E -> N, L, E
        x = self.x_proj(x).transpose(0, 1)
        tgt_len, bsz, embed_dim = x.size()
        # S, N, E -> N, S, E
        y = y.transpose(0, 1)
        x = F.relu(x)
        y = F.relu(y)

        # # to do: self.causal
        # # x_norm = x - torch.mean(x, dim=1, keepdim=True)
        # with torch.no_grad():
        #     mean_y = torch.mean(y, dim=1, keepdim=True)
        # # N, 1, E1
        # # eps = 1.0
        # # N, L, 1
        # # varx = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True))# + eps
        # # N, S, 1
        # with torch.no_grad():
        #     var_y = torch.sqrt(torch.var(y, dim=1, keepdim=True))
        # # N, L, E
        # # x1 = x_norm / varx
        # # N, S, E
        # y1 = (y - mean_y) / var_y

        # layer norm
        # y1 = self.y_layer_norm(y)
        # x1 = self.x_layer_norm(x)
        # y1 = self.y_batch_norm(y)
        y1 = self.y_norm(y)
        # (N, E, S), (N, S, E) -> (N, E, E)
        # 归一化很重要
        yy = torch.bmm(y1.transpose(1, 2), y1) / src_len / embed_dim
        # N, E, E
        # yy = self.get_yy(y)
        # if torch.isnan(yy).int().sum():
        #     print("yy")
        # N, L, E; N, E, E -> N, L, E1
        xyty = torch.bmm(x, yy)
        # if torch.isnan(xyty).int().sum():
        #     print("xyty")
        # L, N, E1 -> L, N, E1
        output = xyty.transpose(0, 1).contiguous()
        # L, N, E1 -> L, N, E1
        if self.has_out:
            output = self.out_proj(output)
        # if torch.isnan(output).int().sum():
        #     print("output")

        return output, None

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

