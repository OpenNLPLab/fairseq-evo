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


# Sparse Relu
@with_incremental_state
class MultiheadSparseReluAttention(nn.Module):
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
        # base
        is_base=True,
        is_ada_q=False,
        is_ada_k=False,
        lambda_=0.99,
        up_fq=16,
        dropout_before=False,
        use_q=False,
        use_k=False,
        # add
        low_d=False,
        has_out=False,
        do_scale=True,
        norm_taylor=True,
        use_relu=False,
        use_elu=False,
        use_leak=False,
        use_square=False,
        use_sigmoid=False,
        use_l2=False,
        # scale
        dim_scale=-1,
        # sparse
        sparse=False,
        d1=32,
        d2=8,
        # for sparse relu
        n_groups=4,
        step=4,
        max_n=3072,
        batch_size=16,
        num=1,
        # 全局感受野
        d_global=16,
        with_global=False,
    ):
        # add
        self.index = index

        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        # add
        self.has_out = has_out
        self.low_d = low_d
        self.do_scale = do_scale
        self.dim_scale = dim_scale

        if self.low_d:
            dim = embed_dim // 2
        elif self.dim_scale != -1:
            dim = self.dim_scale * embed_dim
        else:
            dim = embed_dim
        self.dim = dim
        

        # for test
        self.num = num
        scale_dim = embed_dim // self.num
        
        # sparse relu
        self.n_groups = n_groups
        self.step = step
        self.max_n = max_n
        self.batch_size = batch_size
        self.d_global = d_global
        self.with_global = with_global


        
        self.new_dim = scale_dim * self.n_groups

        self.mask = self.get_mask(self.max_n, scale_dim, self.n_groups, self.step)

        if self.with_global:
            self.front_mask, self.back_mask = self.global_mask(self.max_n, self.new_dim, self.d_global)

        print(f"self.num {self.num}")
        print(self.mask.shape)
        print(f"self.d_global {self.d_global}")
        print(f"self.with_global {self.with_global}")
        if self.with_global:
            print(self.front_mask.shape, self.back_mask.shape)

        self.scaling = self.new_dim ** -0.5
        
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, self.new_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, self.new_dim, bias=bias), q_noise, qn_block_size
        )

        # for sparse v1
        # self.n_groups = n_groups
        # self.step = step
        # self.max_n = max_n
        # self.batch_size = batch_size
        # self.row_index, self.col_index = self.get_index(max_n, embed_dim, self.n_groups, self.step)
        # self.q1 = Parameter(torch.zeros(self.batch_size, self.max_n, embed_dim * self.n_groups))
        # self.k1 = Parameter(torch.zeros(self.batch_size, self.max_n, embed_dim * self.n_groups))
        # print(n_groups, step)
        # print(self.n_groups, self.step)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False



        # print(dim, embed_dim)
        # print(f"do scale {self.do_scale}")
        # print(f"taylor {self.norm_taylor}")
        # print(f"use relu {self.use_relu}")
        # print(f"use elu {self.use_elu}")
        # print(f"use leak {self.use_leak}")
        # print(f"use square {self.use_square}")
        # print(f"use sigmoid {self.use_sigmoid}")
        # print(f"use l2 {self.use_l2}")
        # print(f"sparse {self.sparse}")
        # print(f"d1 {self.d1}")
        # print(f"d2 {self.d2}")

    def get_mask(self, n, e, n_groups, step):
        row_index = np.arange(n).reshape(-1, 1)
        col_index = []
        e1 = e * n_groups
        mask = torch.ones(n, e1, dtype=torch.bool)

        row_num = (e1 - e) // step + 1
        group1 = [range(i * step, i * step + e) for i in range(row_num)]
        group2 = [range(i * step, i * step + e) for i in range(row_num - 1, -1, -1)]
        group_index = [group1, group2]
        l = n // row_num
        k = 0
        for i in range(l):
            start = i * row_num
            end = (i + 1) * row_num
            for j in range(start, end):
                col_index.append(group_index[k][j % row_num])
            k = 1 - k
        if n % row_num != 0:
            start = l * row_num
            end = n
            for j in range(start, end):
                col_index.append(group_index[k][j % row_num])
            k = 1 - k

        mask[row_index, col_index] = 0
        
        return Parameter(mask, requires_grad=False)

    def global_mask(self, n, e, d_global):
        front_mask = torch.ones(n, e, dtype=torch.bool)
        back_mask = torch.ones(n, e, dtype=torch.bool)
        start = max(0, d_global)
        end = min(n - 1, n - d_global)
        front_mask[:start, :] = 0
        back_mask[end:, :] = 0

        return Parameter(front_mask, requires_grad=False), Parameter(back_mask, requires_grad=False)

    def get_global_mask(self, n):
        return self.front_mask[:n, :] & self.back_mask[-n:, :]


    def get_index(self, n, e, n_groups, step):
        row_index = np.arange(n).reshape(-1, 1)
        col_index = []
        e1 = e * n_groups
        row_num = (e1 - e) // step + 1
        group1 = [range(i * step, i * step + e) for i in range(row_num)]
        group2 = [range(i * step, i * step + e) for i in range(row_num - 1, -1, -1)]
        group_index = [group1, group2]
        l = n // row_num
        k = 0
        for i in range(l):
            start = i * row_num
            end = (i + 1) * row_num
            for j in range(start, end):
                col_index.append(group_index[k][j % row_num])
            k = 1 - k
        if n % row_num != 0:
            start = l * row_num
            end = n
            for j in range(start, end):
                col_index.append(group_index[k][j % row_num])
            k = 1 - k
        
        return row_index, col_index

    def transformation(self, x, n_groups, step):
        '''
        ....
        ....
        ....
        ....
            ....
            ....
        ....
        ....
        ....
        ....
        '''
        # print(x.shape)
        b, n, e = x.shape
        # print(b, n, e)
        e1 = e * n_groups
        # print(n_groups)
        # print(b, n, e1)
        x_transform = torch.zeros(b, n, e1).to(x)
        # row_index = np.arange(n).reshape(-1, 1)
        # col_index = []

        # # 每组的数量
        # # 希望row_num > n, 这样就不会出现周期
        # row_num = (e1 - e) // step + 1
        # group1 = [range(i * step, i * step + e) for i in range(row_num)]
        # group2 = [range(i * step, i * step + e) for i in range(row_num - 1, -1, -1)]
        # group_index = [group1, group2]
        # l = n // row_num
        # k = 0
        # for i in range(l):
        #     start = i * row_num
        #     end = (i + 1) * row_num
        #     for j in range(start, end):
        #         col_index.append(group_index[k][j % row_num])
        #     k = 1 - k
        # if n % row_num != 0:
        #     start = l * row_num
        #     end = n
        #     for j in range(start, end):
        #         col_index.append(group_index[k][j % row_num])
        #     k = 1 - k
        x_transform[:, self.row_index[:n], self.col_index[:n]] = x
        
        return x_transform

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
        
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def build_mask(self, src_len, tgt_len):
        d_diag = min(self.d1, tgt_len, src_len)
        d_col = min(self.d2, tgt_len)
        d_row = min(self.d2, src_len)
        mask = torch.ones((src_len, tgt_len), dtype=torch.bool)
        mask1 = torch.tril(mask, diagonal=d_diag)
        mask2 = torch.triu(mask, diagonal=-d_diag)
        diag_mask = (mask1 & mask2)
        diag_mask[:d_col, :] = True
        diag_mask[:, :d_row] = True

        # return ~diag_mask
        return nn.Parameter(~diag_mask, requires_grad=False)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
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

        assert key is not None and value is not None

        '''
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        '''
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        tgt_len, bsz, embed_dim = query.size()
        # L, N, E
        q = self.q_proj(query)
        # S, N, E
        k = self.k_proj(key)

        

        q = F.relu(q)
        k = F.relu(k)

        q = q * self.scaling

        # N, L, E
        q = q.transpose(0, 1)
        # N, S, E
        k = k.transpose(0, 1)

        # v1
        # q1 = self.transformation(q, self.n_groups, self.step)
        # k1 = self.transformation(k, self.n_groups, self.step)
        # attn_output_weights = torch.bmm(q1, k1.transpose(1, 2))
        # self.q1[:, self.row_index[:tgt_len], self.col_index[:tgt_len]] = q
        # self.k1[:, self.row_index[:src_len], self.col_index[:src_len]] = k
        # N, L, S
        # attn_output_weights = torch.bmm(self.q1[:, :tgt_len, :], self.q1[:, :src_len, :].transpose(1, 2))

        # v2
        q1 = q.masked_fill(self.mask[:tgt_len], 0)
        k1 = k.masked_fill(self.mask[:src_len], 0)
        attn_output_weights = torch.bmm(q1, k1.transpose(1, 2))
        
        if attn_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)

        attn_output_weights = F.normalize(attn_output_weights, p=1, dim=-1)
        
        attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
        # N, S, E
        value = value.transpose(0, 1)

        # N, L, E
        attn_output = torch.bmm(attn_output_weights, value)
        # L, N, E
        attn_output = attn_output.transpose(0, 1)
        # L, N, E
        attn_output = self.v_proj(attn_output)

        if self.with_global:
            # print(k.shape)
            # print(self.get_global_mask(src_len).shape)
            k = k.masked_fill(self.get_global_mask(src_len), 0)
            attn_output_weights_global = torch.bmm(q, k.transpose(1, 2))
            if attn_mask is not None:
                attn_output_weights_global = attn_output_weights_global.masked_fill(attn_mask==float("-inf"), 0)

            attn_output_weights_global = F.normalize(attn_output_weights, p=1, dim=-1)
            
            attn_output_weights_global = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)

            # N, L, E
            attn_output_global = torch.bmm(attn_output_weights_global, value)
            # L, N, E
            attn_output_global = attn_output_global.transpose(0, 1)
            # L, N, E
            attn_output_global = self.v_proj(attn_output_global)

            attn_output = attn_output + attn_output_global

        # add
        if self.has_out:
            attn_output = self.out_proj(attn_output)


        if need_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output, None

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
