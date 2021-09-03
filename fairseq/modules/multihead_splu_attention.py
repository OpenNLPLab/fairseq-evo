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


@with_incremental_state
class MultiheadSpluAttention(nn.Module):
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
        # for sparse relu
        n_groups=4,
        step=4,
        max_n=3072,
        num=2,
        # 全局感受野
        d_global=16,
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

        # head dim需要除以num, 乘以n_group
        self.origin_head_dim = embed_dim // num_heads // num
        self.qk_head_dim = self.origin_head_dim * n_groups
        self.qk_dim = self.qk_head_dim * num_heads
        self.scaling = self.qk_dim ** -0.5
        self.v_head_dim = embed_dim // num_heads

        # sparse relu
        self.n_groups = n_groups
        self.step = step
        self.max_n = max_n
        self.d_global = d_global

        # global mask
        self.front_mask, self.back_mask = self.global_mask(self.max_n, self.qk_head_dim, self.d_global)
        # local mask
        # self.masks = []
        # for step in range(1, num_heads):
        #     mask = self.get_mask(self.max_n, self.origin_head_dim, self.n_groups, step)
        #     self.masks.append(mask)
        self.masks = self.get_mask(self.max_n, self.origin_head_dim, n_groups, range(1, num_heads))


        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, self.qk_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, self.qk_dim, bias=bias), q_noise, qn_block_size
        )

        # add
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def get_mask(self, n, e, n_groups, steps):
        e1 = e * n_groups
        # mask1
        masks = [np.zeros((n, e1))]

        for step in steps:
            masks.append(self.get_one_mask(n, e, n_groups, step))

        return Parameter(torch.tensor(masks).bool(), requires_grad=False)

    def get_one_mask(self, n, e, n_groups, step):
        row_index = np.arange(n).reshape(-1, 1)
        col_index = []
        e1 = e * n_groups
        mask = np.ones((n, e1))

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
        
        return mask

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

        # add begin
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        # add end
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

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
        # head_dim = self.head_dim

        tgt_len, bsz, embed_dim = query.size()
        # N, L, E1
        q = self.q_proj(query).transpose(0, 1)
        # N, S, E1
        k = self.k_proj(key).transpose(0, 1)
        # N, S, E
        v = self.v_proj(value).transpose(0, 1)

        scaling = float(self.qk_dim) ** -0.5
        
        # N, h, L, d
        q = q.view(bsz, tgt_len, num_heads, self.qk_head_dim).transpose(1, 2)
        # N, h, S, d
        k = k.view(bsz, src_len, num_heads, self.qk_head_dim).transpose(1, 2)
        v = v.contiguous().view(-1, bsz * num_heads, self.v_head_dim).transpose(0, 1)

        q = F.relu(q)
        k = F.relu(k)
        q = q * scaling
        
        # attn_output_weights: N * h, L, S
        # local attention
        q1 = q.masked_fill(self.masks[:, :tgt_len, :], 0).contiguous().view(-1, tgt_len, self.qk_head_dim)
        k1 = k.masked_fill(self.masks[:, :src_len, :], 0).contiguous().view(-1, src_len, self.qk_head_dim)
        attn_output_weights_local = torch.bmm(q1, k1.transpose(1, 2))
        # print(attn_output_weights_local.shape)
        # print("local")
        # print(attn_output_weights_local)

        # global attention
        q = q.contiguous().view(-1, tgt_len, self.qk_head_dim)
        k = k.masked_fill(self.get_global_mask(src_len), 0).contiguous().view(-1, src_len, self.qk_head_dim)
        attn_output_weights_local_global = torch.bmm(q, k.transpose(1, 2))
        # print("global")
        # print(attn_output_weights_local_global.shape)

        attn_output_weights = attn_output_weights_local + attn_output_weights_local_global


        # attn_mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

        if attn_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)
       
        # N * h, L, S
        # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # attn_output_weights = F.normalize(attn_output_weights, p=1, dim=-1)
        eps = 1e-12
        all_weights = torch.sum(attn_output_weights, dim=-1, keepdim=True).clamp_min(eps).expand_as(attn_output_weights)
        attn_output_weights = attn_output_weights / all_weights
        # dropout
        attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
        # N * h, L, d
        attn_output = torch.bmm(attn_output_weights, v)
        # L, N, E
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # L, N, E
        attn_output = self.out_proj(attn_output)

        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights
        else:
            return attn_output, None

    # def forward(
    #     self,
    #     query,
    #     key: Optional[Tensor],
    #     value: Optional[Tensor],
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

    #     assert key is not None and value is not None

    #     '''
    #     - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
    #       the embedding dimension.
    #     '''
    #     num_heads = self.num_heads
    #     tgt_len, bsz, embed_dim = query.size()
    #     src_len = key.size(0)
    #     # head_dim = self.head_dim

    #     tgt_len, bsz, embed_dim = query.size()
    #     # L, N, E1
    #     q = self.q_proj(query)
    #     # S, N, E1
    #     k = self.k_proj(key)
    #     # S, N, E
    #     v = self.v_proj(value)

    #     scaling = float(self.qk_dim) ** -0.5
    #     q = q * scaling

    #     # # N * h, L, d
    #     # q = q.contiguous().view(tgt_len, bsz * num_heads, self.qk_head_dim).transpose(0, 1)
    #     # # N * h, S, d
    #     # k = k.contiguous().view(-1, bsz * num_heads, self.qk_head_dim).transpose(0, 1)
    #     # v = v.contiguous().view(-1, bsz * num_heads, self.v_head_dim).transpose(0, 1)
        
    #     # # attn_output_weights: N * h, L, S
    #     # # local attention
    #     # q1 = q
    #     # k1 = k
    #     # for i in range(self.num_heads - 1):
    #     #     start = i * bsz
    #     #     end  = (i + 1) * bsz
    #     #     q1[start:end].masked_fill_(self.masks[i][:tgt_len], 0)
    #     #     k1[start:end].masked_fill_(self.masks[i][:src_len], 0)
    #     # attn_output_weights_local = torch.bmm(q1, k1.transpose(1, 2))
    #     # print("local")
    #     # print(attn_output_weights_local)

    #     # # global attention
    #     # k = k.masked_fill(self.get_global_mask(src_len), 0)
    #     # attn_output_weights_local_global = torch.bmm(q, k.transpose(1, 2))
    #     # print("global")
    #     # print(attn_output_weights_local_global)

    #     # N * h, L, d
    #     q = q.contiguous().view(tgt_len, bsz * num_heads, self.qk_head_dim).transpose(0, 1)
    #     # N * h, S, d
    #     k = k.contiguous().view(-1, bsz * num_heads, self.qk_head_dim).transpose(0, 1)
    #     v = v.contiguous().view(-1, bsz * num_heads, self.v_head_dim).transpose(0, 1)
        
    #     # attn_output_weights: N * h, L, S
    #     # local attention
    #     q1 = q
    #     k1 = k
    #     for i in range(self.num_heads - 1):
    #         start = i * bsz
    #         end  = (i + 1) * bsz
    #         q1[start:end].masked_fill_(self.masks[i][:tgt_len], 0)
    #         k1[start:end].masked_fill_(self.masks[i][:src_len], 0)
    #     attn_output_weights_local = torch.bmm(q1, k1.transpose(1, 2))
    #     # print("local")
    #     # print(attn_output_weights_local)

    #     # global attention
    #     k = k.masked_fill(self.get_global_mask(src_len), 0)
    #     attn_output_weights_local_global = torch.bmm(q, k.transpose(1, 2))
    #     # print("global")
    #     # print(attn_output_weights_local_global)

    #     attn_output_weights = attn_output_weights_local + attn_output_weights_local_global

    #     # attn_mask
    #     if attn_mask is not None:
    #         if attn_mask.dim() == 2:
    #             attn_mask = attn_mask.unsqueeze(0)
    #             if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
    #                 raise RuntimeError('The size of the 2D attn_mask is not correct.')
    #         elif attn_mask.dim() == 3:
    #             if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
    #                 raise RuntimeError('The size of the 3D attn_mask is not correct.')
    #         else:
    #             raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
    #     # attn_mask's dim is 3 now.

    #     if attn_mask is not None:
    #         attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)
       
    #     # N * h, L, S
    #     # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    #     attn_output_weights = F.normalize(attn_output_weights, p=1, dim=-1)
    #     # dropout
    #     attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
    #     # N * h, L, d
    #     attn_output = torch.bmm(attn_output_weights, v)
    #     # L, N, E
    #     attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    #     # L, N, E
    #     attn_output = self.out_proj(attn_output)

    #     if need_weights:
    #         attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    #         return attn_output, attn_output_weights
    #     else:
    #         return attn_output, None

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