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
from torch.nn import Dropout
import sys
from fairseq.modules import GatedRMSNorm
from fairseq.modules import RMSNorm
from fairseq.modules import Orpe
from fairseq.modules import OrpeV2
# from fast_transformers.causal_product import causal_dot_product
# N, L, H, E, batch, length, head, dim

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

@with_incremental_state
class NormLocalAttention(nn.Module):
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
        use_relu=True,
        use_elu=False,
        use_leak=False,
        use_bound=False,
        max_l=1024,
        has_out=False,
        causal=False,
        weight_type=1,
        c=1.0,
        v_act=False,
        use_dropout=False,
        p=0.5,
        use_layer_norm=False,
        qk_layer_norm=False,
        seq_dropout=False,
        seq_p=0.3,
        act_fun="relu",
        negative_slope=0.1,
        # orpe
        use_orpe=False,
        core_matrix=1, 
        p_matrix=1, 
        max_positions=512,
        theta_type="a",
        theta_learned=False, 
        householder_learned=False,
        # chunk_size
        chunk_size=32,
        left_window=1,
        right_window=1,
        group_type="chunk"
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
        
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.gated_rms_norm = GatedRMSNorm(embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None


        self.reset_parameters()

        # for test
        self.onnx_trace = False

        # add
        self.act_fun = act_fun
        self.negative_slope = negative_slope
        self.act = self.get_act_fun()

        # orpe add
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.max_positions = max_positions
        self.causal = causal
        self.use_orpe = use_orpe
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned
        if self.use_orpe:
            print("=====================================")
            self.orpe = OrpeV2(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
            # self.orpe = Orpe(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
            print("=====================================")

        self.causal = causal
        self.left_window = left_window
        self.right_window = right_window
        self.group_type = group_type

        # chunk
        self.chunk_size = chunk_size
        print("use relu sparse")
        print(f"use orpe {self.use_orpe}")
        print(f"num_heads {self.num_heads}")
        print(f"add_bias_kv {add_bias_kv}")
        print(f"act_fun {self.act_fun}")
        print(f"negative_slope {self.negative_slope}")
        print(f"chunk_size {self.chunk_size}")
        print(f"causal {self.causal}")
        print(f"self.left_window {self.left_window}")
        print(f"self.right_window {self.right_window}")
        print(f"self.group_type {self.group_type}")

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def get_act_fun(self):
        print(self.act_fun)
        if self.act_fun == "gelu":
            return F.gelu
        elif self.act_fun == "relu":
            return F.relu
        elif self.act_fun == "elu":
            return F.elu
        elif self.act_fun == "sigmoid":
            return F.sigmoid
        elif self.act_fun == "exp":
            return torch.exp
        elif self.act_fun == "1+elu":
            def f(x):
                return F.elu(x) + 1
            return f
        elif self.act_fun == "1+relu":
            def f(x):
                return F.relu(x) + 1
            return f
        elif self.act_fun == "2+elu":
            def f(x):
                return F.elu(x) + 2
            return f
        elif self.act_fun == "relu2":
            def f(x):
                return torch.square(torch.relu(x))
            return f
        elif self.act_fun == "leak":
            def f(x):
                return F.leaky_relu(x, negative_slope=self.negative_slope)
            return f
        else:
            def f(x):
                return x
            return f

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
        if self.group_type == "chunk":
            return self.forward_chunk(query, key, value, key_padding_mask, 
                                      incremental_state, need_weights, static_kv,
                                      attn_mask, before_softmax, need_head_weights)
        elif self.group_type == "window":
            return self.forward_window(query, key, value, key_padding_mask, 
                                      incremental_state, need_weights, static_kv,
                                      attn_mask, before_softmax, need_head_weights)


    # reference
    # https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py
    # not used for cross attention
    # 滑动窗口版本
    def forward_window(
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
        # self.cnt += 1
        # if self.cnt == 10:
        #     sys.exit(0)
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        tgt_len, bsz, embed_dim = query.size()

        scaling = float(head_dim) ** -0.5
        # L, N, E1
        q = self.q_proj(query)
        # scale
        q *= scaling
        # S, N, E1
        k = self.k_proj(key)
        # S, N, E2
        v = self.v_proj(value)

        if self.use_orpe:
            q = self.orpe(q)
            k = self.orpe(k)

        # pad至chunk_size整数倍
        tgt_len_pad = (self.chunk_size - tgt_len % self.chunk_size) % self.chunk_size
        src_len_pad = (self.chunk_size - src_len % self.chunk_size) % self.chunk_size

        # 填充0
        orig_t = tgt_len
        q = F.pad(q, (0, 0, 0, 0, 0, tgt_len_pad)).transpose(0, 1)
        k = F.pad(k, (0, 0, 0, 0, 0, src_len_pad)).transpose(0, 1)
        v = F.pad(v, (0, 0, 0, 0, 0, src_len_pad)).transpose(0, 1)

        # b * h, l, e
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        b, t, e = q.shape
        s = k.shape[1]
        windows = t // self.chunk_size
        ticker = torch.arange(t).reshape(1, -1).to(q)
        b_t = ticker.reshape(1, windows, -1)
        # b, windows, window_size, e
        bq = q.reshape(b, -1, self.chunk_size, e)
        bk = k.reshape(b, -1, self.chunk_size, e)
        bv = v.reshape(b, -1, self.chunk_size, e)


        look_around_kwargs = {'backward': self.left_window, 'forward': self.right_window}
        # s1 = window_size * (left + right + 1)
        # b, windows, s1, e
        bk = look_around(bk,  self.left_window, self.right_window, 0)
        # b, windows, s1, e
        bv = look_around(bv,  self.left_window, self.right_window, 0)
        bq_t = b_t
        bq_k = look_around(b_t, self.left_window, self.right_window, -1)

        # b, windows, window_size, s1
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)

        self.causal = True
        if self.causal:
            # 表示每个位置不能看到哪些部分
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
            dots.masked_fill_(mask, 0)
        prob = self.act(dots)
        weights = self.dropout_module(prob)

        # b, windows, window_size, s1; b, windows, s1, e -> b, windows, window_size, e
        output = torch.einsum('bhij,bhje->bhie', weights, bv)
        # b, windows, window_size, e -> b, t, e -> t, b, e -> t, b, h * e
        output = output.reshape(-1, t, e)[:, :orig_t, :]
        output = output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # perform RMSNorm to stabilize running
        output = self.gated_rms_norm(output)
        # outprojection
        output = self.out_proj(output)

        return output, prob

    # 分组版本
    def forward_chunk(
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
        # self.cnt += 1
        # if self.cnt == 10:
        #     sys.exit(0)
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        tgt_len, bsz, embed_dim = query.size()

        scaling = float(head_dim) ** -0.5
        # L, N, E1
        q = self.q_proj(query)
        # scale
        q *= scaling
        # S, N, E1
        k = self.k_proj(key)
        # S, N, E2
        v = self.v_proj(value)

        # pad至chunk_size整数倍
        tgt_len_pad = (self.chunk_size - tgt_len % self.chunk_size) % self.chunk_size
        src_len_pad = (self.chunk_size - src_len % self.chunk_size) % self.chunk_size
        tgt_g = (tgt_len + tgt_len_pad) // self.chunk_size
        src_g = (src_len + src_len_pad) // self.chunk_size

        # 填充0
        q = F.pad(q, (0, 0, 0, 0, 0, tgt_len_pad))
        k = F.pad(k, (0, 0, 0, 0, 0, src_len_pad))
        v = F.pad(v, (0, 0, 0, 0, 0, src_len_pad))

        # N, L, H, E: batch, length, head, dim
        # N, L, E1 -> N * h, L, e1 -> N * h, g, l, e1
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1).contiguous().view(bsz * num_heads, -1, self.chunk_size, head_dim)
        # N, S, E1 -> N * h, S, e1 -> N * h, g, s, e1
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1).contiguous().view(bsz * num_heads, -1, self.chunk_size, head_dim)
        # N, S, E2 -> N * h, S, e2 -> N * h, g, s, e2
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1).contiguous().view(bsz * num_heads, -1, self.chunk_size, head_dim)

        if self.use_orpe:
            q = self.orpe(q)
            k = self.orpe(k)

        # (N * h, g, l, e1), (N * h, g, s, e1) -> (N * h, g, l, s)
        logits = torch.einsum("bgle,bgse->bgls", q, k)
        prob = self.act(logits)

        if self.causal:
            attn_mask = (torch.triu(torch.ones(self.chunk_size, self.chunk_size)) == 1).transpose(0, 1)
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).to(q)
            prob = prob.masked_fill(attn_mask==float("-inf"), 0)
        weights = self.dropout_module(prob)

        # (N * h, g, l, s), (N * h, g, s, e2) -> (N * h, g, l, e2)
        output = torch.einsum("bgls,bgsd->bgld", weights, v)
        # (N * h, g, l, e2) -> (N * h, L, e2) -> (L, N * h, e2) -> (L, N, E2)
        output = output.contiguous().view(bsz * num_heads, tgt_len + tgt_len_pad, -1).transpose(0, 1).contiguous().view(tgt_len + tgt_len_pad, bsz, -1)[:tgt_len, ...]
        # perform RMSNorm to stabilize running
        output = self.gated_rms_norm(output)
        # outprojection
        output = self.out_proj(output)

        output = output
        return output, prob

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