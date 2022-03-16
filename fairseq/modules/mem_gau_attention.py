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
# from fast_transformers.causal_product import causal_dot_product
# N, L, H, E, batch, length, head, dim


class ScaleNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean_square = (x ** 2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.scala
        return x

# memory attention
@with_incremental_state
class GmuAttention(nn.Module):
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
        lambda_=0.001,
        use_gelu=False,
        mem_use_gelu=False,
        mem_use_grad=True,
        mem_use_q=True,
        mem_use_k=False,
        attention_use_layer_norm=True,
        model_update_freq=1,
        out_use_act=True,
        init_type="default",
        # add
        act_fun="silu",
        expansion_factor=2,
        eps=1e-5,
        norm_type="layer_norm",
    ):
        # add
        self.index = index

        super().__init__()
        # self.e = int(embed_dim * expansion_factor)
        # self.uv = nn.Linear(expansion_factor, 2 * self.e + self.s)
        self.embed_dim = embed_dim
        self.e1 = int(self.embed_dim * expansion_factor)
        self.e2 = int(self.embed_dim // expansion_factor)
        self.u_proj = nn.Linear(embed_dim, self.e1)
        # self.base_proj = nn.Linear(embed_dim, self.s)
        # self.q_weight = nn.Parameter(torch.randn(1, self.s))
        # self.q_bias = nn.Parameter(torch.zeros(1, self.s))
        # self.k_weight = nn.Parameter(torch.randn(1, self.s))
        # self.k_bias = nn.Parameter(torch.zeros(1, self.s))
        # self.q_proj = nn.Linear(embed_dim, embed_dim)
        # self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.memory_proj = nn.Linear(self.e2, self.e1)
        self.qk_proj = nn.Linear(embed_dim, self.e2)
        self.q_weight = nn.Parameter(torch.randn(1, self.e2))
        self.q_bias = nn.Parameter(torch.zeros(1, self.e2))
        self.k_weight = nn.Parameter(torch.randn(1, self.e2))
        self.k_bias = nn.Parameter(torch.zeros(1, self.e2))
        self.o = nn.Linear(self.e1, self.embed_dim)
        # role as v
        self.memory = nn.Parameter(torch.zeros(max_l, self.e2))
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embed_dim, eps=eps)
            self.after_norm = nn.LayerNorm(self.e1, eps=eps)
        elif norm_type == "scale_norm":
            self.norm = ScaleNorm(eps=eps)
            self.after_norm = ScaleNorm(eps=eps)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embed_dim, eps=eps)
            self.after_norm = RMSNorm(self.e1, eps=eps)
        if act_fun == "silu":
            self.act_fn = F.silu
        else:
            self.act_fn = F.gelu

        self.attention_use_layer_norm = attention_use_layer_norm
        self.norm_type = norm_type
        # if self.attention_use_layer_norm:
        #     if self.norm_type == "rmsnorm":
        #         self.layer_norm = RMSNorm(embed_dim)
        #     else:
        #         self.layer_norm = nn.LayerNorm(embed_dim)

        # memory
        self.i = 0
        self.model_update_freq = model_update_freq
        self.lambda_ = lambda_

        # add begin

        self.use_relu = use_relu
        self.use_elu = use_elu
        self.use_leak = use_leak
        self.use_bound = use_bound
        self.bound = embed_dim ** -0.5
        self.causal = causal
        self.use_gelu = use_gelu
        self.mem_use_gelu = mem_use_gelu
        self.has_out = has_out
        self.mem_use_q = mem_use_q
        self.mem_use_k = mem_use_k
        self.act_fun = act_fun
        self.out_use_act = out_use_act
        self.init_type = init_type
        self.seq_dropout = seq_dropout
        self.seq_p = seq_p

        nn.init.normal_(self.q_weight, std=0.02)
        nn.init.normal_(self.k_weight, std=0.02)

        print("mem gau attention")
        print(f"causal {self.causal}")
        print(f"mem_use_q {self.mem_use_q}")
        print(f"mem_use_k {self.mem_use_k}")
        print(f"attention_use_layer_norm {self.attention_use_layer_norm}")
        print(f"model_update_freq {self.model_update_freq}")
        print(f"act_fun_type: {act_fun}")
        print(f"norm_type {self.norm_type}")
        print(f"seq_dropout {self.seq_dropout}")
        print(f"seq_p {self.seq_p}")
        print(f"norm_type {norm_type}")
        print(f"eps {eps}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # if self.init_type == "gelu":
        #     self.gelu_reset()
        # else:
        #     self.reset_parameters()

    def get_act_fun(self):
        if self.act_fun == "gelu":
            return F.gelu
        elif self.act_fun == "relu":
            return F.relu
        else:
            return None

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        print("normal init")
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

    def gelu_reset(self):
        print("use gelu init")
        # std gelu
        c = 0.5874
        d1, d2 = self.k_proj.weight.shape
        nn.init.normal_(self.k_proj.weight, std=c * np.sqrt(2 / (d1 + d2)))
        d1, d2 = self.q_proj.weight.shape
        nn.init.normal_(self.q_proj.weight, std=c * np.sqrt(2 / (d1 + d2)))
        d1, d2 = self.out_proj.weight.shape
        nn.init.normal_(self.out_proj.weight, std=np.sqrt(2 / (d1 + d2)))
        # if self.bias_k is not None:
        #     nn.init.xavier_normal_(self.bias_k)

    # def get_weight(self, max_l):
    #     if (self.weight_type == 1):
    #         a = np.pi / 2
    #         index = a * torch.arange(1, max_l + 1).reshape(1, -1, 1, 1)

    #         return nn.Parameter(index, requires_grad=False)
    #     elif (self.weight_type == 2) or (self.weight_type == 3):
    #         index = torch.arange(1, max_l + 1).reshape(1, -1, 1, 1)

    #         return nn.Parameter(index, requires_grad=False)
    def fft_coef(self, k):
        return (1 - ((-1) ** k) * np.exp(-1)) / (1 + (np.pi * k) ** 2)

    def get_weight(self, max_l):
        # cosformer
        if (self.weight_type == 1):
            a = np.pi / 2
            index = a * torch.arange(1, max_l + 1).reshape(1, -1, 1)

            return nn.Parameter(index, requires_grad=False)
        elif (self.weight_type == 2) or (self.weight_type == 3) or (self.weight_type == 4):
            # 1 - x^2
            index = torch.arange(1, max_l + 1).reshape(1, -1, 1)

            return nn.Parameter(index, requires_grad=False)

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
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        # N, L, E
        x = query.transpose(0, 1)
        # N, S, E
        y = key.transpose(0, 1)
        # normalize
        shortcut, x = x, self.norm(x)
        y = self.norm(y)
        # N, L, E1
        u = self.act_fn(self.u_proj(x))
        # N, L, E2
        q_base = self.act_fn(self.qk_proj(x))
        # N, L, E1
        # q_memory = self.act_fn(self.q_memory(x))
        # N, S, E2
        k_base = self.act_fn(self.qk_proj(y))
        # transform
        # N, S, E2
        q = q_base * self.q_weight + self.q_bias
        k = k_base * self.k_weight + self.k_bias

        if self.mem_use_q:
            if self.mem_use_gelu:
                memory = self.lambda_ * self.memory.unsqueeze(0)
            else:
                memory = self.lambda_ * self.memory.unsqueeze(0)
            # b, L, E'
            memory = memory.repeat(bsz, 1, 1)

            memory[:, :tgt_len] = memory[:, :tgt_len] + (1 - self.lambda_) * q
            memory = memory[:, :src_len]
            memory = self.memory_proj(memory)
        else:
            # 会oom, Qk^TK形式反传有问题
            if self.mem_use_gelu:
                memory = (1 - self.lambda_) * k + self.lambda_ * F.gelu(self.memory[:src_len].unsqueeze(0))
            else:
                memory = (1 - self.lambda_) * k + self.lambda_ * self.memory[:src_len].unsqueeze(0)

        if self.training and self.seq_dropout:
            N = k.shape[0]
            rand = torch.rand(N, max(tgt_len, src_len))
            index = rand < self.seq_p
            q[index[:, :tgt_len]] = 0
            k[index[:, :src_len]] = 0
            # 保持期望不变
            k = k / (1 - (1 - self.seq_p) ** 2)

        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).to(q)
            
            weights = torch.bmm(q, k.transpose(1, 2))
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
            output = torch.bmm(weights, memory)
        else:
            o1 = torch.matmul(k.transpose(1, 2), memory)
            output = torch.bmm(q, o1)

        # B, N, e2
        if self.attention_use_layer_norm:
            output = self.after_norm(output)

        # GLU
        # N, L, E'
        output = u * output
        # N, L, E
        output = self.o(output)
        # N, L, E
        output = x + shortcut
        # L, N, E
        output = output.contiguous().transpose(0, 1)
        
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