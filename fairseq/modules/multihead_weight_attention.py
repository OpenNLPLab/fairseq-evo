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
import sys
from fast_transformers.causal_product import causal_dot_product
# N, L, H, E, batch, length, head, dim

# cosformer
@with_incremental_state
class MultiheadWeightAttention(nn.Module):
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
        max_l=1024,
        has_out=False,
        causal=False,
        weight_type=1,
        c=1.0,
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


        # add begin
        self.use_relu = use_relu
        self.use_elu = use_elu
        self.use_leak = use_leak
        self.max_l = max_l
        self.has_out = has_out
        self.causal = causal
        self.weight_type = weight_type
        self.weight_index = self.get_weight(self.max_l)
        self.add_zero_attn = add_zero_attn

        if (self.weight_type == 1):
            print("cos")
        elif (self.weight_type == 2):
            self.c = c
            print(f"1 - {self.c} * x^2")
        elif (self.weight_type == 3):
            a0 = 1 - np.exp(-1)
            a2 = 25 / 2 - 35 * np.exp(-1)
            self.b0 = 3 * a2 / 2
            self.b1 = a0 - a2 / 2
            
            print("e^-|x|")
        elif (self.weight_type == 4):
            self.c0 = 1 - np.exp(-1)
            self.c1 = self.fft_coef(1)
            self.c2 = self.fft_coef(2)
            print("fourier")
            print("e^-|x|")

        if self.has_out:
            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )

        self.reset_parameters()

        # for test
        self.cnt = 0

        self.onnx_trace = False

        print(f"causal {self.causal}")
        print(f"use relu {self.use_relu}")

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.v_proj.weight)

        if self.has_out:
            nn.init.xavier_uniform_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)

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
        if (self.weight_type == 1):
            a = np.pi / 2
            index = a * torch.arange(1, max_l + 1).reshape(1, -1, 1)

            return nn.Parameter(index, requires_grad=False)
        elif (self.weight_type == 2) or (self.weight_type == 3) or (self.weight_type == 4):
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
        # self.cnt += 1
        # if self.cnt == 10:
        #     sys.exit(0)
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        tgt_len, bsz, embed_dim = query.size()
        m = max(src_len, tgt_len)

        scaling = float(embed_dim) ** -0.5
        # q *= self.scaling
        # L, N, E1
        q = self.q_proj(query)
        # S, N, E1
        k = self.k_proj(key)
        # S, N, E2
        v = self.v_proj(value)


        # N, L, H, E, batch, length, head, dim
        # # N * b, L, e1
        # q = q.contiguous().view(tgt_len,  bsz * num_heads, head_dim).transpose(0, 1)
        # # N * b, S, e2
        # if k is not None:
        #     k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # # N * b, S, e2
        # if v is not None:
        #     v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        # N, L, H, E, batch, length, head, dim
        # N * b, L, e1
        # # N, L, H, D
        # q = q.view(tgt_len, bsz, num_heads, head_dim).transpose(0, 1)
        # # N * b, S, e2
        # # N, S, H, D
        # if k is not None:
        #     k = k.view(-1, bsz, num_heads, head_dim).transpose(0, 1)
        # # N * b, S, e2
        # # N, S, H, D
        # if v is not None:
        #     v = v.view(-1, bsz, num_heads, head_dim).transpose(0, 1)

        # no head
        # N, L, E
        q = q.transpose(0, 1)
        # N, S, E
        if k is not None:
            k = k.transpose(0, 1)
        # N, S, E
        if v is not None:
            v = v.transpose(0, 1)

        if self.use_relu:
            q = F.relu(q)
            k = F.relu(k)
        elif self.use_elu:
            q = F.elu(q)
            k = F.elu(k)
        elif self.use_leak:
            q = F.leaky_relu(q)
            k = F.leaky_relu(k)

        with torch.autograd.profiler.record_function("multihead-weight-attention"):
            q_index = self.weight_index[:, :tgt_len, :] / m
            k_index = self.weight_index[:, :src_len, :] / m
            if (self.weight_type == 1):
                q_ = torch.cat([q * torch.sin(q_index), q * torch.cos(q_index)], dim=-1)
                k_ = torch.cat([k * torch.sin(k_index), k * torch.cos(k_index)], dim=-1)
            if (self.weight_type == 2):
                q_ = torch.cat([(1 - self.c * torch.square(q_index)) * q, 2 * self.c * q_index * q, self.c * q], dim=-1)
                k_ = torch.cat([k, k_index * k, -torch.square(k_index) * k], dim=-1)
            elif (self.weight_type == 3):
                q_ = torch.cat([(self.b1 + self.b0 * torch.square(q_index)) * q, - 2 * self.b0 * q_index * q, self.b0 * q], dim=-1)
                k_ = torch.cat([k, k_index * k, torch.square(k_index) * k], dim=-1)
            elif (self.weight_type == 4):
                q_ = torch.cat([self.c0 * q, self.c1 * q * torch.sin(np.pi * q_index), self.c1 * q * torch.cos(np.pi * q_index), \
                                self.c2 * q * torch.sin(2 * np.pi * q_index), self.c2 * q * torch.cos(2 * np.pi * q_index)], dim=-1)
                k_ = torch.cat([k, k * torch.sin(np.pi * k_index), k * torch.cos(np.pi * k_index), \
                                k * torch.sin(2 * np.pi * k_index), k * torch.cos(2 * np.pi * k_index)], dim=-1)
            # v_ = torch.cat([v, v], dim=-1)
            eps = 1e-6


            # with torch.profiler.profile() as p:
            if self.causal:
                # to do
                # N, L, H, D
                qkv_cos_sin = causal_linear(q_, k_, v)

                # 分母
                # N, L, H
                z_cos_sin = 1 / torch.clamp_min(torch.einsum('nlhi,nlhi->nlh', q_, torch.cumsum(k_, dim=1)), eps)

                # (N * b, S, e1)
                # N, L, H, D
                # N, L, H, D -> L, N, H, D -> L, N, E
                attn_output = (qkv_cos_sin * z_cos_sin.unsqueeze(-1)).transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            else:
                # (N * b, e1, e2)
                kv_ = torch.einsum('nsd,nsm->nmd', k_, v)
                # (N * b, S, e1) (N * b, e1) -> (N * b, S)
                z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
                # (N * b, S, e1) (N * b, e1, e2) (N * b, S)
                attn_output = torch.einsum('nld,nmd,nl->nlm', q_, kv_, z_)

                # N, L, H, D -> L, N, H, D -> L, N, E
                attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            # print(attn_output.shape)
            # add
            if self.has_out:
                attn_output = self.out_proj(attn_output)

        # with torch.autograd.profiler.record_function("multihead-weight-attention"):
        #     q_index = self.weight_index[:, :tgt_len, :, :] / m
        #     k_index = self.weight_index[:, :src_len, :, :] / m
        #     if (self.weight_type == 1):
        #         q_ = torch.cat([q * torch.sin(q_index), q * torch.cos(q_index)], dim=-1)
        #         k_ = torch.cat([k * torch.sin(k_index), k * torch.cos(k_index)], dim=-1)
        #     if (self.weight_type == 2):
        #         q_ = torch.cat([(1 - torch.square(q_index)) * q, 2 * q_index * q, q], dim=-1)
        #         k_ = torch.cat([k, k_index * k, -torch.square(k_index) * k], dim=-1)
        #     elif (self.weight_type == 3):
        #         q_ = torch.cat([(self.b1 + self.b0 * torch.square(q_index)) * q, - 2 * self.b0 * q_index * q, self.b0 * q], dim=-1)
        #         k_ = torch.cat([k, k_index * k, torch.square(k_index) * k], dim=-1)
        #     # v_ = torch.cat([v, v], dim=-1)
        #     eps = 1e-6

        #     # with torch.profiler.profile() as p:
        #     if self.causal:
        #         # N, L, H, D
        #         qkv_cos_sin = causal_linear(q_, k_, v)

        #         # 分母
        #         # N, L, H
        #         z_cos_sin = 1 / torch.clamp_min(torch.einsum('nlhi,nlhi->nlh', q_, torch.cumsum(k_, dim=1)), eps)

        #         # (N * b, S, e1)
        #         # N, L, H, D
        #         # N, L, H, D -> L, N, H, D -> L, N, E
        #         attn_output = (qkv_cos_sin * z_cos_sin.unsqueeze(-1)).transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        #     else:
        #         # (N * b, e1, e2)
        #         kv_ = torch.einsum('nshd,nshm->nhmd', k_, v)
        #         # (N * b, S, e1) (N * b, e1) -> (N * b, S)
        #         z_ = 1 / torch.clamp_min(torch.einsum('nlhd,nhd->nlh', q_, torch.sum(k_, axis=1)), eps)
        #         # (N * b, S, e1) (N * b, e1, e2) (N * b, S)
        #         attn_output = torch.einsum('nlhd,nhmd,nlh->nlhm', q_, kv_, z_)

        #         # N, L, H, D -> L, N, H, D -> L, N, E
        #         attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        #     # print(attn_output.shape)
        #     # add
        #     if self.has_out:
        #         attn_output = self.out_proj(attn_output)

        # sys.exit(0)

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