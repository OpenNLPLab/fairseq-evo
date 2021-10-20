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


# cosformer
@with_incremental_state
class MultiheadCosformerAttention(nn.Module):
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
        causal=False
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

        self.weight_index = self.get_alpha_beta(self.max_l)

        self.add_zero_attn = add_zero_attn

        if self.has_out:
            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )

        self.reset_parameters()

        self.onnx_trace = False

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

    def get_alpha_beta(self, max_l):
        a = np.pi / 2
        index = a * torch.arange(1, max_l + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

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

        scaling = float(embed_dim) ** -0.5
        # q *= self.scaling
        # L, N, E1
        q = self.q_proj(query)
        # S, N, E1
        k = self.k_proj(key)
        # S, N, E2
        v = self.k_proj(value)

        # N * b, L, e1
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        # N * b, S, e2
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # N * b, S, e2
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if self.use_relu:
            q = F.relu(q)
            k = F.relu(k)
        elif self.use_elu:
            q = F.elu(q)
            k = F.elu(k)
        elif self.use_leak:
            q = F.leaky_relu(q)
            k = F.leaky_relu(k)

        # 1, L, 1
        qsin = torch.sin(self.weight_index[:, :tgt_len, :] / tgt_len)
        ksin = torch.sin(self.weight_index[:, :src_len, :] / src_len)
        qcos = torch.cos(self.weight_index[:, :tgt_len, :] / tgt_len)
        kcos = torch.cos(self.weight_index[:, :src_len, :] / src_len)
        print(qsin.requires_grad)
        print(ksin.requires_grad)
        print(qcos.requires_grad)
        print(kcos.requires_grad)

        # N * b, L, e1
        q_sin = q * qsin
        q_cos = q * qcos
        # N * b, S, e2
        k_sin = k * ksin
        k_cos = k * kcos

        eps = 1e-6

        # # N * b, L, S
        # attn_output_weights = torch.bmm(q_sin, k_sin.transpose(1, 2)) + torch.bmm(q_cos, k_cos.transpose(1, 2))
        # if attn_mask is not None:
        #     attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)

        # # N * b, L, S
        # attn_output_weights = F.normalize(attn_output_weights, p=1, dim=-1, eps=1e-8)

        # # print(attn_output_weights[0][0])
        # attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)

        # # N, L, E
        # # attn_output = torch.bmm(attn_output_weights, value)
        # # N * b, L, e2
        # attn_output = torch.matmul(attn_output_weights, v)
        # # L, N, E2
        # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # print(attn_output.shape)

        if self.causal:
            ## cos
            # vi * ki^T, (N, E2, E1)
            kv_cos = torch.einsum('nsd,nsm->nmd', k_cos, v)
            # sum(i < j) vi * ki^T, (N, E2, E1)
            kv_cos_cum = torch.cumsum(kv_cos, dim=0)
            # q * sum(i < j) vi * ki^T, (N, L, E2)
            qkv_cos = torch.einsum("nld,nmd->nlm", q_cos, kv_cos_cum)
            ## sin
            kv_sin = torch.einsum('nsd,nsm->nmd', k_sin, v)
            kv_sin_cum = torch.cumsum(kv_sin, dim=0)
            qkv_sin = torch.einsum("nld,nmd->nlm", q_sin, kv_sin_cum)
            ## sum, (N, L, E2)
            qkv_cos_sin = qkv_cos + qkv_sin

            # 分母
            # N, L, E1
            z_cos = torch.cumsum(k_cos, dim=0)
            # N, L, E1
            z_sin = torch.cumsum(k_sin, dim=0)
            # N, L
            z_cos_sin = 1 / (
                torch.einsum('nld,nld->nl', q_cos, z_cos) + \
                torch.einsum('nld,nld->nl', q_sin, z_sin) + \
                eps)

            # (N, L, E2)
            qz = qkv_cos_sin / z_cos_sin.unsqueeze(-1)
        else:
            print(1)
            # eps = 1e-6
            kv_cos = torch.einsum('nsd,nsm->nmd', k_cos, v)
            kv_sin = torch.einsum('nsd,nsm->nmd', k_sin, v)
            print(kv_cos.shape)
            z_cos_sin = 1 / (
                torch.einsum('nld,nd->nl', q_cos, torch.sum(k_cos, axis=1)) + \
                torch.einsum('nld,nd->nl', q_sin, torch.sum(k_sin, axis=1)) + \
                eps)
            print(z_cos_sin.shape)
            # N, L, E
            attn_output = torch.einsum('nld,nmd,nl->nlm', q_cos, kv_cos, z_cos_sin) + \
                        torch.einsum('nld,nmd,nl->nlm', q_sin, kv_sin, z_cos_sin)
            # L, N, E
            attn_output = attn_output.transpose(0, 1)

        print(q.shape)
        print(k.shape)
        print(value.shape)
        print(attn_output.shape)


        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

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
