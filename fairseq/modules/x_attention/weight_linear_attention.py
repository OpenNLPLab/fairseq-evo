import math
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules import (GatedRMSNorm, RMSNorm, SimpleRMSNorm, Toeplizt,
                             Urpe)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.helpers import (get_activation_fn, get_norm_fn,
                                     logging_info, print_params)
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Dropout, Parameter


@with_incremental_state
class WeightLinearAttention(nn.Module):
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
        act_fun="relu",
        weight_type=-1,
        causal=False,
        # urpe
        use_urpe=True,
        core_matrix=1, 
        p_matrix=1, 
        theta_type="a",
        theta_learned=False, 
        householder_learned=False,
        # norm
        use_norm=False,
        norm_type="simplermsnorm",
        use_sigmoid=False,
        # cos weight type=5
        cos_prenorm=False,
        cos_postnorm=False,
    ):
        super().__init__()
        # add
        self.index = index
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

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
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        
        self.causal = causal
        self.weight_type = weight_type
        if (self.weight_type == 1):
            logging_info("cos")
        if self.weight_type == 2:
            logging_info("1 - c x ^ 2")
            self.alpha = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))
        elif (self.weight_type == 3):
            a0 = 1 - np.exp(-1)
            a2 = 25 / 2 - 35 * np.exp(-1)
            self.b0 = 3 * a2 / 2
            self.b1 = a0 - a2 / 2
            logging_info("e^-|x|")
        elif (self.weight_type == 4):
            self.c0 = 1 - np.exp(-1)
            self.c1 = self.fft_coef(1)
            self.c2 = self.fft_coef(2)
            logging_info("e^-|x|: fourier")
        elif self.weight_type == 5:
            logging_info("cos(q - k)")
            # h, 1, 1
            self.freq = nn.Parameter(np.pi / 2 * (2 ** torch.arange(num_heads).reshape(1, -1, 1, 1)), requires_grad=False)
            self.forward = self.forward_cos
            self.cos_prenorm = cos_prenorm
            self.cos_postnorm = cos_postnorm

        self.act = get_activation_fn(act_fun)
        
        # urpe
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.use_urpe = use_urpe
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned
        self.use_sigmoid = use_sigmoid
        if self.use_urpe:
            self.urpe = Urpe(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
        
        # norm
        self.use_norm = use_norm
        self.norm_type = norm_type
        if self.use_norm:
            self.norm = get_norm_fn(norm_type)(embed_dim)
            
        if self.weight_type == 5:
            self.act = None
            if not self.cos_postnorm:
                self.norm = None

        self.reset_parameters()
        
    def fft_coef(self, k):
        return (1 - ((-1) ** k) * np.exp(-1)) / (1 + (np.pi * k) ** 2)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        logging_info("normal init")
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward_cos(
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

        # L, N, E1
        q = self.q_proj(query)
        # S, N, E1
        k = self.k_proj(key)
        v = self.v_proj(value)

        # N, L, H, E, batch, length, head, dim
        # N, L, e1
        head_dim = embed_dim // num_heads

        q, k, v = map(lambda x: rearrange(x, 'n b (h d) -> b h n d', h=num_heads), [q, k, v])
        
        if self.cos_prenorm:
            # print("use_norm")
            q = F.normalize(q)
            k = F.normalize(k)
        
        q = self.freq * q
        k = self.freq * k
        q = torch.cat([torch.cos(q), torch.sin(q)], dim=-1)
        k = torch.cat([torch.cos(k), torch.sin(k)], dim=-1)
    
        eps = 1e-4
        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).to(q)
            weights = torch.einsum('...nd,...md->...nm', q, k)
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
            if not self.use_norm:
                # print("with denom")
                denorm = weights.sum(dim=-1, keepdim=True)
                # denorm = torch.clamp_min(denorm, eps)
                weights = weights / denorm
            # print(weights[0][:][:5, :5])
            output = torch.einsum('...nm,...md->...nd', weights, v)
        else:
            o1 = torch.einsum('...nd,...ne->...de', k, v)
            output = torch.einsum('...nd,...de->...ne', q, o1)
            if not self.use_norm:
                # print("with denom")
                denorm = torch.einsum('...nd,...d->...n', q, torch.sum(k, axis=-2)).unsqueeze(-1)
                # denorm = torch.clamp_min(denorm, eps)
                output = output / denorm

        output = rearrange(output, 'b h n e -> n b (h e)')
        
        if self.use_norm and self.cos_postnorm:
            # print("withpost")
            output = self.norm(output)

        # L, N, e1
        output = self.out_proj(output)

        return output, None

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
        eps = 1e-4

        # L, N, E1
        q = self.q_proj(query)
        # S, N, E1
        k = self.k_proj(key)
        v = self.v_proj(value)

        # N, L, H, E: batch, length, head, dim
        # N, L, e1
        head_dim = embed_dim // num_heads

        l = max(src_len, tgt_len)

        q, k, v = map(lambda x: rearrange(x, 'n b (h d) -> b h n d', h=num_heads), [q, k, v])
        q = self.act(q)
        k = self.act(k)

        if self.use_urpe:
            q = self.urpe(q)
            k = self.urpe(k)

        if self.weight_type == 1:
            # logging_info("cos")
            m = max(tgt_len, src_len)
            index = torch.arange(m).reshape(1, 1, -1, 1).to(q)
            q_index = np.pi / 2 * index[:, :, :tgt_len, :] / m
            k_index = np.pi / 2 * index[:, :, :src_len, :] / m
            q = torch.cat([q * torch.sin(q_index), q * torch.cos(q_index)], dim=-1)
            k = torch.cat([k * torch.sin(k_index), k * torch.cos(k_index)], dim=-1)
        elif self.weight_type == 2:
            # logging_info("1 - a x^2")
            m = max(tgt_len, src_len)
            index = torch.arange(m).reshape(1, 1, -1, 1).to(q)
            # 1, h, n, 1
            q_index = index[:, :tgt_len, :] / m
            k_index = index[:, :src_len, :] / m
            alpha = self.alpha
            if self.use_sigmoid:
                alpha = F.sigmoid(alpha)
            q = torch.cat([(1 - alpha * (q_index ** 2)) * q, 2 * alpha * q_index * q, q], dim=-1)
            k = torch.cat([k, k_index * k, alpha * (k_index ** 2) * k], dim=-1)
        elif (self.weight_type == 3):
            # logging_info("e ^ -|x|")
            m = max(tgt_len, src_len)
            index = torch.arange(m).reshape(1, 1, -1, 1).to(q)
            q_index = np.pi / 2 * index[:, :, :tgt_len, :] / m
            k_index = np.pi / 2 * index[:, :, :src_len, :] / m
            q = torch.cat([(self.b1 + self.b0 * torch.square(q_index)) * q, - 2 * self.b0 * q_index * q, self.b0 * q], dim=-1)
            k = torch.cat([k, k_index * k, torch.square(k_index) * k], dim=-1)
        elif (self.weight_type == 4):
            # logging_info("e ^ -|x| fourier")
            m = max(tgt_len, src_len)
            index = torch.arange(m).reshape(1, 1, -1, 1).to(q)
            q_index = np.pi / 2 * index[:, :, :tgt_len, :] / m
            k_index = np.pi / 2 * index[:, :, :src_len, :] / m
            q = torch.cat([self.c0 * q, self.c1 * q * torch.sin(np.pi * q_index), self.c1 * q * torch.cos(np.pi * q_index), \
                            self.c2 * q * torch.sin(2 * np.pi * q_index), self.c2 * q * torch.cos(2 * np.pi * q_index)], dim=-1)
            k = torch.cat([k, k * torch.sin(np.pi * k_index), k * torch.cos(np.pi * k_index), \
                            k * torch.sin(2 * np.pi * k_index), k * torch.cos(2 * np.pi * k_index)], dim=-1)

        eps = 1e-4
        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).to(q)
            weights = torch.einsum('...nd,...md->...nm', q, k)
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
            if not self.use_norm:
                denorm = weights.sum(dim=-1, keepdim=True)
                denorm = torch.clamp_min(denorm, eps)
                weights = weights / denorm
            output = torch.einsum('...nm,...md->...nd', weights, v)
        else:
            o1 = torch.einsum('...nd,...ne->...de', k, v)
            output = torch.einsum('...nd,...de->...ne', q, o1)
            if not self.use_norm:
                denorm = torch.einsum('...nd,...d->...n', q, torch.sum(k, axis=-2)).unsqueeze(-1)
                denorm = torch.clamp_min(denorm, eps)
                output = output / denorm

        output = rearrange(output, 'b h n e -> n b (h e)')
        
        if self.use_norm:
            output = self.norm(output)

        # L, N, e1
        output = self.out_proj(output)

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

