import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.rope import rope
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.modules import Orpe
from fairseq.modules import SineSPE, SPEFilter
from einops import rearrange
from fairseq.modules import T5RPE


@with_incremental_state
class MultiheadAttention_(nn.Module):
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
        # change
        norm_taylor=True,
        use_relu=False,
        use_elu=False,
        use_leak=False,
        use_square=False,
        use_sigmoid=False,
        use_linear=False,
        use_softplus=False,
        use_basic=True,
        use_abs=False,
        # 因子
        alpha_beta=False,
        max_l=1024,
        weight_type=1,
        use_rope=False,
        # add
        use_orpe=False,
        core_matrix=1, 
        p_matrix=1, 
        max_positions=512,
        theta_type="a",
        theta_learned=False, 
        householder_learned=False,
        # spe
        use_spe=False,
        use_permutate=False,
        max_seq_len=512,
        # t5
        causal=False,
        use_t5=False,
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
        self.scaling = self.head_dim ** -0.5

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

        self.weight_type = weight_type
        self.use_rope = use_rope
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.max_positions = max_positions
        self.use_orpe = use_orpe
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned
        if self.use_orpe:
            self.orpe = Orpe(self.core_matrix, self.p_matrix, embedding_dim=self.head_dim, theta_type=theta_type, theta_learned=theta_learned, householder_learned=householder_learned)
        self.use_spe = use_spe
        if self.use_spe:
            self.spe_encoder = SineSPE(num_heads=self.num_heads,          # Number of attention heads
                                       in_features=self.head_dim,       # Dimension of keys and queries
                                       num_realizations=self.head_dim,  # New dimension of keys and queries
                                       num_sines=1)          # Number of sinusoidal components
            self.spe_filter = SPEFilter(gated=True, code_shape=self.spe_encoder.code_shape)
        self.use_permutate = use_permutate
        if self.use_permutate:
            raw_permutation = self.generate_random_permutation(self.num_heads, self.kdim // self.num_heads, 0xdeadbeefdeadbeef)
            permutation = self.expand_permutation(max_seq_len, raw_permutation)
            self.register_buffer("permutation", permutation.unsqueeze(0))
            self.register_buffer("ratio", torch.sigmoid(torch.arange(self.num_heads) / self.num_heads * 3 + 2))
        self.use_t5 = use_t5
        if self.use_t5:
            bidirectional = not causal
            self.rpe = T5RPE(bidirectional)

        print(f"weight_type {weight_type}")
        print(f"use_rope {use_rope}")
        print(f"use_orpe {self.use_orpe}")
        print(f"use_spe {self.use_spe}")
        print(f"use_permutate {self.use_permutate}")
        print(f"use_t5 {self.use_t5}")

    # https://github.com/cpcp1998/PermuteFormer/blob/master/language_model/permute/__init__.py
    def generate_random_permutation(self, num_head, head_size, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        permutate = torch.randperm(head_size, generator=rng)
        permutation = [permutate for _ in range(num_head)]
        # permutation = [torch.randperm(head_size, generator=rng) for _ in range(num_head)]
        # change to the same setting in orpe
        permutation = torch.stack(permutation, dim=0)
        return permutation

    # https://github.com/cpcp1998/PermuteFormer/blob/master/language_model/permute/__init__.py
    def expand_permutation(self, max_seq_length, permutation):
        num_head, head_size = permutation.shape
        expanded = [torch.arange(head_size).unsqueeze(0).expand(num_head, head_size)]
        for _ in range(max_seq_length - 1):
            previous = expanded[-1]
            current = previous.gather(-1, permutation)
            expanded.append(current)
        expanded = torch.stack(expanded, dim=1)
        return expanded

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

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_matrix(self, l1, l2):
        i1 = torch.arange(1, l1 + 1).reshape(1, -1, 1)
        i2 = torch.arange(1, l2 + 1).reshape(1, 1, -1)
        weight = np.pi / 2 * (i1 - i2) / max(l1, l2)
        m = torch.cos(weight)

        return nn.Parameter(m, requires_grad=False)


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
        # S, N, E
        v = self.v_proj(value)

        if self.use_spe:
            q = rearrange(q, 'l n (h d) -> l n h d', h=num_heads)
            k = rearrange(k, 'l n (h d) -> l n h d', h=num_heads)
            v = rearrange(v, 'l n (h d) -> l (n h) d', h=num_heads)
            v = rearrange(v, 'l n d -> n l d')
            pos_codes = self.spe_encoder(q.shape[:2])  # pos_codes is a tuple (qbar, kbar)
            q, k = self.spe_filter(q, k, pos_codes)
            q = rearrange(q, 'l n h d -> (n h) l d')
            k = rearrange(k, 'l n h d -> (n h) l d')
        elif self.use_permutate:
            q = rearrange(q, 'l n (h d) -> n l h d', h=num_heads)
            q = rearrange(q, 'n l h d -> n h l d')
            k = rearrange(k, 'l n (h d) -> n l h d', h=num_heads)
            k = rearrange(k, 'n l h d -> n h l d')
            q = q.gather(-1, self.permutation[:, :, :q.shape[2]].expand_as(q))
            k = k.gather(-1, self.permutation[:, :, :k.shape[2]].expand_as(k))
            # act
            q *= (self.ratio.unsqueeze(-1) ** torch.arange(q.shape[2], device=q.device).unsqueeze(0)).unsqueeze(-1)
            k *= ((1 / self.ratio).unsqueeze(-1) ** torch.arange(k.shape[2], device=k.device).unsqueeze(0)).unsqueeze(-1)
            # change shape
            q = rearrange(q, 'n h l d -> (n h) l d')
            k = rearrange(k, 'n h l d -> (n h) l d')
            v = rearrange(v, 'l n (h d) -> l (n h) d', h=num_heads)
            v = rearrange(v, 'l n d -> n l d')
        else:
            # N * h, L, d
            q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            # N * h, S, d
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            # N * h, S, d
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if self.use_rope:
            q = rope(q, dim=1)
            k = rope(k, dim=1)
        # orpe
        if self.use_orpe:
            q = self.orpe(q)
            k = self.orpe(k)
            
        scaling = float(embed_dim) ** -0.5
        q = q * scaling

        # cos transform
        if self.weight_type == 1:
            m = max(src_len, tgt_len)
            # get index and send to cuda
            weight_index = self.get_index(m).to(q)
            # (N * h, L, 2 * d)
            q = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
            # (N * h, S, 2 * d)
            k = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        # N * h, L, S
        if self.use_orpe and self.orpe.core_matrix == 4:
            q = torch.cat([q.real, q.imag], dim=-1)
            k = torch.cat([k.real, k.imag], dim=-1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

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

        if self.use_t5:
            # print("here")
            attn_output_weights = self.rpe(attn_output_weights)

        if attn_mask is not None:
            attn_output_weights += attn_mask
        # print(attn_output_weights[0])

    
        # N * h, L, S
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if self.weight_type == 2:
            matrix = self.get_matrix(tgt_len, tgt_len).to(q)
            attn_output_weights = attn_output_weights * matrix
            attn_output_weights_sum = torch.sum(attn_output_weights, dim=-1, keepdim=True)
            attn_output_weights = attn_output_weights / attn_output_weights_sum
            # print(attn_output_weights)
            # print(torch.sum(attn_output_weights, dim=-1))

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


# @with_incremental_state
# class MultiheadAttention_(nn.Module):
#     """Multi-headed attention.

#     See "Attention Is All You Need" for more details.
#     """

#     def __init__(
#         self,
#         embed_dim,
#         num_heads,
#         kdim=None,
#         vdim=None,
#         dropout=0.0,
#         bias=True,
#         add_bias_kv=False,
#         add_zero_attn=False,
#         self_attention=False,
#         encoder_decoder_attention=False,
#         q_noise=0.0,
#         qn_block_size=8,
#         # add
#         index=0,
#     ):
#         # add
#         self.index = index

#         super().__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

#         self.num_heads = num_heads
#         self.dropout_module = FairseqDropout(
#             dropout, module_name=self.__class__.__name__
#         )

#         self.head_dim = embed_dim // num_heads
#         assert (
#             self.head_dim * num_heads == self.embed_dim
#         ), "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim ** -0.5

#         self.self_attention = self_attention
#         self.encoder_decoder_attention = encoder_decoder_attention

#         assert not self.self_attention or self.qkv_same_dim, (
#             "Self-attention requires query, key and " "value to be of the same size"
#         )

#         self.k_proj = quant_noise(
#             nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
#         )
#         self.v_proj = quant_noise(
#             nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
#         )
#         self.q_proj = quant_noise(
#             nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
#         )

#         # add
#         self.out_proj = quant_noise(
#             nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
#         )

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
#             self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         # add begin
#         # 1 * E
#         self.sigma2 = Parameter(torch.ones(1, self.embed_dim), requires_grad=False)
#         self.lambda_ = 0.99
#         # add end

#         self.reset_parameters()

#         self.onnx_trace = False

#     def prepare_for_onnx_export_(self):
#         self.onnx_trace = True

#     def reset_parameters(self):
#         if self.qkv_same_dim:
#             # Empirically observed the convergence to be much better with
#             # the scaled initialization
#             nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
#             nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
#             nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
#         else:
#             nn.init.xavier_uniform_(self.k_proj.weight)
#             nn.init.xavier_uniform_(self.v_proj.weight)
#             nn.init.xavier_uniform_(self.q_proj.weight)

#         # add begin
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         if self.out_proj.bias is not None:
#             nn.init.constant_(self.out_proj.bias, 0.0)
#         # add end
#         if self.bias_k is not None:
#             nn.init.xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             nn.init.xavier_normal_(self.bias_v)

#     def forward(
#         self,
#         query,
#         key: Optional[Tensor],
#         value: Optional[Tensor],
#         key_padding_mask: Optional[Tensor] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         need_weights: bool = True,
#         static_kv: bool = False,
#         attn_mask: Optional[Tensor] = None,
#         before_softmax: bool = False,
#         need_head_weights: bool = False,
#     ) -> Tuple[Tensor, Optional[Tensor]]:
#         """Input shape: Time x Batch x Channel

#         Args:
#             key_padding_mask (ByteTensor, optional): mask to exclude
#                 keys that are pads, of shape `(batch, src_len)`, where
#                 padding elements are indicated by 1s.
#             need_weights (bool, optional): return the attention weights,
#                 averaged over heads (default: False).
#             attn_mask (ByteTensor, optional): typically used to
#                 implement causal attention, where the mask prevents the
#                 attention from looking forward in time (default: None).
#             before_softmax (bool, optional): return the raw attention
#                 weights and values before the attention softmax.
#             need_head_weights (bool, optional): return the attention
#                 weights for each head. Implies *need_weights*. Default:
#                 return the average attention weights over all heads.
#         """
#         if need_head_weights:
#             need_weights = True

#         assert key is not None and value is not None

#         '''
#         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         '''
#         num_heads = self.num_heads
#         tgt_len, bsz, embed_dim = query.size()
#         src_len = key.size(0)
#         head_dim = embed_dim // num_heads

#         tgt_len, bsz, embed_dim = query.size()
#         # L, N, E
#         q = self.q_proj(query)
#         # S, N, E
#         k = self.k_proj(key)
#         # S, N, E
#         v = self.v_proj(value)

#         if self.training:
#             # L * N, E -> (1, E)
#             # sigma2 = torch.nn.Parameter(torch.var(q.view(-1, self.embed_dim), dim=0, keepdim=True), requires_grad=False)
#             with torch.no_grad():
#                 sigma2 = torch.var(q.view(-1, self.embed_dim), dim=0, keepdim=True)
#                 # sigma2 = sigma2.to(self.sigma2)
#                 self.sigma2 *= self.lambda_
#                 self.sigma2 += (1 - self.lambda_) * sigma2
                
#         # print(torch.mean(self.sigma2), torch.mean(sigma2))
#         # print(self.sigma2.requires_grad)
#         q = q * self.scaling / torch.sqrt(self.sigma2)

#         # N * h, L, d
#         q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#         # N * h, S, d
#         k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#         v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

#         # scaling = float(embed_dim) ** -0.5


#         # N * h, L, S
#         attn_output_weights = torch.bmm(q, k.transpose(1, 2))

#         # attn_mask
#         if attn_mask is not None:
#             if attn_mask.dim() == 2:
#                 attn_mask = attn_mask.unsqueeze(0)
#                 if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
#                     raise RuntimeError('The size of the 2D attn_mask is not correct.')
#             elif attn_mask.dim() == 3:
#                 if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
#                     raise RuntimeError('The size of the 3D attn_mask is not correct.')
#             else:
#                 raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
#         # attn_mask's dim is 3 now.

#         if attn_mask is not None:
#             attn_output_weights += attn_mask
       
#         # N * h, L, S
#         attn_output_weights = F.softmax(attn_output_weights, dim=-1)
#         # dropout
#         attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
#         # N * h, L, d
#         attn_output = torch.bmm(attn_output_weights, v)
#         # L, N, E
#         attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#         # L, N, E
#         attn_output = self.out_proj(attn_output)

#         if need_weights:
#             attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#             return attn_output, attn_output_weights
#         else:
#             return attn_output, None

#     @staticmethod
#     def _append_prev_key_padding_mask(
#         key_padding_mask: Optional[Tensor],
#         prev_key_padding_mask: Optional[Tensor],
#         batch_size: int,
#         src_len: int,
#         static_kv: bool,
#     ) -> Optional[Tensor]:
#         # saved key padding masks have shape (bsz, seq_len)
#         if prev_key_padding_mask is not None and static_kv:
#             new_key_padding_mask = prev_key_padding_mask
#         elif prev_key_padding_mask is not None and key_padding_mask is not None:
#             new_key_padding_mask = torch.cat(
#                 [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
#             )
#         # During incremental decoding, as the padding token enters and
#         # leaves the frame, there will be a time when prev or current
#         # is None
#         elif prev_key_padding_mask is not None:
#             filler = torch.zeros(
#                 (batch_size, src_len - prev_key_padding_mask.size(1)),
#                 device=prev_key_padding_mask.device,
#             )
#             new_key_padding_mask = torch.cat(
#                 [prev_key_padding_mask.float(), filler.float()], dim=1
#             )
#         elif key_padding_mask is not None:
#             filler = torch.zeros(
#                 (batch_size, src_len - key_padding_mask.size(1)),
#                 device=key_padding_mask.device,
#             )
#             new_key_padding_mask = torch.cat(
#                 [filler.float(), key_padding_mask.float()], dim=1
#             )
#         else:
#             new_key_padding_mask = prev_key_padding_mask
#         return new_key_padding_mask

#     @torch.jit.export
#     def reorder_incremental_state(
#         self,
#         incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
#         new_order: Tensor,
#     ):
#         """Reorder buffered internal state (for incremental generation)."""
#         input_buffer = self._get_input_buffer(incremental_state)
#         if input_buffer is not None:
#             for k in input_buffer.keys():
#                 input_buffer_k = input_buffer[k]
#                 if input_buffer_k is not None:
#                     if self.encoder_decoder_attention and input_buffer_k.size(
#                         0
#                     ) == new_order.size(0):
#                         break
#                     input_buffer[k] = input_buffer_k.index_select(0, new_order)
#             incremental_state = self._set_input_buffer(incremental_state, input_buffer)
#         return incremental_state

#     def _get_input_buffer(
#         self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
#     ) -> Dict[str, Optional[Tensor]]:
#         result = self.get_incremental_state(incremental_state, "attn_state")
#         if result is not None:
#             return result
#         else:
#             empty_result: Dict[str, Optional[Tensor]] = {}
#             return empty_result

#     def _set_input_buffer(
#         self,
#         incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
#         buffer: Dict[str, Optional[Tensor]],
#     ):
#         return self.set_incremental_state(incremental_state, "attn_state", buffer)

#     def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
#         return attn_weights

#     def upgrade_state_dict_named(self, state_dict, name):
#         prefix = name + "." if name != "" else ""
#         items_to_add = {}
#         keys_to_remove = []
#         for k in state_dict.keys():
#             if k.endswith(prefix + "in_proj_weight"):
#                 # in_proj_weight used to be q + k + v with same dimensions
#                 dim = int(state_dict[k].shape[0] / 3)
#                 items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
#                 items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
#                 items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

#                 keys_to_remove.append(k)

#                 k_bias = prefix + "in_proj_bias"
#                 if k_bias in state_dict.keys():
#                     dim = int(state_dict[k].shape[0] / 3)
#                     items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
#                     items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
#                         dim : 2 * dim
#                     ]
#                     items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

#                     keys_to_remove.append(prefix + "in_proj_bias")

#         for k in keys_to_remove:
#             del state_dict[k]

#         for key, value in items_to_add.items():
#             state_dict[key] = value

# @with_incremental_state
# class MultiheadAttention_(nn.Module):
#     """Multi-headed attention.

#     See "Attention Is All You Need" for more details.
#     """

#     def __init__(
#         self,
#         embed_dim,
#         num_heads,
#         kdim=None,
#         vdim=None,
#         dropout=0.0,
#         bias=True,
#         add_bias_kv=False,
#         add_zero_attn=False,
#         self_attention=False,
#         encoder_decoder_attention=False,
#         q_noise=0.0,
#         qn_block_size=8,
#         # add
#         index=0,
#     ):
#         # add
#         self.index = index

#         super().__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

#         self.num_heads = num_heads
#         self.dropout_module = FairseqDropout(
#             dropout, module_name=self.__class__.__name__
#         )

#         self.head_dim = embed_dim // num_heads
#         assert (
#             self.head_dim * num_heads == self.embed_dim
#         ), "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim ** -0.5

#         self.self_attention = self_attention
#         self.encoder_decoder_attention = encoder_decoder_attention

#         assert not self.self_attention or self.qkv_same_dim, (
#             "Self-attention requires query, key and " "value to be of the same size"
#         )

#         self.k_proj = quant_noise(
#             nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
#         )
#         self.v_proj = quant_noise(
#             nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
#         )
#         self.q_proj = quant_noise(
#             nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
#         )

#         # add
#         self.out_proj = quant_noise(
#             nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
#         )

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
#             self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         self.reset_parameters()

#         self.onnx_trace = False

#     def prepare_for_onnx_export_(self):
#         self.onnx_trace = True

#     def reset_parameters(self):
#         if self.qkv_same_dim:
#             # Empirically observed the convergence to be much better with
#             # the scaled initialization
#             nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
#             nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
#             nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
#         else:
#             nn.init.xavier_uniform_(self.k_proj.weight)
#             nn.init.xavier_uniform_(self.v_proj.weight)
#             nn.init.xavier_uniform_(self.q_proj.weight)

#         # add begin
#         nn.init.xavier_uniform_(self.out_proj.weight)
#         if self.out_proj.bias is not None:
#             nn.init.constant_(self.out_proj.bias, 0.0)
#         # add end
#         if self.bias_k is not None:
#             nn.init.xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             nn.init.xavier_normal_(self.bias_v)

#     def forward(
#         self,
#         query,
#         key: Optional[Tensor],
#         value: Optional[Tensor],
#         key_padding_mask: Optional[Tensor] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         need_weights: bool = True,
#         static_kv: bool = False,
#         attn_mask: Optional[Tensor] = None,
#         before_softmax: bool = False,
#         need_head_weights: bool = False,
#     ) -> Tuple[Tensor, Optional[Tensor]]:
#         """Input shape: Time x Batch x Channel

#         Args:
#             key_padding_mask (ByteTensor, optional): mask to exclude
#                 keys that are pads, of shape `(batch, src_len)`, where
#                 padding elements are indicated by 1s.
#             need_weights (bool, optional): return the attention weights,
#                 averaged over heads (default: False).
#             attn_mask (ByteTensor, optional): typically used to
#                 implement causal attention, where the mask prevents the
#                 attention from looking forward in time (default: None).
#             before_softmax (bool, optional): return the raw attention
#                 weights and values before the attention softmax.
#             need_head_weights (bool, optional): return the attention
#                 weights for each head. Implies *need_weights*. Default:
#                 return the average attention weights over all heads.
#         """
#         if need_head_weights:
#             need_weights = True

#         assert key is not None and value is not None

#         '''
#         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         '''
#         num_heads = self.num_heads
#         tgt_len, bsz, embed_dim = query.size()
#         src_len = key.size(0)
#         head_dim = embed_dim // num_heads

#         tgt_len, bsz, embed_dim = query.size()
#         # L, N, E
#         q = self.q_proj(query)
#         # S, N, E
#         k = self.k_proj(key)
#         # S, N, E
#         v = self.v_proj(value)

#         # N * h, L, d
#         q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#         # N * h, S, d
#         k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#         v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

#         scaling = float(embed_dim) ** -0.5
#         q = q * scaling

#         # N * h, L, S
#         attn_output_weights = torch.bmm(q, k.transpose(1, 2))

#         # attn_mask
#         if attn_mask is not None:
#             if attn_mask.dim() == 2:
#                 attn_mask = attn_mask.unsqueeze(0)
#                 if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
#                     raise RuntimeError('The size of the 2D attn_mask is not correct.')
#             elif attn_mask.dim() == 3:
#                 if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
#                     raise RuntimeError('The size of the 3D attn_mask is not correct.')
#             else:
#                 raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
#         # attn_mask's dim is 3 now.

#         if attn_mask is not None:
#             attn_output_weights += attn_mask
       
#         # N * h, L, S
#         attn_output_weights = F.softmax(attn_output_weights, dim=-1)
#         # dropout
#         attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
#         # N * h, L, d
#         attn_output = torch.bmm(attn_output_weights, v)
#         # L, N, E
#         attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#         # L, N, E
#         attn_output = self.out_proj(attn_output)

#         if need_weights:
#             attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#             return attn_output, attn_output_weights
#         else:
#             return attn_output, None

#     @staticmethod
#     def _append_prev_key_padding_mask(
#         key_padding_mask: Optional[Tensor],
#         prev_key_padding_mask: Optional[Tensor],
#         batch_size: int,
#         src_len: int,
#         static_kv: bool,
#     ) -> Optional[Tensor]:
#         # saved key padding masks have shape (bsz, seq_len)
#         if prev_key_padding_mask is not None and static_kv:
#             new_key_padding_mask = prev_key_padding_mask
#         elif prev_key_padding_mask is not None and key_padding_mask is not None:
#             new_key_padding_mask = torch.cat(
#                 [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
#             )
#         # During incremental decoding, as the padding token enters and
#         # leaves the frame, there will be a time when prev or current
#         # is None
#         elif prev_key_padding_mask is not None:
#             filler = torch.zeros(
#                 (batch_size, src_len - prev_key_padding_mask.size(1)),
#                 device=prev_key_padding_mask.device,
#             )
#             new_key_padding_mask = torch.cat(
#                 [prev_key_padding_mask.float(), filler.float()], dim=1
#             )
#         elif key_padding_mask is not None:
#             filler = torch.zeros(
#                 (batch_size, src_len - key_padding_mask.size(1)),
#                 device=key_padding_mask.device,
#             )
#             new_key_padding_mask = torch.cat(
#                 [filler.float(), key_padding_mask.float()], dim=1
#             )
#         else:
#             new_key_padding_mask = prev_key_padding_mask
#         return new_key_padding_mask

#     @torch.jit.export
#     def reorder_incremental_state(
#         self,
#         incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
#         new_order: Tensor,
#     ):
#         """Reorder buffered internal state (for incremental generation)."""
#         input_buffer = self._get_input_buffer(incremental_state)
#         if input_buffer is not None:
#             for k in input_buffer.keys():
#                 input_buffer_k = input_buffer[k]
#                 if input_buffer_k is not None:
#                     if self.encoder_decoder_attention and input_buffer_k.size(
#                         0
#                     ) == new_order.size(0):
#                         break
#                     input_buffer[k] = input_buffer_k.index_select(0, new_order)
#             incremental_state = self._set_input_buffer(incremental_state, input_buffer)
#         return incremental_state

#     def _get_input_buffer(
#         self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
#     ) -> Dict[str, Optional[Tensor]]:
#         result = self.get_incremental_state(incremental_state, "attn_state")
#         if result is not None:
#             return result
#         else:
#             empty_result: Dict[str, Optional[Tensor]] = {}
#             return empty_result

#     def _set_input_buffer(
#         self,
#         incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
#         buffer: Dict[str, Optional[Tensor]],
#     ):
#         return self.set_incremental_state(incremental_state, "attn_state", buffer)

#     def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
#         return attn_weights

#     def upgrade_state_dict_named(self, state_dict, name):
#         prefix = name + "." if name != "" else ""
#         items_to_add = {}
#         keys_to_remove = []
#         for k in state_dict.keys():
#             if k.endswith(prefix + "in_proj_weight"):
#                 # in_proj_weight used to be q + k + v with same dimensions
#                 dim = int(state_dict[k].shape[0] / 3)
#                 items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
#                 items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
#                 items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

#                 keys_to_remove.append(k)

#                 k_bias = prefix + "in_proj_bias"
#                 if k_bias in state_dict.keys():
#                     dim = int(state_dict[k].shape[0] / 3)
#                     items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
#                     items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
#                         dim : 2 * dim
#                     ]
#                     items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

#                     keys_to_remove.append(prefix + "in_proj_bias")

#         for k in keys_to_remove:
#             del state_dict[k]

#         for key, value in items_to_add.items():
#             state_dict[key] = value


# @with_incremental_state
# class MultiheadAttention_(nn.Module):
#     """Multi-headed attention.

#     See "Attention Is All You Need" for more details.
#     """

#     def __init__(
#         self,
#         embed_dim,
#         num_heads,
#         kdim=None,
#         vdim=None,
#         dropout=0.0,
#         bias=True,
#         add_bias_kv=False,
#         add_zero_attn=False,
#         self_attention=False,
#         encoder_decoder_attention=False,
#         q_noise=0.0,
#         qn_block_size=8,
#         # add
#         index=0,
#         # base
#         is_base=True,
#         is_ada_q=False,
#         is_ada_k=False,
#         lambda_=0.99,
#         up_fq=16,
#         dropout_before=False,
#         use_q=False,
#         use_k=False,
#         # add
#         low_d=False,
#         has_out=False,
#         do_scale=True,
#         # change
#         norm_taylor=True,
#         use_relu=False,
#         use_elu=False,
#         use_leak=False,
#         use_square=False,
#         use_sigmoid=False,
#         use_linear=False,
#         use_softplus=False,
#         use_basic=True,
#         use_abs=False,
#         # 因子
#         alpha_beta=False,
#         max_l=1024,
#     ):
#         # add
#         self.index = index

#         super().__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

#         self.num_heads = num_heads
#         self.dropout_module = FairseqDropout(
#             dropout, module_name=self.__class__.__name__
#         )

#         self.head_dim = embed_dim // num_heads
#         assert (
#             self.head_dim * num_heads == self.embed_dim
#         ), "embed_dim must be divisible by num_heads"
#         self.scaling = self.head_dim ** -0.5

#         self.self_attention = self_attention
#         self.encoder_decoder_attention = encoder_decoder_attention

#         assert not self.self_attention or self.qkv_same_dim, (
#             "Self-attention requires query, key and " "value to be of the same size"
#         )

#         self.k_proj = quant_noise(
#             nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
#         )
#         self.v_proj = quant_noise(
#             nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
#         )
#         self.q_proj = quant_noise(
#             nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
#         )

#         # add begin
#         self.is_ada_q = is_ada_q
#         self.is_ada_k = is_ada_k
#         self.lambda_ = lambda_
#         self.scaling = self.head_dim ** -0.5
#         self.up_fq = up_fq
#         self.cnt = 0
#         self.dropout_before = dropout_before
#         self.has_out = has_out
#         self.use_q = use_q
#         self.use_k = use_k
#         self.norm_taylor = norm_taylor
#         self.use_relu = use_relu
#         self.use_elu = use_elu
#         self.use_leak = use_leak
#         self.use_square = use_square
#         self.use_sigmoid = use_sigmoid
#         self.use_linear = use_linear
#         self.use_softplus = use_softplus
#         self.use_basic = use_basic
#         self.use_abs = use_abs
#         self.do_scale = do_scale
#         # 1 * E
#         if self.is_ada_q:
#             self.qsigma2 = Parameter(torch.ones(1, self.embed_dim), requires_grad=False)
#         if self.is_ada_k:
#             self.ksigma2 = Parameter(torch.ones(1, self.embed_dim), requires_grad=False)

#         if self.has_out:
#             self.out_proj = quant_noise(
#                 nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
#             )

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
#             self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn


#         self.alpha_beta = alpha_beta
#         self.max_l = max_l
#         # if self.alpha_beta:
#         #     self.row1, self.col1, self.row2, self.col2 = self.get_alpha_beta(self.max_l)
#         if self.alpha_beta:
#             self.weight_index = self.get_alpha_beta(self.max_l)


#         self.reset_parameters()

#         self.onnx_trace = False

#         print(embed_dim)
#         print(f"do scale {self.do_scale}")
#         print(f"taylor {self.norm_taylor}")
#         print(f"use relu {self.use_relu}")
#         print(f"use elu {self.use_elu}")
#         print(f"use leak {self.use_leak}")
#         print(f"use square {self.use_square}")
#         print(f"use sigmoid {self.use_sigmoid}")
#         print(f"use linear {self.use_linear}")
#         print(f"use softplus {self.use_softplus}")
#         print(f"use basic {self.use_basic}")
#         print(f"use abs {self.use_abs}")

#         print(f"args.is_ada_q {self.is_ada_q}")
#         print(f"args.is_ada_k {self.is_ada_k}")
#         print(f"args.do_scale {self.do_scale}")
#         print(f"args.norm_taylor {self.norm_taylor}")
#         print(f"args.lambda_ {self.lambda_}")
#         print(f"args.use_q {self.use_q}")
#         print(f"args.use_k {self.use_k}")
#         print(f"args.has_out {self.has_out}")
#         print(f"self.alpha_beta {self.alpha_beta}")
#         print(f"self.max_l {self.max_l}")

#     def prepare_for_onnx_export_(self):
#         self.onnx_trace = True

#     def reset_parameters(self):
#         if self.qkv_same_dim:
#             # Empirically observed the convergence to be much better with
#             # the scaled initialization
#             nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
#             nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
#             nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
#         else:
#             nn.init.xavier_uniform_(self.k_proj.weight)
#             nn.init.xavier_uniform_(self.v_proj.weight)
#             nn.init.xavier_uniform_(self.q_proj.weight)

#         # add begin
#         if self.has_out:
#             nn.init.xavier_uniform_(self.out_proj.weight)
#             if self.out_proj.bias is not None:
#                 nn.init.constant_(self.out_proj.bias, 0.0)
#         # add end
#         if self.bias_k is not None:
#             nn.init.xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             nn.init.xavier_normal_(self.bias_v)

#     def get_alpha_beta(self, src_len, tgt_len):
#         a = np.pi / 2
#         row_index = torch.arange(1, src_len + 1)
#         col_index = torch.arange(1, tgt_len + 1)
#         row1 = torch.cos(a * row_index / src_len).reshape(1, -1, 1)
#         col1 = torch.cos(a * col_index / tgt_len).reshape(1, -1, 1)

#         row2 = torch.sin(a * row_index / src_len).reshape(1, -1, 1)
#         col2 = torch.sin(a * col_index / tgt_len).reshape(1, -1, 1)

#         # mask = row1 * col1 + row2 * col2

#         return nn.Parameter(row1, requires_grad=False), nn.Parameter(col1, requires_grad=False), nn.Parameter(row2, requires_grad=False), nn.Parameter(col2, requires_grad=False)

#     def get_alpha_beta(self, l):
#         a = np.pi / 2
#         index = a * torch.arange(1, l + 1).reshape(1, -1, 1) / l

#         # mask = row1 * col1 + row2 * col2

#         return nn.Parameter(index, requires_grad=False)


#     def forward(
#         self,
#         query,
#         key: Optional[Tensor],
#         value: Optional[Tensor],
#         key_padding_mask: Optional[Tensor] = None,
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         need_weights: bool = True,
#         static_kv: bool = False,
#         attn_mask: Optional[Tensor] = None,
#         before_softmax: bool = False,
#         need_head_weights: bool = False,
#     ) -> Tuple[Tensor, Optional[Tensor]]:
#         """Input shape: Time x Batch x Channel

#         Args:
#             key_padding_mask (ByteTensor, optional): mask to exclude
#                 keys that are pads, of shape `(batch, src_len)`, where
#                 padding elements are indicated by 1s.
#             need_weights (bool, optional): return the attention weights,
#                 averaged over heads (default: False).
#             attn_mask (ByteTensor, optional): typically used to
#                 implement causal attention, where the mask prevents the
#                 attention from looking forward in time (default: None).
#             before_softmax (bool, optional): return the raw attention
#                 weights and values before the attention softmax.
#             need_head_weights (bool, optional): return the attention
#                 weights for each head. Implies *need_weights*. Default:
#                 return the average attention weights over all heads.
#         """
#         if need_head_weights:
#             need_weights = True

#         assert key is not None and value is not None

#         '''
#         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         '''
#         num_heads = self.num_heads
#         tgt_len, bsz, embed_dim = query.size()
#         src_len = key.size(0)
#         head_dim = embed_dim // num_heads

#         tgt_len, bsz, embed_dim = query.size()
#         # L, N, E
#         q = self.q_proj(query)
#         # S, N, E
#         k = self.k_proj(key)
#         # S, N, E
#         v = self.v_proj(value)

#         if self.training:
#             # L * N, E -> (1, E)
#             # sigma2 = torch.nn.Parameter(torch.var(q.view(-1, self.embed_dim), dim=0, keepdim=True), requires_grad=False)
#             self.cnt += 1
#             if self.cnt % self.up_fq == 0:
#                 # print(self.cnt, self.up_fq)
#                 with torch.no_grad():
#                     if self.is_ada_q:
#                         # print("q")
#                         qsigma2 = torch.var(q.view(-1, self.embed_dim), dim=0, keepdim=True)
#                         self.qsigma2 *= self.lambda_
#                         self.qsigma2 += (1 - self.lambda_) * qsigma2

#                     if self.is_ada_k:
#                         # print("k")
#                         ksigma2 = torch.var(k.view(-1, self.embed_dim), dim=0, keepdim=True)
#                         self.ksigma2 *= self.lambda_
#                         self.ksigma2 += (1 - self.lambda_) * ksigma2

#         if self.do_scale:
#             q = q * self.scaling

#         if self.use_q:
#             # print("q1")
#             q /= torch.sqrt(self.qsigma2)
#         if self.use_k:
#             # print("k1")
#             k /= torch.sqrt(self.ksigma2)

#         if self.use_relu:
#             q = F.relu(q)
#             k = F.relu(k)
#         elif self.use_elu:
#             q = F.elu(q)
#             k = F.elu(k)
#         elif self.use_leak:
#             q = F.leaky_relu(q)
#             k = F.leaky_relu(k)
#         elif self.use_square:
#             q = torch.square(q)
#             k = torch.square(k)
#         elif self.use_sigmoid:
#             q = F.sigmoid(q)
#             k = F.sigmoid(k)

#         # N * h, L, d
#         q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#         # N * h, S, d
#         k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#         v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

#         if self.norm_taylor:
#             # |q| ^ (-1) q
#             q = F.normalize(q, p=2, dim=-1)
#             # |k| ^ (-1) k
#             k = F.normalize(k, p=2, dim=-1)
#             # N * h, L, S
#             # 1 + |q| ^ (-1) * q * k ^ T * |k| ^ (-1) 
#             attn_output_weights = 1 + torch.bmm(q, k.transpose(1, 2))
#             if attn_mask is not None:
#                 attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)

#             attn_output_weights = F.normalize(attn_output_weights, p=1, dim=-1)
#         elif self.use_sigmoid:
#             # N * h, L, S
#             # sum_{i=1}^{embed_dim}, 每行src_len个
#             attn_output_weights = torch.bmm(q, k.transpose(1, 2)) / src_len / embed_dim
#             if attn_mask is not None:
#                 attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)
#         elif self.use_linear:
#             if self.alpha_beta:
#                 # with torch.no_grad():
#                 #     a = np.pi / 2
#                 #     m = max(src_len, tgt_len)
#                 #     index = (a * torch.arange(1, m + 1) / self.max_l).to(q)
#                 #     row_index = torch.arange(1, src_len + 1)
#                 #     col_index = torch.arange(1, tgt_len + 1)
#                 #     row1 = torch.cos(index[:tgt_len]).reshape(1, -1, 1)
#                 #     col1 = torch.cos(index[:src_len]).reshape(1, -1, 1)

#                 #     row2 = torch.sin(index[:tgt_len]).reshape(1, -1, 1)
#                 #     col2 = torch.sin(index[:tgt_len]).reshape(1, -1, 1)
#                 # q1 = q * row1[:, :tgt_len, :]
#                 # k1 = k * col1[:, :src_len, :]
#                 # q2 = q * row2[:, :tgt_len, :]
#                 # k2 = k * col2[:, :src_len, :]
#                 # attn_output_weights = torch.bmm(q1, k1.transpose(1, 2)) + torch.bmm(q2, k2.transpose(1, 2))
                
#                 # q1 = q * self.row1[:, :tgt_len, :]
#                 # k1 = k * self.col1[:, :src_len, :]
#                 # q2 = q * self.row2[:, :tgt_len, :]
#                 # k2 = k * self.col2[:, :src_len, :]
#                 # attn_output_weights = torch.bmm(q1, k1.transpose(1, 2)) + torch.bmm(q2, k2.transpose(1, 2))

#                 row1 = torch.cos(self.weight_index[:, :tgt_len, :])
#                 col1 = torch.cos(self.weight_index[:, :src_len, :])
#                 row2 = torch.sin(self.weight_index[:, :tgt_len, :])
#                 col2 = torch.sin(self.weight_index[:, :src_len, :])

#                 q1 = q * row1
#                 k1 = k * col1
#                 q2 = q * row2
#                 k2 = k * col2
#                 attn_output_weights = torch.bmm(q1, k1.transpose(1, 2)) + torch.bmm(q2, k2.transpose(1, 2))
            
#             else:
#                 # N * h, L, S
#                 attn_output_weights = torch.bmm(q, k.transpose(1, 2))

#             if attn_mask is not None:
#                 attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)

#             attn_output_weights = F.normalize(attn_output_weights, p=1, dim=-1)
#         elif self.use_basic:

#             # attn_mask
#             if attn_mask is not None:
#                 if attn_mask.dim() == 2:
#                     attn_mask = attn_mask.unsqueeze(0)
#                     if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
#                         raise RuntimeError('The size of the 2D attn_mask is not correct.')
#                 elif attn_mask.dim() == 3:
#                     if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
#                         raise RuntimeError('The size of the 3D attn_mask is not correct.')
#                 else:
#                     raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
#             # attn_mask's dim is 3 now.

#             if attn_mask is not None:
#                 attn_output_weights += attn_mask
        
#             # N * h, L, S
#             attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        
#         # dropout
#         attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
#         # N * h, L, d
#         attn_output = torch.bmm(attn_output_weights, v)
#         # L, N, E
#         attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#         # add
#         if self.has_out:
#             attn_output = self.out_proj(attn_output)

#         if need_weights:
#             attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#             return attn_output, attn_output_weights
#         else:
#             return attn_output, None

#     @staticmethod
#     def _append_prev_key_padding_mask(
#         key_padding_mask: Optional[Tensor],
#         prev_key_padding_mask: Optional[Tensor],
#         batch_size: int,
#         src_len: int,
#         static_kv: bool,
#     ) -> Optional[Tensor]:
#         # saved key padding masks have shape (bsz, seq_len)
#         if prev_key_padding_mask is not None and static_kv:
#             new_key_padding_mask = prev_key_padding_mask
#         elif prev_key_padding_mask is not None and key_padding_mask is not None:
#             new_key_padding_mask = torch.cat(
#                 [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
#             )
#         # During incremental decoding, as the padding token enters and
#         # leaves the frame, there will be a time when prev or current
#         # is None
#         elif prev_key_padding_mask is not None:
#             filler = torch.zeros(
#                 (batch_size, src_len - prev_key_padding_mask.size(1)),
#                 device=prev_key_padding_mask.device,
#             )
#             new_key_padding_mask = torch.cat(
#                 [prev_key_padding_mask.float(), filler.float()], dim=1
#             )
#         elif key_padding_mask is not None:
#             filler = torch.zeros(
#                 (batch_size, src_len - key_padding_mask.size(1)),
#                 device=key_padding_mask.device,
#             )
#             new_key_padding_mask = torch.cat(
#                 [filler.float(), key_padding_mask.float()], dim=1
#             )
#         else:
#             new_key_padding_mask = prev_key_padding_mask
#         return new_key_padding_mask

#     @torch.jit.export
#     def reorder_incremental_state(
#         self,
#         incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
#         new_order: Tensor,
#     ):
#         """Reorder buffered internal state (for incremental generation)."""
#         input_buffer = self._get_input_buffer(incremental_state)
#         if input_buffer is not None:
#             for k in input_buffer.keys():
#                 input_buffer_k = input_buffer[k]
#                 if input_buffer_k is not None:
#                     if self.encoder_decoder_attention and input_buffer_k.size(
#                         0
#                     ) == new_order.size(0):
#                         break
#                     input_buffer[k] = input_buffer_k.index_select(0, new_order)
#             incremental_state = self._set_input_buffer(incremental_state, input_buffer)
#         return incremental_state

#     def _get_input_buffer(
#         self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
#     ) -> Dict[str, Optional[Tensor]]:
#         result = self.get_incremental_state(incremental_state, "attn_state")
#         if result is not None:
#             return result
#         else:
#             empty_result: Dict[str, Optional[Tensor]] = {}
#             return empty_result

#     def _set_input_buffer(
#         self,
#         incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
#         buffer: Dict[str, Optional[Tensor]],
#     ):
#         return self.set_incremental_state(incremental_state, "attn_state", buffer)

#     def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
#         return attn_weights

#     def upgrade_state_dict_named(self, state_dict, name):
#         prefix = name + "." if name != "" else ""
#         items_to_add = {}
#         keys_to_remove = []
#         for k in state_dict.keys():
#             if k.endswith(prefix + "in_proj_weight"):
#                 # in_proj_weight used to be q + k + v with same dimensions
#                 dim = int(state_dict[k].shape[0] / 3)
#                 items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
#                 items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
#                 items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

#                 keys_to_remove.append(k)

#                 k_bias = prefix + "in_proj_bias"
#                 if k_bias in state_dict.keys():
#                     dim = int(state_dict[k].shape[0] / 3)
#                     items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
#                     items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
#                         dim : 2 * dim
#                     ]
#                     items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

#                     keys_to_remove.append(prefix + "in_proj_bias")

#         for k in keys_to_remove:
#             del state_dict[k]

#         for key, value in items_to_add.items():
#             state_dict[key] = value