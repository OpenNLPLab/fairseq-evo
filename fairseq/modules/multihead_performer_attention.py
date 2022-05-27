# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
from einops import rearrange
from functools import partial

# add
# from performer_pytorch import Attention, SelfAttention

# @with_incremental_state
# class MultiheadPerformerAttention(nn.Module):
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
#         causal=True,
#         local_heads=0,
#         local_window_size=256,
#         nb_features=None,
#         feature_redraw_interval=1000,
#         generalized_attention=False,
#         kernel_fn=nn.ReLU(),
#         no_projection=False,
#         qkv_bias=False,
#         attn_out_bias=True
#     ):
#         '''
#         dim = embed_dim
#         heads = num_heads
#         dim_head = head_dim
#         修改causal默认为true
#         '''
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

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
#             self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         self.onnx_trace = False

#         # add performer
#         dim = self.embed_dim
#         heads = self.num_heads
#         dim_head = self.head_dim

#         self.attention = SelfAttention(dim, causal, heads, dim_head, local_heads,
#                                        local_window_size, nb_features, feature_redraw_interval,
#                                        generalized_attention, kernel_fn, dropout, no_projection,
#                                        qkv_bias, attn_out_bias)

#     def prepare_for_onnx_export_(self):
#         self.onnx_trace = True

#     def reset_parameters(self):
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
#         attn_weights = None
#         # todo mask的对应关系存疑
#         # random_matrices = torch.randn(self.num_heads, self.proj_dim, self.head_dim)
#         # print("mask")
#         # if (key_padding_mask != None):
#         #     print(key_padding_mask.dtype)
#         #     print(key_padding_mask)
#         # print("context_mask")
#         # if (attn_mask != None):
#         #     print(attn_mask.dtype)
#         #     print(attn_mask)
#         # attn = self.attention(x=query, mask=key_padding_mask, context_mask=attn_mask)
        
#         # print("query")
#         # print(query.dtype)
#         with torch.autograd.profiler.record_function("multihead-performer-attention"):
#             attn = self.attention(x=query)

#         return attn, attn_weights

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

def orthogonal_matrix_chunk(cols, device = None, dtype=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t().to(dtype)

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, seed=0, device=None, dtype=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []
    cur_seed = seed

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device, dtype=dtype)
        block_list.append(q)
        cur_seed = cur_seed + 1

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device, dtype=dtype)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    multiplier = torch.randn((nb_rows, nb_columns), device=device, dtype=dtype).norm(dim=1)

    return torch.diag(multiplier) @ final_matrix

def create_proj_matrix(num_heads, proj_dim, input_dim, ortho=False, seed=0, device=None, dtype=None):
    if ortho:
        return torch.stack(
            [
                gaussian_orthogonal_random_matrix(proj_dim, input_dim, seed=seed + h * 1000, device=device, dtype=dtype)
                for h in range(num_heads)
            ], dim=0)
    else:
        return torch.randn(num_heads, proj_dim, input_dim, device=device, dtype=dtype)

def favorp_projection(
        data: torch.Tensor,
        projection_matrix: torch.Tensor,
        is_query: bool,
        eps: float=0.0001):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      batch_dims_t: tuple of batch dimensions
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = (data.shape[-1] ** -0.25)
    ratio = (projection_matrix.shape[1] ** -0.5)
    data_dash = torch.einsum('bh...d,hjd->bh...j', 
                            (data_normalizer * data),
                            projection_matrix)
    diag_data = torch.sum(data ** 2, dim=-1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = diag_data.unsqueeze(-1)
    
    if is_query:
        data_dash_log = data_dash - diag_data
        stabilizer = torch.amax(data_dash, dim=-1, keepdim=True).detach()
        data_dash = ratio * torch.exp(data_dash_log - stabilizer) + eps
    else:
        data_dash_log = data_dash - diag_data
        stabilizer = torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()
        data_dash = ratio * torch.exp(data_dash_log - stabilizer) + eps
    return data_dash

@with_incremental_state
class MultiheadPerformerAttention(nn.Module):
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
        approx_attn_dim=64,
        causal=False,
    ):
        '''
        dim = embed_dim
        heads = num_heads
        dim_head = head_dim
        修改causal默认为true
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

		# q, k, v projection
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.approx_attn_dim = approx_attn_dim
        self.causal = causal
        self.use_random_proj = True
        self.register_buffer('eval_proj', create_proj_matrix(
            self.num_heads, 
            self.approx_attn_dim, 
            self.head_dim, 
            ortho=True
            )
        )

        print(f"self.approx_attn_dim {self.approx_attn_dim}")
        print(f"self.causal {self.causal}")

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
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
        # attn_weights = None
        # l, b, e -> b l e
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # (B, H, L, D)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        # print("========================")
        # print(q.shape)

        if self.training:
            projection_matrix = create_proj_matrix(
                self.num_heads, self.approx_attn_dim, self.head_dim, ortho=False, device=q.device, dtype=q.dtype)
        else:
            projection_matrix = self.eval_proj
        q_prime, k_prime = self.q_k_projection(q, k, projection_matrix)
        # print(q_prime.shape)

        eps = 1e-2
        if self.causal:
            # b, h, n, m
            weights = torch.einsum('...nd,...md->...nm', q_prime, k_prime)
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
            # (N * h, L, S) -> (N * h, L, S)
            denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
            # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
            attn_weights = weights / denom
            # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
            output = torch.einsum('...nm,...md->...nd', attn_weights, v)
        else:
            kv = torch.einsum('...nm,...nd->...md', k_prime, v)
            qkv = torch.einsum('...nm,...md->...nd', q_prime, kv)
            normalizer = torch.einsum('...nm,...m->...n', q_prime, k_prime.sum(dim=-2))
            output = qkv / normalizer.unsqueeze(-1).clamp(min=eps)
            # print(output.shape)

        # attn_output = output.contiguous().view(B, N, self.num_heads, self.head_dim)
        attn_output = rearrange(output, 'b h n d -> b n (h d)', h=self.num_heads)
        attn_output = self.out_proj(attn_output).transpose(0, 1)
        return attn_output, None

    def q_k_projection(self, q, k, random_proj=None):
        assert random_proj is not None
        feature_proj = partial(favorp_projection, projection_matrix = random_proj)

        q = feature_proj(q, is_query = True)
        k = feature_proj(k, is_query = False)
        return q, k

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