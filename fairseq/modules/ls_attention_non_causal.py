# Copyright (c) 2021 NVIDIA CORPORATION. Licensed under the MIT license.
# Written by Chen Zhu during an internship at NVIDIA, zhuchen.eric@gmail.com
# https://github.com/NVIDIA/transformer-ls/blob/master/imagenet/models/layers/transformer_ls.py
# https://github.com/NVIDIA/transformer-ls/blob/master/lra/attention_transformer_ls.py

from torch import nn
import torch
# from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from torch import Tensor, nn
import numpy as np


# class LSAttentionNonCausal(nn.Module):
#     """Implementation for long-short term attention.
#     Flexible options for using window attention, global token and dynamic projection.
#     Args:
#         dim: input and output feature dimension.
#         num_heads: number of attention heads.
#         qkv_bias: whether to use bias for the projection of query, key and values.
#         qk_scale: scale factor on query and key for numerical stability.
#                   By default, set to square root of head dimensions.
#         attn_drop: dropout probability for attention matrix.
#         proj_drop: dropout probability for the final output.
#         rpe: whether to use relative position encoding.
#         nglo: number of global tokens (e.g., CLS).
#     """
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
#                  attn_drop=0., proj_drop=0., rpe=False, nglo=1,
#                  dp_rank=2, w=2):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.nglo = nglo

#         # Equals to segment size (w) in the paper.
#         self.window_size = w
#         # Equals to r in the paper.
#         self.dp_rank = dp_rank

#         if self.dp_rank > 0:
#             self.to_dynamic_projection = nn.Linear(dim, dp_rank * num_heads)
#         # The LN of DualLN corresponding to dynamic projection
#         self.dual_ln_dp = nn.LayerNorm(dim)
#         # The LN of DualLN corresponding to all the tokens
#         self.dual_ln_full = nn.LayerNorm(dim)

#         # Adapted from ViL: https://github.com/microsoft/vision-longformer/blob/main/src/models/layers/longformer2d.py#L55-L100
#         # We only add RPE to window attention.
#         # Unnecessary to add bias for global tokens, since DualLN already adds biases.
#         self.rpe = rpe
#         if rpe:
#             # handle the boarder conditions...
#             w_pad = int(w*0.5)
#             self.local_relative_position_bias_table = nn.Parameter(
#                 torch.zeros(2 * (w + w_pad - 1) * (2 * w_pad + w + 1) + 1, num_heads))
#             trunc_normal_(self.local_relative_position_bias_table, std=.02)

#             # get pair-wise relative position index
#             coords_h = torch.arange(-w_pad, w_pad + w)
#             coords_w = torch.arange(-w_pad, w_pad + w)
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, 2w, 2w
#             coords = coords.view(2, (w + w_pad * 2)**2).transpose(0, 1).unsqueeze(0) # 1, 4w**2, 2
#             q_coords_hw = torch.arange(0, w)
#             q_coords = torch.stack(torch.meshgrid([q_coords_hw, q_coords_hw])) # 2, w, w
#             q_coords = q_coords.view(2, w**2).transpose(0, 1).unsqueeze(1) # w**2, 1, 2
#             relative_coords = q_coords - coords
#             relative_coords += w_pad + w - 1  # shift to start from 0
#             relative_coords[:, :, 0] *= 2 * w_pad + w
#             relative_position_index = relative_coords.sum(-1)  # w^2, 4w^2
#             self.register_buffer("relative_position_index", relative_position_index)

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
#         eps=1e-6,
#         nx=None, 
#         ny=None,
#     ):
#         # tgt_len, bsz, embed_dim
#         print(query.shape)
#         x = query.transpose(0, 1)
#         print(x.shape)
#         B, N, C = x.shape
#         N_feat = N - self.nglo
#         self.img_size = nx
#         qkv = self.qkv(x)
#         # query, key, value
#         q, k, v = qkv.chunk(3, dim=2)
#         q = q.mul(self.scale)

#         # Layer norm on the projected keys and values
#         k = self.dual_ln_full(k)
#         v = self.dual_ln_full(v)

#         # output size: bsz x n_heads x seqlen x d
#         if self.nglo > 0:
#             q_cls, q = q[:, :self.nglo], q[:, self.nglo:]
#             k_cls, k = k[:, :self.nglo], k[:, self.nglo:]
#             v_cls, v = v[:, :self.nglo], v[:, self.nglo:]

#             q_cls = q_cls.reshape(B, self.nglo, self.num_heads, C // self.num_heads).transpose(1, 2)
#             k_cls = k_cls.reshape(B, self.nglo, self.num_heads, C // self.num_heads).transpose(1, 2)
#             v_cls = v_cls.reshape(B, self.nglo, self.num_heads, C // self.num_heads).transpose(1, 2)

#         q = q.reshape(B, N_feat, self.num_heads, C//self.num_heads).transpose(1, 2)
#         k = k.reshape(B, N_feat, self.num_heads, C//self.num_heads).transpose(1, 2)
#         v = v.reshape(B, N_feat, self.num_heads, C//self.num_heads).transpose(1, 2)

#         print(q.shape)

#         # Long-range Attention (Dynamic Projection)
#         if self.dp_rank > 0:
#             # b x h x r x (l w)
#             # Compute the projection matrix (P_i in the paper)
#             c_scores = self.to_dynamic_projection(x[:, self.nglo:]).transpose(1, 2).contiguous().view(
#                 B, self.num_heads, self.dp_rank, -1)
#             c_scores = c_scores.softmax(dim=-1, dtype=torch.float32).to(x)
#             # b x h x r x d
#             k_lms = c_scores.matmul(k)
#             k_lms = k_lms.transpose(1, 2).contiguous().view(B, self.dp_rank, -1)
#             k_lms = self.dual_ln_dp(k_lms).view(B, self.dp_rank, self.num_heads, -1).contiguous().permute(0, 2, 3, 1)
#             # b x h x (lw) x r
#             dots_all = q.matmul(k_lms)
#             print(q.shape, k_lms.shape, dots_all.shape)

#             if self.window_size > 0:
#                 # Switch the order of dimensions if using window attention.
#                 dots_all = self.group_dots(dots_all)
#         else:
#             dots_all = None

#         # Short-term Attention (Window Attention)
#         # In our window attention, each token attends to at most (4w^2) tokens.
#         if self.window_size > 0:
#             dots_win = self.compute_window_scores(q, k)
#             w2 = int(self.window_size*self.window_size)

#             if self.rpe:
#                 w_pad = int(0.5 * self.window_size)
#                 local_relative_position_bias = self.local_relative_position_bias_table[
#                     self.relative_position_index.view(-1)].view(1, w2, (w_pad*2 + self.window_size)**2, -1)  # w^2, kv_nums,H
#                 local_relative_position_bias = local_relative_position_bias.permute(
#                     0, 3, 1, 2).expand(B, -1, -1, -1).unsqueeze(2).unsqueeze(2)

#                 dots_win += local_relative_position_bias
#             if dots_all is None:
#                 dots_all = dots_win
#             else:
#                 dots_all = torch.cat([dots_all, dots_win], dim=-1)

#         # Global token.
#         if self.nglo > 0:
#             # and compute the scores of queries on CLS
#             dots_q_cls = q.matmul(k_cls.transpose(-1, -2))

#             if self.window_size > 0:
#                 dots_q_cls = self.group_dots(dots_q_cls)
#             dots_all = torch.cat([dots_all, dots_q_cls], dim=-1)

#         attn = dots_all.softmax(dim=-1, dtype=torch.float32).to(x)
#         attn = self.attn_drop(attn)
#         out = 0
#         if self.window_size > 0:
#             offset = max(0, self.dp_rank)
#             kv_group_size = self.window_size
#             total_win_size = max(1, self.window_size // 2) * 2 + kv_group_size
#             attn_win = attn[:, :, :, :, :, offset:offset + total_win_size ** 2]
#             out += self.compute_window_pv(attn_win, v)
#             attn = self.ungroup_dots(attn)

#         # attn will be b x h x lw x n_k from now on
#         if self.dp_rank > 0:
#             attn_lm = attn[:, :, :, :self.dp_rank]
#             v_lms = c_scores.matmul(v.float()).to(v).transpose(1, 2).contiguous().view(B, self.dp_rank, -1)
#             v_lms = self.dual_ln_dp(v_lms).view(B, self.dp_rank, self.num_heads, -1).contiguous().transpose(1, 2)

#             out += attn_lm.matmul(v_lms)

#         if self.nglo > 0:
#             attn_cls = attn[:, :, :, -self.nglo:]
#             out += attn_cls.mul(v_cls)

#             # b x h x 1 x lw
#             cls_inner = q_cls.matmul(k_cls.transpose(-1, -2))
#             cls_dots = q_cls.matmul(out.transpose(-1, -2))
#             cls_dots = torch.cat([cls_inner, cls_dots], dim=-1)

#             cls_dots = cls_dots.softmax(dim=-1, dtype=torch.float32).to(x)
#             cls_next = cls_dots[:, :, :, self.nglo:].matmul(out) # the post_cls variant
#             cls_next += cls_dots[:, :, :, :self.nglo].matmul(v_cls)

#             out = torch.cat([cls_next, out], dim=2)
#         out = out.transpose(1, 2).contiguous().view(B, N, -1)

#         # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         out = self.proj(out)
#         out = self.proj_drop(out)
#         return out

#     def compute_window_scores(self, q, k):
#         """Compute the inner products for the window attention.
#         Frist, divide the query into non-overlapping windows.
#         Then, use torch.as_trided (implemented in self.get_overlapping_tiles) to create a view of the keys
#         that corresponds to the windows with at most 2x memory overhead.
#         Finally, compute the inner product.
#         """
#         # q: b h (l w) d
#         b, h, _, d = q.shape
#         side_size = max(self.window_size//2, 1)
#         # q_group_size: segment size
#         kv_width = 2 * side_size + self.window_size # assuming q_stride=1
#         q_n_group = self.img_size // self.window_size
#         q_tiles = q.reshape(b, h, q_n_group, self.window_size, q_n_group, self.window_size, d).permute(
#             0, 1, 2, 4, 3, 5, 6)
#         # q_tiles: b x h x n_group x n_group x w^2 x d
#         q_tiles = q_tiles.contiguous().view(b, h, q_n_group, q_n_group, -1, d)

#         # k_tiles: b x h x n_group x n_group x 9w^2 x d
#         k_tiles = self.get_overlapping_tiles(k).contiguous().view(b, h, q_n_group, q_n_group, -1, d)
#         # dot_tiles: b x h x n_group x n_group x w^2 x 9w^2
#         dot_tiles = q_tiles.matmul(k_tiles.transpose(-1, -2))

#         # fill "-inf" into the zero-padding parts
#         dot_tiles = dot_tiles.view(b, h, q_n_group, q_n_group, -1, kv_width, kv_width)

#         dot_tiles[:, :, 0, :, :, :side_size].fill_(float('-inf'))
#         dot_tiles[:, :, -1, :, :, -side_size:].fill_(float('-inf'))
#         dot_tiles[:, :, :, 0, :, :, :side_size].fill_(float('-inf'))
#         dot_tiles[:, :, :, -1, :, :, -side_size:].fill_(float('-inf'))

#         dot_tiles = dot_tiles.view(b, h, q_n_group, q_n_group, -1, kv_width ** 2)
#         return dot_tiles

#     def get_overlapping_tiles(self, x):
#         """Get overlapping tiles in the 2D spatial domain, ensuring each query computes correlation with all neighbors
#         """
#         # x: b h (l w) d
#         b, h, _, d = x.shape
#         side_size = max(self.window_size // 2, 1)
#         total_size = 2 * side_size + self.window_size
#         kv_group_size = self.window_size
#         kv_width = self.img_size

#         x = x.view(b, h, kv_width, kv_width, d)
#         x = F.pad(x, [0, 0, side_size, side_size, side_size, side_size], value=0)

#         out_shape = [b, h, kv_width // kv_group_size, kv_width // kv_group_size,
#                      total_size, total_size, d]
#         in_stride = x.stride()
#         out_stride = [in_stride[0], in_stride[1], in_stride[2] * kv_group_size, in_stride[3] * kv_group_size,
#                       in_stride[2], in_stride[3], in_stride[4]]

#         # note we ignored the boundary here
#         return x.as_strided(size=out_shape, stride=out_stride)

#     def compute_window_pv(self, attn, v):
#         """Compute the inner product of attention matrix and the values for the window attention.
#         """
#         b, h, n_group, _, w2, n_k = attn.shape
#         d = v.shape[-1]
#         v_tiles = self.get_overlapping_tiles(v).contiguous().view(b, h, n_group, n_group, -1, d)

#         # b x h x n_group x n_group x w^2 x d
#         pv = attn.matmul(v_tiles)
#         # return: b x h x (lw) x d
#         ret = self.ungroup_dots(pv)

#         return ret

#     def group_dots(self, dots):
#         b, h = dots.shape[:2]
#         print(dots.shape)
#         n_group = self.img_size // self.window_size
#         dots = dots.reshape(b, h, n_group, self.window_size, n_group, self.window_size,
#                             -1).permute(0, 1, 2, 4, 3, 5, 6)
#         dots = dots.contiguous().view(b, h, n_group, n_group, self.window_size * self.window_size, -1)
#         return dots

#     def ungroup_dots(self, dots):
#         b, h, n_group, _, _, n_keys = dots.shape
#         dots = dots.reshape(b, h, n_group, n_group, self.window_size, self.window_size,
#                             -1).permute(0, 1, 2, 4, 3, 5, 6)
#         dots = dots.contiguous().view(b, h, -1, n_keys)
#         return 

class LSAttentionNonCausal(nn.Module):
    """The long-short term attention for bidirectional language modelling
    """

    def __init__(self, 
                 dim, num_heads, max_seq_len, 
                 dropout, num_landmarks=32, window_size=8):
        super().__init__()

        self.cls_from_seq = False

        self.num_head = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        self.seq_len = max_seq_len
        self.dim = dim
        self.fp32 = True

        self.drop_attn = torch.nn.Dropout(dropout)

        self.window_size = window_size

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_o = nn.Linear(self.dim, self.num_head * self.head_dim)


        self.dual_ln_s = nn.LayerNorm(self.num_head * self.head_dim)
        self.dual_ln_l = nn.LayerNorm(self.num_head * self.head_dim)

        self.dconv_fc = nn.Linear(self.dim, self.num_head * self.num_landmarks)

        self.use_conv = -1 > 0
        # if self.use_conv:
        #     self.conv = nn.Conv1d(
        #         in_channels=self.num_head, out_channels=self.num_head,
        #         kernel_size=(config.conv_kernel_size, 1), padding=(config.conv_kernel_size // 2, 0),
        #         bias=False,
        #         groups=self.num_head)
        #     nn.init.zeros_(self.conv.weight)

    def get_tiles(self, x, transpose=False):
        # x: bsz x n_heads x seqlen x d_head
        bsz, n_heads, seqlen, d_h = x.shape
        out_shape = (bsz, n_heads, seqlen//self.window_size-1, 2 * self.window_size, d_h)
        in_strides = x.stride()
        out_strides = (in_strides[0], in_strides[1], in_strides[2]*self.window_size, in_strides[2], 1)

        x_main = x.as_strided(size=out_shape, stride=out_strides)
        x_last = x[:, :, None, -2*self.window_size:, :]
        x = torch.cat([x_main, x_last], dim=2)
        if transpose:
            return x.transpose(-1, -2)
        else:
            #  bsz x n_heads x seqlen//wlen x 2*wlen x d_h
            return x

    def get_tiled_mask(self, mask):
        bsz, seqlen = mask.shape
        out_shape = (bsz, seqlen//self.window_size-1, 2*self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1]*self.window_size, in_stride[1])
        mask_main = mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, :]
        mask_last = mask[:, None, None, -2*self.window_size:]

        return torch.cat([mask_main, mask_last], dim=2)[:, :, :, None, :]

    def sliding_chunks_matmul_qk(self, Q, K, padding_mask):
        # Q, K: bsz x num_heads x seqlen x d_head
        # padding_mask: bsz x seqlen
        bsz, num_heads, seqlen, d_h = Q.shape
        mask_tiles = self.get_tiled_mask(padding_mask)
        K_tiles = self.get_tiles(K, transpose=True)
        Q_tiles = Q.view(bsz, num_heads, seqlen//self.window_size, self.window_size, d_h)
        # bsz x num_heads x seqlen//winsize x winsize x 2winsize
        qk_scores = Q_tiles.matmul(K_tiles)
        qk_scores.masked_fill_(mask_tiles, float('-inf'))
        return qk_scores.view(bsz, num_heads, seqlen, 2*self.window_size)

    def get_tiles_v2(self, x, transpose=False):
        if self.window_size <= 0:
            return x

        bsz, n_heads, seqlen, d_h = x.shape
        n_groups = seqlen // self.window_size
        ext_len = max(self.window_size//2, 1)
        x = F.pad(x, (0, 0, ext_len, ext_len), value=0)
        strides = x.stride()
        if transpose:
            out_shape = (bsz, n_heads, n_groups, d_h, 2 * ext_len + self.window_size)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[3], strides[2])
        else:
            out_shape = (bsz, n_heads, n_groups, 2 * ext_len + self.window_size, d_h)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)

    def get_tiled_mask_v2(self, mask):
        # only mask along the key dimension
        bsz, seqlen = mask.shape
        ext_len = max(self.window_size//2, 1)
        mask = F.pad(mask, (ext_len, ext_len), value=True)
        out_shape = (bsz, seqlen//self.window_size, 2*ext_len + self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1]*self.window_size, in_stride[1])
        return mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, None, :]

    def sliding_chunks_matmul_qk_v2(self, Q, K, padding_mask):
        bsz, num_heads, seqlen, d_h = Q.shape
        if self.window_size > 0:
            # Q, K: bsz x num_heads x seqlen x d_head
            # padding_mask: bsz x seqlen

            mask_tiles = self.get_tiled_mask_v2(padding_mask)
            K_tiles = self.get_tiles_v2(K, transpose=True)
            Q_tiles = Q.view(bsz, num_heads, seqlen//self.window_size, self.window_size, d_h)
            # bsz x num_heads x seqlen//winsize x winsize x 2winsize
            qk_scores = Q_tiles.matmul(K_tiles)
            qk_scores = qk_scores.masked_fill(mask_tiles, float('-inf'))
            return qk_scores.view(bsz, num_heads, seqlen, -1)
        else:
            qk_scores = torch.sum(Q*K, dim=-1, keepdim=True)
            return qk_scores

    # def forward(self, X, mask, cls_embed=None):
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
        eps=1e-6,
        cls_embed=None
    ):
        # input: tgt_len, bsz, embed_dim
        assert not (self.num_landmarks <= 0 and cls_embed is None and self.window_size <= 0)
        # bsz, tgt_len, embed_dim
        # print(query.shape)
        X = query.transpose(0, 1)
        bsz, tgtlen, d_model = X.shape
        len_pad = (self.window_size - tgtlen % self.window_size) % self.window_size
        # print(X.shape)
        X = F.pad(X, (0, 0, 0, len_pad, 0, 0))
        # print(X.shape)
        mask = attn_mask
        if self.cls_from_seq:
            cls_embed = X[:,:1].contiguous()
            X = X[:,1:].contiguous()
            mask = mask[:,1:].contiguous()

        bsz, seqlen, d_model = X.shape
        # bsz x n_head x length x head_dim
        Q = self.split_heads(self.W_q(X)).mul(1./np.sqrt(self.head_dim))

        K = self.split_heads(self.dual_ln_l(self.W_k(X)))
        V = self.split_heads(self.dual_ln_l(self.W_v(X)))
        if self.fp32:
            Q, K, V = Q.float(), K.float(), V.float()

        # bsz x length x num_head*num_lms
        mask = torch.ones(X.shape[:-1]).to(X)
        padding_mask = ~mask.bool()
        # padding_mask = torch.ones(1, 1, seqlen, seqlen) == 0
        # print(padding_mask.shape)
        # print(X.shape)
        # print(self.dconv_fc(X).shape)

        K_compress = V_compress = None
        if self.num_landmarks > 0:
            head_scores = self.dconv_fc(X).masked_fill(padding_mask[:, :, None], float('-inf'))
            head_scores = F.softmax(head_scores, dim=1, dtype=torch.float32) #.to(X)
            if not self.fp32:
                head_scores = head_scores.to(X)
            # bsz x num_head x num_lms x length
            head_scores = head_scores.view(bsz, seqlen, self.num_head, self.num_landmarks).permute(0, 2, 3, 1)
            K_compress = head_scores.matmul(K)
            V_compress = head_scores.matmul(V)

        if cls_embed is not None:
            Q_cls = self.split_heads(self.W_q(cls_embed)).mul(1. / np.sqrt(self.head_dim))
            K_cls = self.split_heads(self.W_k(cls_embed))
            V_cls = self.split_heads(self.W_v(cls_embed))
            if self.num_landmarks > 0:
                K_compress = torch.cat([K_cls, K_compress], dim=2)
                V_compress = torch.cat([V_cls, V_compress], dim=2)
            else:
                K_compress = K_cls
                V_compress = V_cls

        if self.dual_ln_s is not None and K_compress is not None:
            K_compress = self.dual_ln_s(K_compress.transpose(1, 2).contiguous().view(bsz, -1, d_model))
            K_compress = self.split_heads(K_compress)
            V_compress = self.dual_ln_s(V_compress.transpose(1, 2).contiguous().view(bsz, -1, d_model))
            V_compress = self.split_heads(V_compress)

        if self.num_landmarks > 0 or (cls_embed is not None):
            # bsz x num_head x length x num_lms
            attn_compress = Q.matmul(K_compress.transpose(-1, -2))
        else:
            attn_compress = None

        if self.window_size > 0 or self.num_landmarks == 0:
            # First, compute the compressed part, or the attentions on the landmarks
            # First use window attention to attend to the diagonals
            # V: bsize, self.seq_len, self.num_head, self.head_dim
            # win_attn_weights = self.sliding_chunks_matmul_qk(Q, K, padding_mask)
            win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K, padding_mask)
        else:
            win_attn_weights = None

        if attn_compress is None:
            all_attn_ = win_attn_weights
        elif win_attn_weights is None:
            all_attn_ = attn_compress
        else:
            all_attn_ = torch.cat([attn_compress, win_attn_weights], dim=-1)

        all_attn = all_attn_.float().softmax(dim=-1).to(win_attn_weights)
        # If one of the rows are all -inf, then it will be NaN!
        all_attn = all_attn.masked_fill(padding_mask[:,None,:,None], 0)
        if not self.fp32:
            all_attn = all_attn.to(X)
        all_attn = self.drop_attn(all_attn)

        C = 0
        if attn_compress is not None:
            C += all_attn[:,:,:,:K_compress.shape[2]].matmul(V_compress)

        if win_attn_weights is not None:
            win_attn_probs = all_attn[:,:,:,-win_attn_weights.shape[-1]:]
            if self.window_size > 0:
                win_attn_probs = win_attn_probs.view(bsz, self.num_head, seqlen // self.window_size, self.window_size,-1)
                V_tiles = self.get_tiles_v2(V, transpose=False)
                C += win_attn_probs.matmul(V_tiles).view(bsz, self.num_head, seqlen, self.head_dim)
            else:
                C += win_attn_probs * V

        if cls_embed is not None:
            if self.fp32:
                Q_cls, K_cls, V_cls = Q_cls.float(), K_cls.float(), V_cls.float()
            # bsz x n_heads x 1 x (1+seqlen)
            cls_scores = torch.cat([Q_cls.matmul(K_cls.transpose(-1, -2)),
                                    Q_cls.matmul(C.transpose(-1, -2)).masked_fill(padding_mask[:,None,None,:], float('-inf'))],
                                   dim=-1)
            cls_probs = torch.softmax(cls_scores, dim=-1, dtype=torch.float32)#.to(X)
            if not self.fp32:
                cls_probs = cls_probs.to(X)
            out_cls_embed = V_cls * cls_probs[:,:,:,:1] + cls_probs[:,:,:,1:].matmul(C)

        if self.use_conv:
            V = V.masked_fill(padding_mask[:, None, :, None], 0)
            C = C + self.conv(V)

        if cls_embed is not None:
            C = torch.cat([out_cls_embed, C], dim=2)

        if self.fp32:
            # Finally convert it back, same as Nystromformer
            C = C.to(X)
        # print("here")
        # print(self.W_o(self.combine_heads(C)).shape)
        out = self.W_o(self.combine_heads(C)).transpose(0, 1)
        # print(out.shape)
        out = out[:tgtlen, ...]
        # print(out.shape)
        return out, None

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, window_size={self.window_size}'

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X