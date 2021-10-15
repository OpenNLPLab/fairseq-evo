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


# taylor
@with_incremental_state
class MultiheadTaylorAttention(nn.Module):
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
        # res
        has_res=False,
        # right_weight
        has_right_weight=False,
        do_softmax=False,
        with_right_weight=False,
        has_right_weight_not_share=False,
        # 因子
        alpha_beta=False,
        max_l=1024,
        # linear
        use_linear=False
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
        # self.with_right_weight = with_right_weight

        if self.low_d:
            dim = embed_dim // 2
        elif self.dim_scale != -1:
            dim = self.dim_scale * embed_dim
        else:
            dim = embed_dim
        self.dim = dim
        self.scaling = dim ** -0.5
        

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, self.dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, self.dim, bias=bias), q_noise, qn_block_size
        )

        # add begin
        self.is_ada_q = is_ada_q
        self.is_ada_k = is_ada_k
        self.lambda_ = lambda_
        self.scaling = dim ** -0.5
        self.up_fq = up_fq
        self.cnt = 0
        self.dropout_before = dropout_before
        self.has_out = has_out
        self.use_q = use_q
        self.use_k = use_k
        self.norm_taylor = norm_taylor
        self.use_relu = use_relu
        self.use_elu = use_elu
        self.use_leak = use_leak
        self.use_square = use_square
        self.use_sigmoid = use_sigmoid
        self.use_l2 = use_l2
        self.sparse = sparse
        self.d1 = d1
        self.d2 = d2
        self.has_res = has_res
        self.has_right_weight = has_right_weight
        self.do_softmax = do_softmax
        self.has_right_weight_not_share = has_right_weight_not_share
        self.alpha_beta = alpha_beta
        self.max_l = max_l
        self.use_linear = use_linear

        if self.has_right_weight_not_share:
            self.k_proj1 = quant_noise(
                nn.Linear(self.kdim, self.dim, bias=bias), q_noise, qn_block_size
            )
            self.q_proj1 = quant_noise(
                nn.Linear(embed_dim, self.dim, bias=bias), q_noise, qn_block_size
            )
        
        if self.sparse:
            self.diag_mask = self.build_mask(5120, 5120)
        # 1 * E
        if self.is_ada_q:
            self.qsigma2 = Parameter(torch.ones(1, self.dim), requires_grad=False)
        if self.is_ada_k:
            self.ksigma2 = Parameter(torch.ones(1, self.dim), requires_grad=False)

        if self.has_out:
            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )

        if self.alpha_beta:
            self.row1, self.col1, self.row2, self.col2 = self.get_alpha_beta(self.max_l, self.max_l)
        # add end

        # print(self.is_ada_q, self.is_ada_k, self.dropout_before, self.has_out)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        print(dim, embed_dim)
        print(f"self.index {self.index}")
        print(f"do scale {self.do_scale}")
        print(f"taylor {self.norm_taylor}")
        print(f"use relu {self.use_relu}")
        print(f"use elu {self.use_elu}")
        print(f"use leak {self.use_leak}")
        print(f"use square {self.use_square}")
        print(f"use sigmoid {self.use_sigmoid}")
        print(f"use l2 {self.use_l2}")
        print(f"sparse {self.sparse}")
        print(f"d1 {self.d1}")
        print(f"d2 {self.d2}")
        print(f"self.has res {self.has_res}")
        print(f"self.has_right_weight {self.has_right_weight}")
        print(f"self.has_right_weight_not_share {self.has_right_weight_not_share}")
        print(f"self.do_softmax {self.do_softmax}")
        print(f"self.alpha_beta {self.alpha_beta}")
        print(f"self.max_l {self.max_l}")
        print(f"self.use_linear {self.use_linear}")

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

        if self.has_right_weight_not_share:
            nn.init.xavier_uniform_(self.k_proj1.weight)
            nn.init.xavier_uniform_(self.q_proj1.weight)
        
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def get_alpha_beta(self, src_len, tgt_len):
        a = np.pi / 2
        row_index = torch.arange(1, src_len + 1)
        col_index = torch.arange(1, tgt_len + 1)
        row1 = torch.cos(a * row_index / src_len).reshape(1, -1, 1)
        col1 = torch.cos(a * col_index / tgt_len).reshape(1, -1, 1)

        row2 = torch.sin(a * row_index / src_len).reshape(1, -1, 1)
        col2 = torch.sin(a * col_index / tgt_len).reshape(1, -1, 1)

        # mask = row1 * col1 + row2 * col2

        return nn.Parameter(row1, requires_grad=False), nn.Parameter(col1, requires_grad=False), nn.Parameter(row2, requires_grad=False), nn.Parameter(col2, requires_grad=False)

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

        if self.training:
            # L * N, E -> (1, E)
            # sigma2 = torch.nn.Parameter(torch.var(q.view(-1, self.embed_dim), dim=0, keepdim=True), requires_grad=False)
            self.cnt += 1
            if self.cnt % self.up_fq == 0:
                # print(self.cnt, self.up_fq)
                with torch.no_grad():
                    if self.is_ada_q:
                        # print("q")
                        qsigma2 = torch.var(q.view(-1, self.embed_dim), dim=0, keepdim=True)
                        self.qsigma2 *= self.lambda_
                        self.qsigma2 += (1 - self.lambda_) * qsigma2

                    if self.is_ada_k:
                        # print("k")
                        ksigma2 = torch.var(k.view(-1, self.embed_dim), dim=0, keepdim=True)
                        self.ksigma2 *= self.lambda_
                        self.ksigma2 += (1 - self.lambda_) * ksigma2

    
        # scaling = float(embed_dim) ** -0.5
        # q *= self.scaling
        if self.do_scale:
            q = q * self.scaling

        if self.use_q:
            # print("q1")
            q /= torch.sqrt(self.qsigma2)
        if self.use_k:
            # print("k1")
            k /= torch.sqrt(self.ksigma2)

        if self.use_relu:
            q = F.relu(q)
            k = F.relu(k)
        elif self.use_elu:
            q = F.elu(q)
            k = F.elu(k)
        elif self.use_leak:
            q = F.leaky_relu(q)
            k = F.leaky_relu(k)
        elif self.use_square:
            q = torch.square(q)
            k = torch.square(k)
        elif self.use_sigmoid:
            q = F.sigmoid(q)
            k = F.sigmoid(k)
        elif self.use_linear:
            # print("self.use_linear")
            q = 1 + F.elu(q)
            k = 1 + F.elu(k)

        if self.norm_taylor:
            # N, L, E
            q = q.transpose(0, 1)
            # |q| ^ (-1) q
            q = F.normalize(q, p=2, dim=-1)
            # q *= scaling
            # N, S, E
            k = k.transpose(0, 1)
            # |k| ^ (-1) k
            k = F.normalize(k, p=2, dim=-1)
            # N, L, S
            # 1 + |q| ^ (-1) * q * k ^ T * |k| ^ (-1) 
            attn_output_weights = 1 + torch.bmm(q, k.transpose(1, 2))
            if attn_mask is not None:
                attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)
            # print(attn_mask)
            # print(attn_output_weights)
            # N, L, S
            # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            # tmp = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = F.normalize(attn_output_weights, p=1, dim=-1)
            # print(torch.mean(torch.norm(tmp - attn_output_weights)))
            # print(tmp, attn_output_weights)
            # tmp = torch.sum(attn_output_weights, axis=-1)
            # print(tmp)
            # dropout
            # attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
        elif self.use_sigmoid:
            # print(1)
            # N, L, E
            q = q.transpose(0, 1)
            # |q| ^ (-1) q
            # q = F.normalize(q, p=2, dim=-1)
            # q *= scaling
            # N, S, E
            k = k.transpose(0, 1)
            # |k| ^ (-1) k
            # k = F.normalize(k, p=2, dim=-1)
            # N, L, S
            # sum_{i=1}^{embed_dim}, 每行src_len个
            attn_output_weights = torch.bmm(q, k.transpose(1, 2)) / src_len / embed_dim
            if attn_mask is not None:
                attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)
        elif self.use_l2:
            # print(1)
            ## 需要修改causal形式
            # N, L, E
            q = q.transpose(0, 1)
            # N, S, E
            k = k.transpose(0, 1)

            attn_output_weights = torch.bmm(q, k.transpose(1, 2))

            # N, L, 1
            q_norm = torch.sum(torch.pow(q, 2), dim=-1, keepdim=True)
            # N, S, 1
            k_norm = torch.sum(torch.pow(k, 2), dim=-1, keepdim=True)
            # N, 1, 1
            k_norm_sum = torch.sum(k_norm, dim=-2, keepdim=True)
            # norm
            # N, L, 1
            qk_norm = src_len * q_norm + k_norm_sum
            eps = 1e-8
            
            tmp = 2 * attn_output_weights / qk_norm
            # print(torch.mean(torch.abs(tmp) <= 1))

            qk_norm[qk_norm < eps] = eps
            
            attn_output_weights = 2 * attn_output_weights / qk_norm

        else:
            # N, L, E
            q = q.transpose(0, 1)
            # N, S, E
            k = k.transpose(0, 1)
            # N, L, S
            if self.alpha_beta:
                q1 = q * self.row1[:, :tgt_len, :]
                k1 = k * self.col1[:, :src_len, :]
                q2 = q * self.row2[:, :tgt_len, :]
                k2 = k * self.col2[:, :src_len, :]
                attn_output_weights = torch.bmm(q1, k1.transpose(1, 2)) + torch.bmm(q2, k2.transpose(1, 2))
            else:
                # 1 + |q| ^ (-1) * q * k ^ T * |k| ^ (-1) 
                attn_output_weights = torch.bmm(q, k.transpose(1, 2))

            if self.sparse:
                # d_diag = min(self.d1, tgt_len, src_len)
                # d_col = min(self.d2, tgt_len)
                # d_row = min(self.d2, src_len)
                # mask = torch.ones((src_len, tgt_len), dtype=torch.bool)
                # mask1 = torch.tril(mask, diagonal=d_diag)
                # mask2 = torch.triu(mask, diagonal=-d_diag)
                # diag_mask = (mask1 & mask2)
                # # colmask = torch.ones((src_len, tgt_len), dtype=torch.bool)
                # # colmask[:, d_col:] = False
                # # rowmask = torch.ones((src_len, tgt_len), dtype=torch.bool)
                # # rowmask[d_row:, :] = False
                # diag_mask[:d_col, :] = True
                # diag_mask[:, :d_row] = True
                # # allmask = torch.unsqueeze(diag_mask, 0)
                # # allmask = torch.unsqueeze(diag_mask | colmask | rowmask, 0)

                # # attn_output_weights[~allmask] = 0
                # attn_output_weights[:, ~diag_mask] = 0

                # print(attn_output_weights)
                # with torch.autograd.profiler.record_function("sparse"):
                #     # attn_output_weights[:, self.diag_mask[:tgt_len, :src_len]] = 0
                #     attn_output_weights.masked_fill(self.diag_mask[:tgt_len, :src_len], 0)
                # attn_output_weights[:, self.diag_mask[:tgt_len, :src_len]] = 0
                attn_output_weights = attn_output_weights.masked_fill(self.diag_mask[:tgt_len, :src_len], 0)
            
            if attn_mask is not None:
                attn_output_weights = attn_output_weights.masked_fill(attn_mask==float("-inf"), 0)

            # eps = 1e-8
            # all_weights = torch.sum(attn_output_weights, dim=-1, keepdim=True).clamp_min(eps).expand_as(attn_output_weights)
            # attn_output_weights = attn_output_weights / all_weights
            attn_output_weights = F.normalize(attn_output_weights, p=1, dim=-1, eps=1e-8)

            # with open(f"{self.index}.npy", "ab+") as f:
            #     np.save(f, attn_output_weights.cpu().detach().numpy())

            # with open(f"q_{self.index}.npy", "ab+") as f:
            #     np.save(f, q.cpu().detach().numpy())

            # with open(f"k_{self.index}.npy", "ab+") as f:
            #     np.save(f, k.cpu().detach().numpy())


        # print(attn_output_weights[0][0])
        attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
        # attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, True)
        # N, S, E
        value = value.transpose(0, 1)

        # N, L, E
        attn_output = torch.bmm(attn_output_weights, value)
        # right
        if self.has_right_weight:
            # N, 1, E
            q_mean = torch.mean(q, dim=1, keepdim=True)
            # N, 1, E
            k_mean = torch.mean(k, dim=1, keepdim=True)
            qk = torch.bmm(q_mean.transpose(1, 2), k_mean)
            if self.do_softmax:
                qk = F.softmax(qk, dim=1)
            attn_output = torch.bmm(attn_output, qk)
        elif self.has_right_weight_not_share:
            # not share
            # - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            # the embedding dimension.
            # - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            # the embedding dimension.
            # - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            # the embedding dimension.
            # N, 1, E
            query_mean = torch.mean(query, dim=0, keepdim=True).transpose(0, 1)
            key_mean = torch.mean(key, dim=0, keepdim=True).transpose(0, 1)
            q_mean = self.q_proj1(query_mean)
            k_mean = self.k_proj1(key_mean)
            # N, E, E
            qk = torch.bmm(q_mean.transpose(1, 2), k_mean)
            if self.do_softmax:
                # qk = F.softmax(qk, dim=1)
                qk = F.normalize(qk, p=1, dim=1, eps=1e-8)
            # N, L, E
            attn_output = torch.bmm(attn_output, qk)

        if self.has_res:
            attn_output = attn_output + value
        # L, N, E
        attn_output = attn_output.transpose(0, 1)
        # L, N, E
        attn_output = self.v_proj(attn_output)

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
