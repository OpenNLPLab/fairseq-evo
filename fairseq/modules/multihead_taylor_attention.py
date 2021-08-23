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

        if self.low_d:
            dim = embed_dim // 2
        else:
            dim = embed_dim
        self.scaling = dim ** -0.5
        

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, dim, bias=bias), q_noise, qn_block_size
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
        # 1 * E
        if self.is_ada_q:
            self.qsigma2 = Parameter(torch.ones(1, self.embed_dim), requires_grad=False)
        if self.is_ada_k:
            self.ksigma2 = Parameter(torch.ones(1, self.embed_dim), requires_grad=False)

        if self.has_out:
            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
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
        print(self.do_scale)
        print(self.norm_taylor)

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
        else:
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
            # 1 + |q| ^ (-1) * q * k ^ T * |k| ^ (-1) 
            attn_output_weights = torch.bmm(q, k.transpose(1, 2))

            # grad
            # 1 , S
            tmp = attn_output_weights[0][0]
            l1 = torch.norm(tmp, p=1)
            print(l1)
            print(l1 < 1e-12)
            print(l1.shape)
            grad = -torch.abs(tmp) / (l1 ** 2)
            print(grad)
            grad[0] += l1
            print(grad)
            
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
            # t1 = torch.abs(attn_output_weights)
            # t2 = torch.sum(t1, axis=-1)
            # print(attn_output_weights)
            # print(t2)
            # print(tmp)
            # dropout
        attn_output_weights = F.dropout(attn_output_weights, self.dropout_module.p, training=self.training)
        # N, S, E
        value = value.transpose(0, 1)

        # N, L, E
        attn_output = torch.bmm(attn_output_weights, value)
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
