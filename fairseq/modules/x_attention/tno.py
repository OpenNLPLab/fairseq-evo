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
from fairseq.modules import SimpleRMSNorm
from fairseq.modules import GatedRMSNorm
from fairseq.modules import RMSNorm
from fairseq.modules import Urpe
from fairseq.modules import UrpeV2
from fairseq.modules import ToepliztMultihead
from einops import rearrange

@with_incremental_state
class TNO(nn.Module):
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
        act_fun="silu",
        causal=False,
        expand_ratio=2,
        resi_param=False,
        # norm
        use_norm=False,
        norm_type="simplermsnorm",
        # Toeplizt
        use_exp=False,
        toep_type=1,
        max_l=512,
        use_decay=False,
    ):
        # add
        self.index = index

        super().__init__()
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
        
        self.toep_type = toep_type
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        print(f"self.resi_param {self.resi_param}")
        print(f"self.expand_ratio {self.expand_ratio}")
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(self.embed_dim))
            
        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        d2 = embed_dim
        if self.toep_type == 1:
            self.head_dim = d2 // num_heads
            # d^2
            self.v_proj = quant_noise(
                nn.Linear(embed_dim, d1, bias=bias), q_noise, qn_block_size
            )
            # d^2
            self.u_proj = quant_noise(
                nn.Linear(embed_dim, d1, bias=bias), q_noise, qn_block_size
            )
            # d^2
            self.o = quant_noise(
                nn.Linear(d1, embed_dim, bias=bias), q_noise, qn_block_size
            )
            self.forward = self.forward1
        elif self.toep_type == 2:

            self.head_dim = d2 // num_heads
            # d^2
            self.v_proj = quant_noise(
                nn.Linear(embed_dim, d1, bias=bias), q_noise, qn_block_size
            )
            # d^2
            self.o = quant_noise(
                nn.Linear(d1, embed_dim, bias=bias), q_noise, qn_block_size
            )
            self.forward = self.forward2
            
        self.causal = causal
        self.act = self.get_act_fun(act_fun)
        print(f"act_fun {act_fun}")
        print(f"causal {self.causal}")
        
        # toep
        self.max_l = max_l
        self.use_exp = use_exp
        self.use_decay = use_decay
        self.toep = ToepliztMultihead(h=self.num_heads, n=self.max_l, causal=self.causal, use_exp=self.use_exp, use_decay=self.use_decay)
        print(f"self.num_heads {self.num_heads}")
        print(f"self.max_l {self.max_l}")
        print(f"self.use_exp {self.use_exp}")
        print(f"self.use_decay {self.use_decay}")
        
        # norm
        self.norm_type = norm_type
        self.pre_norm = self.get_norm_fun(self.norm_type, d2)
        
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = self.get_norm_fun(norm_type, d1)
        print(f"use_norm {self.use_norm}")
        print(f"norm_type {self.norm_type}")

        self.par_init()
        
    def par_init(self):
        if self.toep_type == 1:
            nn.init.normal_(self.u_proj.weight, std=0.02)
            nn.init.normal_(self.u_proj.bias, std=0.02)
            nn.init.normal_(self.v_proj.weight, std=0.02)
            nn.init.normal_(self.v_proj.bias, std=0.02)
            nn.init.normal_(self.o.weight, std=0.02)
            nn.init.normal_(self.o.bias, std=0.02)

    def get_norm_fun(self, norm_type, embed_dim):
        if norm_type == "rmsnorm":
            print("here! rmsnorm")
            return RMSNorm(embed_dim)
        elif norm_type == "gatedrmsnorm":
            print("here! gatedrmsnorm")
            return GatedRMSNorm(embed_dim)
        elif norm_type == "simplermsnorm":
            print("here! simple rmsnorm")
            return SimpleRMSNorm(embed_dim)
        elif norm_type == "scalenorm":
            print("here! scale norm")
            return ScaleNorm(embed_dim)
        else:
            print("here! layer norm")
            return nn.LayerNorm(embed_dim)

    def get_act_fun(self, act_fun):
        print(act_fun)
        if act_fun == "gelu":
            return F.gelu
        elif act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return F.elu
        elif act_fun == "sigmoid":
            return F.sigmoid
        elif act_fun == "exp":
            return torch.exp
        elif act_fun == "leak":
            def f(x):
                return F.leaky_relu(x, negative_slope=self.negative_slope)
            return f
        elif act_fun == "1+elu":
            def f(x):
                return 1 + F.elu(x)
            return f
        elif act_fun == "silu":
            return F.silu
        elif self.act_fun == "relu2":
            def f(x):
                return torch.square(torch.relu(x))
            return f
        else:
            return lambda x: x

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        print("normal init")
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

    def forward1(
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

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'n b (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2)
        output = rearrange(output, 'b h n d -> n b (h d)')
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output) + shortcut
        
        return output, None
    
    def forward2(
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

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'n b (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2)
        output = rearrange(output, 'b h n d -> n b (h d)')
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output) + shortcut
        
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

