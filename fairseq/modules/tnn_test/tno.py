import math
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Dropout, Parameter

from ..helpers import (get_activation_fn, get_norm_fn, logging_info,
                       print_params)
from ..others import SEBlock
from ..positional_encoding import Urpe
from ..positional_encoding_test import (DynamicToepliztMultihead,
                                        DynamicToepliztMultiheadV2,
                                        DynamicToepliztMultiheadV3,
                                        DynamicToepliztMultiheadV4,
                                        NonDynamicToepliztMultihead,
                                        ToepliztMultihead)


@with_incremental_state
class TNO(nn.Module):
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
        shrink_ratio=1,
        resi_param=False,
        # norm
        use_norm=False,
        norm_type="simplermsnorm",
        # Toeplizt
        use_exp=False,
        use_neg_exp=False, 
        toep_type=1,
        max_l=512,
        use_decay=False,
        use_multi_decay=False,
        use_dynamic=False,
        dpb_embedding=512,
        use_dynamic_v2=False,
        dpb_act="relu",
        dpb_use_pad=True,
        normalize=False,
        use_dynamic_v3=False,
        par_type=1,
        dpb_type=1,
        dynamic_type=1,
        residual=False,
        l=1, 
        transform_type=1,
        gamma=0.999,
        tno_act_type="none",
        dpb_h=-1,
        dpb_dim=-1,
        dpb_layers=3,
        decay_type=-1,
        # SE
        use_se=False,
        se_ratio=16,
        # token shift
        token_shift_type=-1,
        use_norm1=True,
        use_norm2=True,
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

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        
        self.toep_type = toep_type
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        if self.resi_param:
            self.d = nn.Parameter(torch.randn(self.embed_dim))
            
        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        d2 = embed_dim
        self.shrink_ratio = shrink_ratio
        self.head_dim = d1 // num_heads
        if self.toep_type == 1 or self.toep_type == 5 or self.toep_type == 7:
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
            if self.toep_type == 5:
                self.forward = self.forward5
            elif self.toep_type == 7:
                self.forward = self.forward7
            else:
                self.forward = self.forward1
        elif self.toep_type == 2:
            # d^2
            self.v_proj = quant_noise(
                nn.Linear(embed_dim, d1, bias=bias), q_noise, qn_block_size
            )
            # d^2
            self.o = quant_noise(
                nn.Linear(d1, embed_dim, bias=bias), q_noise, qn_block_size
            )
            self.forward = self.forward2
        elif self.toep_type == 3:
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
            self.forward = self.forward3
        elif self.toep_type == 4:
            d2 = embed_dim // self.shrink_ratio
            d2 = (d2 // self.num_heads) * self.num_heads
            self.head_dim = d2 // self.num_heads
            # d^2
            self.v_proj = quant_noise(
                nn.Linear(embed_dim, d2, bias=bias), q_noise, qn_block_size
            )
            # d^2
            self.u_proj = quant_noise(
                nn.Linear(embed_dim, d1, bias=bias), q_noise, qn_block_size
            )
            # d^2
            self.o1 = quant_noise(
                nn.Linear(d2, d1, bias=bias), q_noise, qn_block_size
            )
            self.o2 = quant_noise(
                nn.Linear(d1, embed_dim, bias=bias), q_noise, qn_block_size
            )
            self.forward = self.forward4
        elif self.toep_type == 6:
            self.shrink_ratio = shrink_ratio
            d2 = embed_dim // self.shrink_ratio
            d2 = (d2 // self.num_heads) * self.num_heads
            self.head_dim = d2 // self.num_heads
            d3 = d1 // self.num_heads
            # d^2
            self.v_proj = quant_noise(
                nn.Linear(embed_dim, d2, bias=bias), q_noise, qn_block_size
            )
            # d^2
            self.u_proj = quant_noise(
                nn.Linear(embed_dim, d1, bias=bias), q_noise, qn_block_size
            )
            # d^2
            self.o1 = quant_noise(
                nn.Linear(self.head_dim, d3, bias=bias), q_noise, qn_block_size
            )
            self.o2 = quant_noise(
                nn.Linear(d1, embed_dim, bias=bias), q_noise, qn_block_size
            )
            self.forward = self.forward6
        elif self.toep_type == 8:
            self.q_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
            self.k_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
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
            self.forward = self.forward8
        
        self.causal = causal
        self.act = get_activation_fn(act_fun)
        
        # toep
        self.max_l = max_l
        self.use_exp = use_exp
        self.use_neg_exp = use_neg_exp
        self.use_decay = use_decay
        self.use_multi_decay = use_multi_decay
        self.use_dynamic = use_dynamic
        self.dpb_embedding = dpb_embedding
        self.use_dynamic_v2 = use_dynamic_v2
        self.dpb_act = dpb_act
        self.dpb_use_pad = dpb_use_pad
        self.normalize = normalize
        self.use_dynamic_v3 = use_dynamic_v3
        self.par_type = par_type
        self.dpb_type = dpb_type
        self.dynamic_type = dynamic_type
        self.residual = residual
        self.l = l
        self.transform_type = transform_type
        self.gamma = gamma
        self.bias = bias
        self.tno_act_type = tno_act_type
        self.dpb_layers = dpb_layers
        self.decay_type = decay_type
        self.use_norm1 = use_norm1
        self.use_norm2 = use_norm2
        if self.toep_type != 7:
            if self.use_dynamic:
                self.toep = DynamicToepliztMultihead(
                    h=self.num_heads, 
                    n=self.max_l, 
                    d=self.dpb_embedding, 
                    causal=self.causal, 
                    use_exp=self.use_exp, 
                    use_neg_exp=self.use_neg_exp,
                    use_decay=self.use_decay,
                    use_multi_decay=self.use_multi_decay,
                    bias=self.bias,
                )
            elif self.use_dynamic_v2:
                self.toep = DynamicToepliztMultiheadV2(
                    h=self.num_heads, 
                    n=self.max_l, 
                    d=self.dpb_embedding, 
                    causal=self.causal, 
                    use_exp=self.use_exp,
                    use_neg_exp=self.use_neg_exp,
                    use_decay=self.use_decay, 
                    use_multi_decay=self.use_multi_decay,
                    act=self.dpb_act,
                    use_pad=self.dpb_use_pad,
                    bias=self.bias,
                )
            elif self.use_dynamic_v3:
                self.toep = DynamicToepliztMultiheadV3(
                    h=self.num_heads, 
                    n=self.max_l, 
                    dim=self.head_dim,
                    dpb_dim=self.dpb_embedding, 
                    causal=self.causal, 
                    use_exp=self.use_exp,
                    use_neg_exp=self.use_neg_exp,
                    use_decay=self.use_decay, 
                    use_multi_decay=self.use_multi_decay,
                    use_pad=self.dpb_use_pad,
                    par_type=self.par_type,
                    dpb_type=self.dpb_type,
                    bias=self.bias,
                )
            elif self.dynamic_type == 4:
                if dpb_h == -1:
                    dpb_h = self.num_heads
                if dpb_dim == -1:
                    dpb_dim = self.head_dim
                logging_info(f"dpb_h {dpb_h}")
                logging_info(f"dpb_dim {dpb_dim}")
                self.toep = DynamicToepliztMultiheadV4(
                    h=dpb_h, 
                    n=self.max_l, 
                    dim=dpb_dim,
                    dpb_dim=self.dpb_embedding, 
                    causal=self.causal, 
                    use_exp=self.use_exp,
                    use_neg_exp=self.use_neg_exp,
                    use_decay=self.use_decay, 
                    use_multi_decay=self.use_multi_decay,
                    use_pad=self.dpb_use_pad,
                    act=self.dpb_act,
                    par_type=self.par_type,
                    residual=self.residual,
                    dpb_type=self.dpb_type,
                    layers=self.dpb_layers,
                    l=self.l,
                    transform_type=self.transform_type,
                    gamma=self.gamma,
                    bias=self.bias,
                    act_type=self.tno_act_type,
                    decay_type=self.decay_type,
                    use_norm1=self.use_norm1, 
                    use_norm2=self.use_norm2,
                )
            elif self.dynamic_type == 5:
                if dpb_h == -1:
                    dpb_h = self.num_heads
                if dpb_dim == -1:
                    dpb_dim = self.head_dim
                logging_info(f"dpb_h {dpb_h}")
                logging_info(f"dpb_dim {dpb_dim}")
                self.toep = NonDynamicToepliztMultihead(
                    h=dpb_h, 
                    n=self.max_l, 
                    dim=dpb_dim,
                    dpb_dim=self.dpb_embedding, 
                    causal=self.causal, 
                    use_exp=self.use_exp,
                    use_neg_exp=self.use_neg_exp,
                    use_decay=self.use_decay, 
                    use_multi_decay=self.use_multi_decay,
                    use_pad=self.dpb_use_pad,
                    act=self.dpb_act,
                    par_type=self.par_type,
                    residual=self.residual,
                    dpb_type=self.dpb_type,
                    layers=self.dpb_layers,
                    l=self.l,
                    transform_type=self.transform_type,
                    gamma=self.gamma,
                    bias=self.bias,
                    act_type=self.tno_act_type,
                    decay_type=self.decay_type,
                )
            else:
                self.toep = ToepliztMultihead(h=self.num_heads, n=self.max_l, causal=self.causal, use_exp=self.use_exp, use_decay=self.use_decay)
        
        # norm
        self.norm_type = norm_type
        self.pre_norm = get_norm_fn(norm_type)(d2)
        # self.pre_norm = self.get_norm_fun(self.norm_type, d2)
        
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(norm_type)(d1)
            # self.norm = self.get_norm_fun(norm_type, d1)
        
        # se
        self.use_se = use_se
        self.se_ratio = se_ratio
        if self.use_se:
            self.se = SEBlock(d1, self.se_ratio, self.causal)
        
        # token shift
        self.token_shift_type = token_shift_type

        if self.token_shift_type == 1:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
        elif self.token_shift_type == 2:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.coef = 0.5

        self.par_init()
        
    def par_init(self):
        if self.toep_type == 1 or self.toep_type == 5:
            nn.init.normal_(self.u_proj.weight, std=0.02)
            if self.bias:
                nn.init.normal_(self.u_proj.bias, std=0.02)
            nn.init.normal_(self.v_proj.weight, std=0.02)
            if self.bias:
                nn.init.normal_(self.v_proj.bias, std=0.02)
            nn.init.normal_(self.o.weight, std=0.02)
            if self.bias:
                nn.init.normal_(self.o.bias, std=0.02)
        elif self.toep_type == 3:
            nn.init.normal_(self.u_proj.weight, std=0.02)
            if self.bias:
                nn.init.normal_(self.u_proj.bias, std=0.02)
            nn.init.normal_(self.v_proj.weight, std=0.02)
            if self.bias:
                nn.init.normal_(self.v_proj.bias, std=0.02)
            nn.init.normal_(self.o.weight, std=0.02)
            if self.bias:
                nn.init.normal_(self.o.bias, std=0.02)
        elif self.toep_type == 4:
            return
        elif self.toep_type == 8:
            ##### new
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

            if self.bias:
                nn.init.normal_(self.q_proj.bias, std=0.02)
            if self.bias:
                nn.init.normal_(self.k_proj.bias, std=0.02)
            if self.bias:
                nn.init.normal_(self.v_proj.bias, std=0.02)
                
            nn.init.normal_(self.u_proj.weight, std=0.02)
            if self.bias:
                nn.init.normal_(self.u_proj.bias, std=0.02)
            nn.init.normal_(self.o.weight, std=0.02)
            if self.bias:
                nn.init.normal_(self.o.bias, std=0.02)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
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
        
        if self.token_shift_type == 1:
            query = self.token_shift(query)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(query)
            query = self.coef * q1 + (1 - self.coef) * query

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'n b (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> n b (h d)')
        if self.use_se:
            output = self.se(output)
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
        
        if self.token_shift_type == 1:
            query = self.token_shift(query)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(query)
            query = self.coef * q1 + (1 - self.coef) * query

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'n b (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> n b (h d)')
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output) + shortcut
        
        return output, None
    
    def mean(self, x, dim):
        if self.causal:
            x = torch.cumsum(x, dim=dim)
            n = x.shape[dim]
            l = len(x.shape)
            denom = torch.arange(1, n + 1).to(x)
            for i in range(dim):
                denom = denom.unsqueeze(0)
            for i in range(dim + 1, l):
                denom = denom.unsqueeze(-1)
            x = x / denom
        else:
            x = torch.mean(x, dim=dim, keepdims=True)
            
        return x
    
    def forward3(
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
        
        if self.token_shift_type == 1:
            query = self.token_shift(query)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(query)
            query = self.coef * q1 + (1 - self.coef) * query

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        u = self.mean(u, dim=0)
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'n b (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> n b (h d)')
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output) + shortcut
        
        return output, None
    
    def forward4(
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
        
        if self.token_shift_type == 1:
            query = self.token_shift(query)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(query)
            query = self.coef * q1 + (1 - self.coef) * query

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'n b (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> n b (h d)')
        output = self.o1(output)
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o2(output) + shortcut
        
        return output, None

    def forward5(
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
        
        if self.token_shift_type == 1:
            query = self.token_shift(query)

        shortcut, x = query, query
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'n b (h d) -> b h n d', h=num_heads)
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = rearrange(output, 'b h n d -> n b (h d)')
        if self.use_se:
            output = self.se(output)
        output = u * output
        if self.use_norm:
            output = self.norm(output)
            
        output = self.pre_norm(self.o(output)) + shortcut
        
        return output, None
    
    # forward4的完全的多头版本
    def forward6(
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
        
        if self.token_shift_type == 1:
            query = self.token_shift(query)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(query)
            query = self.coef * q1 + (1 - self.coef) * query

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        u, v = map(lambda x: rearrange(x, 'n b (h d) -> b h n d', h=num_heads), [u, v])
        output = self.toep(v, dim=-2, normalize=self.normalize)
        output = self.o1(output)
        output = u * output
        output = rearrange(output, 'b h n d -> n b (h d)')
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o2(output) + shortcut
        
        return output, None
    
    # forward 1多头版本
    def forward7(
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
        
        if self.token_shift_type == 1:
            query = self.token_shift(query)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(query)
            query = self.coef * q1 + (1 - self.coef) * query

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        u, v = map(lambda x: rearrange(x, 'n b (h d) -> b h n d', h=num_heads), [u, v])
        output = self.toep(v, dim=-2, normalize=self.normalize)
        if self.use_se:
            output = self.se(output)
        output = u * output
        output = rearrange(output, 'b h n d -> n b (h d)')
        if self.use_norm:
            output = self.norm(output)
            
        output = self.o(output) + shortcut
        
        return output, None
    
    # tno->attention
    def forward8(
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
        
        if self.token_shift_type == 1:
            query = self.token_shift(query)
        elif self.token_shift_type == 2:
            q1 = self.token_shift(query)
            query = self.coef * q1 + (1 - self.coef) * query

        shortcut, x = query, self.pre_norm(query)
        if self.resi_param:
            shortcut = shortcut * self.d
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        q = self.q_proj(x)
        k = self.k_proj(x)
        # reshape
        q, k, v = map(lambda x: rearrange(x, 'n b (h d) -> b h n d', h=num_heads), [q, k, v])
        scale = q.shape[-1] ** -0.5
        score = torch.einsum('bhnd,bhmd->bhnm', q, k) * scale
        prob = F.softmax(score, dim=-1)
        output = torch.einsum('bhnm,bhmd->bhnd', prob, v)
        output = rearrange(output, 'b h n d -> n b (h d)')
        if self.use_se:
            output = self.se(output)
        output = u * output
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

