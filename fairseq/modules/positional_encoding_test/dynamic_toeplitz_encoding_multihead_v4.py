# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..helpers import get_activation_fn, logging_info, print_params
from .dpb import DynamicPosBias
from .dpb_v4 import DynamicPosBiasV4
from .dpb_v5 import DynamicPosBiasV5
from .dpb_v6 import DynamicPosBiasV6
from .dpb_v7 import DynamicPosBiasV7
from .dpb_v8 import DynamicPosBiasV8
from .dpb_v9 import DynamicPosBiasV9


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
    else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


class DynamicToepliztMultiheadV4(nn.Module):
    def __init__(
        self, 
        h, 
        n, 
        dim, 
        dpb_dim, 
        causal=False, 
        use_exp=False, 
        use_neg_exp=False, 
        use_decay=False, 
        use_multi_decay=False, 
        residual=False, 
        act="relu", 
        use_pad=False, 
        par_type=1, 
        dpb_type=4,
        l=10,
        transform_type=1,
        gamma=0.999,
        bias=True,
        act_type="none",
        layers=3,
        decay_type=-1,
        use_norm1=True, 
        use_norm2=True,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.h = h
        self.n = n
        self.dim = dim
        self.causal = causal
        self.use_exp = use_exp
        self.use_neg_exp = use_neg_exp
        self.use_pad = use_pad
        self.par_type = par_type
        self.dpb_type = dpb_type
        self.gamma = gamma
        if self.use_exp:
            self.zero_value = float("-inf")
        else:
            self.zero_value = 0

        self.use_decay = use_decay
        if self.use_decay:
            if decay_type == 1:
                logging_info("alibi")
                t = torch.exp(-torch.Tensor(get_slopes(self.h))).reshape(self.h, 1)
                t = repeat(t, 'h 1 -> h 1 d', d=self.dim)
                self.gamma = nn.Parameter(t, requires_grad=False)
            elif decay_type == 2:
                logging_info("double decay")
            elif decay_type == -1:
                logging_info(f"gamma {self.gamma}")
                self.gamma = nn.Parameter(torch.ones(self.h, 1, self.dim) * gamma, requires_grad=False)
            else:
                logging_info(f"gamma {self.gamma}")
                self.gamma = nn.Parameter(torch.ones(self.h, 1, self.dim) * gamma, requires_grad=False)
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.lambda_ = gamma
            # self.gamma = nn.Parameter(torch.randn(self.h, 1, self.dim))
            self.gamma = nn.Parameter(3 + torch.randn(self.h, 1, self.dim))
            logging_info("learnable sigmoid")
            logging_info(f"self.lambda_ {self.lambda_}")

        if self.dpb_type == 4:
            self.dpb = DynamicPosBiasV4(dim=dpb_dim, outdim=self.h * self.dim, residual=residual, bias=bias, act=act)
        elif self.dpb_type == 5:
            self.dpb = DynamicPosBiasV5(dim=dpb_dim, outdim=self.h * self.dim, residual=residual, l=l, transform_type=transform_type, bias=bias, act=act)
        elif self.dpb_type == 6:
            self.dpb = DynamicPosBiasV6(dim=dpb_dim, outdim=self.h * self.dim, residual=residual, l=l, transform_type=transform_type, bias=bias, act=act)
        elif self.dpb_type == 7:
            self.dpb = DynamicPosBiasV7(dim=dpb_dim, outdim=self.h * self.dim, residual=residual, bias=bias, act=act)
        elif self.dpb_type == 8:
            self.dpb = DynamicPosBiasV8(dim=dpb_dim, outdim=self.h * self.dim, residual=residual, bias=bias, layers=layers, act=act, use_norm1=use_norm1, use_norm2=use_norm2)
        elif self.dpb_type == 9:
            self.dpb = DynamicPosBiasV9(dim=dpb_dim, outdim=self.h * self.dim, residual=residual, bias=bias, layers=layers, act=act, use_norm1=use_norm1, use_norm2=use_norm2, transform_type=transform_type, l=l)
        else:
            self.dpb = DynamicPosBiasV4(dim=dpb_dim, outdim=self.h * self.dim, residual=residual, bias=bias, act=act)

        if self.causal:
            self.forward = self.forward_causal
        else:
            self.forward = self.forward_non_causal
            
        self.act_type = act_type
        self.act_fun = get_activation_fn(self.act_type)

    def get_pos(self, n):
        if self.par_type == 1:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0
        elif self.par_type == 2:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n
        elif self.par_type == 3:
            index = torch.exp(torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n)
        
        return index
        
    def get_zero(self):
        index = torch.zeros(1).reshape(1, -1) * 1.0
        if self.par_type == 3:
            index = torch.exp(index)
            
        return index

    def get_neg(self, n):
        if self.causal:
            index = torch.ones(self.h * n * self.dim).reshape(self.h, n, self.dim) * self.zero_value
        else:
            if self.par_type == 1:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0
            elif self.par_type == 2:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0 / n

        return index
    
    def dpb_transform(self, x):
        # n, 1 -> n, (d * h)
        res = self.dpb(x)
        # n, (d * h) -> h, n, d
        res = rearrange(res, 'n (h d) -> h n d', h=self.h)

        return res
    
    def forward_causal(self, x, dim=-2, normalize=False):
        # shape of x: b, h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.dpb_transform(self.get_zero().to(x))
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, d, 1 -> h, n, d
            pos_dpb = self.dpb_transform(self.get_pos(l1).to(x))
            # padding to seq len
            pos = torch.cat([pos_dpb, torch.ones(self.h, l2, self.dim).to(x) * self.zero_value], dim=-2)
        else:
            pos = self.dpb_transform(self.get_pos(n - 1).to(x))

        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            if self.use_exp:
                gamma = torch.log(gamma) * coef
                pos = gamma + pos
            else:
                gamma = gamma ** coef
                pos = gamma * pos
        if self.use_exp:
            a = torch.exp(torch.clamp(torch.cat([zero, pos, zero], dim=1), max=30, min=-60))
        else:
            a = torch.cat([zero, pos, zero], dim=1)
            a = self.act_fun(a)

        # a = F.pad(a, (0, 0, 0, n - 1, 0, 0, ))
        # a: h, n, d
        # x: ..., h, n, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        #### for test
        # matrix = self.toeplizt_matrix(n)
        # res = torch.einsum('...nme,...me->...ne', matrix, x)
        # logging_info(torch.norm(res - output))
        ##### for test
        
    def forward_non_causal(self, x, dim=-2, normalize=False):
        # shape of x: b, h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.dpb_transform(self.get_zero().to(x))
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, d, 1 -> h, n, d
            pos_dpb = self.dpb_transform(self.get_pos(l1).to(x))
            neg_index = self.get_neg(l1).to(x)
            if self.causal:
                neg_dpb = neg_index
            else:
                neg_dpb = self.dpb_transform(neg_index)
            # padding to seq len
            pos = torch.cat([pos_dpb, torch.ones(self.h, l2, self.dim).to(x) * self.zero_value], dim=-2)
            neg = torch.cat([torch.ones(self.h, l2, self.dim).to(x) * self.zero_value, neg_dpb], dim=-2)
        else:
            pos = self.dpb_transform(self.get_pos(n - 1).to(x))
            neg_index = self.get_neg(n - 1).to(x)
            if self.causal:
                neg = neg_index
            else:
                neg = self.dpb_transform(neg_index)

        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)
            if not self.causal:
                neg = -torch.exp(neg)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            if self.use_exp:
                gamma = torch.log(gamma) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = gamma ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        if self.use_exp:
            a = torch.exp(torch.clamp(torch.cat([zero, pos, zero, neg], dim=1), max=30, min=-60))
        else:
            a = torch.cat([zero, pos, zero, neg], dim=1)
            a = self.act_fun(a)
        # a: h, n, d
        # x: ..., h, n, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
    
    def compute(self, x, a, dim, n):
        # x: b, h, n, d
        # a: h, n, d
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        v = torch.fft.rfft(a, 2 * n, dim=dim).unsqueeze(0)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

        return output

    def toeplizt_matrix(self, n):
        # c: first col, r: first row
        # 1, d, 1 -> h, 1, d
        zero = self.dpb_transform(self.get_zero())
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, d, 1 -> h, n, d
            pos_dpb = self.dpb_transform(self.get_pos(l1))
            neg_index = self.get_neg(l1)
            if self.causal:
                neg_dpb = neg_index
            else:
                neg_dpb = self.dpb_transform(neg_index)
            # padding to seq len
            pos = torch.cat([pos_dpb, torch.ones(self.h, l2, self.dim) * self.zero_value], dim=-2)
            neg = torch.cat([torch.ones(self.h, l2, self.dim) * self.zero_value, neg_dpb], dim=-2)
        else:
            pos = self.dpb_transform(self.get_pos(n - 1))
            neg_index = self.get_neg(n - 1)
            if self.causal:
                neg = neg_index
            else:
                neg = self.dpb_transform(neg_index)
                
        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)
            if not self.causal:
                neg = -torch.exp(neg)
                
        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            if self.use_exp:
                gamma = torch.log(gamma) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = gamma ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        if self.use_exp:
            c = torch.exp(torch.cat([zero, pos], dim=-2))
            r = torch.exp(torch.cat([zero, neg.flip(1)], dim=-2))
        else:
            zero = self.act_fun(zero)
            pos = self.act_fun(pos)
            if not self.causal:
                neg = self.act_fun(neg)
            c = torch.cat([zero, pos], dim=-2)
            r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        res = vals[:, j - i].reshape(self.h, n, n, -1)

        return res
