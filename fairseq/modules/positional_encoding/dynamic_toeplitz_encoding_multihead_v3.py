# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from einops import rearrange
from .dpb import DynamicPosBias

class DynamicToepliztMultiheadV3(nn.Module):
    def __init__(self, h, n, dim, dpb_dim, causal=False, use_exp=False, use_neg_exp=False, use_decay=False, use_multi_decay=False, residual=False, use_pad=False, par_type=1):
        super().__init__()
        self.h = h
        self.n = n
        self.dim = dim
        self.causal = causal
        self.use_exp = use_exp
        self.use_neg_exp = use_neg_exp
        self.use_pad = use_pad
        self.par_type = par_type
        if self.use_exp:
            self.zero_value = float("-inf")
        else:
            self.zero_value = 0

        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(torch.randn(self.h, 1, self.dim))
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.gamma = nn.Parameter(torch.randn(self.h, 1, self.dim))

        self.dpb = DynamicPosBias(dpb_dim, h, residual)

    def get_pos(self, n):
        if self.par_type == 1:
            index = torch.arange(1, 1 + n * self.dim).reshape(n, self.dim, -1) * 1.0 / (n * self.dim)
        
        return index
        
    def get_zero(self):
        if self.par_type == 1:
            index = torch.zeros(1 * self.dim).reshape(1, self.dim, -1) * 1.0
            
        return index

    def get_neg(self, n):
        if self.par_type == 1:
            if self.causal:
                index = torch.ones(self.h * n * self.dim).reshape(self.h, n, self.dim) * self.zero_value
            else:
                index = -torch.arange(1, 1 + n * self.dim).flip(0).reshape(n, self.dim, -1) * 1.0 / (n * self.dim)

        return index
    
    def dpb_transform(self, x):
        # n, d, 1 -> n, d, h
        res = self.dpb(x)
        # n, d, h -> h, n, d
        res = rearrange(res, 'n d h -> h n d')
        return res

    def forward(self, x, dim=-2, normalize=False):
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
            if self.use_exp:
                gamma = torch.log(torch.sigmoid(self.gamma)) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = torch.sigmoid(self.gamma) ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        if self.use_exp:
            a = torch.exp(torch.clamp(torch.cat([zero, pos, zero, neg], dim=1), max=30, min=-60))
        else:
            a = torch.cat([zero, pos, zero, neg], dim=1)
        # a: h, n, d
        # x: ..., h, n, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        ##### for test
        # matrix = self.toeplizt_matrix(n)
        # res = torch.einsum('...nme,...me->...ne', matrix, x)
        # print(torch.norm(res - output))
        ##### for test
    
    def compute(self, x, a, dim, n):
        # x: b, h, n, d
        # y: h, n, d
        y = torch.fft.rfft(x, 2 * n, dim=dim, norm="ortho")
        v = torch.fft.rfft(a, dim=dim).unsqueeze(0)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim, norm="ortho")[:, :, :n, :]

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
            if self.use_exp:
                gamma = torch.log(torch.sigmoid(self.gamma)) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = torch.sigmoid(self.gamma) ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        if self.use_exp:
            c = torch.exp(torch.cat([zero, pos], dim=-2))
            r = torch.exp(torch.cat([zero, neg.flip(1)], dim=-2))
        else:
            c = torch.cat([zero, pos], dim=-2)
            r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        res = vals[:, j - i].reshape(self.h, n, n, -1)

        return res
    