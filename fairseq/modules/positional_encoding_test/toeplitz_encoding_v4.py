# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..helpers import print_params


class ToepliztV4(nn.Module):
    def __init__(
        self, 
        h, 
        n, 
        dim, 
        causal=False, 
        use_exp=False, 
        use_neg_exp=False, 
        use_decay=False, 
        use_multi_decay=False, 
        gamma=0.999,
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
        if self.use_exp:
            self.zero_value = float("-inf")
        else:
            self.zero_value = 0

        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(torch.ones(self.h, 1, self.dim) * gamma)
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.gamma = nn.Parameter(torch.randn(self.h, 1, self.dim))

        # [1,...,n-1]
        self.pos = nn.Parameter(torch.randn(self.h, self.n - 1, self.dim))
        # [0]
        self.zero = nn.Parameter(torch.randn(self.h, 1, self.dim))
        # [-(n-1),...,-1]
        if self.causal:
            self.neg = nn.Parameter(torch.ones(self.h, self.n - 1, self.dim) * self.zero_value, requires_grad=False)
        else:
            self.neg = nn.Parameter(torch.randn(self.h, self.n - 1, self.dim))

        if self.causal:
            self.forward = self.forward_causal
        else:
            self.forward = self.forward_non_causal
    
    def forward_causal(self, x, dim=-2, normalize=False):
        # shape of x: b, h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # h, 1, d
        zero = self.zero
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        # padding to seq len
        pos = torch.cat([self.pos[:, :l1, :], torch.ones(self.h, l2, self.dim).to(x) * self.zero_value], dim=-2)

        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
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

        # a: h, n, d
        # x: ..., h, n, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        
    def forward_non_causal(self, x, dim=-2, normalize=False):
        # shape of x: b, h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # h, 1, d
        zero = self.zero
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        # padding to seq len
        pos = torch.cat([self.pos[:, :l1, :], torch.ones(self.h, l2, self.dim).to(x) * self.zero_value], dim=-2)
        neg = torch.cat([torch.ones(self.h, l2, self.dim).to(x) * self.zero_value, self.neg[:, :l1, :]], dim=-2)

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
        # h, 1, d
        zero = self.zero
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        # padding to seq len
        pos = torch.cat([self.pos[:, :l1, :], torch.ones(self.h, l2, self.dim) * self.zero_value], dim=-2)
        neg = torch.cat([torch.ones(self.h, l2, self.dim) * self.zero_value, self.neg[:, :l1, :]], dim=-2)
        
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
            c = torch.cat([zero, pos], dim=-2)
            r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        res = vals[:, j - i].reshape(self.h, n, n, -1)

        return res
