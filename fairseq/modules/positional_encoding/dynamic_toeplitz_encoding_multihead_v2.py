# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from .dpb_v2 import DynamicPosBiasV2

class DynamicToepliztMultiheadV2(nn.Module):
    def __init__(self, h, n, d, causal=False, use_exp=False, use_neg_exp=False, use_decay=False, use_multi_decay=False, residual=False, act="relu", use_pad=False):
        super().__init__()
        self.h = h
        self.n = n
        self.causal = causal
        self.use_exp = use_exp
        self.use_neg_exp = use_neg_exp
        self.use_pad = use_pad
        if self.use_exp:
            self.zero_value = float("-inf")
        else:
            self.zero_value = 0

        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(torch.ones(1) * 10)
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.gamma = nn.Parameter(torch.randn(self.h, 1))

        self.dpb = DynamicPosBiasV2(d, h, residual, act)
        
    def get_pos(self, n):
        tmp = torch.arange(1, n + 1).reshape(-1, 1) * 1.0
        return tmp
        # return torch.arange(1, n).reshape(-1, 1) * 1.0
        
    def get_zero(self):
        return torch.zeros(1).reshape(-1, 1) * 1.0

    def get_neg(self, n):
        if self.causal:
            res = torch.ones(n, self.h) * self.zero_value
        else:
            res = -torch.arange(1, n + 1).flip(0).reshape(-1, 1) * 1.0

        return res

    def forward(self, x, dim=-2, normalize=False):
        # shape of x: b h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        zero = self.dpb(self.get_zero().to(x)).transpose(0, 1)
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, 1 -> n, h -> h, n
            pos_dpb = self.dpb(self.get_pos(l1).to(x)).transpose(0, 1)
            neg_index = self.get_neg(l1).to(x)
            if self.causal:
                neg_dpb = neg_index.transpose(0, 1)
            else:
                neg_dpb = self.dpb(neg_index).transpose(0, 1)
            # padding to seq len
            pos = torch.cat([pos_dpb, torch.ones(self.h, l2).to(x) * self.zero_value], dim=-1)
            neg = torch.cat([torch.ones(self.h, l2).to(x) * self.zero_value, neg_dpb], dim=-1)
        else:
            pos = self.dpb(self.get_pos(n - 1)).to(x).transpose(0, 1)
            neg_index = self.get_neg(n - 1).to(x)
            if self.causal:
                neg = neg_index.transpose(0, 1)
            else:
                neg = self.dpb(neg_index).transpose(0, 1)
        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)
            if not self.causal:
                neg = -torch.exp(neg)
        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1).to(x)
            if self.use_exp:
                gamma = torch.log(torch.sigmoid(self.gamma)) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = torch.sigmoid(self.gamma) ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        if self.use_exp:
            a = torch.exp(torch.clamp(torch.cat([zero, pos, zero, neg], dim=-1), max=30, min=-60))
        else:
            a = torch.cat([zero, pos, zero, neg], dim=-1)
        # h, n
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
        ##### for test
        # no exp
        # matrix = self.toeplizt_matrix_no_exp(n)
        # res = torch.einsum('...nm,...me->...ne', matrix, x)
        # print(torch.norm(res - output))
        # size = list(x.shape[:-1]) + [1]
        # ones = torch.ones(size).to(x)
        # denorm = self.compute(ones, a, dim, n)
        # print(torch.sum(matrix / denorm, dim=-1))
        # exp
        # matrix = self.toeplizt_matrix_exp(n)
        # res = torch.einsum('...nm,...me->...ne', matrix, x)
        # print(torch.norm(res - output))
        # size = list(x.shape[:-1]) + [1]
        # ones = torch.ones(size).to(x)
        # denorm = self.compute(ones, a, dim, n)
        # print(torch.sum(matrix / denorm, dim=-1))
        ##### for test
    
    def compute(self, x, a, dim, n):
        # x: b, h, n, 1
        # y: h, n
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        v = torch.fft.rfft(a, dim=-1).unsqueeze(0).unsqueeze(-1)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

        return output

    def toeplizt_matrix_exp(self, n):
        # c: first col, r: first row
        zero = self.dpb(self.get_zero()).transpose(0, 1)
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, 1 -> n, h -> h, n
            pos_dpb = self.dpb(self.get_pos(l1)).transpose(0, 1)
            neg_index = self.get_neg(l1)
            if self.causal:
                neg_dpb = neg_index.transpose(0, 1)
            else:
                neg_dpb = self.dpb(neg_index).transpose(0, 1)
            # padding to seq len
            pos = torch.cat([pos_dpb, torch.ones(self.h, l2) * self.zero_value], dim=-1)
            neg = torch.cat([torch.ones(self.h, l2) * self.zero_value, neg_dpb], dim=-1)
        else:
            pos = self.dpb(self.get_pos(n - 1)).transpose(0, 1)
            neg_index = self.get_neg(n - 1)
            if self.causal:
                neg = neg_index.transpose(0, 1)
            else:
                neg = self.dpb(neg_index).transpose(0, 1)
        if self.use_exp and self.use_neg_exp:
            zero = -torch.exp(zero)
            pos = -torch.exp(pos)
            if not self.causal:
                neg = -torch.exp(neg)
        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1)
            if self.use_exp:
                gamma = torch.log(torch.sigmoid(self.gamma)) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = torch.sigmoid(self.gamma) ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        c = torch.exp(torch.cat([zero, pos], dim=-1))
        r = torch.exp(torch.cat([zero, neg.flip(1)], dim=-1))
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-1)
        shape = self.h, c.shape[-1], r.shape[-1]
        i, j = torch.ones(*(shape[1:])).nonzero().T
        res = vals[:, j - i].reshape(*shape)

        return res
    
    def toeplizt_matrix_no_exp(self, n):
        # c: first col, r: first row
        zero = self.dpb(self.get_zero()).transpose(0, 1)
        if self.use_pad:
            l1 = min(n - 1, self.n - 1)
            l2 = max(0, n - 1 - l1)
            # n, 1 -> n, h -> h, n
            pos_dpb = self.dpb(self.get_pos(l1)).transpose(0, 1)
            neg_index = self.get_neg(l1)
            if self.causal:
                neg_dpb = neg_index.transpose(0, 1)
            else:
                neg_dpb = self.dpb(neg_index).transpose(0, 1)
            # padding to seq len
            pos = torch.cat([pos_dpb, torch.ones(self.h, l2) * self.zero_value], dim=-1)
            neg = torch.cat([torch.ones(self.h, l2) * self.zero_value, neg_dpb], dim=-1)
        else:
            pos = self.dpb(self.get_pos(n - 1)).transpose(0, 1)
            neg_index = self.get_neg(n - 1)
            if self.causal:
                neg = neg_index.transpose(0, 1)
            else:
                neg = self.dpb(neg_index).transpose(0, 1)
        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1)
            if self.use_exp:
                gamma = torch.log(torch.sigmoid(self.gamma)) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = torch.sigmoid(self.gamma) ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        c = torch.cat([zero, pos], dim=-1)
        r = torch.cat([zero, neg.flip(1)], dim=-1)
        if self.use_neg_exp:
            c = -c
            r = -r
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-1)
        shape = self.h, c.shape[-1], r.shape[-1]
        i, j = torch.ones(*(shape[1:])).nonzero().T
        res = vals[:, j - i].reshape(*shape)

        return res