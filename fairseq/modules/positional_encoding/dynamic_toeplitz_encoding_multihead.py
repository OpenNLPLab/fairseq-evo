# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from .dpb import DynamicPosBias

class DynamicToepliztMultihead(nn.Module):
    def __init__(self, h, n, d, causal=False, use_exp=False, use_decay=False, use_multi_decay=False, residual=False):
        super().__init__()
        self.h = h
        self.n = n
        self.causal = causal
        self.use_exp = use_exp
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
            
        # [1,...,(n-1)]
        self.pos = nn.Parameter(torch.arange(1, n).reshape(-1, 1) * 1.0, requires_grad=False)
        # [0]
        self.zero = nn.Parameter(torch.zeros(1).reshape(-1, 1) * 1.0, requires_grad=False)
        # [-(n-1),...,-1]
        if self.causal:
            self.neg = nn.Parameter(torch.ones(h, n - 1) * self.zero_value, requires_grad=False)
        else:
            self.neg = nn.Parameter(-torch.arange(1, n).flip(0).reshape(-1, 1) * 1.0, requires_grad=False)
            
        self.dpb = DynamicPosBias(d, h, residual)

    def forward(self, x, dim=-2, normalize=False):
        # shape of x: b h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        # n, 1 -> n, h -> h, n
        pos_dpb = self.dpb(self.pos[:l1]).transpose(0, 1)
        if self.causal:
            neg_dpb = self.neg[:, -l1:]
        else:
            neg_dpb = self.dpb(self.pos[-l1:]).transpose(0, 1)
        zero_dpb = self.dpb(self.zero).transpose(0, 1)
        # padding to seq len
        pos = torch.cat([pos_dpb, torch.ones(self.h, l2).to(x) * self.zero_value], dim=-1)
        neg = torch.cat([torch.ones(self.h, l2).to(x) * self.zero_value, neg_dpb], dim=-1)
        if self.use_decay:
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
            a = torch.exp(torch.clamp(torch.cat([zero_dpb, pos, zero_dpb, neg], dim=-1), max=30, min=-60))
        else:
            a = torch.cat([zero_dpb, pos, zero_dpb, neg], dim=-1)
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
        y = torch.fft.rfft(x, 2 * n, dim=dim, norm="ortho")
        v = torch.fft.rfft(a, dim=1).unsqueeze(0).unsqueeze(-1)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim, norm="ortho")[:, :, :n, :]

        return output

    def toeplizt_matrix_exp(self, n):
        # c: first col, r: first row
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        # n, 1 -> n, h -> h, n
        pos_dpb = self.dpb(self.pos[:l1]).transpose(0, 1)
        if self.causal:
            neg_dpb = self.neg[:, -l1:]
        else:
            neg_dpb = self.dpb(self.pos[-l1:]).transpose(0, 1)
        zero_dpb = self.dpb(self.zero).transpose(0, 1)
        # padding to seq len
        pos = torch.cat([pos_dpb, torch.ones(self.h, l2).to(x) * self.zero_value], dim=-1)
        neg = torch.cat([torch.ones(self.h, l2).to(x) * self.zero_value, neg_dpb], dim=-1)
        if self.use_decay:
            coef = torch.arange(1, n).reshape(1, -1).to(x)
            if self.use_exp:
                gamma = torch.log(torch.sigmoid(self.gamma)) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = torch.sigmoid(self.gamma) ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        c = torch.exp(torch.cat([zero_dpb, pos], dim=-1))
        r = torch.exp(torch.cat([zero_dpb, neg.flip(1)], dim=-1))
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-1)
        shape = self.h, c.shape[-1], r.shape[-1]
        i, j = torch.ones(*(shape[1:])).nonzero().T
        res = vals[:, j - i].reshape(*shape)

        return res
    
    def toeplizt_matrix_no_exp(self, n):
        # c: first col, r: first row
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        # n, 1 -> n, h -> h, n
        pos_dpb = self.dpb(self.pos[:l1]).transpose(0, 1)
        if self.causal:
            neg_dpb = self.neg[:, -l1:]
        else:
            neg_dpb = self.dpb(self.pos[-l1:]).transpose(0, 1)
        zero_dpb = self.dpb(self.zero).transpose(0, 1)
        # padding to seq len
        pos = torch.cat([pos_dpb, torch.ones(self.h, l2).to(x) * self.zero_value], dim=-1)
        neg = torch.cat([torch.ones(self.h, l2).to(x) * self.zero_value, neg_dpb], dim=-1)
        if self.use_decay:
            coef = torch.arange(1, n).reshape(1, -1).to(x)
            if self.use_exp:
                gamma = torch.log(torch.sigmoid(self.gamma)) * coef
                pos = gamma + pos
                neg = torch.flip(gamma, dims=[1]) + neg
            else:
                gamma = torch.sigmoid(self.gamma) ** coef
                pos = gamma * pos
                neg = torch.flip(gamma, dims=[1]) * neg
        c = torch.cat([zero_dpb, pos], dim=-1)
        r = torch.cat([zero_dpb, neg.flip(1)], dim=-1)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-1)
        shape = self.h, c.shape[-1], r.shape[-1]
        i, j = torch.ones(*(shape[1:])).nonzero().T
        res = vals[:, j - i].reshape(*shape)

        return res
    
# multi head
# h = 8
# b = 1
# n = 200
# e = 4
# d = 16
# x = torch.rand(b, h, n, e)
# model = DynamicToepliztMultihead(h, 100, d, causal=True)
# y = model.forward(x, dim=-2)
# model = DynamicToepliztMultihead(h, 100, d, causal=True, use_exp=True)
# y = model.forward(x, dim=-2)