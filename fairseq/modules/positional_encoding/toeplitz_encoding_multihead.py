# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class ToepliztMultihead(nn.Module):
    def __init__(self, h, n, causal=False, use_exp=False, use_decay=False):
        super().__init__()
        self.h = h
        self.n = n
        self.causal = causal
        self.use_exp = use_exp
        if self.use_exp:
            self.zero_value = float("-inf")
        else:
            self.zero_value = 0
        # [1,...,(n-1)]
        self.pos = nn.Parameter(torch.ones(h, n - 1))
        # [0]
        self.zero = nn.Parameter(torch.ones(h, 1))
        # [-(n-1),...,-1]
        if self.causal:
            self.neg = nn.Parameter(torch.ones(h, n - 1) * self.zero_value, requires_grad=False)
        else:
            self.neg = nn.Parameter(torch.ones(h, n - 1))
            
        self.use_decay = use_decay
        if self.use_decay == 1:
            self.gamma = nn.Parameter(torch.zeros(1))
            
        # [1,...,(n-1)]
        self.pos = nn.Parameter(torch.rand(h, n - 1))
        # [0]
        self.zero = nn.Parameter(torch.rand(h, 1))
        # [-(n-1),...,-1]
        if self.causal:
            self.neg = nn.Parameter(torch.ones(h, n - 1) * self.zero_value, requires_grad=False)
        else:
            self.neg = nn.Parameter(torch.rand(h, n - 1))

    def forward(self, x, dim=-2, normalize=False):
        # shape of x: b h, n, e
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        # padding to seq len
        pos = torch.cat([self.pos[:, :l1], torch.ones(self.h, l2).to(x) * self.zero_value], dim=-1)
        neg = torch.cat([torch.ones(self.h, l2).to(x) * self.zero_value, self.neg[:, -l1:]], dim=-1)
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
            a = torch.exp(torch.clamp(torch.cat([self.zero, pos, self.zero, neg], dim=-1), max=30, min=-60))
        else:
            a = torch.cat([self.zero, pos, self.zero, neg], dim=-1)
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
        y = torch.fft.fft(x, 2 * n, dim=dim, norm="ortho")
        v = torch.fft.fft(a, dim=1).unsqueeze(0).unsqueeze(-1)
        u = v * y
        output = torch.fft.ifft(u, 2 * n, dim=dim, norm="ortho")[:, :, :n, :].real

        return output

    def toeplizt_matrix_exp(self, n):
        # c: first col, r: first row
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        pos = torch.clamp(torch.cat([self.pos[:, :l1], torch.ones(self.h, l2) * self.zero_value], dim=-1), max=30, min=-60)
        neg = torch.clamp(torch.cat([torch.ones(self.h, l2) * self.zero_value, self.neg[:, -l1:]], dim=-1), max=30, min=-60)
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
        c = torch.exp(torch.cat([self.zero, pos], dim=-1))
        r = torch.exp(torch.cat([self.zero, neg.flip(1)], dim=-1))
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-1)
        shape = self.h, c.shape[-1], r.shape[-1]
        i, j = torch.ones(*(shape[1:])).nonzero().T
        res = vals[:, j - i].reshape(*shape)

        return res
    
    def toeplizt_matrix_no_exp(self, n):
        # c: first col, r: first row
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        pos = torch.cat([self.pos[:, :l1], torch.ones(self.h, l2) * self.zero_value], dim=-1)
        neg = torch.cat([torch.ones(self.h, l2) * self.zero_value, self.neg[-l1:]], dim=-1)
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
        c = torch.cat([self.zero, pos], dim=-1)
        r = torch.cat([self.zero, neg.flip(1)], dim=-1)
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
# x = torch.rand(b, h, n, e)
# model = ToepliztMultihead(h, 100, causal=True)
# y = model.forward(x, dim=-2)
# model = ToepliztMultihead(h, 100, causal=True, use_decay=True)
# y = model.forward(x, dim=-2)
# model = ToepliztMultihead(h, 100, causal=True, use_exp=True)
# y = model.forward(x, dim=-2)
# model = ToepliztMultihead(h, 100, causal=True, use_exp=True, use_decay=True)
# y = model.forward(x, dim=-2)