# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

from ..helpers import print_params


class ToepliztV3(nn.Module):
    def __init__(self, n, causal=False, use_exp=False):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.n = n
        self.causal = causal
        self.use_exp = use_exp
        if self.use_exp:
            self.zero_value = float("-inf")
        else:
            self.zero_value = 0
        # [1,...,n-1]
        self.pos = nn.Parameter(torch.ones(n - 1))
        # [0]
        self.zero = nn.Parameter(torch.ones(1))
        # [-(n-1),...,-1]
        if self.causal:
            self.neg = nn.Parameter(torch.ones(n - 1) * self.zero_value, requires_grad=False)
        else:
            self.neg = nn.Parameter(torch.ones(n - 1))
            
        # [1,...,n-1]
        # self.pos = nn.Parameter(torch.arange(1, n).float())
        # # [0]
        # self.zero = nn.Parameter(torch.zeros(1))
        # # [-(n-1),...,-1]
        # if self.causal:
        #     self.neg = nn.Parameter(torch.arange(n, 0, -1).float() * self.zero_value, requires_grad=False)
        # else:
        #     self.neg = nn.Parameter(torch.arange(n, 0, -1).float())
        
        # # [1,...,n-1]
        # self.pos = nn.Parameter(torch.rand(n - 1))
        # # [0]
        # self.zero = nn.Parameter(torch.rand(1))
        # # [-(n-1),...,-1]
        # if self.causal:
        #     self.neg = nn.Parameter(torch.ones(n - 1).float() * self.zero_value, requires_grad=False)
        # else:
        #     self.neg = nn.Parameter(torch.rand(n - 1))

    def forward(self, x, dim=-2, normalize=False):
        # shape of x: b, n, e
        b = x.shape[0]
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        # padding to seq len
        pos = torch.cat([self.pos[:l1], torch.ones(l2).to(x) * self.zero_value])
        neg = torch.cat([torch.ones(l2).to(x) * self.zero_value, self.neg[-l1:]])
        if self.use_exp:
            a = torch.exp(torch.clamp(torch.cat([self.zero, pos, self.zero, neg]), max=30, min=-60))
        else:
            a = torch.cat([self.zero, pos, self.zero, neg])
        # 1, 1, n, 1
        a = a.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        output = self.compute(x, a, dim, n)

        if normalize:
            # ones = torch.ones(b, n, 1).to(x)
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
    
    def compute(self, x, a, dim, n):
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        v = torch.fft.rfft(a, 2 * n, dim=dim)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

        return output

    def toeplizt_matrix_exp(self, n):
        # c: first col, r: first row
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        pos = torch.clamp(torch.cat([self.pos[:l1], torch.ones(l2) * self.zero_value]), max=30, min=-60)
        neg = torch.clamp(torch.cat([torch.ones(l2) * self.zero_value, self.neg[-l1:]]), max=30, min=-60)
        c = torch.exp(torch.cat([self.zero, pos]))
        r = torch.exp(torch.cat([self.zero, neg.flip(0)]))
        vals = torch.cat([r, c[1:].flip(0)])
        shape = len(c), len(r)
        i, j = torch.ones(*shape).nonzero().T
        res = vals[j - i].reshape(*shape)

        return res
    
    def toeplizt_matrix_no_exp(self, n):
        # c: first col, r: first row
        l1 = min(n - 1, self.n - 1)
        l2 = max(0, n - 1 - l1)
        pos = torch.cat([self.pos[:l1], torch.ones(l2) * self.zero_value])
        neg = torch.cat([torch.ones(l2) * self.zero_value, self.neg[-l1:]])
        c = torch.cat([self.zero, pos])
        r = torch.cat([self.zero, neg.flip(0)])
        vals = torch.cat([r, c[1:].flip(0)])
        shape = len(c), len(r)
        i, j = torch.ones(*shape).nonzero().T
        res = vals[j - i].reshape(*shape)

        return res
