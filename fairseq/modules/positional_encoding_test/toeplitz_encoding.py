# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

from ..helpers import print_params


class Toeplizt(nn.Module):
    def __init__(self, n, type_num=-1, causal=False):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.type_num = type_num
        self.n = n
        self.causal = causal
        if self.type_num == -1:
            # [1,...,n-1]
            self.pos = nn.Parameter(torch.ones(n - 1))
            # [0]
            self.zero = nn.Parameter(torch.ones(1))
            # [-(n-1),...,-1]
            if self.causal:
                self.neg = nn.Parameter(torch.zeros(n - 1), requires_grad=False)
            else:
                self.neg = nn.Parameter(torch.ones(n - 1))
            # # [1,...,n-1]
            # self.pos = nn.Parameter(torch.arange(1, n).float())
            # # [0]
            # self.zero = nn.Parameter(torch.zeros(1))
            # # [-(n-1),...,-1]
            # self.neg = nn.Parameter(-torch.arange(n, 0, -1).float())
        elif self.type_num == 1:
            # exp(-|i - j|)
            # [1,...,n-1]
            vec = torch.clamp(-torch.ones(n - 1).float(), max=30, min=-60)
            self.pos = nn.Parameter(torch.exp(vec), requires_grad=False)
            # [0]
            self.zero = nn.Parameter(torch.ones(1), requires_grad=False)
            # [-(n-1),...,-1]
            vec = torch.clamp(-torch.arange(n, 0, -1).float(), max=30, min=-60)
            if self.causal:
                self.neg = nn.Parameter(torch.zeros(n - 1), requires_grad=False)
            else:
                self.neg = nn.Parameter(torch.exp(vec), requires_grad=False)


    def forward(self, x):
        # shape of x: b, n, e
        b, n, e = x.shape
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        a = torch.cat([self.zero, self.pos[:n-1], self.zero, self.neg[-(n-1):]])
        
        y = torch.fft.fft(x, 2 * n, dim=1, norm="ortho")
        v = torch.fft.fft(a).unsqueeze(0).unsqueeze(-1)
        u = v * y
        output = torch.fft.ifft(u, 2 * n, dim=1, norm="ortho")[:, :n, :].real

        return output

    def toeplizt_matrix(self, n):
        # c: first col, r: first row
        c = torch.cat([self.zero, self.pos[:n-1]])
        r = torch.cat([self.zero, self.neg[-(n-1):].flip(0)])
        vals = torch.cat([r, c[1:].flip(0)])
        shape = len(c), len(r)
        i, j = torch.ones(*shape).nonzero().T
        res = vals[j - i].reshape(*shape)

        return res
