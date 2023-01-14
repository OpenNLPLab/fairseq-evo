# https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html
# https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch
# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..helpers import get_activation_fn, print_params
from ..others import ActLayer


class Ctno(nn.Module):
    def __init__(
        self, 
        h, # num of heads
        dim, # dim per heads
        k, # number of cos components
        causal=False, 
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.h = h
        self.dim = dim
        self.causal = causal
        self.coef = nn.Parameter(torch.randn(h, 1, k // 2, dim), requires_grad=True)
        
    def forward(self, x, decay, cos, index):
        # x: ..., h, n, d
        # causal:
        # decay: 1, n, 1; lambda ^ (0, 1, ..., n - 1, 0, -(n-1), ... , -1)
        # cos: 1, n, k, 1
        # non causal:
        # decay: 1, 2n - 1, 1; lambda ^ (0, 1, ..., n - 1, 0, -(n-1), ... , -1)
        # cos: 1, 2n - 1, k, 1
        n = x.shape[-2]
        # (h, 1, k, d), (1, n, k, 1) -> (h, n, k, d) -> (h, n, d)
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        a = decay * torch.sum(self.coef * cos, dim=-2)
        # x: ..., h, n, d
        # a: h, n, d
        output = self.compute(x, a, n, index)

        return output
        
    def compute(self, x, a, n, index):
        # x: ..., h, n, d
        # a: h, n, d
        s = len(x.shape)
        t = len(a.shape)
        for i in range(s - t):
            a = a.unsqueeze(0)
        y = torch.fft.rfft(x, 2 * n, dim=-2)
        v = torch.fft.rfft(a, 2 * n, dim=-2)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=-2)
        output = torch.index_select(output, -2, index)

        return output

    def mm(self, x, decay, cos):
        # shape of x: ..., h, n, d
        n = x.shape[-2]
        # (h, 1, k, d), (1, n, k, 1) -> (h, n, k, d) -> (h, n, d)
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        a = decay * torch.sum(self.coef * cos, dim=-2)
        zero = a[:, 0, None]
        pos = a[:, 1: n]
        if self.causal:
            neg = torch.zeros_like(pos)
        else:
            neg = a[:, n + 1:]
        c = torch.cat([zero, pos], dim=-2)
        r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        T = vals[:, j - i].reshape(self.h, n, n, -1)

        res = torch.einsum('h n m d, b h m d -> b h n d', T, x)
        
        return res, T

