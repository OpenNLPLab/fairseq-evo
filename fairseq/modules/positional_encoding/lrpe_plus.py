# - https://github.com/zh217/torch-dct
# - https://github.com/zh217/torch-dct/issues/15

import logging
import math

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from fairseq.modules import print_params

from einops import repeat, rearrange

# from alibi
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

class Lrpe_plus(nn.Module):
    def __init__(
        self, 
        core_matrix, 
        p_matrix, 
        embedding_dim=64, 
        num_heads=12,
        dims=[-2],
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dims = dims
        self.index = torch.empty(0)
        self.max_len = 0

        if self.core_matrix == 1:
            d = embedding_dim
            # self.ratio = nn.Parameter((torch.arange(self.num_heads) / self.num_heads * 3 + 2).reshape(-1, 1, 1), requires_grad=True)
            # ratio = torch.exp(-torch.Tensor(get_slopes(self.num_heads)).reshape(-1, 1, 1))
            # self.ratio = nn.Parameter(ratio, requires_grad=False)
            self.ratio = nn.Parameter(torch.sigmoid(torch.arange(self.num_heads) / self.num_heads * 3 + 2).reshape(-1, 1, 1), requires_grad=False)
            theta = 10000 ** (-2 / d * torch.arange(d))
            theta = repeat(theta, 'd -> h n d', h=self.num_heads, n=1)
            self.theta = nn.Parameter(theta, requires_grad=True)
            logging.info("Core: Diag")
        elif self.core_matrix == 2:
            d = embedding_dim // 2
            # self.ratio = nn.Parameter((torch.arange(self.num_heads) / self.num_heads * 3 + 2).reshape(-1, 1, 1), requires_grad=True)
            self.ratio = nn.Parameter(torch.sigmoid(torch.arange(self.num_heads) / self.num_heads * 3 + 2).reshape(-1, 1, 1), requires_grad=False)
            # ratio = torch.exp(-torch.Tensor(get_slopes(self.num_heads)).reshape(-1, 1, 1))
            # self.ratio = nn.Parameter(ratio, requires_grad=False)
            theta = 10000 ** (-2 / d * torch.arange(d))
            # h, 1, d / 2, 1, 1
            theta = repeat(theta, 'd -> h n d', h=self.num_heads, n=1).unsqueeze(-1).unsqueeze(-1)
            self.theta = nn.Parameter(theta, requires_grad=True)
            # 2, 2
            self.identity = nn.Parameter(torch.eye(2), requires_grad=False)
            # 2, 2
            array = [[0, 1], [0, 0]]
            self.cycle1 = nn.Parameter(torch.Tensor(array), requires_grad=False)
            array = [[0, 0], [1, 0]]
            self.cycle2 = nn.Parameter(torch.Tensor(array), requires_grad=False)
            logging.info("Core: Block 2")

        if self.p_matrix == 1:
            logging.info("P: Identity")
        elif self.p_matrix == 2:
            logging.info("P: Diag")
        # elif self.p_matrix == 3:
        #     logging.info("P: Unitary")

        self.p_transform = self.get_p()
        self.core_transform = self.get_core_transform()

    def forward(self, x, is_q=True):
        n = x.shape[-2]
        if n > self.max_len:
            self.max_len = n
            self.index = torch.arange(1, n + 1).reshape(-1, 1).to(x)
        # ..., h, l, d
        x = self.p_transform(x, is_q=is_q)
        for dim in self.dims:
            x = self.core_transform(x, dim, is_q=is_q)
        return x

    def get_p(self):
        if self.p_matrix == 1:
            def f(x, is_q=True):
                return x
            return f
        elif self.p_matrix == 2:
            self.p = nn.Parameter(torch.randn(self.num_heads, 1, self.embedding_dim)) 
            return self.p_diag
        # elif self.p_matrix == 3:
        #     self.unitary = nn.Parameter(torch.zeros(self.h, self.embedding_dim, self.embedding_dim), requires_grad=False) 
        #     self.p = nn.Parameter(torch.randn(self.h, 1, self.embedding_dim)) 
        #     return self.p_unitary
        
    def p_diag(self, x, is_q=True, eps=1e-3):
        if is_q:
            return x * (torch.exp(self.p) + eps)
        else:
            return x / (torch.exp(self.p) + eps)
        
    # def p_unitary(self, x, is_q=True, eps=1e-3):
    #     # h, 1, d
    #     p = F.normalize(p)
    #     # h, 1, 1
    #     cos = p[:, :, :1]
    #     # h, 1, d - 1
    #     y = p[:, :, 1:]
    #     # h, 1, 1
    #     sin = torch.norm(y, dim=-1, keepdim=True)
    #     y /= sin
    #     self.unitary[:, :1, :1] = cos - 1
    #     self.unitary[:, :1, 1:] = -sin * y
    #     self.unitary[:, 1:, :1] = sin * y.transpose(2, 1)
    #     self.unitary[:, 1:, 1:] = (cos - 1) * y * y.transpose(2, 1)
        
        
    #     p_normlize = p / norm
        

    def get_core_transform(self):
        if self.core_matrix == 1:
            return self.diag
        elif self.core_matrix == 2:
            return self.block_2

    def diag(self, x, dim=-2, is_q=True, eps=1e-3):
        # assume the input has dim : ..., h, n, d
        if dim < 0:
            dim += len(x.shape)
        n = x.shape[dim]
        m = len(x.shape)
        if is_q:
            theta = self.theta
            # ratio = torch.sigmoid(self.ratio) + eps
            ratio = self.ratio
        else:
            theta = -self.theta
            # ratio = 1 / (torch.sigmoid(self.ratio) + eps)
            ratio = 1 / self.ratio
        # h, 1, d
        theta = self.theta
        # 调整为相同形状
        # ..., h, 1, d
        for _ in range(m - 3):
            theta = theta.unsqueeze(0)
        # n, 1
        index = self.index[:, :n]
        # ..., n, 1
        m = len(x.shape)
        for _ in range(dim):
            index = index.unsqueeze(0)
        # ..., h, n, d
        phi = theta * index
        r1 = ratio ** index
        
        real = r1 * torch.cos(phi)
        imag = r1 * torch.sin(phi)
               
        return torch.complex(x * real, x * imag)

    # https://stackoverflow.com/questions/63855692/matrix-multiplication-for-complex-numbers-in-pytorch
    def element_wise_complex(self, t1, t2):
        return torch.complex(t1.real * t2.real - t1.imag * t2.imag, t1.real * t2.imag + t1.imag * t2.real)

    def element_wise_complex_real(self, t1, t2):
        return torch.complex(t1 * t2.real, t1 * t2.imag)
    
    def block_2(self, x, dim=-2, is_q=True, eps=1e-3):
        # assume the input has dim : ..., h, n, d
        if dim < 0:
            dim += len(x.shape)
        n = x.shape[dim]
        m = len(x.shape)
        if is_q:
            theta = self.theta
            # ratio = torch.sigmoid(self.ratio) + eps
            ratio = self.ratio
        else:
            theta = -self.theta
            # ratio = 1 / (torch.sigmoid(self.ratio) + eps)
            ratio = 1 / self.ratio
        # theta h, 1, d
        # ..., h, 1, d / 2, 1, 1
        for _ in range(m - 3):
            theta = theta.unsqueeze(0)
        l = len(theta.shape)
        identity = self.identity
        if is_q:
            cycle = self.cycle1
        else:
            cycle = self.cycle2
        for _ in range(l - 2):
            identity = identity.unsqueeze(0)
            cycle = cycle.unsqueeze(0)
        # n, 1
        index = self.index[:, :n]
        # ..., n, 1
        m = len(x.shape)
        for _ in range(m - 2):
            index = index.unsqueeze(0)
        # ..., n, 1, 1, 1
        for _ in range(2):
            index = index.unsqueeze(-1)
        # ..., h, 1, d / 2, 1, 1
        # n * theta
        diag = theta * index
        # (n - 1) * theta
        corner = diag / theta
        # ..., h, 1, 1
        for _ in range(m - 3):
            ratio = ratio.unsqueeze(0)
        for _ in range(2):
            ratio = ratio.unsqueeze(-1)
        if is_q:
            # r ^ n
            r1 = ratio ** index
            # n * r ^ (n - 1)
            r2 = r1 / ratio * index
        else:
            # (1 / r) ^ n
            r1 = ratio ** index
            # - n * (1 / r) ^ (n + 1)
            r2 = -r1 * ratio * index

        # ..., h, 1, d / 2, 2, 2
        real = r1 * torch.cos(diag) * identity + \
               r2 * torch.cos(corner) * cycle
        imag = r1 * torch.sin(diag) * identity + \
               r2 * torch.sin(corner) * cycle

        # ..., h, n, d -> ..., h, n, d / 2, 2
        x = rearrange(x, '... (e n) -> ... e n', n=2)
        out_real = torch.einsum('... d, ... d e -> ... e', x, real)
        out_imag = torch.einsum('... d, ... d e -> ... e', x, imag)
        
        out_real, out_imag = map(lambda x: rearrange(x, '... e n -> ... (e n)', n=2), [out_real, out_imag])

        return torch.complex(out_real, out_imag)