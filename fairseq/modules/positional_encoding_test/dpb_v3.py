# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn

from ..helpers import print_params
from ..norm import SimpleRMSNorm


class DynamicPosBiasV3(nn.Module):
    def __init__(self, dim, num_heads, act="silu", bias=True):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.num_heads = num_heads
        self.pos_dim = dim // 8
        self.act = act
        self.pos_proj = nn.Linear(1, self.pos_dim, bias=bias)
        self.pos1 = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
        )
        self.pos2 = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.num_heads, bias=bias)
        )
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, biases):
        pos = self.pos2(self.pos1(self.pos_proj(biases)))
        
        return pos
    
class SimpleDynamicPosBias(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.index = nn.Parameter(torch.arange(1, self.num_heads + 1).reshape(1, -1), requires_grad=False)

    def forward(self, biases):
        res = biases * self.index
        return res
