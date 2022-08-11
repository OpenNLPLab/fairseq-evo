# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn
import numpy as np
from ..norm import SimpleRMSNorm

class DynamicPosBiasV6(nn.Module):
    def __init__(self, dim, outdim, residual, act="relu", l=10, transform_type=1, bias=True):
        super().__init__()
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.l = l
        if self.l == -1:
            print("scalar")
            d = 1
        else:
            d = 2 * self.l
            self.transform_type = transform_type
            if self.transform_type == 1:
                print("attention pe")
                self.freq = nn.Parameter(1. / (10000 ** (torch.arange(0, d, 2).float().reshape(1, -1) / d)), requires_grad=False)
            else:
                b = self.get_b()
                print(f"b: {b}")
                self.freq = nn.Parameter(np.pi * (b ** torch.arange(l).reshape(1, -1)), requires_grad=False)
        self.pos_proj = nn.Linear(d, self.pos_dim, bias=bias)
        self.pos = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.outdim, bias=bias)
        )
        
    def get_b(self):
        if self.transform_type == 2:
            return np.exp(-1)
        elif self.transform_type == 3:
            return 2
        elif self.transform_type == 4:
            return 1 / 2
        
    def transform(self, x):
        if self.l == -1:
            res = x
        else:
            x = x * self.freq
            res = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            
        return res
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, x):
        pos = self.transform(x)
        pos = self.pos_proj(pos)
        res = self.pos(pos)
        
        return res