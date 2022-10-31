# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import numpy as np
import torch
import torch.nn as nn

from ..helpers import print_params
from ..norm import SimpleRMSNorm


class DynamicPosBiasV5(nn.Module):
    def __init__(self, dim, outdim, residual, act="relu", l=10, transform_type=1, bias=True):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.l = l
        self.transform_type = transform_type
        if self.transform_type == 1:
            self.freq = nn.Parameter(np.pi * (2 ** torch.arange(l).reshape(1, -1)), requires_grad=False)
        elif self.transform_type == 2:
            d = 2 * self.l
            self.freq = nn.Parameter(1. / (10000 ** (torch.arange(0, d, 2).float().reshape(1, -1) / d)), requires_grad=False)
        self.pos_proj = nn.Linear(2 * self.l, self.pos_dim, bias=bias)
        self.pos1 = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
        )
        self.pos2 = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.pos_dim, bias=bias)
        )
        self.pos3 = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.outdim, bias=bias)
        )
        
    def transform(self, x):
        if self.transform_type == 1 or self.transform_type == 2:
            x = x * self.freq
            res = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            
        return res
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, biases):
        biases = self.transform(biases)
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos
