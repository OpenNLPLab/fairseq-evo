# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn

from ..helpers import print_params
from ..norm import SimpleRMSNorm


class DynamicPosBiasV7(nn.Module):
    def __init__(self, dim, outdim, residual, act="relu", bias=True):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.pos_proj = nn.Linear(1, 2, bias=bias)
        self.layers = nn.ModuleList([])
        d = 2
        while (2 * d <= self.pos_dim):
            d1 = 2 * d
            self.layers.append(
                nn.Sequential(
                    SimpleRMSNorm(d),
                    self.get_act(),
                    nn.Linear(d, d1, bias=bias),
                )
            )
            d = d1
        self.out = nn.Sequential(
            SimpleRMSNorm(d),
            self.get_act(),
            nn.Linear(d, self.outdim, bias=bias)
        )
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, biases):
        x = self.pos_proj(biases)
        for m in self.layers:
            x = m(x)
        x = self.out(x)

        return x
