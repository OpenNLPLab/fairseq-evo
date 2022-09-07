# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn
from ..norm import SimpleRMSNorm
from .act_layer import ActLayer

class DynamicPosBiasV8(nn.Module):
    def __init__(self, dim, outdim, residual, act="relu", bias=True, layers=3, use_norm1=True, use_norm2=True):
        super().__init__()
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.pos_proj = nn.Linear(1, self.pos_dim, bias=bias)
        self.layers = nn.ModuleList([])
        self.use_norm1 = use_norm1
        self.use_norm2 = use_norm2
        print(f"self.use_norm1 {self.use_norm1}")
        print(f"self.use_norm2 {self.use_norm2}")
        for i in range(layers):
            if self.use_norm1:
                self.layers.append(
                    nn.Sequential(
                        SimpleRMSNorm(self.pos_dim),
                        self.get_act(),
                        nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        self.get_act(),
                        nn.Linear(self.pos_dim, self.pos_dim, bias=bias),
                    )
                )
        if self.use_norm2:
            self.out = nn.Sequential(
                SimpleRMSNorm(self.pos_dim),
                self.get_act(),
                nn.Linear(self.pos_dim, self.outdim, bias=bias)
            )
        else:
            self.out = nn.Sequential(
                self.get_act(),
                nn.Linear(self.pos_dim, self.outdim, bias=bias)
            )
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        elif self.act == "relu":
            return nn.ReLU(inplace=True)
        else:
            return ActLayer(self.act)

    def forward(self, biases):
        x = self.pos_proj(biases)
        if self.residual:
            for m in self.layers:
                x = m(x) + x
        else:
            for m in self.layers:
                x = m(x)
        x = self.out(x)

        return x