# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn
from ..norm import SimpleRMSNorm
from .act_layer import ActLayer

class DynamicPosBiasV9(nn.Module):
    def __init__(
        self, 
        dim, 
        outdim, 
        residual, 
        act="relu", 
        bias=True, 
        layers=3, 
        use_norm1=True, 
        use_norm2=True, 
        transform_type=1,
        l=-1,
    ):
        super().__init__()
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        self.l = l
        self.transform_type = transform_type
        if self.l == -1:
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
        print(f"self.transform_type {self.transform_type}")
        print(f"self.l {self.l}")
        self.pos_proj = nn.Linear(d, self.pos_dim, bias=bias)
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
        elif self.act == "relu":
            return nn.ReLU(inplace=True)
        else:
            return ActLayer(self.act)

    def forward(self, x):
        pos = self.transform(x)
        x = self.pos_proj(pos)
        if self.residual:
            for m in self.layers:
                x = m(x) + x
        else:
            for m in self.layers:
                x = m(x)
        x = self.out(x)

        return x