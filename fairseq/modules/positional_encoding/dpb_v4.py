# https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py

import torch
import torch.nn as nn
from ..norm import SimpleRMSNorm

class DynamicPosBiasV4(nn.Module):
    def __init__(self, dim, outdim, residual, act="relu"):
        super().__init__()
        self.residual = residual
        self.outdim = outdim
        self.pos_dim = dim
        self.act = act
        print(self.pos_dim, self.outdim)
        self.pos_proj = nn.Linear(1, self.pos_dim)
        self.pos1 = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            SimpleRMSNorm(self.pos_dim),
            self.get_act(),
            nn.Linear(self.pos_dim, self.outdim)
        )
        
    def get_act(self):
        if self.act == "silu":
            return nn.SiLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos