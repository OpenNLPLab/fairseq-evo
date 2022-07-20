# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

import torch
import torch.nn as nn
from einops import rearrange
from einops import repeat

class SEBlock(nn.Module):
    def __init__(self, embedding, reduction=16, causal=False):
        super().__init__()
        self.causal = causal
        self.fc = nn.Sequential(
            nn.Linear(embedding, embedding // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(embedding // reduction, embedding),
            nn.Sigmoid()
        )
        if self.causal:
            self.forward = self.forward_causal
        else:
            self.forward = self.forward_non_causal

    def forward_non_causal(self, x):
        # x: ..., n, d
        n = x.shape[-2]
        # ..., d
        y = torch.mean(x, dim=-2)
        # ..., d
        y = self.fc(y)
        # ..., n, d
        y = repeat(y, '... d -> ... n d', n=n)
        
        return x * y
    
    def forward_causal(self, x):
        # x: ..., n, d
        n = x.shape[-2]
        l = len(x.shape)
        # n, 1
        denorm = torch.arange(1, n + 1).reshape(n, 1).to(x)
        # ..., n, 1
        for i in range(l - 2):
            denorm = denorm.unsqueeze(0)
        # ..., n, d
        y = torch.cumsum(x, dim=-2) / denorm
        # ..., n, d
        y = self.fc(y)
        
        return x * y