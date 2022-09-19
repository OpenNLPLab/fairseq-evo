# https://github.com/antonyvigouret/Pay-Attention-to-MLPs/blob/master/models.py
import einops
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len, causal=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)
        self.seq_len = seq_len
        self.causal = causal
        print(f"gmlp: self.causal {self.causal}")

    def forward(self, x, mask=None):
        # x: b, n, d
        b, n, d = x.shape
        m = min(n, self.seq_len)
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        weight = self.proj.weight[:m, :m]
        if self.causal:
            if (mask == None) or (m < n):
                mask = (torch.triu(torch.ones(m, m)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).to(x)
            # mask = (torch.triu(torch.ones(m, m)) == 1).transpose(0, 1)
            # mask = mask.float().masked_fill(mask == 0, float('-inf')).to(x)
            weight = weight.masked_fill(mask==float("-inf"), 0)
        v = torch.einsum('bnd,mn->bmd', v[:, :m], weight)
        v = F.pad(v, (0, 0, 0, n - m, 0, 0))
        
        return u * v

class GatingMlpBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, causal):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len, causal)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)

    def forward(self, x, mask=None):
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x, mask)
        x = self.proj_2(x)
        return x + shorcut