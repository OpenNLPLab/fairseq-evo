import torch.nn as nn
from einops import rearrange

from ..helpers import get_activation_fn, get_norm_fn, print_params
from .ctno import Ctno


class Gctu(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        k=128,
        bias=True,
        act_fun="silu",
        causal=False,
        expand_ratio=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.num_heads = num_heads

        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        self.head_dim = d1 // num_heads
        # linear projection
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.u_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)
        # Ctno
        self.toep = Ctno(
            h=num_heads, 
            dim=self.head_dim,
            k=k,
            causal=causal, 
        )
    
    def forward(self, x, vander, index):
        # x: b, n, d
        num_heads = self.num_heads
        u = self.act(self.u_proj(x))
        v = self.act(self.v_proj(x))
        # reshape
        v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads)
        output = self.toep(v, vander, index)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = u * output
        output = self.o(output)
        
        return output
