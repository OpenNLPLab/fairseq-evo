import torch
import numpy as np
from torch import nn

class MatrixFFT(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, dim=-1, reverse=False, causal=True):
        return self.transform(x, dim, reverse)#.real
    
    def transform(self, x, dim=-1, reverse=False, causal=True):
        n = x.shape[dim]
        theta = torch.Tensor([-np.pi * 2 / n])
        real = torch.cos(theta)
        imag = torch.sin(theta)
        w = torch.complex(real, imag)
        w_row = w ** torch.arange(n)
        W = torch.vander(w_row, increasing=True).to(x.device) / np.sqrt(n)
        if reverse:
            W = W.conj().t()
        # cusal mask
        if causal:
            attn_mask = (torch.triu(torch.ones(n, n)) == 1).transpose(0, 1).to(x)
            real = W.real
            imag = W.imag
            real = real.masked_fill(attn_mask == 0, 0)
            imag = imag.masked_fill(attn_mask == 0, 0)
            W = torch.complex(real, imag)
        
        if dim == -1:
            output = torch.einsum('ij,...j->...i', W, x.to(W))
        elif dim == -2:
            output = torch.einsum('ij,...jk->...ik', W, x.to(W))
            
        if reverse:
            output = output.real
        
        return output
    
    def test(self):
        x = torch.rand(2, 10, 5)
        y1 = self.transform(x, causal=False)
        y2 = self.transform(y1, reverse=True, causal=False)
        print(torch.norm(y2 - x))
        
# For test
MatrixFFT().test()