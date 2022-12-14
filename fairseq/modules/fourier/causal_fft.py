import numpy as np
import torch
from torch import nn


# The effect is very poor, ignore it for the time being
class MatrixFFT(nn.Module):
    def __init__(self, max_seq=512):
        super().__init__()
        self.max_seq = max_seq
        W = self.build_fft_matrix(self.max_seq)
        self.real = nn.Parameter(W.real, requires_grad=False)
        self.imag = nn.Parameter(W.imag, requires_grad=False)
    
    def build_fft_matrix(self, n):
        theta = torch.Tensor([-np.pi * 2 / n])
        real = torch.cos(theta)
        imag = torch.sin(theta)
        w = torch.complex(real, imag)
        w_row = w ** torch.arange(n)
        W = torch.vander(w_row, increasing=True) / np.sqrt(n)
        
        return W
        
    def forward(self, x, dim=-1, reverse=False, causal=True):
        return self.transform(x, dim, reverse)#.real
    
    def transform(self, x, dim=-1, reverse=False, causal=True):
        n = x.shape[dim]
        if n != self.max_seq:
            W = self.build_fft_matrix(n).to(x.device)
        else:
            W = torch.complex(self.real, self.imag)
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
        return torch.norm(y2 - x)
