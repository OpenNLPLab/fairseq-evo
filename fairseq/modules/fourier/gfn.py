# https://github.com/raoyongming/GFNet/blob/master/gfnet.py
import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from .causal_fft import MatrixFFT


class GlobalFilter(nn.Module):
    def __init__(self, seq_len, dim, causal=False, max_seq=512):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(seq_len, dim, 2, dtype=torch.float32) * 0.02)
        self.seq_len = seq_len
        self.causal = causal
        if self.causal:
            self.fft = MatrixFFT(max_seq=max_seq)

    def forward(self, x):
        b, n, d = x.shape
        if self.causal:
            m = n
        else:
            m = n // 2 + 1
        if m > self.seq_len:
            complex_weight = F.pad(self.complex_weight, (0, 0, 0, 0, 0, m - self.seq_len))
        else:
            complex_weight = self.complex_weight[:m]
        x = x.to(torch.float32)
        if self.causal:
            x = self.fft(x, dim=-2, causal=True)
        else:
            x = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(complex_weight)
        x = x * weight
        if self.causal:
            x = self.fft(x, dim=-2, reverse=True, causal=True)
        else:
            x = torch.fft.irfft(x, dim=1, n=n, norm='ortho')
        
        return x
