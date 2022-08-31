# https://github.com/raoyongming/GFNet/blob/master/gfnet.py
import math
import torch
import torch.fft
import torch.nn as nn

class GlobalFilter(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(seq_len, dim, 2, dtype=torch.float32) * 0.02)
        self.seq_len = seq_len

    def forward(self, x):
        b, n, d = x.shape
        m = n // 2 + 1
        if m > self.seq_len:
            complex_weight = F.pad(self.complex_weigh, (0, 0, 0, 0, 0, m - seq_len))
        else:
            complex_weight = self.complex_weight[:m]
        x = x.to(torch.float32)
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(complex_weight)
        x = x * weight
        x = torch.fft.irfft(x, dim=1, n=n, norm='ortho')
        
        return x