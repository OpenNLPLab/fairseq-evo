import torch
import torch.nn as nn
import torch.nn.functional as F

class SynthesizerDense(nn.Module):
    def __init__(self, dim, max_seq_len, causal=False):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, max_seq_len)
        self.act = nn.ReLU()
        self.causal = causal
        print(f"self.causal {self.causal}")

    def forward(self, x, mask=None):
        # x: b, n, d
        b, n, d = x.shape
        energy = self.w2(self.act(self.w1(x)))[:, :, :n]
        if self.causal:
            if (mask == None):
                mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).to(x)
            mask = mask.masked_fill(mask==float("-inf"), 0)

        dense_attn = F.softmax(dense_attn, dim=-1)
        output = torch.matmul(dense_attn, x)
        
        return output
    
class SynthesizerRandom(nn.Module):
    def __init__(self, max_seq_len, causal=False):
        super().__init__()
        self.w = nn.Parameter(torch.randn(max_seq_len, max_seq_len), requires_grad=True)
        self.causal = causal
        print(f"self.causal {self.causal}")

    def forward(self, x, mask=None):
        # x: b, n, d
        b, n, d = x.shape
        energy = self.w[:n, :n]
        if self.causal:
            if (mask == None):
                mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).to(x)
            mask = mask.masked_fill(mask==float("-inf"), 0)

        dense_attn = F.softmax(dense_attn, dim=-1)
        output = torch.matmul(dense_attn, x)
        
        return output