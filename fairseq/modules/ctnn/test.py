import torch
from fairseq.modules.ctnn import Ctno

b = 2
h = 2
n = 20
dim = 32
k = 10
gamma = 0.99

# test
x = torch.rand(b, h, n, dim).cuda()
coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
zero = torch.ones(1, 1, 1).cuda()
pos = gamma ** coef
neg = torch.flip(pos, dims=[1])
index = torch.arange(n).cuda()

# causal
decay = torch.cat([zero, pos], dim=1)
cos = torch.rand(1, n, k, 1).cuda()
ctno = Ctno(
    h=h,
    dim=dim,
    k=k,
    causal=True
).cuda()

y1 = ctno(x, decay, cos, index)
y2, T = ctno.mm(x, decay, cos)
print("causal")
print(torch.norm(y1 - y2))

# non causal
decay = torch.cat([zero, pos, zero, neg], dim=1)
cos = torch.rand(1, 2 * n, k, 1).cuda()
ctno = Ctno(
    h=h,
    dim=dim,
    k=k,
).cuda()

y1 = ctno(x, decay, cos, index)
y2, T = ctno.mm(x, decay, cos)
print("non causal")
print(torch.norm(y1 - y2))
