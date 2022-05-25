import torch
import torch.nn as nn
import numpy as np

class Urpe(nn.Module):
    def __init__(self, core_matrix, p_matrix, max_positions=512, embedding_dim=768, 
                 theta_type="a", theta_learned=False, householder_learned=False):
        super().__init__()
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix
        self.theta_type = theta_type
        self.theta_learned = theta_learned
        self.householder_learned = householder_learned

        # Lambda matrix
        if self.core_matrix == 1:
            if self.theta_learned:
                print("Learn theta!")
                self.theta = nn.Parameter(10000 ** (-2 / embedding_dim * torch.arange(embedding_dim // 2)).reshape(1, 1, -1))
            else:
                print(f"Theta_type {self.theta_type}")
        elif self.core_matrix == 2:
            print("Mixed")
        elif self.core_matrix == 3:
            print("Permutation")
            permutation = self.get_permutation(max_positions, embedding_dim)
            self.register_buffer("permutation", permutation)
        elif self.core_matrix == 4:
            print("Complex exp")
            if self.theta_learned:
                print("Learn theta!")
                self.theta = nn.Parameter(10000 ** (-2 / embedding_dim * torch.arange(embedding_dim)).reshape(1, 1, -1))
            else:
                print(f"Theta_type {self.theta_type}")

        # P matrix
        if self.p_matrix == 1:
            print("Identity")
        elif self.p_matrix == 2:
            print("Householder")
            if self.householder_learned:
                print("learn householder!")
                self.v = nn.Parameter(torch.randn(1, embedding_dim, 1))
            else:
                v = torch.randn(1, embedding_dim, 1)
                v = v / torch.norm(v)
                print(f"Householder norm is {torch.norm(v)}")
                self.v = nn.Parameter(v, requires_grad=False)
        elif self.p_matrix == 3:
            print("Fourier")
        elif self.p_matrix == 4:
            print("Odd_even")

        self.p = self.get_p()
        self.core_transform = self.get_core_transform()

    def forward(self, x):
        '''
        input shape: (b, l, e), b stands for batch size, l stands for sequence length, e stands for embedding dimension.
        '''
        x = self.p(x)
        x = self.core_transform(x)
        return x

    def get_p(self):
        if self.p_matrix == 1:
            def f(x):
                return x
            return f
        elif self.p_matrix == 2:
            return self.householder
        elif self.p_matrix == 3:
            def f(x):
                return torch.fft.fft(x, norm="ortho")
            return f
        elif self.p_matrix == 4:
            return self.odd_even_permutation

    def get_core_transform(self):
        if self.core_matrix == 1:
            return self.reflect
        elif self.core_matrix == 2:
            return self.mix_reflect
        elif self.core_matrix == 3:
            return self.do_permutation
        elif self.core_matrix == 4:
            return self.complex_exp

    def get_permutation(self, max_positions, embedding_dim):
        permutation = torch.randperm(embedding_dim).reshape(1, -1)
        expanded = [torch.arange(embedding_dim).unsqueeze(0)]
        for _ in range(max_positions - 1):
            previous = expanded[-1]
            current = previous.gather(-1, permutation)
            expanded.append(current)
        expanded = torch.stack(expanded, dim=1)
        return expanded

    def odd_even_permutation(self, x):
        # 2k->k, 2k+1->d+k
        e = x.shape[-1]
        d = e - e // 2
        permutation = torch.arange(e)
        index = torch.arange(e)
        permutation[::2] = index[::2] // 2
        permutation[1::2] = (index[1::2] - 1) // 2 + d
        permutation = permutation.to(x.device)
        x = x.gather(-1, permutation.expand_as(x))

        return x

    def do_permutation(self, x):
        b, l, e = x.shape
        x = x.gather(-1, self.permutation[:, :l, :].expand_as(x))

        return x

    def reflect(self, x):
        b, l, d = x.shape
        e = d - 1 if d % 2 == 1 else d
        return self.transform(x, e)

    def mix_reflect(self, x):
        b, l, d = x.shape
        assert d >= 3
        # split
        e = d // 2
        # to even
        if e % 2:
            e += 1
        return self.transform(x, e)

    def transform(self, x, e):
        assert e % 2 == 0
        b, l, d = x.shape
        # do identity transformation
        x1 = x[:, :, e:]
        # do reflection
        x = x[:, :, :e]
        if self.theta_learned:
            theta = self.theta
        else:
            if self.theta_type == "a":
                theta = 10000 ** (-2 / e * torch.arange(e // 2))
            elif self.theta_type == "b":
                theta = np.pi / 2 / l / (e // 2) * torch.arange(1, e // 2 + 1)
            elif self.theta_type == "c":
                theta = np.pi / 2 / l / torch.arange(1, e // 2 + 1)
            theta = theta.reshape(1, 1, -1).to(x)
        theta = torch.stack([theta, theta], dim=-1).reshape(1, 1, e)
        theta = theta * torch.arange(l).reshape(1, -1, 1).to(x)
        # (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
        x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x_transform = x * torch.cos(theta) + x_half * torch.sin(theta)
        # merge
        if e != d:
            x_transform = torch.cat([x_transform, x1], dim=-1)

        return x_transform

    def complex_exp(self, x):
        b, l, e = x.shape
        if self.theta_learned:
            theta = self.theta
        else:
            if self.theta_type == "a":
                theta = 10000 ** (-2 / e * torch.arange(e))
            theta = theta.reshape(1, 1, -1).to(x.device)
        matrix = theta * torch.arange(l).reshape(1, -1, 1).to(x.device)

        sin_cos = torch.complex(torch.cos(matrix),torch.sin(matrix)).to(x.device)
        x = self.element_wise_complex(x, sin_cos)
        return x

    def element_wise_complex(self, t1, t2):
        return torch.complex(t1.real * t2.real - t1.imag * t2.imag, t1.real * t2.imag + t1.imag * t2.real)

    def householder(self, x, eps=1e-6):
        if self.householder_learned:
            v = self.v / (torch.norm(self.v) + eps)
        else:
            v = self.v
        # (b, n, e), (1, e, 1) -> (1, n, 1)
        y = torch.matmul(x, v)
        # (1, n, 1), (1, 1, e) -> (1, n, e)
        y = torch.matmul(y, v.transpose(1, 2))

        return x - 2 * y