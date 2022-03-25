# - https://github.com/zh217/torch-dct
# - https://github.com/zh217/torch-dct/issues/15

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class Orpe(nn.Module):
    def __init__(self, core_matrix, p_matrix, max_positions=512):
        super().__init__()
        self.core_matrix = core_matrix
        self.p_matrix = p_matrix

        print("==================================")
        print("Orpe Module!")
        print(f"core_matrix {self.core_matrix}")
        print(f"p_matrix {self.p_matrix}")
        if self.core_matrix == 1:
            print("rope")
        elif self.core_matrix == 2:
            print("mixed")
        elif self.core_matrix == 3:
            print("permutation")

        if self.p_matrix == 1:
            print("Identity")
        elif self.p_matrix == 2:
            print("DCT")
        elif self.p_matrix == 3:
            print("Householder")
        print("==================================")

        self.p = self.get_p()
        self.core_transform = self.get_core_transform()
        self.p_transpose = self.get_p_transpose()

    def forward(self, x):
        # b, l, e
        x = self.p(x)
        x = self.core_transform(x)
        x = self.p_transpose(x)

        return x

    def get_p(self):
        if self.p_matrix == 1:
            def f(x):
                return x
            return f
        elif self.p_matrix == 2:
            return self.dct
        elif self.p_matrix == 3:
            # todo
            return self.dct

    def get_p_transpose(self):
        if self.p_matrix == 1:
            def f(x):
                return x
            return f
        elif self.p_matrix == 2:
            return self.idct
        elif self.p_matrix == 3:
            # todo
            return self.idct

    def get_core_transform(self):
        if self.core_matrix == 1:
            return self.rope
        elif self.core_matrix == 2:
            return self.mix_rope
        elif self.core_matrix == 3:
            # todo
            return self.mix_rope

    def rope(self, x):
        b, l, e = x.shape
        theta = 10000 ** (-2 / e * torch.arange(e // 2))
        theta = theta.reshape(1, 1, -1).to(x)
        # theta = torch.cat([theta, theta], dim=-1)
        theta = torch.stack([theta, theta], dim=-1).reshape(1, 1, e)
        theta = theta * torch.arange(l).reshape(1, -1, 1).to(x)
        x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x_transform = x * torch.sin(theta) + x_half * torch.cos(theta)

        return x_transform

    def mix_rope(self, x):
        b, l, d = x.shape
        # split
        e = d // 2
        x1 = x[:, :, :e]
        x = x[:, :, e:]
        # transform x
        theta = 10000 ** (-2 / e * torch.arange(e // 2)).to(x)
        theta = theta.reshape(1, 1, -1)
        theta = torch.stack([theta, theta], dim=-1).reshape(1, 1, e)
        theta = theta * torch.arange(l).reshape(1, -1, 1).to(x)
        x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)

        x_transform = x * torch.sin(theta) + x_half * torch.cos(theta)

        x_transform = torch.cat([x1, x_transform], dim=-1)

        return x_transform

    def dct(self, x):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        # Vc = torch.rfft(v, 1, onesided=False)
        Vc = torch.view_as_real(torch.fft.fft(v, dim=1)) 

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def idct(self, X):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        # norm
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        # v = torch.fft.irfft(V, 1, onesided=False)
        v= torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)