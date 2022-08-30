# https://github.com/erksch/fnet-pytorch/blob/master/fnet.py
import torch
from scipy import linalg
from torch import nn

# only for test
class FourierMMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dft_mat_seq = nn.Parameter(torch.tensor(linalg.dft(config['max_position_embeddings'])), requires_grad=False)
        self.dft_mat_hidden = nn.Parameter(torch.tensor(linalg.dft(config['hidden_size'])), requires_grad=False)

    def forward(self, hidden_states):
        hidden_states_complex = hidden_states.type(torch.complex128)
        return torch.einsum(
            "...ij,...jk,...ni->...nk",
            hidden_states_complex,
            self.dft_mat_hidden,
            self.dft_mat_seq
        ).real.type(torch.float32)

class FourierFFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states.float(), dim=-1), dim=-2).real
    
class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fft = FourierMMLayer(config) if config['fourier'] == 'matmul' else FourierFFTLayer()
        self.mixing_layer_norm = nn.LayerNorm(config['hidden_size'])
        self.feed_forward = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output_dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.output_layer_norm = nn.LayerNorm(config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        fft_output = self.fft(hidden_states)
        fft_output = self.mixing_layer_norm(fft_output + hidden_states)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = self.output_layer_norm(output + fft_output)
        return output

class FNetFairseqLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fnet = FNetLayer(config)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = self.fnet(x)
        x = x.transpose(0, 1)
        
        return x, None