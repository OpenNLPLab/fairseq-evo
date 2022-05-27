# https://github.com/lucidrains/long-short-transformer/blob/main/long_short_transformer/long_short_transformer.py

from math import gcd, ceil
import functools
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, Optional, Tuple
from fairseq.incremental_decoding_utils import with_incremental_state

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    if m.is_integer():
        return tensor

    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

@with_incremental_state
class LongShortAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        causal = True,
        window_size = 128,
        segment_size = 16,
        r = 1,
        dropout = 0.
    ):
        super().__init__()
        assert not (causal and r >= segment_size), 'r should be less than segment size, if autoregressive'
        dim_head = dim // heads
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal

        self.window_size = window_size
        self.segment_size = segment_size
        self.pad_to_multiple = window_size if not causal else lcm(window_size, segment_size)

        # self.to_dynamic_proj = nn.Linear(dim_head, r, bias = False)
        self.to_dynamic_proj = nn.Linear(dim_head, r)
        self.local_norm = nn.LayerNorm(dim_head)
        self.global_norm = nn.LayerNorm(dim_head)

        self.attn_dropout = nn.Dropout(dropout)

        # self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # self.to_kv = nn.Linear(dim, inner_dim, bias = False)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)

        print(f"self.causal {self.causal}")

    def forward(
        self, 
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        eps=1e-6
    ):
        x = query
        mask = None
        # mask = attn_mask

        b, n, *_, h, device, causal, w, s = *x.shape, self.heads, x.device, self.causal, self.window_size, self.segment_size

        # pad input sequence to multiples of window size (or window size and segment length if causal)

        x = pad_to_multiple(x, self.pad_to_multiple, dim = -2, value = 0.)

        # derive from variables

        padded_len = x.shape[-2]
        windows = padded_len // w
        is_padded = padded_len != n

        mask_value = -torch.finfo(x.dtype).max

        # handle mask if padding was needed and mask was not given

        if is_padded:
            mask = default(mask, torch.ones((b, n), device = device).bool())
            mask = pad_to_multiple(mask, w, dim = -1, value = False)

        # get queries, keys, values

        qkv = (self.to_q(x), self.to_kv(x))

        # get sequence range, for calculating mask

        seq_range = torch.arange(padded_len, device = device)

        # split heads

        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        # scale queries

        q = q * self.scale

        # get local queries and keys similarity scores

        window_fn = lambda t: rearrange(t, 'b (w n) d -> b w n d', n = w)
        lq, lkv = map(window_fn, (q, kv))

        lookaround_kwargs = {'backward': 1, 'forward': (0 if causal else 1)}
        lkv = look_around(lkv, **lookaround_kwargs)

        lkv = self.local_norm(lkv)
        lsim = einsum('b w i d, b w j d -> b w i j', lq, lkv)

        # prepare global key / values

        if self.causal:
            # autoregressive global attention is handled in segments
            # later on, these segments are carefully masked to prevent leakage

            gkv = rearrange(kv, 'b (n s) d -> b n s d', s = s)
            pkv = self.to_dynamic_proj(gkv)

            if exists(mask):
                pmask = repeat(mask, 'b (n s) -> (b h) n s', s = s, h = h)
                pkv.masked_fill_(~pmask[..., None], mask_value)

            pkv = pkv.softmax(dim = -2)

            gkv = einsum('b n s d, b n s r -> b n r d', gkv, pkv)
            gkv = rearrange(gkv, 'b n r d -> b (n r) d')
        else:
            # equation (3) in the paper
            pkv = self.to_dynamic_proj(kv)
            if exists(mask):
                pmask = repeat(mask, 'b n -> (b h) n', h = h)
                pkv.masked_fill_(~pmask[..., None], mask_value)

            pkv = pkv.softmax(dim = -2)

            gkv = einsum('b n d, b n r -> b r d', kv, pkv)

        # calculate global queries and keys similarity scores

        gkv = self.global_norm(gkv)
        gsim = einsum('b n d, b r d -> b n r', q, gkv)

        # concat values together (same as keys)

        gkv = repeat(gkv, 'b r d -> b w r d', w = windows)
        v = torch.cat((gkv, lkv), dim = -2)

        # masking

        buckets, i, j = lsim.shape[-3:]

        if exists(mask):
            mask = repeat(mask, 'b (w n) -> (b h) w n', n = w, h = h)
            mask = look_around(mask, pad_value = False, **lookaround_kwargs)
            mask = rearrange(mask, 'b w n -> b w () n')
            lsim.masked_fill_(~mask, mask_value)

        # mask out padding

        seq_range_windowed = rearrange(seq_range, '(w n) -> () w n', w = windows)
        pad_mask = look_around(seq_range_windowed, pad_value = -1, **lookaround_kwargs) == -1
        lsim.masked_fill_(pad_mask[:, :, None], mask_value)

        # calculate causal masking for both global and local

        if self.causal:
            g_range = rearrange(seq_range, '(n s) -> n s', s = s)
            g_range_max = g_range.amax(dim = -1)
            g_mask = seq_range[:, None] >= g_range_max[None, :]
            g_mask = rearrange(g_mask, 'i j -> () i j')
            gsim.masked_fill_(~g_mask, mask_value)

            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            causal_mask = repeat(causal_mask, 'i j -> () u i j', u = buckets)
            lsim.masked_fill_(causal_mask, mask_value)

        # concat local and global similarities together to ready for attention

        gsim = rearrange(gsim, 'b (w n) r -> b w n r', w = windows)
        sim = torch.cat((gsim, lsim), dim = -1)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values (same as keys, since tied) and project out

        out = einsum('b w i j, b w j d -> b w i d', attn, v)
        out = rearrange(out, '(b h) w n d -> b (w n) (h d)', h = h)
        out = out[:, :n]
        return self.to_out(out), None

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value