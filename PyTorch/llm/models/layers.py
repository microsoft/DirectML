# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, Tuple, Dict

import torch_directml
import torch
import torch.nn as nn
from torch import Tensor
from models.configs import ModelArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.rmsnorm = torch_directml.rmsnorm

    def forward(self, x: Tensor) -> Tensor:
        output = self.rmsnorm(x.float(), self.weight.float(), self.eps)
        return output.type_as(x)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: Union[int, float] = 10000,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dtype = dtype
        device = device if device is not None else torch_directml.device(torch_directml.default_device())

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0).to(self.dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0).to(self.dtype), persistent=False)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor] :
        return (
            self.cos_cached,
            self.sin_cached
        )

class LlamaAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: Dict[str, torch.Tensor], prefix: str, *argspy):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def _init_rope( self, max_position_embeddings: int = 4096, rope_base: Union[int, float] = 10000.0, dtype: torch.dtype = torch.float16) -> None:
        self.min_position = 0
        self.past_key_tensor = None
        self.past_value_tensor = None
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_base, dtype=dtype
        )

    def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.reshape(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb()

        q, k = torch_directml.apply_rotary_position_emb(
            q, k, cos, sin, self.min_position, seqlen, self.head_dim)
        self.min_position += seqlen

        if self.n_head != self.n_local_heads:
            k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        q, k, v = map(lambda x: x.transpose(1, 2).reshape(bsz, -1, self.dim), (q, k, v))

        y, self.past_key_tensor, self.past_value_tensor = torch_directml.multi_head_attention(
            q, k, v, self.dim, self.n_head, self.past_key_tensor, self.past_value_tensor, mask
        )
        y = self.wo(y)
        return y

class LlamaTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, feed_forward_module: nn.Module):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.feed_forward = feed_forward_module(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
