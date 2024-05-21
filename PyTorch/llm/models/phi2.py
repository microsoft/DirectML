from dataclasses import dataclass
from typing import Optional
import math
import torch_directml
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 51200
    n_layer: int = 32
    n_head: int = 32
    dim: int = 2560
    intermediate_size: int = 10240
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        else:
            self.n_kv_groups = self.n_head // self.n_local_heads
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head
    
    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]
        assert len(config) == 1, name
        return cls(**transformer_configs[config[0]])

transformer_configs = {
    "phi-2": dict(block_size=2048, n_layer=32, n_head=32, dim=2560, intermediate_size=10240, rope_base=10000),
}


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size)

        self.max_batch_size = -1
        self.max_seq_length = -1
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype=torch.float32):
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention._init_rope(dtype=dtype)

        self.causal_mask = torch.tril(
            torch.ones(self.config.n_head, self.max_seq_length, self.max_seq_length, dtype=torch.int32)
        )

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        mask = self.causal_mask[None, :, input_pos, :input_pos[-1].item()+1]
        x = self.tok_embeddings(idx)

        for _, layer in enumerate(self.layers):
            x = layer(x, input_pos, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        hidden_states = self.attention_norm(x)
        attn_outputs = self.attention(hidden_states, mask, input_pos)
        ffn_hidden_states = self.feed_forward(hidden_states)
        return attn_outputs + ffn_hidden_states + x


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        self.wqkv = nn.Linear(config.dim, 3 * config.dim)
        self.wo = nn.Linear(config.dim, config.dim)
        self.kv_cache = None
        
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *argspy):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def _init_rope(self, dtype=torch.float32):
        self.min_position = 0
        self.past_key_tensor = None
        self.past_value_tensor = None
        self.rotary_emb = PhiRotaryEmbedding(self.n_head, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        q = q.reshape(bsz, seqlen, self.n_head, self.head_dim).transpose(1,2)
        k = k.reshape(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1,2)
        
        cos, sin = self.rotary_emb()
        
        q, k = torch_directml.apply_rotary_position_emb(
            q, k, cos, sin, self.min_position, seqlen, self.rotary_emb.dim)

        self.min_position += seqlen

        q, k = map(lambda x: x.transpose(1, 2).reshape(bsz, -1, self.dim), (q, k))
        y, self.past_key_tensor, self.past_value_tensor = torch_directml.multi_head_attention(
            q,k,v, self.dim, self.n_head, self.past_key_tensor, self.past_value_tensor, mask
        )

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.dim)
        self.mlp = torch_directml.mlp_phi2

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x, self.w1.weight, self.w2.weight, self.w1.bias, self.w2.bias)


class PhiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, dtype=torch.float32, device=None):
        super().__init__()
        device = device if device is not None else torch_directml.device(torch_directml.default_device())
        self.dtype = dtype
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=dtype).to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.dtype)

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0).to(dtype), persistent=False)

    def forward(self):
        return (
            self.cos_cached,
            self.sin_cached,
        )
