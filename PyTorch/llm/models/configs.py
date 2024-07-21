# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    partial_rotary_factor: float = 1.0

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
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

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match

        if not config:
            raise ValueError(f"No configuration found for the model named '{name}'. Supported models: {list(transformer_configs.keys())}")

        return cls(**transformer_configs[config[0]])

transformer_configs = {
    "7B": dict(block_size=4096, n_layer=32, n_head=32, dim=4096),
    "phi-2": dict(block_size=2048, n_layer=32, n_head=32, dim=2560, intermediate_size=10240, rope_base=10000, vocab_size=51200, partial_rotary_factor=0.4),
    "Phi-3-mini-4k-instruct": dict(block_size=4096, n_layer=32, n_head=32, dim=3072, intermediate_size=8192, rope_base=10000, vocab_size=32064),
    "Mistral-7B": dict(block_size=4096, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "Llama-3-8B": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000.0),
}

default_models = {
    "llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-2": "microsoft/phi-2",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}
