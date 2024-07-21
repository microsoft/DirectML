# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch_directml
import torch.nn as nn
from torch import Tensor

from models.configs import ModelArgs
from models.layers import RMSNorm
from models.layers import LlamaTransformerBlock as TransformerBlock
from models.base import Transformer


class Phi3Transformer(Transformer):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__(config)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, FeedForward) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        mask = self.causal_mask[None, :, input_pos, :input_pos[-1].item()+1]
        x = self.tok_embeddings(idx)

        for _, layer in enumerate(self.layers):
            x = layer(x, input_pos, mask)

        x = self.norm(x)
        logits = self.output(x)
        return logits

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, 2*config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.mlp = torch_directml.mlp_phi3

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x, self.w1.weight, self.w2.weight)

