import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask : Optional[torch.Tensor] = None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            if attn.dtype == torch.float16:
                """
                -1e9 is overflow in fp16. It needs to be set a min.
                Theoretically, the mask for empty token needs to be set as -inf. Check https://arxiv.org/pdf/1706.03762.pdf
                """
                min_mask = -65504.0 #torch.finfo(torch.float16).min == -65504.0. jit scripting could handle finfo
            else:
                min_mask = -1e9
            attn = attn.masked_fill(mask == 0, min_mask)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
