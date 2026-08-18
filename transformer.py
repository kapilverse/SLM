"""
Phase 13/14/15 — Feed-forward network, LayerNorm, and the full
Transformer decoder block (pre-norm, GQA + FFN with residuals).

Block:
    x -> LN -> GQA -> + residual -> LN -> FFN -> + residual -> out
"""

import torch
import torch.nn as nn

from gqa import GroupedQueryAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int,
                 context_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, context_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual pattern: normalize BEFORE the sublayer, add the
        # sublayer's output back onto the unnormalized residual stream.
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == "__main__":
    x = torch.randn(2, 16, 128)
    block = TransformerBlock(d_model=128, n_heads=4, n_kv_heads=2, d_ff=512, context_size=16)
    out = block(x)
    print(out.shape)  # expect [2, 16, 128] -- same shape in and out, always
