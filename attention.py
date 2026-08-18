"""
Phase 9/10/11 — Self-attention, causal masking, multi-head attention.
Built from scratch. Do NOT use nn.MultiheadAttention.

Single head:
    Q = X @ Wq        [B, T, d_head]
    K = X @ Wk        [B, T, d_head]
    V = X @ Wv        [B, T, d_head]

    scores = Q @ K.transpose(-2, -1) / sqrt(d_head)   [B, T, T]
    scores = scores.masked_fill(causal_mask, -inf)    # Phase 10
    attn = softmax(scores, dim=-1)                    [B, T, T]
    out = attn @ V                                     [B, T, d_head]

Multi-head:
    d_model split into n_heads heads of size d_model // n_heads.
    Run the above per head in parallel, concat, then one output projection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention (n_heads == n_kv_heads)."""

    def __init__(self, d_model: int, n_heads: int, context_size: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # TODO(you): one fused linear for Q, K, V is a common trick, but for
        # learning purposes define them separately first — it's clearer.
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Phase 10: causal mask, precomputed once. Lower-triangular = allowed.
        mask = torch.tril(torch.ones(context_size, context_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # C == d_model

        # TODO(you): project to Q, K, V. Shapes start as [B, T, C].
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # TODO(you): reshape [B, T, C] -> [B, n_heads, T, d_head] so each
        # head attends independently. Hint: .view(B, T, n_heads, d_head)
        # then .transpose(1, 2).
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, D]
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, D]

        # TODO(you): implement the scaled dot-product attention equation
        # yourself (don't call F.scaled_dot_product_attention here — the
        # point of this phase is to write it out).
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)     # [B, H, T, T]
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)                              # [B, H, T, T]
        attn = self.attn_dropout(attn)

        out = attn @ v                                                # [B, H, T, D]

        # TODO(you): merge heads back: [B, H, T, D] -> [B, T, H*D] == [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(out))


if __name__ == "__main__":
    x = torch.randn(2, 16, 128)  # [B, T, C]
    attn = CausalSelfAttention(d_model=128, n_heads=4, context_size=16)
    out = attn(x)
    print(out.shape)  # expect [2, 16, 128]
