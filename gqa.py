"""
Phase 12 — Grouped Query Attention (GQA).

MHA: n_heads query heads, n_heads key heads, n_heads value heads (1:1:1).
GQA: n_heads query heads, but only n_kv_heads (< n_heads) key/value heads.
     Each group of (n_heads // n_kv_heads) query heads shares one K/V head.

Example from config: n_heads=4, n_kv_heads=2
    Q1, Q2 -> share K1, V1
    Q3, Q4 -> share K2, V2

Why: K/V is what you cache during autoregressive generation (KV-cache).
Fewer K/V heads => smaller cache => less memory/bandwidth at inference,
with only a small quality cost vs full MHA.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, context_size: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads   # query heads per kv head
        self.d_head = d_model // n_heads

        # Q projects to full n_heads * d_head, same as MHA.
        self.Wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        # K/V project to only n_kv_heads * d_head — this is where GQA saves params + KV-cache size.
        self.Wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_size, context_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)       # [B, Hq, T, D]
        k = self.Wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)    # [B, Hkv, T, D]
        v = self.Wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)    # [B, Hkv, T, D]

        # TODO(you): expand K and V so each has n_heads "views", by repeating
        # each kv head n_groups times along the head dimension. This is the
        # crux of GQA — think carefully about *why* repeat_interleave (not
        # plain repeat) gives the correct Q-to-KV grouping described above.
        k = k.repeat_interleave(self.n_groups, dim=1)   # [B, Hq, T, D]
        v = v.repeat_interleave(self.n_groups, dim=1)   # [B, Hq, T, D]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)   # [B, Hq, T, T]
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                               # [B, Hq, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)

        return self.resid_dropout(self.out_proj(out))


if __name__ == "__main__":
    x = torch.randn(2, 16, 128)
    gqa = GroupedQueryAttention(d_model=128, n_heads=4, n_kv_heads=2, context_size=16)
    out = gqa(x)
    print(out.shape)  # expect [2, 16, 128]

    # TODO(you): compare parameter counts vs a plain CausalSelfAttention
    # with n_heads=4 to see GQA's K/V savings directly.
    n_params = sum(p.numel() for p in gqa.parameters())
    print("GQA params:", n_params)
