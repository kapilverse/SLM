"""
Phase 7/8 — Token embeddings + positional embeddings.

Token IDs -> [B, T] of ints
Token embedding table -> [vocab_size, d_model]
Position embedding table -> [context_size, d_model]

output = token_emb(x) + pos_emb(positions)   -> [B, T, d_model]
"""

import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T]
        B, T = token_ids.shape

        # TODO(you): build a [T] tensor of positions 0..T-1 on the same
        # device as token_ids, then look it up in self.pos_emb.
        positions = torch.arange(T, device=token_ids.device)

        tok = self.token_emb(token_ids)      # [B, T, d_model]
        pos = self.pos_emb(positions)        # [T, d_model] -> broadcasts over B
        return self.dropout(tok + pos)       # [B, T, d_model]


if __name__ == "__main__":
    emb = GPTEmbeddings(vocab_size=50257, context_size=128, d_model=128)
    x = torch.randint(0, 50257, (32, 128))   # [B=32, T=128]
    out = emb(x)
    print(out.shape)  # expect [32, 128, 128]
