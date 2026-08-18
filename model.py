"""
Phase 16/17/18 — Assemble SmallGPT and count parameters.

Token Embedding + Position Embedding
    -> TransformerBlock x n_layers
    -> final LayerNorm
    -> Linear LM head -> [B, T, vocab_size] logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from embeddings import GPTEmbeddings
from transformer import TransformerBlock


class SmallGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.embeddings = GPTEmbeddings(cfg.vocab_size, cfg.context_size, cfg.d_model, cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff,
                              cfg.context_size, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # TODO(you, optional): weight-tying self.lm_head.weight with
        # self.embeddings.token_emb.weight is a common GPT trick — try it
        # once the untied version trains correctly, and see how param count
        # changes in count_parameters() below.

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        # token_ids: [B, T]
        x = self.embeddings(token_ids)          # [B, T, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Phase 18: flatten [B, T, vocab] -> [B*T, vocab] and
            # [B, T] -> [B*T] so cross_entropy can compare each position's
            # distribution to its target token independently.
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def count_parameters(self) -> dict:
        """Phase 17: rough parameter breakdown by component."""
        counts = {}
        counts["token_emb"] = self.embeddings.token_emb.weight.numel()
        counts["pos_emb"] = self.embeddings.pos_emb.weight.numel()
        counts["blocks"] = sum(p.numel() for p in self.blocks.parameters())
        counts["ln_f"] = sum(p.numel() for p in self.ln_f.parameters())
        counts["lm_head"] = self.lm_head.weight.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


if __name__ == "__main__":
    cfg = GPTConfig()
    model = SmallGPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.context_size))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.context_size))

    logits, loss = model(x, y)
    print("logits shape:", logits.shape)   # [2, context_size, vocab_size]
    print("loss:", loss.item())

    for k, v in model.count_parameters().items():
        print(f"{k:>10}: {v:,}")
