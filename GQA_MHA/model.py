"""
Self-contained GPT model for the paper/experiment folder.

Includes GQA/MHA attention (unified: standard MHA is just the
n_kv_heads == n_heads special case), Transformer block, and full model
assembly. Independent implementation from the parent project's files.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class Attention(nn.Module):
    """Causal attention with optional grouped-query attention.

    Set n_kv_heads == n_heads for standard multi-head attention, or
    n_kv_heads < n_heads (dividing n_heads evenly) for GQA.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, context_size: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.Wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_size, context_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, context_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, context_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size, context_size, d_model, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # nn.Embedding's default init (std=1.0) is ~sqrt(d_model) larger than
        # nn.Linear's default (std=1/sqrt(in_features)). Left uncorrected,
        # tying this table to the LM head (a Linear layer) inflates logits
        # and destabilizes loss from step 0. Rescale to match Linear's scale.
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=1.0 / (d_model ** 0.5))

    def forward(self, token_ids):
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)
        return self.dropout(self.token_emb(token_ids) + self.pos_emb(positions))


class SmallGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = GPTEmbeddings(cfg.vocab_size, cfg.context_size, cfg.d_model, cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff, cfg.context_size, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # Share the token embedding matrix with the LM head instead of
            # learning two separate [vocab_size, d_model] matrices -- halves
            # the two largest parameter blocks in this model.
            self.lm_head.weight = self.embeddings.token_emb.weight

    def forward(self, token_ids, targets=None):
        x = self.embeddings(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def count_parameters(self) -> dict:
        kv_params = sum(p.numel() for block in self.blocks
                         for p in [block.attn.Wk.weight, block.attn.Wv.weight])
        return {
            "token_emb": self.embeddings.token_emb.weight.numel(),
            "pos_emb": self.embeddings.pos_emb.weight.numel(),
            "blocks": sum(p.numel() for p in self.blocks.parameters()),
            "kv_projection_params": kv_params,
            "ln_f": sum(p.numel() for p in self.ln_f.parameters()),
            "lm_head": self.lm_head.weight.numel(),
            "total": sum(p.numel() for p in self.parameters()),
        }
