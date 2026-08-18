"""
Phase 25 — Autoregressive text generation with temperature + top-k
sampling.
"""

import torch
import torch.nn.functional as F

from tokenizer import Tokenizer


@torch.no_grad()
def generate(model, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.8, top_k: int | None = 40, device=None):
    model.eval()
    device = device or next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)  # [1, T]

    context_size = model.cfg.context_size

    for _ in range(max_new_tokens):
        # Model only has positional embeddings up to context_size, so we
        # must truncate to the last context_size tokens before each step.
        x_cond = x[:, -context_size:]

        logits, _ = model(x_cond)          # [1, T, vocab]
        logits = logits[:, -1, :]          # last position's logits: [1, vocab]

        # TODO(you): temperature scaling — dividing logits by temperature
        # before softmax controls how "peaked" the distribution is.
        # temperature < 1 -> more confident/conservative, > 1 -> more random.
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            # TODO(you): zero out (set to -inf) everything except the top_k
            # highest-probability tokens before sampling.
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)          # [1, vocab]
        next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


if __name__ == "__main__":
    from config import GPTConfig
    from model import SmallGPT

    cfg = GPTConfig()
    model = SmallGPT(cfg)
    tok = Tokenizer()

    # NOTE: with random (untrained) weights this will produce gibberish —
    # that's expected. Run this again after training to compare.
    out = generate(model, tok, "Once upon a time", max_new_tokens=20)
    print(out)
