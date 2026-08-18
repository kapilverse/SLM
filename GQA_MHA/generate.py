"""Self-contained autoregressive generation for the experiment folder."""

import torch
import torch.nn.functional as F

from tokenizer import Tokenizer


@torch.no_grad()
def generate(model, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.8, top_k: int | None = 40, device=None):
    model.eval()
    device = device or next(model.parameters()).device

    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    context_size = model.cfg.context_size

    for _ in range(max_new_tokens):
        x_cond = x[:, -context_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())
