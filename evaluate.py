"""
Phase 26 — Evaluate the model: loss + perplexity, and a helper to compare
checkpoints over training.

Perplexity = exp(loss). Loss is the average negative log-likelihood the
model assigns to the correct next token; perplexity converts that back
into an intuitive "effective number of equally-likely choices" the model
is choosing among. Lower is better; a random model over a 50k vocab has
perplexity near 50,000, a good model gets down to double/triple digits.
"""

import math

import torch


@torch.no_grad()
def perplexity(model, loader, device, max_batches: int | None = None) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()

    avg_loss = sum(losses) / max(len(losses), 1)
    return math.exp(avg_loss)


if __name__ == "__main__":
    # TODO(you): load a checkpoint with checkpoint.load_checkpoint, run this
    # against your val_loader, and record perplexity at 1K/10K/50K steps to
    # build the comparison table from Phase 26.
    pass
