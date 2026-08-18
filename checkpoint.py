"""
Phase 23 — Checkpointing. Mandatory on free Colab (sessions disconnect).

Save model weights, optimizer state, step, loss, and config together so
training can resume exactly where it left off.
"""

import os

import torch


def save_checkpoint(path: str, model, optimizer, step: int, loss: float, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "config": cfg,
    }, path)
    print(f"Saved checkpoint at step {step} -> {path}")


def load_checkpoint(path: str, model, optimizer=None, map_location=None):
    # weights_only=False: safe here because we only ever load checkpoints
    # this project wrote itself (they contain a GPTConfig dataclass, which
    # torch's default weights_only=True mode refuses to unpickle).
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0), ckpt.get("loss", None), ckpt.get("config", None)
