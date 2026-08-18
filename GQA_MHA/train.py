"""
Self-contained training loop for the GQA-vs-MHA controlled experiment.

Always seeds torch before model construction, so two runs differing only
in n_kv_heads get the same initialization/data-order treatment as far as
torch's RNG allows (embedding/linear init order is identical since the
model structure/parameter shapes only differ in Wk/Wv, which are
constructed at the same point in both configs).
"""

import os
import time
import json

import torch

from config import GPTConfig, TrainConfig
from model import SmallGPT
from dataset import build_dataloaders, load_text_file
from checkpoint import save_checkpoint, load_checkpoint


@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters: int):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(data_path: str, gpt_cfg: GPTConfig, train_cfg: TrainConfig,
          run_name: str, resume_from: str | None = None, log_path: str | None = None):
    torch.manual_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_name}] Using device:", device)

    text = load_text_file(data_path)
    train_loader, val_loader, tok = build_dataloaders(
        text, gpt_cfg.context_size, train_cfg.batch_size, train_split=train_cfg.train_split
    )

    model = SmallGPT(gpt_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )

    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step, _, _ = load_checkpoint(resume_from, model, optimizer, map_location=device)
        print(f"[{run_name}] Resumed from step {start_step}")

    step = start_step
    t0 = time.time()
    history = []

    train_iter = iter(train_loader)
    while step < train_cfg.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        step += 1

        if step % train_cfg.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, device, train_cfg.eval_iters)
            elapsed = time.time() - t0
            print(f"[{run_name}] step {step:6d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | {elapsed:.1f}s")
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss, "elapsed": elapsed})
            if log_path:
                with open(log_path, "w") as f:
                    json.dump(history, f, indent=2)

        if step % train_cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"{run_name}_step{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, loss.item(), gpt_cfg)

    print(f"[{run_name}] Training complete.")
    return model, tok, history
