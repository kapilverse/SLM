"""
Phase 19-24 — Training loop, GPU device handling, checkpointing,
monitoring. Start by overfitting a tiny dataset (Phase 19) before moving
to TinyStories (Phase 20) on Colab's T4 (Phase 21).
"""

import os
import time

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


def train(data_path: str = "data/tiny.txt", resume_from: str | None = None,
          gpt_cfg: GPTConfig | None = None, train_cfg: TrainConfig | None = None):
    gpt_cfg = gpt_cfg or GPTConfig()
    train_cfg = train_cfg or TrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
        print(f"Resumed from step {start_step}")

    step = start_step
    t0 = time.time()

    train_iter = iter(train_loader)
    while step < train_cfg.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # TODO(you): this is the training loop from Phase 22 — trace every
        # line and make sure you can explain what each call does to the
        # model's weights.
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        step += 1

        if step % train_cfg.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, device, train_cfg.eval_iters)
            elapsed = time.time() - t0
            print(f"step {step:6d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | {elapsed:.1f}s")

        if step % train_cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, f"ckpt_step{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, loss.item(), gpt_cfg)

    print("Training complete.")
    return model, tok


if __name__ == "__main__":
    # Phase 19: point data_path at a ridiculously small file first and
    # confirm training loss drops toward ~0 (the model should be able to
    # memorize it). Only move to TinyStories once that works.
    #
    # A tiny text file has few tokens, so context_size/batch_size must
    # shrink too or the train/val split won't contain a single full window.
    tiny_gpt_cfg = GPTConfig(context_size=8, n_layers=2, n_heads=2, n_kv_heads=1, d_model=32, d_ff=64)
    tiny_train_cfg = TrainConfig(batch_size=4, max_steps=300, eval_interval=50,
                                  eval_iters=5, checkpoint_interval=300, train_split=0.8)
    train(data_path="data/tiny.txt", gpt_cfg=tiny_gpt_cfg, train_cfg=tiny_train_cfg)
