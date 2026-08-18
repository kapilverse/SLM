"""
Three controlled comparisons at fixed model scale, same seed/data/steps:

  - gqa:         n_heads=4, n_kv_heads=2   (grouped-query attention)
  - mha:         n_heads=4, n_kv_heads=4   (standard multi-head attention)
  - weight_tied: n_heads=4, n_kv_heads=2, tie_weights=True (GQA + tied
                 embedding/LM-head weights)

Each config changes exactly one architectural knob relative to "gqa"
(the baseline), isolating that knob's effect on validation loss,
parameter count, and generation speed.

Run on a Colab T4 (or locally on CPU for a quick smoke test with reduced
max_steps).
"""

import argparse
import json
import math
import time

import torch

from config import GPTConfig, TrainConfig
from train import train
from generate import generate
from tokenizer import Tokenizer


def measure_generation_speed(model, tok, device, n_tokens: int = 200) -> float:
    model.eval()
    t0 = time.time()
    generate(model, tok, "Once upon a time", max_new_tokens=n_tokens, temperature=0.8, top_k=40, device=device)
    elapsed = time.time() - t0
    return n_tokens / elapsed


def run(data_path: str, max_steps: int, seed: int, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    configs = {
        "gqa": GPTConfig(n_heads=4, n_kv_heads=2),
        "mha": GPTConfig(n_heads=4, n_kv_heads=4),
        "weight_tied": GPTConfig(n_heads=4, n_kv_heads=2, tie_weights=True),
    }

    for name, gpt_cfg in configs.items():
        train_cfg = TrainConfig(max_steps=max_steps, seed=seed, checkpoint_dir=f"checkpoints/{name}")
        model, tok, history = train(
            data_path=data_path,
            gpt_cfg=gpt_cfg,
            train_cfg=train_cfg,
            run_name=name,
            log_path=f"logs_{name}.json",
        )

        final_val_loss = history[-1]["val_loss"] if history else None
        perplexity = math.exp(final_val_loss) if final_val_loss is not None else None
        tokens_per_sec = measure_generation_speed(model, tok, device)
        param_counts = model.count_parameters()

        results[name] = {
            "n_kv_heads": gpt_cfg.n_kv_heads,
            "final_val_loss": final_val_loss,
            "val_perplexity": perplexity,
            "kv_projection_params": param_counts["kv_projection_params"],
            "total_params": param_counts["total"],
            "generation_tokens_per_sec": tokens_per_sec,
            "history": history,
        }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Comparison ===")
    for name, r in results.items():
        print(f"{name}: val_loss={r['final_val_loss']:.4f}  ppl={r['val_perplexity']:.2f}  "
              f"total_params={r['total_params']:,}  kv_params={r['kv_projection_params']:,}  "
              f"tok/s={r['generation_tokens_per_sec']:.1f}")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/tinystories.txt")
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="comparison_results.json")
    args = parser.parse_args()
    run(args.data_path, args.max_steps, args.seed, args.out)
