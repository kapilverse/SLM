"""
Self-contained model/training config for the GQA-vs-MHA experiment.
Independent of the parent project's config.py by design.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2   # 2 = GQA, 4 = standard MHA (n_heads == n_kv_heads)
    d_ff: int = 512
    dropout: float = 0.1
    tie_weights: bool = False   # if True, LM head shares the token embedding matrix


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 30000
    eval_interval: int = 250
    eval_iters: int = 50
    checkpoint_interval: int = 1000
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    train_split: float = 0.9
    seed: int = 1337
