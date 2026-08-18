"""
Model + training hyperparameters in one place.

Phase 16/17: these numbers directly determine parameter count and
compute cost. Change them here, nowhere else.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257     # gpt2 tiktoken vocab
    context_size: int = 128     # T: max sequence length the model can attend over
    d_model: int = 128          # C: embedding / residual stream dimension
    n_layers: int = 4
    n_heads: int = 4            # query heads
    n_kv_heads: int = 2         # key/value heads (n_heads must be divisible by this) -> GQA
    d_ff: int = 512             # feed-forward hidden dimension
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5000
    warmup_steps: int = 100
    eval_interval: int = 250
    eval_iters: int = 50
    checkpoint_interval: int = 500
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    train_split: float = 0.9
