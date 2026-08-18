"""
Phase 5/6 — Turn a flat stream of token ids into (input, target) training
pairs, and wrap that in a PyTorch Dataset + DataLoader.

Core idea:
    tokens = [10, 20, 30, 40, 50]
    context_size = 4
    x = [10, 20, 30, 40]
    y = [20, 30, 40, 50]        # y is x shifted right by one

Every window of context_size+1 tokens gives one training example.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], context_size: int):
        # TODO(you): store as a single 1D LongTensor for fast slicing.
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_size = context_size

    def __len__(self):
        # TODO(you): how many distinct (x, y) windows fit in self.tokens?
        return len(self.tokens) - self.context_size

    def __getitem__(self, idx):
        # TODO(you): slice out x and y (y shifted by one position).
        x = self.tokens[idx: idx + self.context_size]
        y = self.tokens[idx + 1: idx + 1 + self.context_size]
        return x, y


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dataloaders(text: str, context_size: int, batch_size: int, train_split: float = 0.9):
    """Tokenize raw text, split into train/val, wrap in DataLoaders."""
    tok = Tokenizer()
    ids = tok.encode(text)

    split_idx = int(len(ids) * train_split)
    train_ids, val_ids = ids[:split_idx], ids[split_idx:]

    train_ds = TextDataset(train_ids, context_size)
    val_ds = TextDataset(val_ids, context_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tok


if __name__ == "__main__":
    # TODO(you): point this at data/tiny.txt once you've created Phase 5's
    # ridiculously small dataset, then print one batch's x/y shapes.
    sample_text = "hello world\nhello there\nhow are you\nI like machine learning\n"
    train_loader, val_loader, tok = build_dataloaders(sample_text, context_size=4, batch_size=2)
    x, y = next(iter(train_loader))
    print("x shape:", x.shape)  # expect [B, T]
    print("y shape:", y.shape)
