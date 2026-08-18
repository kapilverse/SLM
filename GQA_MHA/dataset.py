"""Self-contained dataset/dataloader construction for the experiment."""

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], context_size: int):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_size = context_size

    def __len__(self):
        return len(self.tokens) - self.context_size

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.context_size]
        y = self.tokens[idx + 1: idx + 1 + self.context_size]
        return x, y


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_dataloaders(text: str, context_size: int, batch_size: int, train_split: float = 0.9):
    tok = Tokenizer()
    ids = tok.encode(text)

    split_idx = int(len(ids) * train_split)
    train_ids, val_ids = ids[:split_idx], ids[split_idx:]

    train_ds = TextDataset(train_ids, context_size)
    val_ds = TextDataset(val_ids, context_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tok
