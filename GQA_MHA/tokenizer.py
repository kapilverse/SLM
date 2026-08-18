"""Self-contained tokenizer wrapper (GPT-2 BPE via tiktoken). No pretrained
model weights involved -- this is a fixed encoding scheme only."""

import tiktoken


class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: list[int]) -> str:
        return self.enc.decode(token_ids)
