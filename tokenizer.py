"""
Phase 4 — Tokenization.

We reuse the GPT-2 BPE tokenizer via tiktoken (encoding/decoding only —
no pretrained weights, no pretrained model). Everything downstream
(embeddings, attention, training) is built from scratch.
"""

import tiktoken


class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: list[int]) -> str:
        return self.enc.decode(token_ids)


if __name__ == "__main__":
    # TODO(you): sanity check — encode("hello world") then decode it back
    # and confirm you get the original string. Print the token ids too.
    tok = Tokenizer()
    ids = tok.encode("hello world")
    print(ids)
    print(tok.decode(ids))
