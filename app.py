"""
Phase 27 — Gradio web app for the trained SmallGPT model.
"""

import glob
import os

import gradio as gr
import torch

from config import GPTConfig
from model import SmallGPT
from tokenizer import Tokenizer
from checkpoint import load_checkpoint
from generate import generate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt")))
    return ckpts[-1] if ckpts else None


def load_model():
    cfg = GPTConfig()
    model = SmallGPT(cfg).to(DEVICE)
    tok = Tokenizer()

    ckpt_path = find_latest_checkpoint()
    if ckpt_path:
        load_checkpoint(ckpt_path, model, map_location=DEVICE)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found — using randomly initialized weights.")

    model.eval()
    return model, tok


MODEL, TOKENIZER = load_model()


def run_generate(prompt: str, temperature: float, max_tokens: int):
    if not prompt.strip():
        return ""
    return generate(
        MODEL, TOKENIZER, prompt,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=40,
        device=DEVICE,
    )


demo = gr.Interface(
    fn=run_generate,
    inputs=[
        gr.Textbox(label="Prompt", value="Once upon a time"),
        gr.Slider(0.1, 1.5, value=0.8, label="Temperature"),
        gr.Slider(10, 300, value=100, step=10, label="Max tokens"),
    ],
    outputs=gr.Textbox(label="Output"),
    title="My Small GPT",
    description="A GPT-style language model built from scratch in PyTorch, trained on TinyStories.",
)


if __name__ == "__main__":
    demo.launch()
