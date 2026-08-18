"""
Phase 27 — Gradio web app for the trained SmallGPT model.

Weights are hosted in a separate Hugging Face *model* repo (not bundled
into this Space's git repo) and downloaded once at startup via
huggingface_hub. Set HF_MODEL_REPO / HF_MODEL_FILENAME below (or via env
vars) to point at your own checkpoint repo.
"""

import glob
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from config import GPTConfig
from model import SmallGPT
from tokenizer import Tokenizer
from checkpoint import load_checkpoint
from generate import generate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HF Hub model repo holding the trained checkpoint (see prepare_tinystories.py /
# train.py for how it was produced). Override via env vars if you fork this.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "kapilverse/nexa-smallgpt")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "ckpt_step30000.pt")


def find_local_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt")))
    return ckpts[-1] if ckpts else None


def resolve_checkpoint_path() -> str | None:
    """Prefer a local checkpoint (e.g. during local dev); otherwise download
    the trained weights from the HF model repo."""
    local = find_local_checkpoint()
    if local:
        return local

    print(f"Downloading checkpoint from {HF_MODEL_REPO}/{HF_MODEL_FILENAME} ...")
    return hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)


def load_model():
    cfg = GPTConfig()
    model = SmallGPT(cfg).to(DEVICE)
    tok = Tokenizer()

    ckpt_path = resolve_checkpoint_path()
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


n_params = sum(p.numel() for p in MODEL.parameters())

demo = gr.Interface(
    fn=run_generate,
    inputs=[
        gr.Textbox(label="Prompt", value="Once upon a time"),
        gr.Slider(0.1, 1.5, value=0.8, label="Temperature"),
        gr.Slider(10, 300, value=100, step=10, label="Max tokens"),
    ],
    outputs=gr.Textbox(label="Output", lines=8),
    title="🧠 Nexa — A GPT Built From Scratch",
    description=(
        f"A {n_params/1e6:.1f}M-parameter GPT-style language model, implemented from scratch in "
        "PyTorch (causal self-attention, grouped-query attention, transformer decoder blocks — "
        "no pretrained weights) and trained on the TinyStories dataset on a Colab T4 GPU.\n\n"
        "Try prompts like *\"Once upon a time\"* or *\"One day, a girl named\"* — lower temperature "
        "gives safer, more repetitive text; higher temperature gives more varied (and more incoherent) text."
    ),
    examples=[
        ["Once upon a time", 0.8, 100],
        ["The little dog", 0.7, 100],
        ["One day, a girl named", 0.9, 120],
    ],
)


if __name__ == "__main__":
    demo.launch()
