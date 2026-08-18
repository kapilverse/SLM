<div align="center">

# 🧠 MySmallGPT

### A GPT-Style Language Model Built From Scratch in PyTorch

*No pretrained weights. No `GPT2LMHeadModel`. Every layer hand-written and verified.*

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![tiktoken](https://img.shields.io/badge/tiktoken-BPE-4B8BBE?logo=openai&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?logo=gradio&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-T4-76B900?logo=nvidia&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-TinyStories-9146FF)
![License](https://img.shields.io/badge/License-MIT-informational)

</div>

---

A GPT-style autoregressive language model built from scratch in PyTorch —
no `AutoModelForCausalLM`, no `GPT2LMHeadModel`, no pretrained Transformer
implementation. Implements token embeddings, positional embeddings, causal
self-attention, grouped-query attention (GQA), Transformer decoder blocks,
AdamW optimization, checkpointed GPU training, autoregressive generation,
and Gradio deployment. Trained on TinyStories using an NVIDIA T4 GPU
(Google Colab free tier).

## Project structure

- `config.py` — model + training hyperparameters
- `tokenizer.py` — GPT-2 BPE tokenizer (tiktoken, encode/decode only)
- `dataset.py` — text -> token windows -> DataLoader
- `embeddings.py` — token + positional embeddings
- `attention.py` — causal self-attention, causal masking, multi-head attention
- `gqa.py` — grouped-query attention
- `transformer.py` — feed-forward network, LayerNorm, Transformer decoder block
- `model.py` — SmallGPT: full model assembly + parameter counting
- `train.py` — training loop, GPU device handling, checkpointing
- `checkpoint.py` — save/load model + optimizer state
- `generate.py` — autoregressive generation with temperature + top-k sampling
- `evaluate.py` — loss / perplexity evaluation
- `app.py` — Gradio web app

## How to work through this

Follow the phased plan (see conversation / project notes) in order:
Phase 0-3 (foundations) -> Phase 4-8 (data pipeline) -> Phase 9-15
(attention & Transformer block) -> Phase 16-18 (full model) -> Phase 19-24
(training on Colab T4) -> Phase 25-26 (generation & evaluation) -> Phase 27
(Gradio deployment).

Each file above has `TODO(you)` comments marking the parts you should
implement/verify yourself rather than just running as-is. Test each file's
`if __name__ == "__main__":` block in isolation before moving to the next.

## Quickstart (Phase 19 — overfit a tiny dataset)

```bash
pip install -r requirements.txt
python train.py
```

This trains on `data/tiny.txt` (4 lines) — training loss should drop close
to 0, confirming the model can memorize a tiny dataset before you scale up
to TinyStories.

## Scaling up (Phase 20-21 — TinyStories on Colab T4)

1. Download the TinyStories dataset (e.g. via `datasets` from Hugging Face —
   only the *data*, not any pretrained model).
2. Point `train.py`'s `data_path` at the combined text, or adapt
   `dataset.py`/`train.py` to stream it directly.
3. Run in Google Colab with a T4 GPU runtime. `train.py` auto-detects CUDA.
4. Checkpoints save to `checkpoints/` every `checkpoint_interval` steps —
   resume via `train(resume_from="checkpoints/ckpt_stepXXXX.pt")`.

## Generating text

```bash
python generate.py
```

## Deploying

```bash
python app.py
```

Then push to a Hugging Face Space for public deployment.
