<div align="center">

#  Nexa

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

A GPT-style autoregressive language model built from scratch in PyTorch -
no `AutoModelForCausalLM`, no `GPT2LMHeadModel`, no pretrained Transformer
implementation. Implements token embeddings, positional embeddings, causal
self-attention, grouped-query attention (GQA), Transformer decoder blocks,
AdamW optimization, checkpointed GPU training, autoregressive generation,
and Gradio deployment. Trained on TinyStories using an NVIDIA T4 GPU
(Google Colab free tier).

## Project structure

- `config.py` - model + training hyperparameters
- `tokenizer.py` - GPT-2 BPE tokenizer (tiktoken, encode/decode only)
- `dataset.py` - text -> token windows -> DataLoader
- `embeddings.py` - token + positional embeddings
- `attention.py` - causal self-attention, causal masking, multi-head attention
- `gqa.py` - grouped-query attention
- `transformer.py` - feed-forward network, LayerNorm, Transformer decoder block
- `model.py` - SmallGPT: full model assembly + parameter counting
- `train.py` - training loop, GPU device handling, checkpointing
- `checkpoint.py` - save/load model + optimizer state
- `generate.py` - autoregressive generation with temperature + top-k sampling
- `evaluate.py` - loss / perplexity evaluation
- `app.py` - Gradio web app

## Build Roadmap

| Stage | Covers | Status |
|---|---|---|
| Foundations | PyTorch tensors, autograd, GPU basics |  Done |
| Data pipeline | Tokenizer, dataset windows, embeddings |  Done |
| Attention | Causal self-attention, GQA, Transformer block |  Done |
| Full model | `SmallGPT` assembly, parameter counting |  Done |
| Training | Colab T4 training, checkpointing |  Done - 30,000 steps on TinyStories |
| Evaluation | Loss, perplexity, generation samples |  Done |
| Deployment | Gradio demo on Hugging Face Spaces |  Done |

Each file has `TODO(you)` comments marking the parts worth tracing
yourself to understand the mechanics. Every file's
`if __name__ == "__main__":` block can be run standalone as a sanity
check.

## Quickstart (overfit a tiny dataset)

```bash
pip install -r requirements.txt
python train.py
```

This trains on `data/tiny.txt` (4 lines) - training loss should drop close
to 0, confirming the model can memorize a tiny dataset before you scale up
to TinyStories.

## Training on TinyStories (Colab T4)

1. Download and flatten TinyStories:
   ```bash
   python prepare_tinystories.py --output data/tinystories.txt --num-stories 500000
   ```
2. Train (auto-detects CUDA):
   ```python
   from train import train
   model, tok = train(data_path="data/tinystories.txt")
   ```
3. Checkpoints save every `checkpoint_interval` steps — resume with:
   ```python
   train(data_path="data/tinystories.txt", resume_from="checkpoints/ckpt_stepXXXX.pt")
   ```

## Results

Trained for 30,000 steps on 500K TinyStories, on a single Colab T4 GPU:

| Metric | Random init | Step 30,000 |
|---|---|---|
| Validation loss | 10.92 | 2.28 |
| Validation perplexity | ~54,900 | 9.60 |

Sample generation (`temperature=0.8, top_k=40`):

> *Once upon a time, there was a little girl named Lily. She had a big,
> soft, fluffy pillow that she loved to sleep safe. One day, Lily's
> mommy told her to be careful by mistake...*

## Generating text

```bash
python generate.py
```

## Live Demo

Try the trained model directly in your browser:

**Live demo (Gradio Space): https://huggingface.co/spaces/yadavkapil7155/nexa-smallgpt**

**Trained model/checkpoint repo: https://huggingface.co/yadavkapil7155/nexa-smallgpt**

**GitHub repo: https://github.com/kapilverse/SLM**

**Technical report (in the repo): https://github.com/kapilverse/SLM/REPORT.md**

**Research Paper: https://zenodo.org/records/22002958**

**TinyStories dataset (if you want to credit it): https://huggingface.co/datasets/roneneldan/TinyStories**
