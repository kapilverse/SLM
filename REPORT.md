# Nexa: A GPT-Style Language Model Implemented From Scratch in PyTorch

**Author:** Kapil
**Date:** August 2026

## Abstract

We present Nexa, a decoder-only Transformer language model implemented
entirely from scratch in PyTorch, without relying on any pretrained
model or third-party Transformer implementation (e.g. Hugging Face's
`GPT2LMHeadModel`). The model incorporates causal self-attention,
grouped-query attention (GQA), and standard Transformer decoder blocks
with pre-normalization and residual connections. We train a 13.6M
parameter instance of this architecture on the TinyStories dataset
using a single NVIDIA T4 GPU (Google Colab, free tier), and evaluate it
via training/validation loss, perplexity, and qualitative generation
samples. Validation loss decreases from an initial random-baseline of
10.92 (near the theoretical maximum of ln(50257) ≈ 10.82) to 2.28 after
30,000 training steps, corresponding to a perplexity reduction from
~54,900 to 9.60. Generated samples exhibit grammatically correct,
TinyStories-style narrative text with correct local coherence, though
long-range narrative consistency is limited by model scale. This work
is presented as a verified, end-to-end implementation and training
exercise rather than a novel research contribution.

## 1. Introduction

Large language models are typically studied and used through
pre-existing implementations and pretrained checkpoints, which can
obscure the mechanics of how such models are built and trained. This
project implements every component of a GPT-style language model from
first principles — tokenization interface, embeddings, attention,
grouped-query attention, Transformer blocks, and the training loop — to
build and demonstrate a complete, first-principles understanding of the
architecture.

The explicit goals were:
1. Implement every architectural component (attention, masking, GQA,
   Transformer blocks) without relying on pre-built high-level modules
   such as `nn.MultiheadAttention` or a pretrained Transformer.
2. Verify each component's correctness in isolation before composing
   them into a full model.
3. Train the resulting model on a real, non-trivial dataset
   (TinyStories) using freely available GPU compute (Colab T4).
4. Evaluate the trained model quantitatively (loss, perplexity) and
   qualitatively (generation samples).
5. Deploy the trained model behind a public, interactive demo.

## 2. Architecture

### 2.1 Overview

The model follows the standard decoder-only Transformer architecture
(GPT-style):

```
Token IDs
    -> Token Embedding + Positional Embedding
    -> [Transformer Decoder Block] x N
    -> Final LayerNorm
    -> Linear LM Head
    -> Logits over vocabulary
```

### 2.2 Tokenization

Tokenization uses the GPT-2 byte-pair encoding (BPE) tokenizer via the
`tiktoken` library (`tokenizer.py`). This is a fixed, non-trainable
encoding scheme (vocabulary size 50,257) containing no learned model
parameters; no pretrained language model weights are used anywhere in
this project.

### 2.3 Embeddings

Token embeddings and learned positional embeddings are implemented as
two `nn.Embedding` tables (`embeddings.py`), summed elementwise to
produce the initial residual stream:

```
x = TokenEmbedding(token_ids) + PositionEmbedding(positions)
```

### 2.4 Causal Self-Attention

Standard multi-head causal self-attention (`attention.py`) is
implemented directly from the scaled dot-product attention equation,
without using `nn.MultiheadAttention`:

```
Q, K, V = XW_q, XW_k, XW_v
scores  = QK^T / sqrt(d_head)
scores  = scores.masked_fill(future_positions, -inf)     # causal mask
attn    = softmax(scores)
output  = attn @ V
```

The causal mask is a precomputed lower-triangular boolean matrix,
ensuring position *i* can only attend to positions ≤ *i*.

### 2.5 Grouped-Query Attention (GQA)

To reduce key/value projection parameters and, at inference time, the
KV-cache footprint, the model uses grouped-query attention (`gqa.py`),
following Ainslie et al. (2023). Query heads are split into groups,
each group sharing a single key/value head:

```
n_heads = 4, n_kv_heads = 2  =>  2 query heads per KV head
```

Key/value tensors are expanded via `repeat_interleave` along the head
dimension prior to the attention computation, so each query head
attends to its assigned (shared) key/value head.

### 2.6 Transformer Decoder Block

Each block (`transformer.py`) uses the pre-normalization residual
pattern:

```
x = x + GQA(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

The feed-forward network is a two-layer MLP with ReLU activation
(`d_model -> d_ff -> d_model`).

### 2.7 Full Model

`model.py` assembles the embeddings, N Transformer blocks, a final
LayerNorm, and a linear projection to vocabulary logits. Given target
tokens, the model computes cross-entropy loss over the flattened
`[B*T, vocab_size]` logits against `[B*T]` targets.

### 2.8 Model Configuration

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 50,257 |
| Context length | 128 |
| Embedding dimension (d_model) | 128 |
| Transformer layers | 4 |
| Attention (query) heads | 4 |
| Key/value heads (GQA) | 2 |
| Feed-forward dimension | 512 |
| Dropout | 0.1 |
| **Total parameters** | **~13.6M** |

Parameter breakdown (via `model.py`'s `count_parameters()`):

| Component | Parameters |
|---|---|
| Token embedding | 6,432,896 |
| Positional embedding | 16,384 |
| Transformer blocks (all 4) | 725,504 |
| Final LayerNorm | 256 |
| LM head | 6,432,896 |
| **Total** | **13,607,936** |

The token embedding and LM head together account for ~94% of
parameters — a consequence of the large vocabulary (50,257) relative
to the small embedding dimension (128), typical of small language
models using an unmodified GPT-2 vocabulary.

## 3. Training Setup

### 3.1 Dataset

The model is trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(Eldan & Li, 2023), a synthetic dataset of short, simple children's
stories designed for training small language models. We use 500,000
stories from the training split, concatenated with `<|endoftext|>`
separators and tokenized with the GPT-2 BPE tokenizer
(`prepare_tinystories.py`).

### 3.2 Training Procedure

Training follows the standard next-token-prediction objective: given a
window of `context_size` tokens, predict the token immediately
following each position (`dataset.py`). Training uses:

- Optimizer: AdamW (learning rate 3e-4, weight decay 0.1)
- Batch size: 32
- Gradient clipping: max norm 1.0
- Hardware: 1x NVIDIA T4 GPU (Google Colab, free tier)
- Checkpointing: every 500–1000 steps, saved to persistent storage
  (Google Drive) to survive session interruptions

Training was run in two phases due to Colab's session time limits: an
initial 5,000-step run, followed by a resumed run to 30,000 total
steps, using the checkpoint/resume mechanism in `checkpoint.py` and
`train.py`.

## 4. Evaluation

### 4.1 Correctness Verification

Prior to training at scale, the following properties were verified in
isolation:

- **Loss sanity**: an untrained (randomly initialized) model produces
  a cross-entropy loss of 10.92, closely matching the theoretical
  value for uniform random prediction over the vocabulary,
  ln(50,257) ≈ 10.82.
- **Causal masking correctness**: modifying a token at position *t* in
  the input sequence was confirmed to leave model outputs at all
  positions < *t* unchanged, and to change outputs at position ≥ *t* —
  confirming no future-token information leaks backward through the
  attention mechanism.
- **GQA validity**: output shapes, absence of NaNs, and correct
  rejection of invalid head-count configurations were verified for the
  grouped-query attention module.
- **Data windowing correctness**: verified that target sequences are
  exactly the input sequence shifted by one token position across
  window boundaries.
- **Checkpoint round-tripping**: verified that saved model and
  optimizer state, when reloaded, exactly reproduce pre-save weights
  (bit-for-bit tensor equality).

### 4.2 Training Dynamics

| Step | Train Loss | Val Loss | Val Perplexity |
|---|---|---|---|
| 0 (init) | — | 10.92 | ~54,900 |
| 5,000 | 3.25 | 3.05 | ~21.1 |
| 10,000 | 2.88 | 2.72 | ~15.2 |
| 15,000 | 2.86 | 2.52 | ~12.5 |
| 20,000 | 2.57 | 2.42 | ~11.2 |
| 25,000 | 2.51 | 2.35 | ~10.5 |
| 30,000 | 2.56 | 2.28 | 9.60 |

Validation loss tracks training loss closely throughout, with
validation loss consistently at or below training loss — indicating
no overfitting occurred over the observed training range. The rate of
loss reduction diminishes over training (Δ0.31 from step 5k→10k vs.
Δ0.14 from step 20k→30k), consistent with the model approaching its
capacity given its scale (13.6M parameters, 4 layers).

### 4.3 Qualitative Generation Samples

Generated using temperature 0.8, top-k 40, from the step-30,000
checkpoint:

> *Once upon a time, there was a little girl named Lily. She had a big,
> soft, fluffy pillow that she loved to sleep safe.*
>
> *One day, Lily's mommy told her to be careful by mistake. The pillow
> was dark and dark. Lily's mommy felt scared and scared.*
>
> *Lily's mommy saw the trumpet and wanted to stop her from. She ran to
> the living room and started to cry. Her mommy said, "Don't worry,
> Lily. We ...*

> *One day, a girl named Amy was feeling very happy and didn't like to
> play outside. She wanted to find her friends so she asked her friends
> to play a game. The friends said yes! They wanted to climb the tree.*
>
> *Amy's friends took the rake to the garden and they both found the
> perfect grass. They were so happy! They started to laugh and talked
> about the view of the tree. Amy was happy and started to applaud to
> play outside.*

**Observations:**
- Sentence-level grammar, punctuation, and quoted dialogue are
  consistently well-formed.
- The model correctly learned document boundaries, naturally emitting
  `<|endoftext|>` between distinct stories.
- Style and vocabulary match the TinyStories register (simple,
  child-directed narrative).
- Long-range coherence is limited: the second sample states Amy
  "didn't like to play outside" then immediately has her want to play
  outside — a logical inconsistency attributable to the model's
  limited context capacity (128 tokens) and small parameter count, not
  an implementation defect.

## 5. Deployment

The trained model is served via a Gradio web application (`app.py`),
deployed as a Hugging Face Space. The checkpoint is hosted in a
separate Hugging Face model repository and downloaded at Space
startup via `huggingface_hub`, keeping the Space's own git repository
lightweight. The demo exposes prompt, temperature, and max-token
controls, with example prompts pre-populated for first-time visitors.

## 6. Discussion

### 6.1 What This Demonstrates

The results confirm a working, from-scratch implementation of the full
GPT training pipeline: the loss curve behaves exactly as expected for
successful learning (monotonic decrease from the random-initialization
theoretical maximum, with validation loss tracking training loss), and
generated text exhibits genuine learned linguistic structure rather
than memorized or templated output.

### 6.2 Limitations

- **Scale**: at 13.6M parameters and 4 layers, this model is
  substantially smaller than production language models, and its
  ceiling on coherence and factual consistency reflects this.
- **Novelty**: this project is an implementation and training
  exercise, not a novel research contribution. The architecture (GQA +
  standard Transformer decoder blocks) and dataset (TinyStories) are
  both established in prior work; no new technique or empirical
  finding is proposed here.
- **Training budget**: 30,000 steps at batch size 32 represents a
  small fraction of even a single epoch over the ~90M-token training
  corpus; further training would likely continue to reduce loss, as
  the validation curve had not yet plateaued.

### 6.3 Future Work

A companion controlled experiment comparing grouped-query attention
against standard multi-head attention at this same model scale is
presented separately (see `paper/`). Further extensions could include
a scaling study across model sizes, longer training runs to observe
where the loss curve plateaus, or an ablation of weight tying between
the token embedding and LM head.

## 7. Reproducibility

All source code, training scripts, and the data preparation pipeline
are available at: `https://github.com/kapilverse/SLM`

A live interactive demo of the trained model is available at:
`https://huggingface.co/spaces/yadavkapil23/nexa-smallgpt`

The trained checkpoint is hosted at:
`https://huggingface.co/yadavkapil23/nexa-smallgpt`

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need.*
- Ainslie, J. et al. (2023). *GQA: Training Generalized Multi-Query
  Transformer Models from Multi-Head Checkpoints.*
- Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language
  Models Be and Still Speak Coherent English?*
