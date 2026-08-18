# Grouped-Query Attention vs. Standard Multi-Head Attention: A Controlled Comparison at Small Scale

**Author:** Kapil
**Date:** August 2026

## Abstract

We present a controlled empirical comparison between grouped-query
attention (GQA) and standard multi-head attention (MHA) in a small
(~13.6M parameter) GPT-style decoder-only Transformer, trained from
scratch on the TinyStories dataset. Holding architecture, random seed,
data, and training budget (30,000 steps) fixed, and varying only the
number of key/value heads (2 for GQA vs. 4 for MHA, with 4 query heads
throughout), we find the two configurations reach nearly identical
validation loss (2.2500 for GQA vs. 2.2451 for MHA; perplexity 9.49 vs.
9.44) while GQA uses exactly half the key/value projection parameters
(65,536 vs. 131,072). This small, controlled result is consistent with
the finding reported for GQA at much larger scale (Ainslie et al.,
2023): the technique gives a real parameter/memory saving at negligible
cost to model quality, and this cost remains negligible even at a scale
three-plus orders of magnitude smaller than where GQA is typically
studied.

## 1. Motivation

The GPT implementation underlying this study (see companion technical
report, `../REPORT.md`) uses grouped-query attention because it is a
well-known efficiency technique in modern LLMs (LLaMA, Mistral, etc.),
not because its cost/benefit tradeoff had been verified independently
at small scale. This experiment isolates that one architectural choice
and measures it directly, rather than assuming the large-scale result
transfers unchanged.

**Research question**: at a fixed, small parameter budget and fixed
training budget, how much does reducing the number of key/value heads
(GQA) cost in validation loss, compared to standard multi-head
attention, and how much does it save in key/value parameter count?

## 2. Experimental Setup

Both configurations share an identical architecture and training
procedure, differing only in `n_kv_heads`:

| | GQA | MHA |
|---|---|---|
| Query heads | 4 | 4 |
| KV heads | 2 | 4 |
| KV group size | 2 (each KV head shared by 2 query heads) | 1 (standard MHA) |

Shared configuration (see `config.py`):

| Hyperparameter | Value |
|---|---|
| d_model | 128 |
| Layers | 4 |
| Context length | 128 |
| Feed-forward dim | 512 |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Dropout | 0.1 |
| Batch size | 32 |
| Optimizer | AdamW (lr=3e-4, weight_decay=0.1) |
| Training steps | 30,000 |
| Random seed | 1337 (identical for both runs) |
| Dataset | 500,000 TinyStories, 90/10 train/val split |
| Hardware | 1x GPU (Kaggle, T4-class) |

Both models were trained sequentially in the same script
(`run_experiment.py`) with the same seed set immediately before model
construction, so both models start from comparable initialization
conditions and see data in the same order.

## 3. Results

### 3.1 Validation Loss and Perplexity

| Configuration | KV heads | Final val loss (step 30,000) | Val perplexity |
|---|---|---|---|
| GQA | 2 | 2.2500 | 9.49 |
| MHA | 4 | 2.2451 | 9.44 |
| **Difference (MHA − GQA)** | | **−0.0049** | **−0.05** |

The two configurations converge to nearly identical validation loss.
The gap (0.0049 nats, well under 1% relative difference in perplexity)
is small enough that, without repeated-seed runs, it cannot be
confidently distinguished from run-to-run noise — but it is clearly not
a large or practically meaningful quality cost.

### 3.2 Full Training Curves

Both models show smooth, monotonic loss decreases with no signs of
instability or overfitting (validation loss tracks training loss
closely throughout for both configurations):

- **GQA**: val loss 6.04 (step 250) → 2.90 (step 8,750) → 2.25 (step 30,000)
- **MHA**: val loss 6.04 (step 250) → 2.79 (step 8,750) → 2.245 (step 30,000)

The two curves are visually near-indistinguishable across the full
30,000-step run (see raw logs in `logs_gqa.json` / `logs_mha.json`
produced by `run_experiment.py`).

### 3.3 Parameter Cost

| Configuration | KV projection params | Total params |
|---|---|---|
| GQA | 65,536 | 13,607,936 |
| MHA | 131,072 | 13,673,472 |
| **Savings (GQA vs. MHA)** | **65,536 (50%)** | **65,536 (0.48% of total)** |

GQA exactly halves the key/value projection parameters, as expected
(2 KV heads vs. 4, everything else identical). At this small model
scale, this is a modest 0.48% reduction in *total* parameters, because
KV projections are a small fraction of a model whose parameter count is
dominated by the token embedding and LM head (see companion report,
Section 2.8). The practical benefit of GQA is understated by this
total-parameter view — its larger real-world payoff is in **KV-cache
memory during autoregressive inference**, which scales with KV
projection size, not total parameter count, and matters increasingly at
longer context lengths and larger batch sizes than used here.

## 4. Discussion

### 4.1 What This Shows

At this small scale (13.6M parameters, 128-token context, 4 layers),
halving the number of key/value heads via GQA produces no meaningfully
detectable quality cost relative to standard multi-head attention,
while providing an exact, verifiable halving of key/value parameters.
This corroborates — independently, at a scale roughly three orders of
magnitude smaller than typically studied — the core claim of the
original GQA work: quality is preserved while attention's KV footprint
shrinks.

### 4.2 Limitations

- **Single seed**: results are from one training run per configuration.
  We report the observed difference (0.0049 loss) but do not have a
  variance estimate across seeds, so we cannot rule out that a
  different seed could show a somewhat larger or smaller gap. The
  difference observed is small enough that it may not be statistically
  distinguishable from seed-to-seed noise.
- **Single scale**: this study covers exactly one model size and one
  GQA group size (2:1). It does not establish how the GQA/MHA gap
  changes at larger scale, longer context, or different group ratios
  (e.g. 4:1, 8:1).
- **A third configuration (weight-tied embeddings) was planned but not
  completed** due to repeated training-session interruptions on the
  compute platform used; this report is scoped to the GQA-vs-MHA
  comparison only.
- **Generation-speed comparison was not completed** for the same
  reason; this report presents the parameter-count and loss comparison
  only, not an inference-latency measurement.

### 4.3 Future Work

A natural extension is repeating this comparison across multiple seeds
to establish whether the observed 0.0049 loss gap is statistically
meaningful, and across multiple GQA group sizes (e.g. 4:1, 8:1 group
ratios) to characterize how the quality/parameter tradeoff curve
behaves as KV sharing increases. Measuring actual inference latency and
KV-cache memory at longer context lengths, where GQA's benefit is
expected to be most visible, would also strengthen the practical case
made here.

## 5. Reproducibility

All code for this experiment is self-contained in this folder
(`paper/`), independent of the parent project's implementation. To
reproduce:

```bash
pip install -r ../requirements.txt
python prepare_tinystories.py --output data/tinystories.txt --num-stories 500000
python run_experiment.py --data-path data/tinystories.txt --max-steps 30000
```

Both configurations use `seed=1337` (see `config.py`'s `TrainConfig`),
fixed via `torch.manual_seed()` immediately before model construction
in `train.py`.

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need.*
- Ainslie, J. et al. (2023). *GQA: Training Generalized Multi-Query
  Transformer Models from Multi-Head Checkpoints.*
- Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can Language Models
  Be and Still Speak Coherent English?*
