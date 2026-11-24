---
layout: post
title: 07-02 Attention Mechanisms in Depth
chapter: '07'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter07
---

Once you see how Bahdanau alignment rescues an RNN translator from the fixed-vector bottleneck, the next natural leap is mathematical abstraction: strip away the RNN entirely and ask what happens when attention operates as a standalone, batched matrix operator with scaled dot products and multiple parallel heads. This module bridges conceptual intuition to the exact linear algebra, variance scaling, and PyTorch multi-head code that power modern Transformers.

## Background Knowledge Map: Deepening Attention from Alignment to Multi-Head Tensors

- **Exact Scaled Dot-Product Algebra**: Deriving why dividing $$Q K^\top$$ by $$\sqrt{d_k}$$ stabilizes softmax variance across growing head dimensions.
- **Masking Geometry**: Constructing padding masks (to ignore `<PAD>` tokens) and causal upper-triangular masks (to enforce autoregressive causality) by injecting $$-\infty$$ *before* softmax.
- **Multi-Head Subspace Decomposition**: Splitting $$d_{\text{model}}$$ into $$h$$ heads of dimension $$d_k = d_{\text{model}}/h$$ so the model can simultaneously track syntactic dependencies, coreference, and positional proximity.

### Step-by-Step Worked Example (Read Slowly): Shape & Cost Audit of Batched Attention

Consider a batch of $$B = 4$$ sentences of length $$T = 16$$ with model dimension $$d_{\text{model}} = 64$$ split into $$h = 8$$ heads ($$d_k = 64 / 8 = 8$$).

1. **Projected Q, K, V shapes after head splitting**:
   `Q, K, V` each have shape `(B, h, T, d_k) = (4, 8, 16, 8)`.
2. **Attention score matrix `Q @ K.T`**:
   Shape is `(4, 8, 16, 16)` — holding $$4 \times 8 \times 256 = 8{,}192$$ pairwise alignment scores.
3. **Weighted value output `attn @ V` and head re-concatenation**:
   `(4, 8, 16, 16) @ (4, 8, 16, 8) -> (4, 8, 16, 8)`, which transposes and reshapes cleanly back to `(4, 16, 64)` before the final output projection $$W_O \in \mathbb{R}^{64 \times 64}$$.

# 07-02 Attention Mechanisms in Depth

This block goes **beyond the fundamentals lesson**. You already know *why* attention exists and the basic formulas. Here we tighten the mathematics, then implement every moving part carefully.

## What you will study

1. **Attention Mathematics**  
   Unified score functions, vectorized QKV forms, self-attention as an $$L\times L$$ interaction, multi-head algebra, padding and causal masks as pre-softmax operations, complexity and path-length arguments.

2. **Attention Implementation**  
   From-scratch Bahdanau and scaled attention, multi-head modules, masking demos, a minimal seq2seq-with-attention sketch, papers, and production pitfalls.

## How to read this block

| Order | Lesson | Outcome |
|-------|--------|---------|
| 1 | **07-02-01 Attention Mathematics** | You can write every formula without looking it up |
| 2 | **07-02-02 Implementation** (overview) | Map math → code modules |
| 3 | **07-02-02-01 Core Implementation** | Working Bahdanau + self-attention code |
| 4 | **07-02-02-02 Multi-Head, Papers, Pitfalls** | Multi-head, causal masks, literature, debugging checklist |

If a formula in 07-02-01 feels abstract, jump to the matching demo in 07-02-02-01, then return. Math and code are meant to reinforce each other, not compete.

## Connection to the rest of the course

- **Back to 07-01** for intuition and historical motivation.  
- **Forward to Chapter 08 (Transformers)** for stacking self-attention with feed-forward blocks, residuals, and positional encodings.

Proceed to **Attention Mathematics** when ready.

## Video tham khảo

- **xem trước** [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown, English. The chapter's deep block: scaled dot-product attention.
- [Attention for Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=PSs6nxngL6k) — StatQuest with Josh Starmer, English. The sequence-model version, so self-attention is not the only pattern.
- [A Dive Into Multihead Attention, Self-Attention and Cross-Attention](https://www.youtube.com/watch?v=mmzRYGCfTzc) — Machine Learning Studio, English. Multi-head, self-attention, and cross-attention, which this block separates.
