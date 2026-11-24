---
layout: post
title: 08-01 The Transformer Architecture
chapter: '08'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter08
---

How did one architecture unify machine translation, code generation, protein folding, and image recognition under a single set of building blocks? Because the Transformer separates *where information flows* (self- and cross-attention) from *how each token transforms it* (position-wise MLPs), wrapped inside a deeply stable residual stream. This module hub connects the full mathematical theory of the Transformer with its step-by-step PyTorch implementation and diagnostic demos.

## Background Knowledge Map: Mastering the Transformer Module

- **Theory & Equations**: Positional encodings, Pre-LN vs. Post-LN residual blocks, FFN capacity, and encoder–decoder vs. decoder-only topologies.
- **From-Scratch Code**: Modular PyTorch layers, mask broadcasting, buffer registration, and unit tests.
- **Training & Diagnostics**: Learning-rate warmup, attention map inspection, and classic failure modes.

### Step-by-Step Worked Example (Read Slowly): Encoder-Only vs. Decoder-Only Attention Complexity

Compare a 6-layer Encoder-only model (BERT-style) and a 6-layer Decoder-only model (GPT-style) on a sequence of length $$T = 128$$:

1. **Active entries in the $$128 \times 128$$ attention matrix per head**:
   - Bidirectional Encoder: every token attends to all $$128$$ tokens $$\implies 128^2 = 16{,}384$$ active entries.
   - Causal Decoder: token $$i$$ attends only to tokens $$0 \dots i$$ $$\implies \frac{128 \times 129}{2} = 8{,}256$$ active entries (roughly $$50\%$$ of the matrix is strictly masked to zero!).
2. **Systems implication**: FlashAttention exploits this exact causal triangle to skip computing the upper-right half of the tile grid—achieving nearly a $$2\times$$ speedup on causal LLMs compared to bidirectional encoders at the same sequence length.

# 08-01 The Transformer Architecture

This section is the heart of the chapter: how attention pieces from Chapter 07 assemble into a full model.

## Lessons

1. **Transformer Theory**  
   Motivation vs RNNs, QKV self-attention inside the model, multi-head packaging, positional encodings, encoder layer, decoder layer (masked self-attn + cross-attn + FFN), residuals/LayerNorm, encoder-only vs decoder-only vs encoder–decoder families.

2. **Transformer Implementation**  
   From-scratch modules, smoke tests, a tiny training demo, papers, and production pitfalls (mask conventions, PE mistakes, KV-cache, $$O(L^2)$$ cost).

## Suggested path

| Step | Lesson | Outcome |
|------|--------|---------|
| 1 | **08-01-01 Theory** | You can write the layer equations without notes |
| 2 | **08-01-02 Implementation overview** | Module map + checklist |
| 3 | **08-01-02-01 Core code** | Working PyTorch blocks |
| 4 | **08-01-02-02 Demos & pitfalls** | Train a toy model; debug confidently |

If theory feels abstract, open the matching class in Core Code, run a forward pass on random tokens, then return to the equations.

Continue with **Transformer Theory**.

## Video tham khảo

- **xem trước** [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, English. The data flow the theory and code lessons share.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, English. The code lesson's destination: a GPT built from that block.
- [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY) — StatQuest with Josh Starmer, English. Use this if you want every tensor named before opening the code.
