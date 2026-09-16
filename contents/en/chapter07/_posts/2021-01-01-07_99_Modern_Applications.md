---
layout: post
title: 07-99 Modern Applications and Updates (2022–2026)
chapter: '07'
order: 8
owner: Deep Learning Course
lang: en
categories:
- chapter07
lesson_type: optional
---

# Optional: attention kernels, serving, and multi-query variants

> This lesson is **optional**. It does **not** replace scaled-dot-product or multi-head math. It covers the systems and architectural variants that made attention deployable at LLM scale.

The score is still

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V.$$

From 2022 to 2026 the fight moved to **IO**, **KV-cache layout**, and **fewer key/value heads**.

## 1. FlashAttention: exact attention, better memory traffic

[Dao et al., 2022](https://arxiv.org/abs/2205.14135) tile $$Q,K,V$$ so the $$T\times T$$ matrix never sits in HBM. [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023) improves parallelism; [FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024) targets Hopper (asynchrony + FP8). PyTorch’s `scaled_dot_product_attention` dispatches to these kernels when shapes allow.

This is the same softmax as the theory lesson. The novelty is the **algorithmic implementation**.

## 2. Concrete applications

### PagedAttention and LLM serving

[Kwon et al., 2023](https://arxiv.org/abs/2309.06180) (vLLM) store the KV cache in paged blocks so many decoded sequences can share GPU memory without reservation waste. That is attention *serving*, not a new score function.

### GQA / MQA

Multi-query ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) and grouped-query attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245), GQA) let many query heads share one or a few KV heads. Llama 2/3 and Mistral use GQA so the cache is

$$\text{KV bytes} \propto T \cdot n_{\text{kv}} \cdot d, \quad n_{\text{kv}} \ll n_{\text{heads}}.$$

### Sliding-window and linear attention

Mistral-style **sliding-window** attention and various linear-attention papers trade the full $$T\times T$$ pattern for $$O(T)$$ cost. Use them when the Chapter 07 quadratic warning becomes a product constraint.

## 3. Widely used software

- PyTorch `F.scaled_dot_product_attention` (SDPA) / [FlashAttention](https://github.com/Dao-AILab/flash-attention).
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention serving.
- [xFormers](https://github.com/facebookresearch/xformers) — memory-efficient attention ops.

## 4. Citations (2022–2026)

- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135).
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691).
- [FlashAttention-3 (Shah et al., 2024)](https://arxiv.org/abs/2407.08608).
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180).
- [GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245).

## 5. How this complements the core notes

Write the NumPy attention and multi-head pitfalls first. This lesson only adds **kernels, KV-cache, and GQA** — the products you will meet in every LLM codebase.
