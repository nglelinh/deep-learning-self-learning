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

Standard attention looks effortless on a whiteboard, yet at sequence length $$T = 32{,}768$$, materializing the $$T \times T$$ score matrix in GPU High-Bandwidth Memory (HBM) consumes gigabytes of VRAM per head and stalls Tensor Cores waiting on memory reads. Between 2022 and 2026, systems breakthroughs—**FlashAttention** (IO-aware tiling + online softmax), **PagedAttention** (virtual memory for KV-caches), and **Grouped-Query Attention (GQA)**—rewrote how attention executes on silicon, making 128K–1M-token LLMs practical without changing a single mathematical digit of exact attention.

## Background Knowledge Map: IO-Aware Kernels, KV-Cache Compression, and Long Context

- **FlashAttention (Online Softmax Tiling)**: GPU SRAM is $$>10\times$$ faster than HBM but tiny (~20 MB). FlashAttention streams blocks of $$Q, K, V$$ into SRAM and computes exact softmax incrementally using running max and normalizer statistics—reducing HBM memory from $$O(T^2)$$ to $$O(T)$$ while cutting wall-clock time $$2\times\text{–}4\times$$.
- **KV-Cache Memory Wall & GQA/MQA**: During autoregressive decoding, caching past $$K, V$$ tensors dominates GPU memory. **Multi-Query Attention (MQA)** shares 1 KV head across all query heads; **Grouped-Query Attention (GQA)** shares $$h_{\text{kv}}$$ KV heads across groups of $$h_q / h_{\text{kv}}$$ query heads (e.g., Llama 3), slashing KV-cache size by $$4\times\text{–}8\times$$ with near-zero quality loss.
- **PagedAttention (vLLM)**: Borrowing virtual memory paging from operating systems, PagedAttention stores KV-caches in non-contiguous fixed-size blocks, eliminating memory fragmentation and enabling dynamic batching and shared-prefix caching.
- **Sliding-Window & Sparse Attention**: Restricting local layers to a window $$W$$ (e.g., Mistral's 4K sliding window) bounds attention compute to $$O(T \cdot W)$$ while stacked layers still propagate information globally.

### Step-by-Step Worked Example (Read Slowly): KV-Cache Memory Savings of GQA vs. MHA

Consider a 32-layer LLM with $$d_{\text{model}} = 4{,}096$$, $$h_q = 32$$ query heads ($$d_k = 128$$), context length $$T = 8{,}192$$ tokens, stored in `bf16` (2 bytes per element) for a single request:

1. **Standard Multi-Head Attention (MHA: $$h_{\text{kv}} = 32$$)**:
   - Per layer, K and V each store $$T \times h_{\text{kv}} \times d_k = 8{,}192 \times 32 \times 128 = 33{,}554{,}432$$ elements.
   - Both K and V in `bf16`: $$2 \times 33{,}554{,}432 \times 2\text{ bytes} = 134{,}217{,}728\text{ bytes} = 128\text{ MiB/layer}$$.
   - Across 32 layers: $$32 \times 128\text{ MiB} = 4{,}096\text{ MiB} = 4.0\text{ GiB}$$ for just *one* user sequence!
2. **Grouped-Query Attention (GQA with $$h_{\text{kv}} = 8$$, group size 4)**:
   - KV heads drop by $$4\times$$ ($$32 \to 8$$).
   - Total KV-cache across 32 layers: $$4.0\text{ GiB} / 4 = 1.0\text{ GiB}$$—immediately allowing $$4\times$$ larger serving batch sizes on the exact same GPU!

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

Write the NumPy attention and multi-head pitfalls first. This lesson only adds **kernels, KV-cache, and GQA** — the products you will meet in every LLM codebase. Chapter **26-07** retells FlashAttention as tiling + online softmax (no CUDA) and puts GQA next to the decoder-block diagram; **26-04** is the serving-time KV-cache.

## Video tham khảo

- **xem trước** [How FlashAttention Accelerates Generative AI Revolution](https://www.youtube.com/watch?v=gBMO1JZav44) — Jia-Bin Huang, English. FlashAttention computes the same attention with less memory traffic, which is this note's claim.
- [FlashAttention - Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4) — Stanford MLSys Seminars, English. Tri Dao's own seminar on that algorithm. It is long and canonical; use it after the short version.
- [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown, English. The exact attention formula FlashAttention is not approximating away.
