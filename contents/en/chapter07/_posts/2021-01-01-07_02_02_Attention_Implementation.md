---
layout: post
title: 07-02-02 Attention Implementation
chapter: '07'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter07
---

Implementing attention cleanly is where conceptual clarity meets production engineering. This implementation hub links the core single-head & Bahdanau reference implementations with full multi-head PyTorch modules, `nn.MultiheadAttention` API comparisons, and a pre-flight debugging checklist covering shapes, masks, scaling, and `NaN` guards.

## Key ideas: Attention Engineering Checklist

- **Shape Discipline**: Consistently tracking `(B, h, T_q, d_k)` vs. `(B, h, T_k, d_k)` across self-attention ($$T_q = T_k$$) and cross-attention ($$T_q \ne T_k$$).
- **Mask Semantics**: Verifying whether a framework API expects `True = masked/ignored` (PyTorch `key_padding_mask`) or `1 = keep` (custom teaching code) before training.
- **Built-in vs. Teaching Modules**: Reconciling your transparent teaching code with `nn.MultiheadAttention` and PyTorch's fused `F.scaled_dot_product_attention` (which automatically dispatches FlashAttention on modern GPUs).

### Worked example: Catching the Silent `.view(B, h, T, d_k)` Bug

Suppose $$B = 1, T = 2, h = 2, d_k = 2$$ ($$d_{\text{model}} = 4$$) and token embeddings are:
$$\text{Token 0} = [1, 2, 3, 4], \quad \text{Token 1} = [5, 6, 7, 8].$$

1. **Correct split (`view(1, 2, 2, 2).transpose(1, 2)`)**:
   - Head 0 sees: Token 0 $$= [1, 2]$$, Token 1 $$= [5, 6]$$.
   - Head 1 sees: Token 0 $$= [3, 4]$$, Token 1 $$= [7, 8]$$.
2. **Buggy split (`view(1, 2, 2, 2)` directly without transpose)**:
   - Head 0 gets `[[1, 2], [3, 4]]` (both from Token 0!) and Head 1 gets `[[5, 6], [7, 8]]` (both from Token 1!)—completely mixing sequence positions with feature channels without throwing any Python exception!

# 07-02-02 Attention Implementation

Mathematics tells you *what* attention computes. Implementation tells you whether you actually built that object—shapes, masks, stability, and APIs included.

## Lessons in this section

1. **Core Implementation**  
   Bahdanau attention from scratch, scaled self-attention, translation-style demos, and a clear NumPy → PyTorch path.

2. **Multi-Head, Papers, and Pitfalls**  
   Multi-head modules, causal masking, a minimal seq2seq-with-attention sketch, canonical papers, and a debugging checklist used in real training runs.

## Implementation checklist (keep this open while coding)

| Check | Why it matters |
|-------|----------------|
| Softmax over the **key** axis | Weights must form a distribution over memory slots |
| Scale by $$\sqrt{d_k}$$ | Prevents saturated attention at large width |
| Mask **before** softmax with large negatives | Correct zeros + correct gradients |
| Causal mask for autoregressive decoders | Stops future leakage |
| Padding mask in batched training | Stops attending to `PAD` |
| Multi-head reshape: `(B,L,H,d)` ↔ `(B,H,L,d)` | Wrong transpose silently scrambles features |

Start with **Core Implementation**, then continue to multi-head and pitfalls.

## Video tham khảo

- **xem trước** [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown, English. So the code lessons are implementing a formula you have already computed once.
- [A Dive Into Multihead Attention, Self-Attention and Cross-Attention](https://www.youtube.com/watch?v=mmzRYGCfTzc) — Machine Learning Studio, English. The module map: self, cross, and multi-head.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, English. Attention written in Python, which is the coding lesson's target.
