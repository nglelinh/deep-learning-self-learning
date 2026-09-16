---
layout: post
title: 03-99 Modern Applications and Updates (2022–2026)
chapter: '03'
order: 14
owner: Deep Learning Course
lang: en
categories:
- chapter03
lesson_type: optional
---

# Optional: backpropagation as a compiler problem

> This lesson is **optional**. It does **not** replace loss-function, gradient-descent, or backpropagation theory. It maps those algorithms onto the compilers and kernels used to train models after 2022.

Backprop is still reverse-mode autodiff of a scalar loss $$\mathcal{L}$$. The 2022–2026 story is that **the tape is compiled**: operators are fused, activations are rematerialized, and mixed precision is the default.

## 1. The same chain rule, a different scheduler

If a layer computes $$\mathbf{y} = f(\mathbf{x}; \theta)$$, reverse mode needs $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$$ and $$\frac{\partial \mathcal{L}}{\partial \theta}$$. A compiler (`torch.compile`, XLA, Triton) decides:

- which intermediates to **keep** versus **recompute** (activation checkpointing),
- which elementwise ops to **fuse** into one kernel,
- whether to run in BF16/FP8 with FP32 accumulation.

That does not change the math of Chapter 03; it changes wall-clock time and memory.

## 2. Concrete applications

### `torch.compile` and JAX `jit`

[Ansel et al., 2024](https://arxiv.org/abs/2404.14294) describe PyTorch 2’s Dynamo + AOTAutograd + inductor path: Python bytecode is captured into an FX graph, then lowered. JAX does the same idea with `jit`/`grad` composition. Both are production answers to “I wrote backprop by hand in NumPy; how does industry run it?”

### Mixed precision and loss scaling

A typical update is

$$\theta \leftarrow \theta - \eta \cdot \mathrm{Cast}_{fp32}(\widehat{\nabla \mathcal{L}}),$$

where $$\widehat{\nabla \mathcal{L}}$$ was computed in BF16/FP16. Overflow is avoided by scaling the loss before the backward pass (native AMP in PyTorch). This is the same SGD/Adam step as the theory notes, with a numeric format policy.

### Kernel-level backward for attention

[Dao et al., 2022](https://arxiv.org/abs/2205.14135) (FlashAttention) derive an **IO-aware backward** that never materializes the full $$N\times N$$ attention matrix. The gradients of softmax-attention are identical to the Chapter 07 formulas; the implementation is a fused backward kernel. We point to it here because it is the most famous “backprop engineering” result of the period.

## 3. Widely used software

- [PyTorch 2](https://pytorch.org/) — `torch.compile`, `torch.func`, `torch.utils.checkpoint`.
- [JAX](https://github.com/google/jax) — `grad`, `value_and_grad`, XLA.
- [Triton](https://github.com/triton-lang/triton) — write the backward kernel when the compiler is not enough.

## 4. Citations (2022–2026)

- [PyTorch 2 (Ansel et al., 2024)](https://arxiv.org/abs/2404.14294) — dynamic graph capture + compiled autograd.
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — exact attention gradients with tiled, IO-aware backward.
- [Online normalizer computation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867) — the streaming softmax used inside those kernels (still the numeric backbone in 2024–2026 code).

## 5. How this complements the core notes

Work the backprop worksheet and the NumPy trainer first. This lesson only answers “what industry added around the tape”: compilers, AMP, and fused backwards — not a new learning rule.
