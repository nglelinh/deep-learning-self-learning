---
layout: post
title: 03-03 Backpropagation Algorithm
chapter: '03'
order: 8
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

> *If you had to estimate the gradient of a 1-billion-parameter network by nudging each weight one by one with finite differences, a single gradient step would require a billion forward passes. Backpropagation computes every single one of those billion exact partial derivatives in just ONE backward pass—a billion-fold algorithmic miracle.*

## Background Knowledge Map: Core Pillars of Backpropagation

1. **Computational Graphs (DAG Decomposition)**: Breaks arbitrarily complex neural architectures into atomic differentiable primitives (`+`, `*`, `matmul`, `ReLU`, `exp`) with trivial local derivatives.
2. **Reverse-Mode Automatic Differentiation**: Propagates upstream scalar-loss sensitivities $$\delta = \frac{\partial L}{\partial z}$$ backward from output to input, sharing every downstream sub-path instead of recomputing it.
3. **Local Caching (The Memory–Compute Trade-Off)**: Stores forward-pass activations $$a^{[l-1]}$$ and pre-activations $$z^{[l]}$$ so local Jacobians can be evaluated instantly during the backward sweep.
4. **Gradient Checking (Finite-Difference Verification)**: Validates hand-derived or custom backward kernels against centered finite differences $$\frac{L(\theta+\varepsilon) - L(\theta-\varepsilon)}{2\varepsilon}$$ to catch sign or transpose bugs.

### Step-by-Step Worked Example (Read Slowly): Reverse-Mode Backprop vs. Finite Differences on a Tiny Graph

Consider the computational graph $$L = (a \cdot b + c)^2$$ at $$a = 2, b = 3, c = 1$$. Define intermediate nodes $$u = a \cdot b$$ and $$v = u + c$$, so $$L = v^2$$.

1. **Forward Pass (Cache Intermediates)**:
   - $$u = 2 \times 3 = 6$$
   - $$v = 6 + 1 = 7$$
   - $$L = 7^2 = 49$$
2. **Single Reverse-Mode Backward Pass (Compute ALL Three Gradients Together)**:
   - Start at output: $$\frac{\partial L}{\partial L} = 1$$.
   - Through $$L = v^2$$: $$\frac{\partial L}{\partial v} = 2v = 2(7) = 14$$.
   - Through $$v = u + c$$:
     $$\frac{\partial L}{\partial u} = 14 \times 1 = 14, \qquad \frac{\partial L}{\partial c} = 14 \times 1 = 14$$
   - Through $$u = a \cdot b$$:
     $$\frac{\partial L}{\partial a} = 14 \cdot b = 14(3) = 42, \qquad \frac{\partial L}{\partial b} = 14 \cdot a = 14(2) = 28$$
3. **Finite-Difference Verification for $$\frac{\partial L}{\partial a}$$ ($$\varepsilon = 0.001$$)**:
   $$\frac{L(2.001, 3, 1) - L(1.999, 3, 1)}{0.002} = \frac{(7.003)^2 - (6.997)^2}{0.002} = \frac{49.042009 - 48.958009}{0.002} = \frac{0.084}{0.002} = 42.000$$
   Exact match to full precision!

Backpropagation — the engine of deep learning:

1. **Theory and Worked Example** — concept, math, intuition
2. **Implementation** — from-scratch and PyTorch code, papers, pitfalls, takeaways

## Video tham khảo

- **xem trước** [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown, English. The intuition: each example nudges every weight.
- [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown, English. The chain-rule calculus behind that nudge.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, English. The same algorithm implemented, which is the second lesson in this block.
