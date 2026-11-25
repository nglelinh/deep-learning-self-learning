---
layout: post
title: 06-01 Long Short-Term Memory Networks
chapter: '06'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter06
---

Why did the LSTM architecture remain one of the most cited inventions in computer science for nearly thirty years? Because it turned an impossible optimization problem—propagating gradients across dozens of nonlinear recurrent steps—into a smooth, controllable additive highway. This chapter hub guides you through both the formal mathematical theory of LSTM/GRU cells and their hands-on code implementation, gate visualization, and pitfall diagnostics.

## Key ideas: Navigating the LSTM Architecture Module

- **Theory & Gradient Flow**: Understanding the Constant Error Carousel, the four-gate equations, and why additive cell updates bypass the vanishing Jacobian product of vanilla RNNs.
- **Core Code & Memory Tasks**: Writing explicit LSTM cells in NumPy and PyTorch and testing them on long-range memory tasks where vanilla RNNs fail.
- **Gate Inspection & GRU Comparison**: Measuring live gate saturation, contrasting 4-gate LSTMs with 3-gate GRUs, and mastering production best practices (forget-bias init, gradient clipping, sequence packing).

### Worked example: Comparing Effective Depth & State Bandwidth

Consider a 2-layer stacked LSTM with $$d_h = 256$$ processing a sequence of length $$T = 60$$:

1. **Recurrent state vector carried between steps**:
   Each layer maintains both $$h_t \in \mathbb{R}^{256}$$ and $$C_t \in \mathbb{R}^{256}$$, so total state per sequence is $$2 \text{ layers} \times 2 \text{ vectors} \times 256 = 1{,}024$$ floats ($$4\text{ KiB}$$ in `float32`).
2. **Computational path length**:
   A token at $$t = 1$$ reaches $$t = 60$$ through $$59$$ additive cell-state hops—constant $$4\text{ KiB}$$ memory regardless of whether $$T = 60$$ or $$T = 60{,}000$$!
3. **Bridge to Chapter 07**: Compressing all 60 tokens into a fixed $$1{,}024$$-float bottleneck is amazingly memory-efficient, yet it eventually causes an **information bottleneck** in machine translation—motivating Attention in the very next chapter.

This lesson covers LSTM and related gated recurrent architectures:

1. **LSTM and GRU Theory** — gates, cell state, math, and intuition
2. **LSTM Implementation** — full code, papers, pitfalls, and takeaways

## Video tham khảo

- **xem trước** [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — StatQuest with Josh Starmer, English. Before the theory and code split.
- [Illustrated Guide to LSTM's and GRU's: A step by step explanation](https://www.youtube.com/watch?v=8HyCNIVRbSU) — The AI Hacker, English. The gate-by-gate picture the code lesson implements.
- [Simple Explanation of GRU (Gated Recurrent Units) | Deep Learning Tutorial 37 (Tensorflow & Python)](https://www.youtube.com/watch?v=tOuXgORsXJ4) — codebasics, English. GRU as the smaller gate set.
