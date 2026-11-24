---
layout: post
title: 05-01 Recurrent Neural Networks - Fundamentals
chapter: '05'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter05
---

Before gated cells (LSTM/GRU) and Transformers entered the scene, the vanilla Recurrent Neural Network established the core mathematical language of sequential deep learning: hidden states as compressed memory, parameter sharing across time, and Backpropagation Through Time. This module pairs the rigorous mathematical theory of vanilla RNNs with hands-on NumPy and PyTorch implementations so you understand both why recurrence works and precisely where its gradient dynamics break down.

## Background Knowledge Map: Vanilla RNN Fundamentals Roadmap

- **Recurrent State-Space Formulation**: Modeling variable-length sequences via $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$ with shared temporal weights.
- **BPTT & Gradient Pathologies**: Understanding how repeated Jacobian products cause vanishing or exploding gradients as sequence length grows.
- **End-to-End Implementation & Language Modeling**: Coding BPTT from scratch, applying gradient clipping, and sampling autoregressive text from a character-level RNN.

### Step-by-Step Worked Example (Read Slowly): Exact Sequence-Length Memory Footprint of BPTT

Suppose you train a batch of $$N = 32$$ sequences of length $$T = 100$$ with hidden dimension $$d_h = 256$$ in `float32` (4 bytes per scalar).

1. **Hidden states stored for full BPTT**:
   $$N \times T \times d_h = 32 \times 100 \times 256 = 819{,}200 \text{ floats} \implies 819{,}200 \times 4 = 3{,}276{,}800\text{ bytes} \approx 3.125\text{ MiB}.$$
2. **If sequence length grows $$10\times$$ to $$T = 1{,}000$$**:
   Activation memory scales strictly linearly $$O(T)$$ to $$31.25\text{ MiB}$$ (for one layer), while compute per forward pass also scales linearly $$O(T)$$—unlike full self-attention, which scales quadratically $$O(T^2)$$!
3. **Why this still matters**: That $$O(T)$$ memory and $$O(1)$$ per-step inference cost is the exact reason modern State Space Models (Mamba, RWKV) revived recurrent dynamics in 2023–2026.

This lesson introduces recurrent neural networks for sequential data:

1. **RNN Theory and Intuition** — concept overview, math, and worked intuition
2. **RNN Implementation** — full code, related concepts, papers, pitfalls, and takeaways

## Video tham khảo

- **xem trước** [Recurrent Neural Networks (RNNs), Clearly Explained!!!](https://www.youtube.com/watch?v=AsNTP8Kwu80) — StatQuest with Josh Starmer, English. Before the theory and code lessons.
- [Lecture 10 | Recurrent Neural Networks](https://www.youtube.com/watch?v=6niqTuYFZLQ) — Stanford University School of Engineering, English. The lecture version of the same recurrence, including backprop through time.
- [Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=L8HKweZIOmg) — StatQuest with Josh Starmer, English. Sequence-to-sequence is the architecture type the introduction lists and the code lesson reaches.
