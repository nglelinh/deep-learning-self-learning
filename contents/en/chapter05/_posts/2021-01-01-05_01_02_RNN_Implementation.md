---
layout: post
title: 05-01-02 RNN Implementation
chapter: '05'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter05
---

Turning recurrent equations into a working sequence model requires mastering two complementary layers of craft: first, building the core forward/BPTT mechanics in NumPy and `nn.RNN`; second, wiring that core into an end-to-end character-level language model with temperature sampling, gradient clipping, and hidden-state management. This section hub organizes both implementation halves so you can move cleanly from toy sequence regression to generative text modeling.

## Background Knowledge Map: Core Implementation Pillars for RNNs

- **Explicit State Propagation & Caching**: Managing `(h_prev -> h_next)` across variable sequence lengths and detaching hidden states across truncated BPTT boundaries.
- **Numerical Stability Controls**: Applying global gradient norm clipping and orthogonal/Xavier initialization to keep the recurrent Jacobian well-conditioned at initialization and during training.
- **Autoregressive Training vs. Sampling Loops**: Contrasting parallel teacher-forced cross-entropy loss over shifted `(input, target)` sequences with sequential token-by-token sampling at inference time.

### Step-by-Step Worked Example (Read Slowly): Counting Parameters in a Character-Level `nn.RNN`

Suppose we build a 1-layer PyTorch character RNN with vocabulary size $$V = 65$$ (one-hot input $$d_x = 65$$), hidden dimension $$d_h = 128$$, and linear output head $$128 \to 65$$.

1. **Input-to-hidden (`weight_ih`, `bias_ih`)**:
   $$128 \times 65 + 128 = 8{,}320 + 128 = 8{,}448.$$
2. **Hidden-to-hidden (`weight_hh`, `bias_hh`)**:
   $$128 \times 128 + 128 = 16{,}384 + 128 = 16{,}512.$$
3. **Output projection (`nn.Linear(128, 65)`)**:
   $$65 \times 128 + 65 = 8{,}320 + 65 = 8{,}385.$$
4. **Total trainable parameters**:
   $$8{,}448 + 16{,}512 + 8{,}385 = 33{,}345.$$
   Note that PyTorch's `nn.RNN` allocates *two* separate bias vectors (`bias_ih` and `bias_hh`, adding 128 extra parameters over a single-bias NumPy implementation)—a classic detail to know when comparing parameter counts!

RNN implementation:

1. **Core Implementation** — from-scratch and sequence prediction demos
2. **Character LM, Papers, and Pitfalls** — language modeling, related concepts, papers, takeaways

## Video tham khảo

- **xem trước** [Lecture 10 | Recurrent Neural Networks](https://www.youtube.com/watch?v=6niqTuYFZLQ) — Stanford University School of Engineering, English. The implementation contract: an RNN cell plus a loss over the sequence.
- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy, English. A character-level language model built in code, which is the lesson's programming target.
- [Recurrent Neural Networks (RNNs), Clearly Explained!!!](https://www.youtube.com/watch?v=AsNTP8Kwu80) — StatQuest with Josh Starmer, English. Keep this beside the code for the shapes of h and x.
