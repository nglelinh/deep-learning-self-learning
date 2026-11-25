---
layout: post
title: 02-02 Neural Network Architecture
chapter: '02'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter02
---

> *Give two engineers the exact same budget of one million parameters—one builds a single wide layer, the other builds a structured 8-layer bottleneck—and their test accuracies can differ by twenty points. Architecture is how we encode assumptions about the geometry of the problem directly into the computational graph.*

## Key ideas: Principles of Neural Architecture Design

1. **Hierarchical Composition**: Early layers extract local, reusable primitives while deeper layers assemble them into invariant global concepts.
2. **Width vs. Depth Trade-Off**: Width expands instantaneous representational bandwidth and eases optimization; depth multiplies compositional expressivity exponentially per parameter.
3. **Information Bottlenecks & Skip Connections**: Bottlenecks force compression of task-relevant signals (as in autoencoders), while residual pathways preserve gradient flow across very deep stacks.
4. **Parameter & Activation Memory Budgeting**: Balances expressivity against overfitting and hardware memory limits by sizing each layer deliberately rather than arbitrarily.

### Worked example: Comparing Two Equal-Parameter Architectures (Wide vs. Deep)

Suppose an input has dimension $$d_{\text{in}} = 64$$ and output has $$d_{\text{out}} = 4$$, and we compare two MLPs with nearly identical parameter counts (ignoring biases for quick estimation):
- **Architecture A (Wide, 1 hidden layer)**: `64 -> 128 -> 4`
- **Architecture B (Deep, 2 hidden layers)**: `64 -> 64 -> 64 -> 4`

1. **Count Parameters of Architecture A (`64 -> 128 -> 4`)**:
   - Layer 1: $$64 \times 128 + 128 = 8{,}320$$
   - Layer 2: $$128 \times 4 + 4 = 516$$
   - **Total A**: $$8{,}836$$ parameters.
2. **Count Parameters of Architecture B (`64 -> 64 -> 64 -> 4`)**:
   - Layer 1: $$64 \times 64 + 64 = 4{,}160$$
   - Layer 2: $$64 \times 64 + 64 = 4{,}160$$
   - Layer 3: $$64 \times 4 + 4 = 260$$
   - **Total B**: $$8{,}580$$ parameters (even fewer than A!).
3. **Architectural Comparison**:
   - **Latency / Parallelism**: Architecture A executes in 2 sequential matrix multiplies; B requires 3.
   - **Representational Capacity**: Architecture B composes two nonlinear warps in 64D space, giving it exponentially higher piecewise-linear region capacity per parameter than Architecture A—showing why moderate depth almost always outperforms a single wide slab at equal parameter cost.

Neural network architecture from neurons to deep systems:

1. **Architecture Theory** — concept overview, math, and intuition
2. **Depth, Width, and Design Patterns** — design choices, representation, implementation, considerations

## Video tham khảo

- **xem trước** [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, English. Depth as a stack of learned functions.
- [The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) — StatQuest with Josh Starmer, English. Makes width concrete: more hidden units, a more flexible squiggle.
- [Lecture 4 | Introduction to Neural Networks](https://www.youtube.com/watch?v=d14TUNcbn1k) — Stanford University School of Engineering, English. Stanford's architecture picture for fully connected nets, before later chapters change the wiring.
