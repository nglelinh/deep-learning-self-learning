---
layout: post
title: 02-03 Activation Functions in Detail
chapter: '02'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter02
---

> *Remove every activation function from a 100-layer neural network and the entire billion-parameter monument collapses mathematically into a single linear regression layer. Activation functions are the sole source of nonlinearity—and simultaneously the primary gatekeepers of gradient flow.*

## Background Knowledge Map: Why Activation Functions Govern Both Expressivity and Trainability

1. **Linear Collapse Theorem**: Without nonlinear activations, $$W_L(\dots(W_2(W_1 \mathbf{x}))\dots) = (W_L \cdots W_1)\mathbf{x} = W_{\text{eff}}\mathbf{x}$$, destroying all benefits of depth.
2. **Derivative Magnitude & Gradient Flow**: Because backprop multiplies by $$g'(z^{[l]})$$ at every layer, activations with $$g'(z) \ll 1$$ (like saturated sigmoid) shrink gradients exponentially to zero across depth.
3. **Zero-Centered Outputs & Optimization Geometry**: Strictly positive outputs (like sigmoid) force all weight gradients entering a neuron to share the same sign, causing inefficient zig-zagging updates.
4. **Modern Smooth Gating (GELU, SiLU/Swish, SwiGLU)**: Trades ReLU's sharp corner at zero for smooth, non-monotonic curves that preserve small negative gradients and improve Transformer convergence.

### Step-by-Step Worked Example (Read Slowly): Proving Linear Collapse Numerically on a 2-Layer Network

Consider two linear layers with **no activation function**:
$$W_1 = \begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix}, \quad W_2 = \begin{bmatrix} 1 & 2 \\ -1 & 1 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 4 \\ 1 \end{bmatrix}$$

1. **Pass $$\mathbf{x}$$ Through Both Layers Sequentially**:
   - Layer 1: $$\mathbf{h} = W_1\mathbf{x} = \begin{bmatrix} 2(4) - 1(1) \\ 0(4) + 3(1) \end{bmatrix} = \begin{bmatrix} 7 \\ 3 \end{bmatrix}$$.
   - Layer 2: $$\mathbf{y} = W_2\mathbf{h} = \begin{bmatrix} 1(7) + 2(3) \\ -1(7) + 1(3) \end{bmatrix} = \begin{bmatrix} 13 \\ -4 \end{bmatrix}$$.
2. **Collapse Into One Single Matrix $$W_{\text{eff}} = W_2 W_1$$**:
   $$W_{\text{eff}} = \begin{bmatrix} 1 & 2 \\ -1 & 1 \end{bmatrix}\begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix} = \begin{bmatrix} 2 & 5 \\ -2 & 4 \end{bmatrix}$$
   Check: $$W_{\text{eff}}\mathbf{x} = \begin{bmatrix} 2(4) + 5(1) \\ -2(4) + 4(1) \end{bmatrix} = \begin{bmatrix} 13 \\ -4 \end{bmatrix}$$. Identical for *every* input $$\mathbf{x} \in \mathbb{R}^2$$!
3. **Insert Just One ReLU Between Layers (Test on $$\mathbf{x}' = [-1, -1]^\top$$)**:
   With ReLU, $$\mathbf{h}' = \text{ReLU}([-1, -3]^\top) = [0, 0]^\top \implies \mathbf{y}' = [0, 0]^\top$$, whereas $$W_{\text{eff}}\mathbf{x}' = [-7, -2]^\top \neq [0, 0]^\top$$. That single elementwise bend breaks linear collapse forever.

Activation functions in depth:

1. **Classic Activation Functions** — why activations matter, common functions, comparison
2. **Modern Activations and Vanishing Gradients** — choosing activations, implementation, vanishing gradients

## Video tham khảo

- **xem trước** [Why Non-linear Activation Functions (C1W3L07)](https://www.youtube.com/watch?v=NkOv_k7r6no) — DeepLearningAI, English. Without a nonlinearity the whole stack collapses to one matrix.
- [Neural Networks Pt. 3: ReLU In Action!!!](https://www.youtube.com/watch?v=68BZ5f7P94E) — StatQuest with Josh Starmer, English. ReLU, the default classic activation the first lesson in this block computes.
- [The GELU, SiLU and SwiGLU activation functions, clearly explained!!!](https://www.youtube.com/watch?v=2FaI2Fen1mQ) — StatQuest with Josh Starmer, English. The modern activations the second lesson compares with ReLU.
