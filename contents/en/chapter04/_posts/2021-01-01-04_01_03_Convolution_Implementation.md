---
layout: post
title: 04-01-03 Convolution Implementation
chapter: '04'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

Knowing the equation of 2D convolution on a whiteboard is only half the craft; translating it into bug-free, GPU-efficient code requires bridging two very different worlds. On one side sits raw NumPy, where explicit indexing exposes every boundary condition and kernel-flipping nuance. On the other sits PyTorch's `nn.Conv2d`, where batched 4D tensors, `bias=False` conventions, and modular Conv–BN–ReLU blocks power modern vision backbones. This hub connects both implementations so you can verify framework outputs against first-principles arithmetic.

## Key ideas: Bridging First-Principles Code and Framework APIs

- **Verification by Differential Testing**: Implementing a slow, transparent NumPy reference and comparing its output against `torch.nn.functional.conv2d` on identical random tensors guarantees that your understanding of padding, stride, and channel summation matches the C++/CUDA backend to floating-point precision ($$< 10^{-6}$$).
- **Tensor Rank & Axis Semantics**: Raw 2D images are `(H, W)`, single multi-channel images are `(C, H, W)`, and PyTorch mini-batches are `(N, C, H, W)`. Mixing up channel-last `(H, W, C)` (NumPy/PIL) with channel-first `(C, H, W)` (PyTorch) is the single most common vision bug.
- **Modular Block Abstraction**: Real CNNs encapsulate `Conv2d -> BatchNorm2d -> ReLU` into `nn.Module` building blocks, allowing clean residual wiring, automated weight initialization, and easy inspection of intermediate feature maps.

### Worked example: Cross-Checking NumPy vs. PyTorch Output on a Minimal Tensor

Suppose we verify that our NumPy cross-correlation matches PyTorch `nn.Conv2d` on a minimal `(N=1, C_in=1, H=3, W=3)` input with a `(C_out=1, C_in=1, kH=2, kW=2)` kernel, `stride=1`, `padding=0`, `bias=0`:

$$X = \begin{bmatrix} 2 & 0 & 1 \\ 1 & 3 & 2 \\ 0 & 2 & 1 \end{bmatrix}, \quad W = \begin{bmatrix} 1 & 0 \\ -1 & 2 \end{bmatrix}.$$

1. **Compute the four output cells by hand (NumPy cross-correlation)**:
   - $$(0, 0)$$: $$1(2) + 0(0) + (-1)(1) + 2(3) = 2 - 1 + 6 = 7$$.
   - $$(0, 1)$$: $$1(0) + 0(1) + (-1)(3) + 2(2) = 0 - 3 + 4 = 1$$.
   - $$(1, 0)$$: $$1(1) + 0(3) + (-1)(0) + 2(2) = 1 + 0 + 4 = 5$$.
   - $$(1, 1)$$: $$1(3) + 0(2) + (-1)(2) + 2(1) = 3 - 2 + 2 = 3$$.
2. **Expected PyTorch tensor**:
   `torch.tensor([[[[7., 1.], [5., 3.]]]])` of shape `(1, 1, 2, 2)`.
3. **Diagnostic check**: If your NumPy code instead returns `[[7, 1], [5, 3]]` transposed or flipped, you immediately catch whether you accidentally implemented true convolution (flipped kernel) or swapped `(H, W)` indices before scaling to real datasets.

Implementing convolutional layers:

1. **NumPy Implementation** — from-scratch convolution and demos
2. **PyTorch and CNN Blocks** — PyTorch APIs, complete CNN block, training example

## Video tham khảo

- **xem trước** [PyTorch Tutorial 14 - Convolutional Neural Network (CNN)](https://www.youtube.com/watch?v=pDdP0TFzsoQ) — Patrick Loeber, English. Nn.Conv2d, which is the PyTorch half of this implementation block.
- [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, English. The shapes that call must respect.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, English. The explicit sum a NumPy implementation is looping over.
