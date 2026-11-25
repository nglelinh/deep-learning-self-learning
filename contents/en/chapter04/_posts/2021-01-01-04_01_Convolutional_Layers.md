---
layout: post
title: 04-01 Convolutional Layers
chapter: '04'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

At the heart of every convolutional neural network—from LeNet reading handwritten checks in 1998 to ConvNeXt competing with Vision Transformers today—sits a single mathematical primitive: the convolutional layer. Before diving into pooling or multi-stage architectures, this module dissects that primitive from four complementary angles: exact discrete math, geometric intuition, hands-on NumPy/PyTorch code, and modern efficiency variants.

## Key ideas: Four Lenses on the Convolutional Layer

- **Algebraic & Dimensional Mechanics**: Exact cross-correlation formulas, multi-channel inner products, and the output-shape equation $$H_{\text{out}} = \lfloor (H + 2p - d(k-1) - 1)/s \rfloor + 1$$ govern every tensor transition.
- **Visual & Geometric Intuition**: Local filters act as template matchers and finite-difference edge detectors whose stacked receptive fields build hierarchical representations from pixels to semantic parts.
- **Systems & Implementation Fluency**: Translating index loops in NumPy into batched `(N, C, H, W)` PyTorch blocks (with `BatchNorm2d` and `bias=False`) turns theory into trainable GPU code.
- **Architectural Efficiency & Trade-offs**: Depthwise separable, dilated, and residual convolutions solve the compute, receptive-field, and optimization bottlenecks of plain stacked layers.

### Worked example: One-Shot Sanity Check of a Conv Layer Specification

Before coding any convolutional layer, verify three numbers by hand for `nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)` applied to a $$64 \times 64$$ RGB image:

1. **Output spatial size**:
   $$H_{\text{out}} = \left\lfloor \frac{64 + 2(2) - 5}{2} \right\rfloor + 1 = \left\lfloor \frac{63}{2} \right\rfloor + 1 = 31 + 1 = 32.$$
   Output tensor shape for a batch of $$16$$ images: `(16, 32, 32, 32)`.
2. **Trainable parameter count (with bias)**:
   $$P = 32 \times (3 \times 5 \times 5 + 1) = 32 \times 76 = 2{,}432.$$
3. **Activation memory (float32, 4 bytes/element) for one batch**:
   $$16 \times 32 \times 32 \times 32 \times 4\text{ bytes} = 2{,}097{,}152\text{ bytes} = 2.0\text{ MiB}.$$
   Notice that activations already occupy over $$200\times$$ more memory than the layer's $$2{,}432 \times 4 \approx 9.5\text{ KiB}$$ of weights—a foundational systems fact of deep vision.

This lesson covers convolutional layers — the core building block of CNNs — organized into focused sections:

1. **Convolution Math and Dimensions** — concept overview, mathematical foundation, multi-channel convolution, output size, and parameter counts
2. **Convolution Intuition and Examples** — edge detection, hierarchical features, and receptive field growth
3. **Convolution Implementation** — NumPy from scratch, PyTorch APIs, and a complete CNN block
4. **Related Concepts, Pitfalls, and Papers** — connections to other layers, common mistakes, practical tricks, and foundational papers

Work through each section in order before moving on to pooling and classic architectures.

## Video tham khảo

- **xem trước** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, English. Before the math, intuition, and code lessons split one operation apart.
- [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, English. The lecture that carries the operation into a layer with channels.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, English. A fully worked image example of the same layer.
