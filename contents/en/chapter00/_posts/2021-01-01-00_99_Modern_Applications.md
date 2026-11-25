---
layout: post
title: 00-99 Modern Applications and Updates (2022–2026)
chapter: '00'
order: 23
owner: Deep Learning Course
lang: en
categories:
- chapter00
lesson_type: optional
---

> *It is tempting to assume that modern foundation-model frameworks have abstracted math away into `loss.backward()`. In reality, the biggest breakthroughs of 2022–2026—Physics-Informed Neural Networks, LoRA low-rank matrices, FlashAttention, and flow-matching diffusion—are pure applied calculus, linear algebra, and probability written in Triton and JAX.*

## Key ideas: Classical Math Inside 2022–2026 AI Stacks

1. **Automatic Differentiation & Scientific ML (PINNs)**: Uses higher-order derivatives ($$\frac{\partial^2 u}{\partial x^2}$$) computed directly through autodiff graphs to enforce partial differential equations as soft loss penalties.
2. **Low-Rank Linear Algebra (LoRA & Quantized Matrices)**: Exploits the insight that fine-tuning weight updates $$\Delta W \in \mathbb{R}^{d \times d}$$ live in a tiny subspace of rank $$r \ll d$$, factoring $$\Delta W = BA$$ to cut trainable parameters by $$100\times$$.
3. **Probability & Score Matching (Diffusion & Flow Matching)**: Trains generative models to estimate the score function $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$ or velocity field transporting Gaussian noise smoothly into complex data distributions.
4. **Differentiable Programming Beyond Neural Nets**: Extends reverse-mode gradients through physics simulators, ray tracers, and convex optimization layers to solve inverse problems end-to-end.

### Worked example: Parameter & FLOP Savings of Low-Rank Adaptation (LoRA)

Consider a Transformer projection matrix $$W_0 \in \mathbb{R}^{4096 \times 4096}$$ adapted via LoRA with rank $$r = 16$$: $$W = W_0 + BA$$ where $$B \in \mathbb{R}^{4096 \times 16}$$ and $$A \in \mathbb{R}^{16 \times 4096}$$.

1. **Full Fine-Tuning Parameter Count**:
   $$\text{Params}_{\text{full}} = 4096 \times 4096 = 16{,}777{,}216$$
2. **LoRA Trainable Parameter Count**:
   $$\text{Params}_{\text{LoRA}} = (4096 \times 16) + (16 \times 4096) = 65{,}536 + 65{,}536 = 131{,}072$$
3. **Compression Ratio**:
   $$\frac{\text{Params}_{\text{LoRA}}}{\text{Params}_{\text{full}}} = \frac{131{,}072}{16{,}777{,}216} = \frac{1}{128} \approx 0.781\%$$
4. **Verify Associative Matrix Order (Avoid Forming $$BA$$ Explicitly!)**:
   For a single token vector $$\mathbf{x} \in \mathbb{R}^{4096}$$, computing $$B(A\mathbf{x})$$ takes $$2 \times 4096 \times 16 = 131{,}072$$ multiply-adds instead of $$16.78\text{M}$$—a $$128\times$$ speedup derived purely from matrix multiplication associativity!

# Optional: calculus, linear algebra, and probability in 2022–2026 stacks

> This lesson is **optional**. It does **not** replace the core notes on calculus, linear algebra, or probability. It shows how those same objects appear in current scientific and production software.

The chapter’s core idea is that gradients, matrices, and distributions are the language of learning. Between 2022 and 2026 that language became *executable*: compilers turn a math graph into fused GPU kernels, and scientific codes share the same autodiff used by deep learning.

## 1. Why the math chapter still matters

Automatic differentiation is the industrial form of the chain rule from the calculus notes. If $$f = g \circ h$$, then

$$Df(\mathbf{x}) = Dg(h(\mathbf{x})) \cdot Dh(\mathbf{x}).$$

Reverse-mode autodiff (what PyTorch and JAX implement) evaluates this product from the output backward, which is exactly backpropagation for a scalar loss. The new work is not a different derivative: it is *how* the product is scheduled, fused, and checked.

## 2. Concrete applications

### Scientific machine learning and PINNs

Physics-informed neural networks still minimize a residual that mixes data loss with PDE residuals. The residual of a differential operator $$\mathcal{N}$$ is

$$\mathcal{L}_{\text{PDE}} = \mathbb{E}_{\mathbf{x}}\big\|\mathcal{N}[\hat{u}](\mathbf{x})\big\|^2,$$

where derivatives of the network $$\hat{u}$$ come from autodiff, not finite differences. NVIDIA’s [Modulus](https://docs.nvidia.com/deeplearning/modulus/index.html) and the JAX ecosystem ([JAX-CFD](https://github.com/google/jax-cfd), [JAX-MD](https://github.com/jax-md/jax-md)) made this pattern routine after 2022.

### Differentiable programming beyond neural nets

JAX’s `grad`, `vmap`, and `jit` treat a simulation step as a function whose Jacobian can be composed. That is linear algebra (Jacobian–vector and vector–Jacobian products) plus the Taylor viewpoint: a single Newton–Schulz or conjugate-gradient iteration is just another differentiable block.

### Probability in foundation-model training

Scaling-law papers treat training as a statistical experiment: loss $$L$$ versus compute, tokens, and parameters. [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) (Chinchilla) fitted power laws of the form

$$L(N, D) \approx E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

which is the same likelihood / expectation language as the probability notes, applied to LLM training budgets.

## 3. Widely used software

- [JAX](https://github.com/google/jax) — composable `grad` / `vmap` / `jit` (Google).
- [PyTorch 2](https://pytorch.org/get-started/pytorch-2.0/) `torch.compile` — graph capture of the same autograd tape ([Ansel et al., 2024](https://arxiv.org/abs/2404.14294)).
- [tinygrad](https://github.com/tinygrad/tinygrad) and [MLX](https://github.com/ml-explore/mlx) — small autodiff engines that make the calculus explicit.

## 4. Citations (2022–2026)

- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — Chinchilla scaling laws: allocate tokens and parameters jointly.
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation (Ansel et al., 2024)](https://arxiv.org/abs/2404.14294) — compiler for the autograd graph.
- [JAX: composable transformations of Python+NumPy programs](https://github.com/google/jax) — the standard research stack for autodiff + SIMD batching.

## 5. How this complements the core notes

Keep using the calculus and linear-algebra lessons to compute $$\nabla f$$ and matrix factorizations by hand. This optional note only shows where those objects now live in compilers and scientific ML — it does not replace the derivations.

## Video tham khảo

- **xem trước** [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown, English. Reverse-mode autodiff is the chain rule this note says PyTorch and JAX still implement.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, English. Builds that reverse-mode product in code, which is the executable form of the calculus chapter.
- [Visualizing the chain rule and product rule | Chapter 4, Essence of calculus](https://www.youtube.com/watch?v=YG15m2VwSjA) — 3Blue1Brown, English. The one-variable chain rule being differentiated here is the identity the note writes as Df = Dg · Dh.
