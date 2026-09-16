---
layout: post
title: 02-99 Modern Applications and Updates (2022–2026)
chapter: '02'
order: 10
owner: Deep Learning Course
lang: en
categories:
- chapter02
lesson_type: optional
---

# Optional: the modern multilayer perceptron block

> This lesson is **optional**. It does **not** rewrite perceptron, architecture, activation, or forward-propagation theory. It shows how those blocks are packaged in 2022–2026 foundation models.

A neuron is still $$z = \mathbf{w}^\top \mathbf{x} + b$$ followed by a nonlinearity. What changed is the **block**: residual paths, gated MLPs, and RMS normalization became the default “layer” that you stack.

## 1. From a single neuron to a Transformer MLP

Most LLMs use a gated feed-forward block (SwiGLU / GeGLU), not a plain `Linear → ReLU → Linear`. With input $$\mathbf{x} \in \mathbb{R}^{d}$$,

$$\mathrm{SwiGLU}(\mathbf{x}) = \big(\mathrm{SiLU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)\big) W_3,$$

where $$\mathrm{SiLU}(z) = z\,\sigma(z)$$. This is the same forward-propagation algebra as Chapter 02, with an extra gate (the Hadamard product). [Shazeer, 2020](https://arxiv.org/abs/2002.05202) introduced GLU variants; they became standard in PaLM, LLaMA, and later open models.

## 2. Concrete applications

### Pre-norm + RMSNorm instead of BatchNorm

LLMs almost never use BatchNorm. They use **RMSNorm**

$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \odot \boldsymbol{\gamma}$$

and place it *before* the residual branch (pre-norm). That is an activation-scale story from the vanishing-gradient lesson, applied at 7B–400B scale.

### MLP-Mixer and all-MLP vision

[Tolstikhin et al., 2021](https://arxiv.org/abs/2105.01601) (Mixer) and later all-MLP papers showed that token-mixing + channel-mixing MLPs can classify ImageNet without convolutions. The 2022–2024 follow-up is practical: Mixers appear as cheap baselines next to ConvNeXt and ViT, not as a replacement for CNNs.

### Kolmogorov–Arnold Networks (KANs)

[Liu et al., 2024](https://arxiv.org/abs/2404.19756) replace the fixed activation on the node with a learnable univariate function on the **edge**. Treat this as an experiment on the “what is a neuron?” question — not as a new default for production vision or NLP.

## 3. Widely used software

- `torch.nn.SiLU`, `RMSNorm`, and SwiGLU implementations inside [LLaMA](https://github.com/meta-llama/llama) / [Hugging Face transformers](https://github.com/huggingface/transformers).
- [timm](https://github.com/huggingface/pytorch-image-models) — modern MLP and ConvNeXt blocks for vision.

## 4. Citations (2022–2026)

- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) — gated MLPs used by later LMs (still the block you will read in 2024–2026 code).
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) — RMSNorm, now the LLaMA-family default.
- [KAN: Kolmogorov–Arnold Networks (Liu et al., 2024)](https://arxiv.org/abs/2404.19756) — learnable edge activations; complementary research, not a replacement for MLPs.

## 5. How this complements the core notes

Use the perceptron and activation lessons to understand $$z$$ and $$\sigma$$. This note only documents the **packaging** (SwiGLU, RMSNorm, residuals) you will see when you open a 2025 LLM or `timm` model card.
