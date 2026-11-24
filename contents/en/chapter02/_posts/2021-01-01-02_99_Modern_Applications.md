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

> *Look inside Llama 3, Qwen, or DeepSeek and you might think the humble MLP was replaced by Attention. In fact, roughly two-thirds of the parameters and FLOPs in every modern Transformer live inside its upgraded MLP blocks—powered by Pre-Norm RMSNorm, residual highways, and SwiGLU gating.*

## Background Knowledge Map: Evolution of the Modern MLP Block (2022–2026)

1. **Pre-Norm + RMSNorm**: Places normalization *before* each sub-layer and drops mean-centering ($$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}$$), stabilizing deep Transformer gradients while cutting normalization overhead.
2. **SwiGLU Feedforward Blocks**: Replaces the 2-matrix `Linear -> ReLU -> Linear` block with a 3-matrix gated block using hidden width $$\approx \frac{8}{3}d$$ so total parameter count stays matched at $$8d^2$$.
3. **All-MLP Vision (MLP-Mixer)**: Demonstrates that alternating token-mixing MLPs and channel-mixing MLPs can rival CNNs without a single convolution or attention layer.
4. **Kolmogorov–Arnold Networks (KANs)**: Explores placing learnable univariate spline functions on *edges* rather than fixed activations on *nodes*, sparking active research in scientific function approximation.

### Step-by-Step Worked Example (Read Slowly): Computing RMSNorm vs. LayerNorm on a Token Vector

Take a 4-dimensional token embedding $$\mathbf{x} = [2, 2, 2, 2]$$ with unit scale weights $$\boldsymbol{\gamma} = [1, 1, 1, 1]$$ and $$\varepsilon = 0$$ (for clean hand arithmetic; in practice $$\varepsilon = 10^{-6}$$).

1. **Compute RMSNorm**:
   $$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{4}\left(2^2 + 2^2 + 2^2 + 2^2\right)} = \sqrt{\frac{16}{4}} = \sqrt{4} = 2.0$$
   $$\text{RMSNorm}(\mathbf{x}) = \frac{[2, 2, 2, 2]}{2.0} = [1, 1, 1, 1]$$
   RMSNorm preserves the direction of $$\mathbf{x}$$ while rescaling its root-mean-square magnitude to exactly $$1.0$$.
2. **Compare with LayerNorm on the Same Vector**:
   Mean $$\mu = 2.0$$, variance $$\sigma^2 = \frac{1}{4}\sum(2 - 2)^2 = 0$$, so LayerNorm subtracts the mean and produces $$[0, 0, 0, 0]$$ (with $$\varepsilon > 0$$).
3. **Now Test on $$\mathbf{v} = [1, -1, 3, -3]$$** (zero-mean):
   - $$\text{RMS}(\mathbf{v}) = \sqrt{\frac{1 + 1 + 9 + 9}{4}} = \sqrt{5} \approx 2.236$$.
   - $$\text{RMSNorm}(\mathbf{v}) = \frac{1}{\sqrt{5}}[1, -1, 3, -3] \approx [0.447, -0.447, 1.342, -1.342]$$.
   Because RMSNorm skips computing and subtracting $$\mu$$, it saves a full memory pass over the tensor while maintaining identical scale invariance on near-zero-mean hidden states.

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

Use the perceptron and activation lessons to understand $$z$$ and $$\sigma$$. This note only documents the **packaging** (SwiGLU, RMSNorm, residuals) you will see when you open a 2025 LLM or `timm` model card. Chapter **26-07** puts that packaging on a full decoder stack (RoPE, GQA, accounting).

## Video tham khảo

- **xem trước** [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, English. The MLP block inside a transformer is the modern fully connected layer this note describes.
- [The GELU, SiLU and SwiGLU activation functions, clearly explained!!!](https://www.youtube.com/watch?v=2FaI2Fen1mQ) — StatQuest with Josh Starmer, English. SwiGLU is the activation that block actually uses.
- [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, English. The original stacked-neuron picture, so the note's 'from one neuron to an MLP' comparison stays grounded.
