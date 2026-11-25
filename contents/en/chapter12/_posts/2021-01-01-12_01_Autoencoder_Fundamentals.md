---
layout: post
title: 12-01 Autoencoder Fundamentals
chapter: '12'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter12
---

Long before transformers dominated representation learning, autoencoders proved a deceptively simple principle: you do not need human labels to learn rich internal features if you ask a network to compress and rebuild the world. Understanding both the mathematical foundations (linear PCA equivalence, sparsity, denoising) and the hands-on PyTorch mechanics gives you the conceptual bridge to VAEs, masked image modeling, and modern neural codecs.

## Key ideas: Unifying Autoencoder Theory and Practice

- **Pillar 1 — Information Bottleneck Principle**: Forces an encoder $f_\phi: \mathbb{R}^d \to \mathbb{R}^k$ ($k \ll d$) to discard redundant variation and retain only the coordinates needed for reconstruction.
- **Pillar 2 — Regularized Families**: Organizes vanilla, denoising, sparse, and contractive autoencoders as different constraints preventing trivial identity copying.
- **Pillar 3 — Representation Geometry**: Connects bottleneck width and architectural inductive bias (MLP vs. CNN) to cluster separation and interpolation quality.
- **Pillar 4 — Bridge to Probabilistic Generation**: Highlights where deterministic autoencoders stop (unstructured latent holes) and where variational autoencoders begin.

### Worked example: Choosing Bottleneck Dimension via Explained Variance

Suppose a 4D dataset has covariance eigenvalues $\lambda_1 = 9.0, \lambda_2 = 5.0, \lambda_3 = 0.8, \lambda_4 = 0.2$.

1. **Total variance of the data**:
   $$\text{Tr}(\Sigma) = 9.0 + 5.0 + 0.8 + 0.2 = 15.0.$$
2. **Minimum possible MSE of a linear autoencoder with bottleneck $k = 2$** (by Eckart–Young–Mirsky, equal to the sum of discarded eigenvalues):
   $$\text{MSE}_{\min}(k=2) = \frac{\lambda_3 + \lambda_4}{d} = \frac{0.8 + 0.2}{4} = 0.25 \text{ per coordinate (total residual } 1.0\text{)}.$$
3. **Fraction of variance retained**:
   $$\frac{\lambda_1 + \lambda_2}{\text{Tr}(\Sigma)} = \frac{14.0}{15.0} = 93.33\%.$$
4. **Takeaway**: A linear bottleneck of size $k=2$ captures $93.33\%$ of total variance; any nonlinear autoencoder that beats $\text{MSE} = 0.25$ on this data is demonstrably exploiting curved manifold structure beyond PCA.

Autoencoder fundamentals:

1. **Theory** — concept overview, math, intuition
2. **Implementation** — code, latent visualization, papers, pitfalls

## Video tham khảo

- **xem trước** [What are Autoencoders?](https://www.youtube.com/watch?v=qiUEgSCyY5o) — IBM Technology, English. Encoder, bottleneck, and decoder.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, English. The reconstruction loss those three pieces train with.
- [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, English. Why a deterministic bottleneck is not yet a generative model, the point of the next chapter.
