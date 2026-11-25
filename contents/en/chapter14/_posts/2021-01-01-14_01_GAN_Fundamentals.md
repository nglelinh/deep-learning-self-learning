---
layout: post
title: 14-01 Generative Adversarial Networks
chapter: '14'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter14
---

Generative Adversarial Networks transformed deep learning by proving you can learn to synthesize crisp, high-resolution data without ever writing down a tractable probability density. Connecting the minimax game theory directly to hands-on PyTorch training loops equips you to understand not only classic DCGANs and StyleGANs, but also why adversarial losses remain essential inside modern VAEs, audio codecs, and distilled diffusion models.

## Key ideas: Core Pillars of GAN Mastery

- **Pillar 1 — Minimax Equilibrium & Density Ratios**: Explains how an optimal discriminator estimates $\frac{p_{\text{data}}}{p_{\text{data}} + p_G}$ and reduces generator optimization to minimizing Jensen–Shannon divergence.
- **Pillar 2 — Stability Theory (WGAN, Gradient Penalty, Spectral Norm)**: Bridges the gap between brittle early GANs and smooth, Lipschitz-constrained critics.
- **Pillar 3 — Practical DCGAN Training Mechanics**: Covers alternating updates, `.detach()`, LeakyReLU, BatchNorm, and Adam $\beta_1 = 0.5$ tuning.
- **Pillar 4 — Diagnostic Discipline**: Distinguishes healthy equilibrium (`D(real) ≈ 0.5–0.7`, `D(fake) ≈ 0.3–0.5`) from mode collapse and discriminator overpowering.

### Worked example: Diagnosing GAN Health from Discriminator Outputs

Consider three training snapshots reporting average discriminator probabilities $(\bar{D}_{\text{real}},\; \bar{D}_{\text{fake}})$:
- **Snapshot A**: $\bar{D}_{\text{real}} = 0.999,\; \bar{D}_{\text{fake}} = 0.001$
- **Snapshot B**: $\bar{D}_{\text{real}} = 0.62,\; \bar{D}_{\text{fake}} = 0.38$
- **Snapshot C**: $\bar{D}_{\text{real}} = 0.50,\; \bar{D}_{\text{fake}} = 0.50$, while generated images are all identical copies of the digit "1".

1. **Compute $\mathcal{L}_D = -\ln \bar{D}_{\text{real}} - \ln(1 - \bar{D}_{\text{fake}})$ for each**:
   - Snapshot A: $-\ln(0.999) - \ln(0.999) \approx 0.002$ (Discriminator has crushed Generator; vanishing gradients!).
   - Snapshot B: $-\ln(0.62) - \ln(0.62) = 2(0.478) = 0.956$ (Healthy informative competition near equilibrium!).
   - Snapshot C: $-\ln(0.5) - \ln(0.5) = 1.386$ (Loss looks "ideal," yet visual inspection exposes mode collapse!).
2. **Takeaway**: Never judge a GAN by loss scalars alone—always pair loss monitoring with generated image grids and diversity metrics like FID.

Generative adversarial networks:

1. **Theory** — minimax formulation and intuition
2. **Implementation** — training loop, papers, pitfalls

## Video tham khảo

- **xem trước** [A Friendly Introduction to Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8) — Luis Serrano Academy, English. Before theory and code diverge.
- [The Math Behind Generative Adversarial Networks Clearly Explained!](https://www.youtube.com/watch?v=Gib_kiXgnvA) — Normalized Nerd, English. The value function both lessons write.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, English. CS231n's GAN segment, including the failure modes the notes name.
