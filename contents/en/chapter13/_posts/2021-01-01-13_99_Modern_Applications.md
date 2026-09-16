---
layout: post
title: 13-99 Modern Applications and Updates (2022–2026)
chapter: '13'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter13
lesson_type: optional
---

# Optional: VAEs as latents for diffusion, video, and compression

> This lesson is **optional**. It does **not** replace ELBO, reparameterization, or the VAE implementation. It shows how the same variational autoencoder became the **codec** in 2022–2026 generative systems.

The ELBO

$$\mathcal{L}(\theta,\phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x)\,\|\,p(z))$$

is still the right objective. Products rarely sample “faces from $$z\sim\mathcal{N}(0,I)$$” anymore; they train a VAE so a **second** model (diffusion or a Transformer) can work in $$z$$-space.

## 1. The Stable Diffusion VAE

[Rombach et al., 2022](https://arxiv.org/abs/2112.10752) train a KL-regularized autoencoder (often called `AutoencoderKL`) whose latent is a downsampled spatial grid, not a single vector. The KL term is the Chapter 13 regularizer; perceptual + adversarial terms keep reconstructions sharp enough for a denoiser. Almost every open text-to-image pipeline still ships this pattern (SD, SDXL, many SD3-class stacks).

## 2. Concrete applications

### Video and 3D latents

Stable Video Diffusion and later video models reuse image VAEs or train **temporal** VAEs so the expensive denoiser sees fewer tokens. The math is the same encoder–decoder + KL; the engineering is 3D/causal convs.

### Consistency and one-step generators in latent space

[Song et al., 2023](https://arxiv.org/abs/2303.01469) (consistency models) and later latent consistency (LCM) distill a diffusion ODE in the VAE latent. The VAE is frozen; the new work is the sampler. This is why Chapter 13 still matters after diffusion “won.”

### Representation learning

$$\beta$$-VAE-style disentanglement is less fashionable than in 2018, but **variational** bottlenecks still appear in world models (DreamerV3, Chapter 21 optional) and compression.

## 3. Widely used software

- [diffusers](https://github.com/huggingface/diffusers) `AutoencoderKL` / `AutoencoderTiny` (TAESD).
- [Stability-AI SD VAE checkpoints](https://huggingface.co/stabilityai) on the Hub.
- [openai/consistency_models](https://github.com/openai/consistency_models).

## 4. Citations (2022–2026)

- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752).
- [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01469).
- [Scaling Rectified Flow Transformers — SD3 (Esser et al., 2024)](https://arxiv.org/abs/2403.03206) — still a latent autoencoder + flow.

## 5. How this complements the core notes

Derive the ELBO and the reparameterization trick first. This lesson only shows **where $$z$$ is consumed** in 2022–2026 products — not a new evidence lower bound.
