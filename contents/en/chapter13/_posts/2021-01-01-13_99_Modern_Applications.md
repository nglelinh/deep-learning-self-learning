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

If you open the checkpoint of Stable Diffusion, Flux, Sora, or modern video generators, you will find a surprisingly familiar component right at the entrance and exit: a KL-regularized Variational Autoencoder. Rather than generating pixels directly, today's frontier generative systems split the problem in two—a VAE handles perceptual compression into a compact continuous latent tensor, while a diffusion or flow-matching Transformer models global semantics in that latent space.

## Background Knowledge Map: VAEs in Modern Multimodal Systems (2022–2026)

- **Pillar 1 — Perceptual Latent VAEs (KL-f8 / SDXL / Flux)**: Uses a very tiny KL weight ($\beta \approx 10^{-6}$) plus LPIPS and PatchGAN losses so the VAE preserves crisp high-frequency textures while keeping latents smoothly Gaussian for downstream diffusion.
- **Pillar 2 — Spatiotemporal 3D Causal VAEs for Video**: Compresses video tensors $(T, H, W, 3)$ across both time (e.g., $4\times$) and space ($8\times \times 8\times$) using causal 3D convolutions so frames can be decoded streamingly without future leakage.
- **Pillar 3 — Latent Scaling Factors**: Normalizes encoded latents by an empirical scale factor $s$ (e.g., $0.18215$ in SD 1.5) so latent variance equals $1.0$, aligning perfectly with standard diffusion noise schedules.
- **Pillar 4 — Channel Width vs. Reconstruction Trade-off**: Explains why modern VAEs increased latent channels from $4$ (SD 1.5) to $16$ (SD3 / Flux)—dramatically improving small text, faces, and fine edges.

### Step-by-Step Worked Example (Read Slowly): Spatiotemporal Compression Factor of a Video VAE

Consider a 1-second HD video clip of $T = 25$ frames at resolution $H = 512, W = 512$ with $C = 3$ RGB channels, encoded by a causal 3D VAE with temporal stride $4$ (keeping the first frame separate: $1 + (T-1)/4$), spatial stride $8$, and $C_z = 16$ latent channels.

1. **Raw video tensor size**:
   $$25 \times 512 \times 512 \times 3 = 19{,}660{,}800\text{ scalar values}.$$
2. **Compressed latent tensor dimensions**:
   - Temporal frames: $T_z = 1 + \frac{25 - 1}{4} = 7$.
   - Spatial size: $H_z = 512/8 = 64$, $W_z = 512/8 = 64$.
   - Total latent scalars: $7 \times 64 \times 64 \times 16 = 458{,}752$.
3. **Overall compression ratio**:
   $$\frac{19{,}660{,}800}{458{,}752} \approx 42.86\times\text{ fewer scalars},$$
   and spatiotemporal tokens (with $2 \times 2$ ViT patching) drop to $7 \times 32 \times 32 = 7{,}168$ tokens—small enough for full 3D self-attention on a single GPU.
4. **Takeaway**: Modern VAEs are no longer standalone blurry image generators; they are the ultra-efficient perceptual codecs that make megapixel image and video synthesis tractable.

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

## Video tham khảo

- **xem trước** [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) — AI Coffee Break with Letitia, English. The Stable Diffusion VAE is the deployed descendant this note describes.
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — Arxiv Insights, English. The ELBO that VAE still trains, even when it is only a compressor in front of diffusion.
- [But how do AI images and videos actually work? | Guest video by Welch Labs](https://www.youtube.com/watch?v=iv-5mZ_9CPY) — 3Blue1Brown, English. What the diffusion model does with the VAE's latent.
