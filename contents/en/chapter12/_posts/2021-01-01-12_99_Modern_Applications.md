---
layout: post
title: 12-99 Modern Applications and Updates (2022–2026)
chapter: '12'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter12
lesson_type: optional
---

# Optional: autoencoders as latents, codecs, and masked reconstructors

> This lesson is **optional**. It does **not** replace vanilla / denoising / sparse autoencoder theory. It shows how the same encoder–decoder idea is used after 2022: **latent compression for diffusion**, **neural audio codecs**, and **masked image modeling**.

The reconstruction objective

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}}\big\|\mathbf{x} - d_\phi(e_\theta(\mathbf{x}))\big\|^2$$

is still the starting point. Products now care about a *structured* latent: quantized, perceptual, or partially masked.

## 1. Latents for diffusion

Stable Diffusion’s first stage is an autoencoder (KL or VQ) that maps a $$512\times512$$ image to a small latent grid. Diffusion (Chapter 11 optional, Chapter 13) then runs in that grid. The AE is trained with reconstruction + adversarial + perceptual losses ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752)). That is this chapter’s architecture, optimized as a **codec for a second generative model**.

## 2. Concrete applications

### Neural audio codecs

[Défossez et al., 2022](https://arxiv.org/abs/2210.13438) (EnCodec) and [Kumar et al., 2023](https://arxiv.org/abs/2306.06546) (DAC) are residual-vector-quantized autoencoders for waveforms. They sit under AudioLM / MusicGen / many TTS stacks: the AE turns audio into discrete tokens a Transformer can model.

### Masked autoencoders (MAE)

[He et al., 2022](https://arxiv.org/abs/2111.06377) mask random patches and reconstruct pixels with a ViT encoder–decoder. This is an AE *pretext* (self-supervised; Chapter 16) rather than a generative prior. Mentioned here because the implementation is literally an autoencoder on patches.

### Vector-quantized AEs

VQ-VAE / VQGAN remain the discrete bottleneck for image and video tokenizers (Make-A-Video, MAGVIT-v2). The commitment loss from the theory notes is the same; the 2022–2025 work is **better codebooks and lookup-free quantization**.

## 3. Widely used software

- [diffusers](https://github.com/huggingface/diffusers) `AutoencoderKL` — SD-style latents.
- [facebookresearch/encodec](https://github.com/facebookresearch/encodec) and [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec).
- [facebookresearch/mae](https://github.com/facebookresearch/mae).

## 4. Citations (2022–2026)

- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) — AE as diffusion latent.
- [Masked Autoencoders Are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377).
- [High Fidelity Neural Audio Compression — EnCodec (Défossez et al., 2022)](https://arxiv.org/abs/2210.13438).
- [High-Fidelity Audio Compression with Improved RVQGAN — DAC (Kumar et al., 2023)](https://arxiv.org/abs/2306.06546).

## 5. How this complements the core notes

Implement the NumPy/PyTorch autoencoder and the latent-space pitfalls first. This lesson only shows **where those latents go** in 2022–2026 systems (diffusion, codecs, MAE) — not a new reconstruction derivation.
