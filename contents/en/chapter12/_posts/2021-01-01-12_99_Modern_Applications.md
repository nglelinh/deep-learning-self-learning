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

You might think classical autoencoders were retired once diffusion models and LLMs took over—in reality, they quietly became the load-bearing compression layer underneath almost every modern multimodal system. Stable Diffusion does not diffuse pixels; it diffuses inside an autoencoder's latent space. Audio language models do not predict raw 48 kHz waveforms; they predict discrete tokens emitted by a vector-quantized neural audio codec.

## Background Knowledge Map: Modern Autoencoders (2022–2026)

- **Pillar 1 — Latent Diffusion Codecs**: Compresses high-resolution RGB images (e.g., $512 \times 512 \times 3$) by a spatial factor of $8\times$ into compact $64 \times 64 \times 4$ latents, cutting quadratic self-attention cost by over $4{,}000\times$.
- **Pillar 2 — Vector Quantization & Residual VQ (RVQ)**: Snaps continuous encoder vectors to nearest codebook entries $e_k$, turning images and audio into discrete token streams compatible with autoregressive Transformers.
- **Pillar 3 — Masked Autoencoders (MAE)**: Drops $75\%$ of image patches at the encoder input and reconstructs missing pixels with a lightweight decoder, scaling self-supervised vision pretraining to billion-parameter ViTs.
- **Pillar 4 — Perceptual & Adversarial Reconstruction Losses**: Replaces pure pixel MSE (which causes blur) with LPIPS perceptual losses plus PatchGAN discriminators so compressed reconstructions stay razor-sharp.

### Step-by-Step Worked Example (Read Slowly): Compute Savings of Latent Compression & Residual Vector Quantization Bitrate

1. **Spatial compression savings for a $512 \times 512 \times 3$ image encoded to $64 \times 64 \times 4$**:
   - Raw pixel elements: $512 \times 512 \times 3 = 786{,}432$.
   - Latent elements: $64 \times 64 \times 4 = 16{,}384$.
   - Element compression factor: $786{,}432 / 16{,}384 = 48\times$.
   - Sequence length for self-attention ($H \times W$): drops from $262{,}144$ to $4{,}096$ ($64\times$ shorter), reducing $O(N^2)$ attention operations by $64^2 = 4{,}096\times$!
2. **Neural audio codec bitrate (EnCodec / DAC style)**:
   Suppose an encoder produces $f_s = 75$ frames/second, each quantized by $N_q = 8$ residual VQ codebooks of size $K = 1024$ entries ($\log_2 1024 = 10$ bits per codebook).
   $$\text{Bitrate} = 75 \text{ frames/s} \times 8 \text{ codebooks} \times 10 \text{ bits} = 6{,}000 \text{ bits/s} = 6.0\text{ kbps}.$$
3. **Takeaway**: Modern autoencoders turn prohibitive raw media signals into $48\times$ smaller image latents and $6\text{ kbps}$ discrete audio tokens—making foundation-scale generation computationally feasible.

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

## Video tham khảo

- **xem trước** [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) — AI Coffee Break with Letitia, English. Stable Diffusion's latent is the autoencoder this note says diffusion now lives in.
- [What are Autoencoders?](https://www.youtube.com/watch?v=qiUEgSCyY5o) — IBM Technology, English. The plain autoencoder that latent was generalized from.
- [But how do AI images and videos actually work? | Guest video by Welch Labs](https://www.youtube.com/watch?v=iv-5mZ_9CPY) — 3Blue1Brown, English. The diffusion process that runs in that latent space.
