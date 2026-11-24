---
layout: post
title: 14-99 Modern Applications and Updates (2022–2026)
chapter: '14'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter14
lesson_type: optional
---

Many observers declared GANs dead once Latent Diffusion took over text-to-image generation in 2022—yet look inside 2024–2026's fastest one-step image generators (SDXL-Turbo, DMD2, LADD) and neural codecs, and you will find an adversarial discriminator doing the heavy lifting. While diffusion owns iterative multi-step sampling, GANs remain unmatched whenever you need single-step sub-50ms inference, razor-sharp local texture, or clean latent-space facial editing.

## Background Knowledge Map: GANs in the Post-Diffusion Era (2022–2026)

- **Pillar 1 — StyleGAN3 & Disentangled Latent Inversion**: Achieves alias-free equivariant synthesis and projects real photographs into $\mathcal{W}^+$ latent space for instant semantic editing (age, lighting, gaze).
- **Pillar 2 — Adversarial Diffusion Distillation (ADD / LADD)**: Combines a pretrained diffusion teacher with a PatchGAN/feature-space discriminator to collapse 50-step diffusion down to 1–4 real-time steps without losing sharpness.
- **Pillar 3 — Perceptual Discriminators Inside Codecs**: Uses multi-scale PatchGAN and multi-period audio discriminators (HiFi-GAN, EnCodec) so VAE and audio-codec decoders reconstruct crisp edges and harmonics.
- **Pillar 4 — Latency vs. Diversity Engineering Trade-offs**: Defines when to choose a GAN or adversarially distilled student (real-time video, on-device filters, super-resolution) over multi-step diffusion.

### Step-by-Step Worked Example (Read Slowly): Inference Latency Speedup of Adversarial Diffusion Distillation

Suppose a standard diffusion U-Net/DiT takes $t_{\text{step}} = 35\text{ ms}$ per denoising step on a consumer GPU, uses Classifier-Free Guidance (doubling batch size so each step costs $2 \times 35 = 70\text{ ms}$), runs for $N = 30$ steps, plus a $40\text{ ms}$ VAE decode. Compare this to an adversarially distilled 1-step GAN-diffusion student (no CFG needed at inference).

1. **Standard 30-step Diffusion with CFG**:
   $$T_{\text{diff}} = 30 \times 70\text{ ms} + 40\text{ ms} = 2{,}140\text{ ms} \approx 2.14\text{ seconds/image}.$$
2. **Adversarially Distilled 1-Step Student**:
   $$T_{\text{1-step}} = 1 \times 35\text{ ms} + 40\text{ ms} = 75\text{ ms/image} \quad (\approx 13.3\text{ frames/sec}).$$
3. **End-to-end speedup**:
   $$\frac{2{,}140}{75} \approx 28.5\times\text{ faster}.$$
4. **Takeaway**: Pairing adversarial discriminators with diffusion teachers preserves modern foundation-model semantics while restoring classic GAN single-pass speed.

# Optional: GANs after diffusion — faces, inversion, and one-step generators

> This lesson is **optional**. It does **not** replace GAN losses, DCGAN/StyleGAN theory, or the training-pitfalls notes. It records what GANs are still *for* after diffusion became the default sampler.

The adversarial pair

$$\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]$$

is unchanged. The market share changed: text-to-image is mostly diffusion/flow; **GANs remain** where you need instant sampling, inversion, or a well-studied face prior.

## 1. StyleGAN3 and the 2022–2023 face stack

[Karras et al., 2021](https://arxiv.org/abs/2106.12423) (StyleGAN3, used throughout 2022–2024) remove texture sticking via alias-free convs. Industry still uses StyleGAN2/3 for identity, aging, and editing because the **W / W+** latent is invertible and editable ([Richardson et al., e4e / pSp](https://github.com/omertov/encoder4editing) pipelines). Diffusion inversion exists, but StyleGAN inversion is cheaper and more mature.

## 2. Concrete applications

### GigaGAN and large-scale text-to-image GANs

[Kang et al., 2023](https://arxiv.org/abs/2303.05511) (GigaGAN) show a GAN can reach diffusion-like text-to-image quality with **one forward pass**. The paper is the existence proof that adversarial training did not die; most open products still chose diffusion for stability.

### Distillation: GAN as a student of diffusion

Many one-step generators (SD-Turbo, LCM-LoRA-style distillations, UFOGen-class work) add an **adversarial** head so a tiny student matches a slow teacher. That is Chapter 14’s discriminator used as a *distillation* tool, not as the only generative model.

### When to pick a GAN in 2026

- Real-time avatars, talking-head, and game sprites.
- Dataset rebalancing when you already have a StyleGAN fit.
- Not your first choice for open-vocabulary “a red cube on the moon.”

## 3. Widely used software

- [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) and [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).
- [mingukkang/GigaGAN](https://github.com/mingukkang/GigaGAN) (paper code).
- [huggingface/diffusers](https://github.com/huggingface/diffusers) — some turbo/adversarial distill checkpoints.

## 4. Citations (2022–2026)

- [Alias-Free Generative Adversarial Networks — StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423).
- [Scaling up GANs for Text-to-Image Synthesis — GigaGAN (Kang et al., 2023)](https://arxiv.org/abs/2303.05511).
- [Adversarial Diffusion Distillation (Sauer et al., 2023)](https://arxiv.org/abs/2311.17042) — SD-Turbo: GAN loss on a diffusion student.

## 5. How this complements the core notes

Train the toy GAN and read the mode-collapse notes first. This lesson only answers “why is StyleGAN still in production while DALL·E-class demos use diffusion?”

## Video tham khảo

- **xem trước** [AI generated faces - StyleGAN explained!](https://www.youtube.com/watch?v=4LNO8nLxF4Y) — AI Bites, English. StyleGAN is the face-synthesis stack this note's 2022–2023 section describes.
- [A Friendly Introduction to Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8) — Luis Serrano Academy, English. The original two-player game StyleGAN did not replace.
- [What are GANs (Generative Adversarial Networks)?](https://www.youtube.com/watch?v=TpMIssRdhco) — IBM Technology, English. A short baseline if StyleGAN's style space is hard to separate from a vanilla GAN.
