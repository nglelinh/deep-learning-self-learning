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
