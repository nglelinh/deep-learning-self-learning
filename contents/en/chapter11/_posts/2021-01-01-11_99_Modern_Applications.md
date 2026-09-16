---
layout: post
title: 11-99 Modern Applications and Updates (2022–2026)
chapter: '11'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter11
lesson_type: optional
---

# Optional: generative modeling after GANs — diffusion and flow matching

> This lesson is **optional**. It does **not** replace the likelihood / MLE / evaluation fundamentals in this chapter, and it does **not** steal the VAE or GAN chapters. It records the 2022–2026 shift: **score-based diffusion** and **flow matching** became the default generative families for images (and then video/audio).

The core idea of the chapter still holds: a generative model is a distribution $$p_\theta(\mathbf{x})$$ you can sample and, ideally, score. What changed is *which* $$p_\theta$$ products ship.

## 1. Denoising diffusion (landscape)

[Ho et al., 2020](https://arxiv.org/abs/2006.11239) (DDPM) and the 2022 latent-diffusion wave ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752), Stable Diffusion) train a denoiser $$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$ on a forward noising process. The usual simplified loss is

$$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\varepsilon}}\big\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\big\|^2, \quad
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\varepsilon}.$$

[Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748) (DiT) put that denoiser on a Transformer; [Esser et al., 2024](https://arxiv.org/abs/2403.03206) (SD3) move toward flow-matching / rectifier objectives in latent space.

## 2. Flow matching

[Lipman et al., 2023](https://arxiv.org/abs/2210.02747) train a vector field $$v_\theta(\mathbf{x}, t)$$ along a probability path between noise and data. A simple linear path $$\mathbf{x}_t = (1-t)\mathbf{z} + t\mathbf{x}$$ yields the regression

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,\mathbf{x},\mathbf{z}}\big\| v_\theta(\mathbf{x}_t, t) - (\mathbf{x} - \mathbf{z}) \big\|^2.$$

This is the 2023–2025 “cleaner ODE” story that many new image/video models advertise (Stable Diffusion 3, Flux-class systems).

## 3. Concrete applications

- **Text-to-image / video**: Stable Diffusion, SDXL, SD3, commercial video diffusion.
- **Data science**: synthetic tabular/image augmentation when privacy forbids real samples (still evaluate with the FID / density ideas from this chapter).
- **Do not skip AE/GAN**: autoencoders (Chapter 12) became the *latent* of diffusion; GANs (Chapter 14) remain for faces, inversion, and fast one-step generators.

## 4. Widely used software

- [Hugging Face diffusers](https://github.com/huggingface/diffusers) — DDPM, LDM, flow-matching pipelines.
- [Stability-AI / SD3](https://github.com/Stability-AI/sd3) and [black-forest-labs/flux](https://github.com/black-forest-labs/flux).
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion) — the 2022 reference trainer.

## 5. Citations (2022–2026)

- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752).
- [Scalable Diffusion Models with Transformers — DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748).
- [Flow Matching for Generative Modeling (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747).
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Esser et al., 2024)](https://arxiv.org/abs/2403.03206) — SD3.

## 6. How this complements the core notes

Learn MLE, implicit vs explicit density, and the toy GAN in this chapter first. Diffusion and flow matching are the **update to the family tree**, not a replacement for those definitions. Details of VAEs and GANs stay in Chapters 13–14.
