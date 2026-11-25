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

Between 2020 and 2026, generative modeling underwent a tectonic shift: unstable adversarial games gave way to denoising diffusion models and flow matching, which turn generation into solving a smooth differential equation from Gaussian noise to data. Rather than replacing likelihood theory, these modern engines directly generalize it—connecting score matching, variational lower bounds, and continuous normalizing flows into one unified framework.

## Key ideas: Diffusion, Score Matching, and Flow Matching

- **Pillar 1 — Forward Corruption & Reverse Denoising (DDPM)**: Gradually adds Gaussian noise to data $x_0 \to x_t$ via a closed-form kernel, then trains a neural network $\epsilon_\theta(x_t, t)$ to predict the noise and reverse the Markov chain.
- **Pillar 2 — Score Functions & Langevin Dynamics**: Interprets denoising as estimating the Stein score $\nabla_x \log p_t(x)$—the vector field pointing uphill toward higher data density—without ever computing the partition function $Z$.
- **Pillar 3 — Conditional Flow Matching (Rectified Flows)**: Replaces curved stochastic diffusion paths with straight-line linear interpolants $x_t = (1-t)x_0 + t x_1$ and regresses a velocity field $v_\theta(x_t, t)$ directly onto $x_1 - x_0$, enabling high-quality generation in far fewer ODE steps.
- **Pillar 4 — Classifier-Free Guidance (CFG)**: Steers conditional generation by extrapolating between conditional and unconditional predictions, trading diversity for sharper prompt adherence.

### Worked example: One Exact Flow-Matching Training & Sampling Step in 2D

Suppose a 2D data point is $x_1 = (4.0, 2.0)$, sampled noise is $x_0 = (0.0, -2.0)$, and we sample time $t = 0.25$ along the linear interpolant $x_t = (1 - t)x_0 + t x_1$.

1. **Compute the noisy interpolant $x_t$**:
   $$x_{0.25} = 0.75\,(0.0, -2.0) + 0.25\,(4.0, 2.0) = (0.0, -1.5) + (1.0, 0.5) = (1.0,\; -1.0).$$
2. **Compute the exact target velocity $u_t = \frac{dx_t}{dt} = x_1 - x_0$**:
   $$u_t = (4.0 - 0.0,\; 2.0 - (-2.0)) = (4.0,\; 4.0).$$
3. **Evaluate the Flow Matching MSE loss** if our network predicts $v_\theta(x_{0.25}, 0.25) = (3.5, 4.5)$:
   $$\mathcal{L}_{\text{FM}} = \|v_\theta - u_t\|_2^2 = (3.5 - 4.0)^2 + (4.5 - 4.0)^2 = 0.25 + 0.25 = 0.50.$$
4. **Take one Euler ODE sampling step from $x_{0.25}$ with step size $\Delta t = 0.25$**:
   $$x_{0.50} = x_{0.25} + \Delta t \cdot v_\theta(x_{0.25}, 0.25) = (1.0, -1.0) + 0.25(3.5, 4.5) = (1.875,\; 0.125).$$
5. **Takeaway**: Unlike GANs, training reduces to simple regression on straight velocity vectors, and because the true path is linear, even coarse Euler steps stay close to the exact trajectory.

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

## Video tham khảo

- **xem trước** [Flow-Matching vs Diffusion Models explained side by side](https://www.youtube.com/watch?v=firXjwZ_6KI) — AI Coffee Break with Letitia, English. Flow matching beside denoising diffusion is the 2022–2026 landscape this note draws.
- [But how do AI images and videos actually work? | Guest video by Welch Labs](https://www.youtube.com/watch?v=iv-5mZ_9CPY) — 3Blue1Brown, English. How a diffusion model actually turns noise into an image, including classifier-free guidance.
- [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) — AI Coffee Break with Letitia, English. Latent diffusion, the practical form those ideas shipped in.
