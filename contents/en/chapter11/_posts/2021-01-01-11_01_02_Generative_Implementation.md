---
layout: post
title: 11-01-02 Generative Models Implementation
chapter: '11'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter11
---

Knowing the math of generative models on a whiteboard is only half the craft; the real intuition arrives the moment you train two competing paradigms—an explicit autoregressive density estimator and an implicit adversarial generator—on the same synthetic dataset and watch how differently their losses behave. This roadmap bridges theory into working PyTorch code, guiding you from exact mixture likelihoods to adversarial pitfalls and diagnostic checks.

## Background Knowledge Map: From Generative Equations to Running Code

- **Pillar 1 — Synthetic Benchmark Design**: Uses low-dimensional multi-modal distributions (like 8 Gaussians on a circle) where ground-truth density and mode locations are known analytically, making every failure mode visually and numerically transparent.
- **Pillar 2 — Explicit Conditional Implementation**: Translates $p_\theta(x_1)p_\theta(x_2 \mid x_1)$ into differentiable PyTorch modules with mixture parameters and stable log-likelihood loss.
- **Pillar 3 — Implicit Adversarial Training Loops**: Implements alternating discriminator/generator updates, highlighting why optimizer state, learning-rate balance, and gradient flow govern stability.
- **Pillar 4 — Quantitative Model Comparison**: Establishes disciplined evaluation habits—comparing held-out negative log-likelihood where available alongside mode-counting diagnostics for implicit models.

### Step-by-Step Worked Example (Read Slowly): Designing the 8-Gaussian Circle Geometry

Let us compute the exact coordinates of the 8 Gaussian centers arranged on a circle of radius $R = 2.0$ at angles $\theta_k = \frac{2\pi k}{8} = \frac{\pi k}{4}$ for $k = 0, \dots, 7$, each with isotropic standard deviation $\sigma = 0.1$.

1. **Center coordinates $c_k = (R\cos\theta_k,\, R\sin\theta_k)$**:
   - $k = 0$: $(2.000,\; 0.000)$
   - $k = 1$: $(2\cos 45^\circ,\; 2\sin 45^\circ) = (\sqrt{2},\; \sqrt{2}) \approx (1.414,\; 1.414)$
   - $k = 2$: $(0.000,\; 2.000)$
2. **Minimum distance between adjacent mode centers**:
   $$d_{\text{adj}} = 2R\sin\left(\frac{\pi}{8}\right) = 4 \times 0.3827 \approx 1.531.$$
3. **Mode separation in standard deviations**:
   $$\frac{d_{\text{adj}}}{\sigma} = \frac{1.531}{0.1} = 15.31\sigma.$$
4. **Takeaway**: Because adjacent clusters sit more than $15\sigma$ apart, a unimodal Gaussian or an independent factorized model cannot bridge them without dumping huge probability mass in the empty interior—making this benchmark an ideal litmus test for both autoregressive and GAN implementations.

Generative models implementation:

1. **Autoregressive Model** — toy mixture data and AR training
2. **Toy GAN, Papers, and Pitfalls** — comparison demos, metrics, takeaways

## Video tham khảo

- **xem trước** [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy, English. The autoregressive code path.
- [A Friendly Introduction to Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8) — Luis Serrano Academy, English. The toy GAN the second code lesson trains.
- [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, English. Which of the two you are implementing, so the losses do not get mixed.
