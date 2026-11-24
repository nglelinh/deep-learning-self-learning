---
layout: post
title: 13-01 Variational Autoencoders
chapter: '13'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter13
---

Variational Autoencoders sit at the intersection of Bayesian statistics and deep neural networks. By replacing deterministic bottleneck codes with learned probability distributions, VAEs turn compression into principled density modeling—giving you a single framework that can reconstruct, interpolate, detect anomalies, and synthesize new data.

## Background Knowledge Map: Roadmap to VAE Mastery

- **Pillar 1 — Probabilistic Encoder–Decoder Duality**: Views the encoder $q_\phi(z \mid x)$ as an inference network and the decoder $p_\theta(x \mid z)$ as a generative likelihood model.
- **Pillar 2 — The ELBO Objective**: Unifies reconstruction fidelity and latent regularization into a single lower bound on $\log p_\theta(x)$.
- **Pillar 3 — Differentiable Sampling**: Uses the reparameterization trick so standard backpropagation trains stochastic latent layers end-to-end.
- **Pillar 4 — Diagnostics & Practical Tuning**: Connects theoretical failure modes (posterior collapse, blurry MSE reconstructions) directly to code-level remedies (KL annealing, balanced reductions).

### Step-by-Step Worked Example (Read Slowly): Mapping Uniform Grid Coordinates to a 2D Gaussian Prior via Inverse CDF

To visualize a 2D VAE latent space cleanly, we sample uniformly spaced quantiles $u \in \{0.1, 0.5, 0.9\}$ along each axis and map them through the standard normal inverse CDF $\Phi^{-1}(u)$ instead of spacing linearly in $[-3, 3]$.

1. **Compute exact Gaussian quantiles**:
   $$\Phi^{-1}(0.1) \approx -1.2816, \qquad \Phi^{-1}(0.5) = 0.0000, \qquad \Phi^{-1}(0.9) \approx +1.2816.$$
2. **Form the $3 \times 3$ equal-probability-mass grid**:
   Each cell corresponds to an equal slice of prior probability mass under $\mathcal{N}(0, I_2)$, with corner codes $(\pm 1.2816, \pm 1.2816)$ and center $(0, 0)$.
3. **Compare density at corner vs. linear grid corner $(3.0, 3.0)$**:
   - At $(1.2816, 1.2816)$: $p(z) = \frac{1}{2\pi}e^{-(1.2816^2)} \approx 0.0308$.
   - At $(3.0, 3.0)$: $p(z) = \frac{1}{2\pi}e^{-9} \approx 1.96 \times 10^{-5}$ ($1{,}570\times$ sparser—far outside training support!).
4. **Takeaway**: Sampling latent grids via $\Phi^{-1}(u)$ keeps every decoded visualization inside the high-density region of the prior where the decoder was actually trained.

Variational autoencoders:

1. **Theory** — ELBO, reparameterization, intuition
2. **Implementation** — training, sampling, papers, pitfalls

## Video tham khảo

- **xem trước** [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — Arxiv Insights, English. Before the theory and code split.
- [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, English. The lecture version of the same graphical model.
- [MaDL - Kullback-Leibler Divergence](https://www.youtube.com/watch?v=vPvnpC0PNVQ) — Tübingen Machine Learning, English. The KL term those lessons will not let you skip.
