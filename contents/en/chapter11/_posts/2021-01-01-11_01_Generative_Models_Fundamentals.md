---
layout: post
title: 11-01 Generative Models - Core Concepts
chapter: '11'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter11
---

Every major breakthrough in modern AI—from GPT's next-token prediction to Stable Diffusion's image synthesis—grows from a single root: learning a probability distribution over complex objects. Before diving into specialized architectures like VAEs, GANs, or diffusion models, mastering the shared mathematical vocabulary of likelihood, latent variables, and distribution divergences gives you a compass that works across the entire generative landscape.

## Key ideas: Core Pillars of Generative Fundamentals

- **Pillar 1 — Joint vs. Conditional Modeling**: Contrasts discriminative models $p_\theta(y \mid x)$ with generative models $p_\theta(x)$ or $p_\theta(x, y)$, clarifying why generative models enable sampling, anomaly detection, and imputation.
- **Pillar 2 — The Likelihood–Sampling Trade-off**: Maps out why autoregressive models give exact likelihoods at slow sequential sampling cost, while implicit models sample fast without tractable densities.
- **Pillar 3 — Latent-Space Geometry**: Introduces continuous and discrete latent codes $z$ as compressed semantic coordinates that explain observable high-dimensional variation.
- **Pillar 4 — Experimental Verification**: Pairs theoretical definitions with toy PyTorch experiments so abstract concepts like mode dropping and KL asymmetry are anchored in concrete code.

### Worked example: Computing Marginal Likelihood via Bayes' Rule on a Discrete Latent Model

Suppose a simple generative model has a binary latent variable $z \in \{0, 1\}$ with prior $p(z=0) = 0.6$, $p(z=1) = 0.4$, and emits a scalar $x$ with conditional densities $p(x \mid z=0) = 0.2$ and $p(x \mid z=1) = 0.8$ at an observed point $x^*$.

1. **Compute the joint probabilities $p(x^*, z) = p(x^* \mid z)\,p(z)$**:
   $$p(x^*, z=0) = 0.2 \times 0.6 = 0.12, \qquad p(x^*, z=1) = 0.8 \times 0.4 = 0.32.$$
2. **Marginalize out $z$ to obtain the data likelihood $p(x^*)$**:
   $$p(x^*) = p(x^*, z=0) + p(x^*, z=1) = 0.12 + 0.32 = 0.44.$$
3. **Infer the posterior distribution $p(z \mid x^*)$ via Bayes' rule**:
   $$p(z=1 \mid x^*) = \frac{0.32}{0.44} = \frac{8}{11} \approx 0.7273, \qquad p(z=0 \mid x^*) = \frac{3}{11} \approx 0.2727.$$
4. **Takeaway**: Even though $z=0$ had the higher prior ($0.6$), the evidence $x^*$ shifts posterior belief decisively to $z=1$ ($72.7\%$). When $z$ becomes a 128-dimensional continuous vector, that innocent sum turns into an intractable integral—setting up the exact motivation for VAEs in Chapter 13.

Generative models fundamentals:

1. **Theory** — concept overview, math, intuition
2. **Implementation** — code, papers, pitfalls, takeaways

## Video tham khảo

- **xem trước** [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, English. The split between density models and implicit samplers.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, English. The same split in the CS231n lecture this block follows.
- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy, English. An autoregressive model, the first implementation in the block.
