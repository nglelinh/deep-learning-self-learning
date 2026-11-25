---
layout: post
title: 00 Introduction
chapter: '00'
order: 1
owner: GitHub Copilot
lang: en
categories:
- chapter00
---

> *Imagine staring at a modern 70-billion-parameter language model generating fluent code and wondering whether deep learning is software engineering or applied mathematics. Underneath PyTorch's clean abstractions, every single forward pass, loss spike, and gradient update is governed by four classical pillars of mathematics working in concert—and the moment a model diverges at step 4,000, it is those four pillars that tell you why.*

## Key ideas: The Mathematical Architecture of Deep Learning

1. **Multivariable Calculus (The Engine of Credit Assignment)**: Measures how infinitesimal perturbations to millions of weights alter a scalar loss, providing the chain rule and local curvature estimates that make gradient-based optimization possible.
2. **Linear Algebra (The Language of Representation)**: Encodes batches of data, layer weights, and attention scores as vectors and matrices, turning neural computation into structured compositions of linear maps and spectral decompositions.
3. **Real Analysis & Set Theory (The Guarantee of Stability)**: Supplies the topological and metric foundations—compactness, continuity, and Lipschitz bounds—that determine whether a loss minimum exists and whether an iterative optimizer can converge to it.
4. **Probability & Information Theory (The Bridge to Uncertainty)**: Casts learning as density estimation and risk minimization, explaining why cross-entropy trains classifiers, why weight decay acts as a Gaussian prior, and how generative models sample new data.

### Worked example: Watching All Four Pillars Fire in One Training Step

Suppose we train a tiny one-parameter linear model $$\hat{y} = w x$$ on a single noisy observation $$(x, y) = (2, 5)$$ under Gaussian noise, starting at $$w_0 = 1.0$$ with learning rate $$\eta = 0.1$$.

1. **Set Theory & Domain Check**: Our parameter space is $$\Theta = \mathbb{R}$$ and our input-output pair lies in $$\mathbb{R} \times \mathbb{R}$$.
2. **Linear Algebra (Forward Pass)**: Treat $$x$$ and $$w_0$$ as $$1 \times 1$$ tensors:
   $$\hat{y}_0 = w_0 x = 1.0 \times 2 = 2.0$$
   The residual error is $$e_0 = \hat{y}_0 - y = 2.0 - 5.0 = -3.0$$.
3. **Probability (Maximum Likelihood Loss)**: Under a unit-variance Gaussian likelihood $$y \mid x \sim \mathcal{N}(wx, 1)$$, minimizing negative log-likelihood is equivalent to minimizing half the squared error:
   $$L(w) = \frac{1}{2}(wx - y)^2 \implies L(1.0) = \frac{1}{2}(-3.0)^2 = 4.5$$
4. **Calculus (Gradient & Update)**: Differentiate $$L(w)$$ via the chain rule:
   $$\frac{dL}{dw} = (wx - y) \cdot x = (-3.0)(2) = -6.0$$
   Applying one step of gradient descent yields:
   $$w_1 = w_0 - \eta \frac{dL}{dw} = 1.0 - 0.1(-6.0) = 1.6$$
5. **Verification of Descent**: Re-evaluate the loss at $$w_1 = 1.6$$:
   $$\hat{y}_1 = 1.6 \times 2 = 3.2, \quad L(1.6) = \frac{1}{2}(3.2 - 5.0)^2 = \frac{1}{2}(-1.8)^2 = 1.62 < 4.5$$
   In five arithmetic lines, all four pillars of Chapter 00 cooperated to cut the loss by 64%.

Deep Learning is at the heart of data science. Whether you're training a neural network, minimizing errors in regression models, or efficiently allocating resources in recommendation systems, you're essentially solving problems that involve finding the "best" solution from a vast set of possibilities. But to do this effectively, you need to speak the language of mathematics. We'll revisit key ideas from linear algebra, set theory, and calculus, ensuring you're equipped to handle gradients, matrices, constraints, and uncertainties that arise in deep-learning tasks.


**Optional (2022–2026):** After the core math notes, see **00-99 Modern Applications and Updates** for autodiff compilers (JAX, `torch.compile`) and how gradients, matrices, and likelihoods show up in scientific ML and scaling-law training.

**Optional (interview practice):** After the probability and calculus notes, see **00-98 Interviews Practice** for original prompts on entropy / KL and autodiff; the full solved bank is the free *Deep Learning Interviews* PDF ([arXiv:2201.00650](https://arxiv.org/abs/2201.00650)).

## Video tham khảo

- **xem trước** [MaDL - Introduction](https://www.youtube.com/watch?v=gq0m_2Glz0g) — Tübingen Machine Learning, English. It is a guided tour of the linear algebra, calculus, and probability this chapter assumes before any network.
- [Vectors | Chapter 1, Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs) — 3Blue1Brown, English. The vector picture here is the one later lessons use for weights, activations, and data points.
- [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, English. The three probability axioms are the rules the probability half of this chapter never leaves.
