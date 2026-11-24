---
layout: post
title: 03-02 Gradient Descent
chapter: '03'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

> *In 1847, Augustin-Louis Cauchy proposed walking downhill step by step to solve astronomical orbit equations. Nearly two centuries later, that exact local descent rule—augmented with mini-batches, momentum, and adaptive step sizes—trains every large neural network on Earth.*

## Background Knowledge Map: Mechanics of Gradient Descent

1. **Iterative Local Minimization**: Updates $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla L(\boldsymbol{\theta}_t)$$, relying on local first-order slopes because closed-form solutions ($$\nabla L = \mathbf{0}$$) are intractable for nonlinear networks.
2. **Step-Size Stability Boundary**: Too small an $$\eta$$ stalls convergence; too large an $$\eta$$ (exceeding $$2/L_{\text{smooth}}$$) overshoots the valley and causes geometric explosion.
3. **Stochastic Mini-Batch Noise**: Approximating the full gradient on $$B$$ samples trades variance for massive speedups per epoch and adds implicit noise that helps escape sharp minima and saddle points.
4. **Curvature & Conditioning Challenges**: Ravines with wildly different slopes across axes cause vanilla GD to oscillate across steep walls while creeping along flat valleys.

### Step-by-Step Worked Example (Read Slowly): Watching Step Size $$\eta$$ Switch Between Convergence, Exact Jump, and Divergence

Minimize $$L(w) = 2w^2$$ (exact minimum at $$w^* = 0$$, gradient $$L'(w) = 4w$$, curvature $$L''(w) = 4$$) starting from $$w_0 = 5$$. Update rule: $$w_{t+1} = w_t - \eta(4w_t) = (1 - 4\eta)w_t$$.

1. **Regime 1: Smooth Convergence ($$\eta = 0.1$$)**:
   Multiplier $$1 - 4(0.1) = 0.6$$:
   $$w_0 = 5 \to w_1 = 3.0 \to w_2 = 1.8 \to w_3 = 1.08 \to 0$$
2. **Regime 2: Exact One-Step Convergence ($$\eta = 1/L'' = 0.25$$)**:
   Multiplier $$1 - 4(0.25) = 0$$:
   $$w_1 = 0 \quad \text{(hits the exact minimum in 1 step!)}$$
3. **Regime 3: Critical Oscillation ($$\eta = 2/L'' = 0.5$$)**:
   Multiplier $$1 - 4(0.5) = -1$$:
   $$w_0 = 5 \to w_1 = -5 \to w_2 = 5 \to w_3 = -5 \quad \text{(bounces forever!)}$$
4. **Regime 4: Divergence ($$\eta = 0.6 > 0.5$$)**:
   Multiplier $$1 - 4(0.6) = -1.4$$:
   $$w_0 = 5 \to w_1 = -7.0 \to w_2 = +9.8 \to w_3 = -13.72 \to \pm\infty$$
   That single threshold $$\eta < 2/\lambda_{\max}$$ governs stability across all of deep learning.

Gradient descent and its practical variants:

1. **Gradient Descent Variants and Learning Rate**
2. **Implementation and Challenges** — code, convergence criteria, common difficulties

## Video tham khảo

- **xem trước** [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown, English. The downhill picture of gradient descent.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, English. A numeric step-by-step of the same update, including the learning-rate choice.
- [Stochastic Gradient Descent, Clearly Explained!!!](https://www.youtube.com/watch?v=vMh0zPT0tLI) — StatQuest with Josh Starmer, English. Stochastic gradient descent is the variant the lesson contrasts with full-batch descent.
