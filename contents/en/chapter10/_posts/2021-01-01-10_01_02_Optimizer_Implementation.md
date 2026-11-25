---
layout: post
title: 10-01-02 Optimizer Implementation
chapter: '10'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter10
---

Knowing optimizer equations is one thing; watching their trajectories race side-by-side across an ill-conditioned 2D valley and a noisy neural network loss surface is what builds lasting engineering judgment. This implementation hub pairs the from-scratch NumPy/PyTorch optimizer classes with empirical benchmarks, Adam vs. AdamW comparisons, learning-rate schedules, and classic convergence pitfalls.

## Key ideas: Practical Optimizer Engineering

- **From-Scratch Verification**: Coding Momentum, NAG, AdaGrad, RMSprop, and Adam in NumPy and confirming identical trajectories against `torch.optim`.
- **Comparative Benchmarking**: Testing convergence speed, oscillation damping, and generalization across both convex/non-convex 2D surfaces and neural classification tasks.
- **Production Recipes**: Combining `AdamW` (for Transformers/NLP) or `SGD + Momentum` (for classic CNNs) with warmup, cosine decay, and gradient clipping.

### Worked example: Computing Cosine Annealing Learning Rate at Quarter Milestones

For `CosineAnnealingLR` decaying from $$\eta_{\max} = 10^{-3}$$ to $$\eta_{\min} = 0$$ over $$T_{\max} = 100$$ epochs:
$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T_{\max}}\right)\right).$$

1. **Epoch $$t = 0$$**: $$\cos(0) = 1 \implies \eta(0) = 1.0 \times 10^{-3}$$.
2. **Epoch $$t = 25$$**: $$\cos(\pi/4) \approx 0.7071 \implies \eta(25) = 0.5(1.7071) \times 10^{-3} \approx 8.54 \times 10^{-4}$$ (stays high early!).
3. **Epoch $$t = 50$$**: $$\cos(\pi/2) = 0 \implies \eta(50) = 5.0 \times 10^{-4}$$.
4. **Epoch $$t = 75$$**: $$\cos(3\pi/4) \approx -0.7071 \implies \eta(75) = 0.5(0.2929) \times 10^{-3} \approx 1.46 \times 10^{-4}$$.
5. **Epoch $$t = 100$$**: $$\cos(\pi) = -1 \implies \eta(100) = 0$$.

Optimizer implementation:

1. **Core Implementation** — Momentum through AdamW from scratch
2. **Comparisons, Papers, and Pitfalls** — 2D trajectories, classification bake-off, takeaways

## Video tham khảo

- **xem trước** [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, English. The Adam update the code lesson implements.
- [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, English. The neighboring updates (RMSProp, AdaGrad) so a bug in one buffer is easier to see.
- [PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) — Patrick Loeber, English. Calling the optimizer from a training step, after the update itself is clear.
