---
layout: post
title: 10-01 Advanced Optimization Algorithms
chapter: '10'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter10
---

Gradient descent in 100 million dimensions is nothing like rolling a marble down a smooth bowl—the landscape is packed with saddle points, flat plateaus, and razor-thin ravines. This module hub connects the mathematical theory of advanced first-order optimizers (Momentum, NAG, AdaGrad, RMSprop, Adam) with full code implementations, trajectory comparisons, and practical hyperparameter playbooks.

## Key ideas: Navigating Advanced Optimization

- **Core Optimizer Theory**: Understanding first-moment velocity, Nesterov look-ahead, second-moment adaptive preconditioning, and Adam's bias correction.
- **Code & Trajectory Analysis**: Implementing every optimizer from scratch and comparing convergence behavior on ill-conditioned surfaces and neural benchmarks.
- **Default Selection Heuristics**: Matching optimizer + weight decay + LR schedule to vision, sequence, and Transformer workloads.

### Worked example: Effective Momentum Horizon (How Many Past Steps Does $$\beta = 0.9$$ Remember?)

In an exponential moving average $$m_t = \beta m_{t-1} + (1 - \beta) g_t$$, the weight placed on a gradient from $$k$$ steps ago decays as $$\beta^k$$.

1. **Effective averaging window**:
   $$N_{\text{eff}} \approx \frac{1}{1 - \beta}.$$
2. **Concrete values**:
   - $$\beta_1 = 0.9 \implies \frac{1}{1 - 0.9} = 10 \text{ steps}$$ (Adam's first moment tracks roughly the last 10 mini-batches).
   - $$\beta_2 = 0.999 \implies \frac{1}{1 - 0.999} = 1{,}000 \text{ steps}$$ (Adam's second moment averages gradient variance over ~1,000 steps!).
   - In LLM training with huge batches, practitioners often lower $$\beta_2 = 0.95$$ ($$N_{\text{eff}} = 20$$ steps) so the optimizer adapts faster to sudden gradient spikes!

This lesson covers modern optimizers beyond vanilla gradient descent:

1. **Optimizer Theory** — Momentum, NAG, AdaGrad, RMSprop, Adam, AdamW, and intuition
2. **Optimizer Implementation** — code comparisons, papers, pitfalls, and takeaways

## Video tham khảo

- **xem trước** [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, English. The family the theory and code lessons split.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, English. Adam's two moment estimates, the update you will code.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering, English. Stanford's optimization lecture for the same family in a full training context.
