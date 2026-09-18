---
layout: post
title: 10-98 Interviews Practice (optimizers)
chapter: '10'
order: 8
owner: Deep Learning Course
lang: en
categories:
- chapter10
lesson_type: optional
---

# Optional: interview practice — optimizers

> This lesson is **optional**. It does **not** replace momentum, RMSprop, or Adam theory. After those notes, use this drill, then open *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) for solved items on adaptive methods, hyperparameters, and training dynamics.

**Book themes for this chapter (study in the PDF, not here):** optimization algorithms (including Adam-family updates), hyperparameter choices, and the training subsection of the expanded deep-learning topic area.

Hub: **01-98 Deep Learning Interviews practice track**.

## Practice prompts (original)

### Q1. Momentum on a ravine

Vanilla SGD oscillates across a narrow valley and crawls along the long axis. In one sentence each: what does a velocity buffer $$ \mathbf{v}\leftarrow \beta\mathbf{v}+\nabla J $$ change about those two directions?

**Hint.** High-frequency flips average out; a consistent gradient accumulates.

**Discussion.** Across the ravine the gradient sign flips, so the momentum average stays small and the oscillation damps. Along the valley the gradient keeps the same sign, so $$\mathbf{v}$$ builds and the crawl speeds up. That is the interview picture; you do not need a specific $$\beta$$ derivation unless asked.

### Q2. Why Adam’s bias correction exists

Adam’s first moment is $$\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t$$ with $$\mathbf{m}_0=\mathbf{0}$$. Why is $$\mathbf{m}_t$$ too small at $$t=1$$, and what does dividing by $$1-\beta_1^t$$ do?

**Hint.** Unroll one step: $$\mathbf{m}_1=(1-\beta_1)\mathbf{g}_1$$.

**Discussion.** At $$t=1$$ you only have a $$(1-\beta_1)$$ fraction of the first gradient (typically $$0.1$$ if $$\beta_1=0.9$$). The factor $$1-\beta_1^t$$ is the sum of the geometric weights; dividing undoes the cold-start shrink. The same story applies to the second moment with $$\beta_2$$. After a few hundred steps $$\beta^t\approx 0$$ and the correction is optional in practice — but interviewers still want the $$t=1$$ calculation.

### Q3. Adam vs AdamW

You want an effective L2 penalty $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$. Why is “add $$\lambda\mathbf{w}$$ to the gradient, then let Adam rescale” not the same as decaying $$\mathbf{w}$$ by $$\lambda\eta$$ outside the adaptive denominator?

**Hint.** Adam divides the gradient (and anything you add to it) by $$\sqrt{\hat{\mathbf{v}}}+\epsilon$$.

**Discussion.** Coupled weight decay is distorted: coordinates with large second moments get *less* decay. AdamW applies decay to the weights directly, so every coordinate shrinks by a comparable $$\lambda\eta$$ (up to the learning-rate schedule). That is the default in modern CNN / Transformer recipes and the distinction Chapter 10 already names; this prompt only checks that you can say *why*.

### Q4. When SGD+momentum can beat Adam

A ConvNet on a large image set generalizes worse with well-tuned Adam than with SGD+momentum at a longer horizon. Give one optimization-centric and one implicit-regularization-centric hypothesis — not “Adam is buggy.”

**Hint.** Adaptive methods rewrite the preconditioner every step; SGD keeps a global scale.

**Discussion.** Optimization: Adam can reach a sharp region of $$J$$ faster and then sit there because the per-coordinate scale stays large on small-gradient coordinates. Implicit bias: the SGD noise + shared learning rate has a different stationary distribution over minima than Adam’s preconditioned noise. Neither hypothesis is a theorem you must prove on a whiteboard; you *should* mention that you would verify with a longer SGD run and a proper val set, not with one seed.

### Q5. A one-line warmup rationale

You start a Transformer with AdamW at the peak learning rate and the loss spikes to NaN. Why might linearly warming $$\eta$$ for a few thousand steps fix it even if the *final* $$\eta$$ is unchanged?

**Hint.** At init, activations and attention logits are poorly scaled; a large step is taken with a meaningless $$\mathbf{m},\mathbf{v}$$.

**Discussion.** Early gradients are huge or incoherent; Adam’s second-moment estimate is also untrustworthy (Q2). A small $$\eta$$ lets the moments and the residual stream settle before you take ImageNet- or LLM-sized steps. Warmup is not magic — if the peak $$\eta$$ is simply too large, you will still diverge after warmup ends.

## Attribution

Kashani, S., and Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Download the PDF from arXiv for the full solved Q&A. This page is original course practice, not a reprint.
