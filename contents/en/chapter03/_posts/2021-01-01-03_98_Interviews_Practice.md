---
layout: post
title: 03-98 Interviews Practice (autodiff, Hessian, logistic)
chapter: '03'
order: 15
owner: Deep Learning Course
lang: en
categories:
- chapter03
lesson_type: optional
---

# Optional: interview practice — autodiff, Hessian, logistic

> This lesson is **optional**. It does **not** replace loss, gradient-descent, or backpropagation theory. After those notes, use this drill, then open *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) for solved items on algorithmic differentiation, curvature, and logistic regression.

**Book themes for this chapter (study in the PDF, not here):** algorithmic (automatic) differentiation, Hessian intuition, and logistic / log-odds classification.

Hub: **01-98 Deep Learning Interviews practice track**.

## Practice prompts (original)

### Q1. Logistic gradient in one line

Binary logistic regression predicts $$\hat y=\sigma(\mathbf{w}^\top\mathbf{x})$$ and uses $$\ell= -y\log\hat y-(1-y)\log(1-\hat y)$$. Show

$$\nabla_{\mathbf{w}}\ell = (\hat y-y)\,\mathbf{x}.$$

**Hint.** Use $$\sigma'(z)=\sigma(z)(1-\sigma(z))$$ and the chain rule from Chapter 00 / the backprop notes.

**Discussion.** Let $$z=\mathbf{w}^\top\mathbf{x}$$. Then $$\partial\ell/\partial z = -\frac{y}{\hat y}\sigma'(z)+\frac{1-y}{1-\hat y}\sigma'(z)$$. Substituting $$\sigma'=\hat y(1-\hat y)$$ collapses the expression to $$\hat y-y$$. Multiplying by $$\partial z/\partial\mathbf{w}=\mathbf{x}$$ gives the claim. This is why “sigmoid + BCE” is numerically nicer when fused: you never materialize a lone $$\sigma'$$ that underflows.

### Q2. Odds, log-odds, and a coefficient

A fitted logistic model has $$z = -1 + 2\,x_{\mathrm{smoke}}$$ where $$x_{\mathrm{smoke}}\in\{0,1\}$$. What are the odds of $$y=1$$ for a non-smoker vs a smoker? What does the $$2$$ mean in log-odds?

**Hint.** Odds are $$\hat y/(1-\hat y)=e^{z}$$.

**Discussion.** Non-smoker: $$z=-1$$, odds $$e^{-1}\approx 0.37$$. Smoker: $$z=1$$, odds $$e^{1}\approx 2.72$$. The coefficient $$2$$ is the *log odds ratio* for the binary feature: smoking multiplies the odds by $$e^{2}\approx 7.4$$. Interviews like this because it checks that you do not treat a logistic weight as a linear probability effect.

### Q3. Forward vs reverse mode on a tiny graph

You need $$\partial f/\partial x$$ and $$\partial f/\partial y$$ for $$f=(x y)+ \sin(x)$$. Count scalar multiplies / transcendental calls in (a) two forward sweeps and (b) one reverse sweep. Which would you pick if $$f$$ later becomes a 50-layer net with millions of weights?

**Hint.** Forward seeds one input; reverse seeds the output.

**Discussion.** Two forward sweeps: seed $$x$$ then $$y$$, each replaying $$xy$$ and $$\sin x$$. One reverse sweep stores the forward values and sends adjoints back through $$+$$, $$\times$$, and $$\sin$$. For a scalar loss and a huge $$\mathbf{w}$$, reverse mode (backprop) is the only practical choice — that is the whole point of Chapter 03. Forward mode remains useful for a few inputs and many outputs (e.g. a Jacobian of a vector-valued layer).

### Q4. Logistic Hessian and convexity

For one example, $$\ell(\mathbf{w})= -y\log\sigma(\mathbf{w}^\top\mathbf{x})-(1-y)\log\bigl(1-\sigma(\mathbf{w}^\top\mathbf{x})\bigr)$$. Show that

$$\nabla^2_{\mathbf{w}}\ell = \hat y(1-\hat y)\,\mathbf{x}\mathbf{x}^\top$$

is positive semidefinite. What does that tell you about local minima of unregularized logistic regression?

**Hint.** $$\mathbf{v}^\top(\mathbf{x}\mathbf{x}^\top)\mathbf{v}=(\mathbf{v}^\top\mathbf{x})^2\ge 0$$ and $$\hat y(1-\hat y)\in(0,1)$$.

**Discussion.** The Hessian is a nonnegative multiple of a rank-1 PSD matrix, so $$\ell$$ is convex in $$\mathbf{w}$$. A sum over a dataset stays convex. Every local minimum is global. Adding $$\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$$ makes the Hessian positive *definite* when $$\lambda>0$$. Deep nets lose this guarantee as soon as you compose nonlinear hidden layers.

### Q5. When the Hessian is a diagnosis, not a solver

Name one training symptom that a *diagonal* Hessian (or just per-parameter second-moment) can explain, and one reason we still do not run full Newton on a modern CNN.

**Hint.** Think “sharp vs flat directions” from Chapter 00 Q5, and think about $$n_{\mathrm{params}}^2$$.

**Discussion.** A large diagonal entry means that coordinate is sharp: a learning rate that is fine elsewhere overshoots here — the story behind RMSprop / Adam (Chapter 10). Full Newton needs a $$P\times P$$ Hessian (or linear solves with it). For millions of parameters that is impossible to store; even Hessian–vector products are used sparingly (damping, sharpness-aware ideas), not as the default step.

## Attribution

Kashani, S., and Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Download the PDF from arXiv for the full solved Q&A. This page is original course practice, not a reprint.
