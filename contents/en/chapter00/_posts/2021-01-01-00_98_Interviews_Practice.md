---
layout: post
title: 00-98 Interviews Practice (information theory and calculus)
chapter: '00'
order: 25
owner: Deep Learning Course
lang: en
categories:
- chapter00
lesson_type: optional
---

# Optional: interview practice — information theory and calculus

> This lesson is **optional**. It does **not** replace the calculus, linear-algebra, or probability notes. After those lessons, use this drill, then open *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) for a larger solved set on the same themes.

**Book themes for this chapter (study in the PDF, not here):** information theory — entropy, KL divergence, mutual information — and calculus / automatic-differentiation intuition.

Hub: **01-98 Deep Learning Interviews practice track**.

## Practice prompts (original)

### Q1. Cross-entropy split

Let $$p$$ be a one-hot label over $$K$$ classes and $$q$$ a softmax prediction. Show

$$\mathrm{CE}(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$$

and explain why $$H(p) = 0$$ in the usual supervised setting.

**Hint.** Expand $$D_{\mathrm{KL}}(p\|q) = \sum_k p_k \log(p_k/q_k)$$ and use $$0\log 0 := 0$$.

**Discussion.** $$\mathrm{CE}(p,q) = -\sum_k p_k\log q_k$$. The KL term adds $$H(p) = -\sum_k p_k\log p_k$$, which is zero when $$p$$ is a point mass. Training therefore minimizes KL to the empirical one-hot, even if we implement CE. For non-one-hot targets (label smoothing), $$H(p)$$ is a positive constant you can ignore for gradient steps but should not ignore when reading reported CE numbers.

### Q2. Mutual information after a perfect classifier

Let $$Y$$ be a balanced binary label and $$\hat Y$$ the model's hard prediction. What happens to $$I(Y;\hat Y)$$ if the classifier is perfect? If it always outputs the majority class?

**Hint.** $$I(Y;\hat Y) = H(Y) - H(Y\mid\hat Y)$$.

**Discussion.** Perfect prediction makes $$H(Y\mid\hat Y) = 0$$, so $$I(Y;\hat Y) = H(Y) = 1$$ bit when $$Y$$ is fair. A constant predictor is independent of $$Y$$, so mutual information is $$0$$. Accuracy can look “high” on an imbalanced set while $$I(Y;\hat Y)$$ stays near zero — a useful interview contrast.

### Q3. One chain-rule gradient

Let $$f(\mathbf{w}) = \sigma(\mathbf{w}^\top\mathbf{x})$$ with $$\sigma(z) = (1+e^{-z})^{-1}$$ and $$\mathbf{x}$$ fixed. Write $$\nabla_{\mathbf{w}} f$$ in a form you could implement in one line.

**Hint.** $$\sigma'(z) = \sigma(z)\bigl(1-\sigma(z)\bigr)$$.

**Discussion.** $$\nabla_{\mathbf{w}} f = \sigma(\mathbf{w}^\top\mathbf{x})\bigl(1-\sigma(\mathbf{w}^\top\mathbf{x})\bigr)\,\mathbf{x}$$. This is the same local factor that appears in logistic regression (Chapter 03). Reverse-mode autodiff evaluates it from the scalar $$f$$ backward; you do not form the Jacobian of $$\mathbf{w}$$ explicitly.

### Q4. Why reverse mode wins for a scalar loss

A network maps $$\mathbb{R}^n\to\mathbb{R}$$ (one loss). Why is reverse-mode autodiff roughly “one forward + one backward,” almost independent of $$n$$, while forward mode scales with $$n$$?

**Hint.** Forward mode seeds one input direction at a time; reverse mode seeds the scalar output once.

**Discussion.** Forward mode computes a Jacobian–vector product $$J\mathbf{v}$$. Getting every partial of a scalar needs $$n$$ such products (the standard basis). Reverse mode computes a vector–Jacobian product $$\mathbf{u}^\top J$$; with $$\mathbf{u}=1$$ you obtain the full gradient in one sweep. That is why deep-learning frameworks implement reverse mode (backprop). Forward mode still wins when there are many outputs and few inputs.

### Q5. Hessian as local curvature

For $$f(\mathbf{x}) = \tfrac12\mathbf{x}^\top A\mathbf{x}$$ with $$A$$ symmetric positive definite, what is $$\nabla^2 f$$? How does a large condition number $$\kappa(A)$$ show up in gradient descent?

**Hint.** The Taylor picture in the calculus notes: the Hessian is the quadratic term.

**Discussion.** $$\nabla^2 f = A$$. Gradient descent zig-zags along the narrow valley when $$\kappa(A)=\lambda_{\max}/\lambda_{\min}$$ is large: steps that are safe on the sharp axis are tiny on the flat axis. Interviewers often want this geometric sentence before they ask about Adam or Newton.

## Attribution

Kashani, S., and Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Download the PDF from arXiv for the full solved Q&A. This page is original course practice, not a reprint.
