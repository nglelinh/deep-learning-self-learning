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

> *In technical ML interviews, interviewers rarely ask you to integrate a textbook formula—they ask a deceptively short question like "Why does reverse-mode autodiff beat forward-mode for neural networks?" to see whether your calculus and information theory are operational rather than memorized.*

## Background Knowledge Map: Core Interview Themes in Math for Deep Learning

1. **Information Theory (Entropy, Cross-Entropy, and KL Divergence)**: Connects classification loss directly to coding theory via the exact decomposition $$H(p, q) = H(p) + D_{\text{KL}}(p \| q)$$, proving that cross-entropy is minimized iff $$q = p$$.
2. **Mutual Information & Data Processing Inequality**: Quantifies shared information $$I(X; Y) = H(Y) - H(Y\mid X)$$ and explains why deterministic post-processing can never create information that wasn't already present.
3. **Automatic Differentiation Modes (Forward vs. Reverse)**: Compares Jacobian-vector products (cost proportional to input dimension $$n$$) against vector-Jacobian products (cost proportional to output dimension $$m$$), explaining why scalar-loss networks ($$m = 1, n = 10^9$$) rely on reverse mode.
4. **Second-Order Geometry (Hessian Eigenvalues & Conditioning)**: Uses the Hessian spectrum to diagnose local minima, saddle points, and ill-conditioned ravines during live whiteboard debugging.

### Step-by-Step Worked Example (Read Slowly): Verifying $$H(p, q) = H(p) + D_{\text{KL}}(p \| q)$$ in Bits

Suppose the true binary label distribution is $$p = (0.75, 0.25)$$ and a classifier predicts $$q = (0.50, 0.50)$$. Compute all three quantities in base-2 logarithms (bits), noting $$\log_2(0.75) \approx -0.4150$$, $$\log_2(0.25) = -2$$, $$\log_2(0.50) = -1$$.

1. **True Entropy $$H(p)$$**:
   $$H(p) = -\big(0.75\log_2 0.75 + 0.25\log_2 0.25\big) = -\big(0.75(-0.4150) + 0.25(-2)\big) = 0.3113 + 0.5000 = 0.8113\text{ bits}$$
2. **Cross-Entropy $$H(p, q)$$**:
   $$H(p, q) = -\big(0.75\log_2 0.50 + 0.25\log_2 0.50\big) = -\big(0.75(-1) + 0.25(-1)\big) = 1.0000\text{ bit}$$
3. **KL Divergence $$D_{\text{KL}}(p \| q)$$**:
   $$D_{\text{KL}}(p \| q) = 0.75\log_2\!\left(\frac{0.75}{0.50}\right) + 0.25\log_2\!\left(\frac{0.25}{0.50}\right) = 0.75(0.5850) + 0.25(-1) = 0.4387 - 0.2500 = 0.1887\text{ bits}$$
4. **Identity Verification**:
   $$H(p) + D_{\text{KL}}(p \| q) = 0.8113 + 0.1887 = 1.0000 = H(p, q)$$
   Because $$H(p) = 0.8113$$ is fixed by the dataset, minimizing cross-entropy loss is mathematically identical to driving $$D_{\text{KL}}(p \| q) \to 0$$.

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

## Video tham khảo

- **xem trước** [MaDL - Information and Entropy](https://www.youtube.com/watch?v=pXh53lqPkHs) — Tübingen Machine Learning, English. Entropy is the information-theory prompt this practice set asks you to compute.
- [MaDL - Kullback-Leibler Divergence](https://www.youtube.com/watch?v=vPvnpC0PNVQ) — Tübingen Machine Learning, English. KL divergence is the second information-theory object in these prompts.
- [The paradox of the derivative | Chapter 2, Essence of calculus](https://www.youtube.com/watch?v=9vKqVkMQHKk) — 3Blue1Brown, English. The calculus prompts are about the derivative; this is the picture to redraw before answering.
