---
layout: post
title: 02-98 Interviews Practice (neurons and activations)
chapter: '02'
order: 11
owner: Deep Learning Course
lang: en
categories:
- chapter02
lesson_type: optional
---

> *Neural-architecture interview questions rarely test obscure papers; they probe whether you can spot subtle pitfalls on a whiteboard in 60 seconds—like why removing nonlinearities collapses depth, why softmax overflows without max-subtraction, or why a dead ReLU neuron can never revive itself.*

## Key ideas: High-Frequency Architecture Interview Concepts

1. **Nonlinearity & Rank Collapse**: Proving concisely why an $$L$$-layer linear MLP has rank bounded by $$\min_l d_l$$ and collapses to one affine map.
2. **Numerical Stability of Softmax & LogSumExp**: Using shift invariance $$\text{softmax}(\mathbf{z} - c) = \text{softmax}(\mathbf{z})$$ to prevent overflow and underflow in `float16`/`float32`.
3. **Saturation & Dead-Neuron Pathology**: Diagnosing why large positive/negative logits freeze sigmoid gradients ($$\sigma'(z) \to 0$$) and why large negative biases permanently silence ReLU units.
4. **Parameter & FLOP Estimation**: Mentally computing parameter counts and memory footprints of dense layers to spot capacity bottlenecks during system design.

### Worked example: Diagnosing a Dead ReLU Neuron After One Aggressive Update

Suppose a ReLU neuron $$a = \text{ReLU}(w x + b)$$ receives positive inputs $$x \in [0, 1]$$, starts with $$w = 0.5, b = 0.2$$, and on a single outlier sample $$(x = 1.0, \frac{\partial L}{\partial a} = +20.0)$$ takes a gradient step with overly large learning rate $$\eta = 0.5$$.

1. **Compute Forward Activation & Gradients Before the Update**:
   $$z = 0.5(1.0) + 0.2 = 0.7 > 0 \implies a = 0.7, \quad \frac{\partial a}{\partial z} = 1$$
   $$\frac{\partial L}{\partial w} = 20.0 \times 1.0 = 20.0, \qquad \frac{\partial L}{\partial b} = 20.0$$
2. **Apply the Aggressive Gradient Step ($$\eta = 0.5$$)**:
   $$w_{\text{new}} = 0.5 - 0.5(20.0) = -9.5, \qquad b_{\text{new}} = 0.2 - 0.5(20.0) = -9.8$$
3. **Check Subsequent Pre-Activations for ANY Input $$x \in [0, 1]$$**:
   $$z_{\text{new}} = -9.5 x - 9.8 \leq -9.8 < 0 \quad \forall x \in [0, 1]$$
   Because $$z_{\text{new}} < 0$$ on the entire dataset, $$\frac{\partial a}{\partial z} = 0$$ forever, so $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial b} = 0$$ on every future step. The neuron is permanently **dead**—directly explaining why gradient clipping, moderate learning rates, and LeakyReLU/GELU exist.

# Optional: interview practice — neurons and activations

> This lesson is **optional**. It does **not** rewrite perceptron, architecture, or activation theory. After those notes, use this drill, then open *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) for solved items on perceptrons, activations, and the “expanded deep learning” architecture basics.

**Book themes for this chapter (study in the PDF, not here):** perceptrons, activation functions, and elementary MLP structure from the long deep-learning topic area.

Hub: **01-98 Deep Learning Interviews practice track**.

## Practice prompts (original)

### Q1. Depth without nonlinearity

You stack three layers $$\mathbf{h} = W_3W_2W_1\mathbf{x}$$ with no activations. Prove that there exist weights $$W$$ such that $$\mathbf{h}=W\mathbf{x}$$. What does that say about “adding layers” as a way to get a richer hypothesis class?

**Hint.** Matrix multiplication is associative.

**Discussion.** $$W := W_3W_2W_1$$ is one linear map. Extra linear layers do not enlarge the set of representable functions; they only factor the same matrix (and can make optimization worse). Nonlinearity is what creates a genuinely deeper model. Interviewers use this to check that “more layers” is not automatically “more expressive.”

### Q2. Softmax shift invariance

Show that $$\mathrm{softmax}(\mathbf{z}+c\mathbf{1}) = \mathrm{softmax}(\mathbf{z})$$ for any scalar $$c$$. Why do frameworks often subtract $$\max_i z_i$$ before the exponentials?

**Hint.** Factor $$e^{c}$$ out of every term.

**Discussion.** Each numerator and the denominator pick up the same $$e^{c}$$, so the ratio is unchanged. Subtracting the max is the same identity with $$c=-\max z_i$$; it keeps $$e^{z_i+c}\le 1$$ and avoids overflow. The invariance also means an extra bias shared by all logits is unidentified — only *differences* of logits matter.

### Q3. Where sigmoid saturates

For $$z=10$$ and $$z=-10$$, estimate $$\sigma'(z)$$ with $$\sigma(z)=(1+e^{-z})^{-1}$$. Why is that a problem in a deep stack of sigmoids?

**Hint.** $$\sigma'(z)=\sigma(z)(1-\sigma(z))$$ peaks at $$1/4$$ when $$z=0$$.

**Discussion.** $$\sigma(10)\approx 1$$ so $$\sigma'(10)\approx 0$$; likewise at $$-10$$. Backprop multiplies these factors. A chain of saturated sigmoids drives hidden gradients to zero (the classical vanishing-gradient story). ReLU avoids saturation on the positive side, at the cost of a hard zero on the negative side (next question).

### Q4. Dead ReLU

A hidden unit computes $$\mathrm{ReLU}(\mathbf{w}^\top\mathbf{x}+b)$$. Give a simple condition on $$(\mathbf{w},b)$$ and the data cloud under which the unit stays at $$0$$ for every training point, and stays that way after every SGD update.

**Hint.** If the pre-activation is negative on the whole batch, the gradient w.r.t. $$(\mathbf{w},b)$$ is zero.

**Discussion.** If $$\mathbf{w}^\top\mathbf{x}+b<0$$ for all training $$\mathbf{x}$$, the ReLU and its local gradient are identically zero, so SGD never moves that unit. A large negative bias or a bad initialization plus a high learning rate can push many units into this state. Leaky ReLU / GELU are common mitigations; so is a smaller step size after a loss spike.

### Q5. Parameter count of a wide MLP

An MLP maps $$784\to 256\to 256\to 10$$ with biases. How many scalars are trained? What changes if you insert a $$256$$-unit residual block that is still affine-plus-ReLU?

**Hint.** A layer $$\mathbb{R}^{m}\to\mathbb{R}^{n}$$ has $$nm+n$$ parameters.

**Discussion.** First layer $$784\cdot 256+256$$, two hidden $$256\cdot 256+256$$ each, last $$256\cdot 10+10$$. That is $$784\cdot256 + 2\cdot256^{2} + 256\cdot10 + (256+256+10)$$. A residual block with the same width adds another $$256^{2}+256$$ (plus the skip, which has no extra weights if dimensions match). Interviews often want the count *and* the comment that residual connections do not by themselves add parameters on the skip.

## Attribution

Kashani, S., and Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Download the PDF from arXiv for the full solved Q&A. This page is original course practice, not a reprint.

## Video tham khảo

- **xem trước** [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, English. Before the neuron and activation prompts.
- [Neural Networks Pt. 3: ReLU In Action!!!](https://www.youtube.com/watch?v=68BZ5f7P94E) — StatQuest with Josh Starmer, English. ReLU's kink is a common prompt; the video computes it.
- [Why Non-linear Activation Functions (C1W3L07)](https://www.youtube.com/watch?v=NkOv_k7r6no) — DeepLearningAI, English. Why the activation cannot be linear is the other standard prompt.
