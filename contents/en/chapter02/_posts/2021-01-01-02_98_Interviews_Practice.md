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
