---
layout: post
title: 09-98 Interviews Practice (overfitting and regularizers)
chapter: '09'
order: 9
owner: Deep Learning Course
lang: en
categories:
- chapter09
lesson_type: optional
---

# Optional: interview practice — overfitting and regularizers

> This lesson is **optional**. It does **not** replace dropout, BatchNorm, or the other regularization notes. After those lessons, use this drill, then open *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) for solved items on overfitting, validation, and capacity control.

**Book themes for this chapter (study in the PDF, not here):** overfitting / generalization, validation, and the regularizers that appear in the expanded deep-learning topic area.

Hub: **01-98 Deep Learning Interviews practice track**.

## Practice prompts (original)

### Q1. Train loss down, val loss up

Training cross-entropy is still falling after epoch 40; validation cross-entropy rose from epoch 25. Name two *different* interventions (not “get more GPUs”) and what each assumes about the gap.

**Hint.** One changes the hypothesis class or the effective objective; one changes *when* you stop.

**Discussion.** (1) Stronger regularization (higher weight decay, more dropout, more augmentation) assumes the model can still fit the training set but is too flexible. (2) Early stopping / model selection on a true validation split assumes you already passed the useful-capacity point. Collecting more labeled data attacks the same gap from the sample-size side. Retraining from scratch with a smaller net is a capacity change, not a training-time regularizer.

### Q2. L2 vs L1 on a linear head

For a linear layer, compare adding $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$ vs $$\lambda\|\mathbf{w}\|_1$$ to the loss. Which one drives individual weights to *exact* zero more readily, and why is that only a partial story in a ReLU net?

**Hint.** The L1 subgradient contains an interval at 0; L2’s gradient is linear and shrinks but rarely hits zero in SGD.

**Discussion.** L1 (lasso-style) can produce exact zeros; L2 shrinks weights toward the origin without a sparsity incentive. In a deep ReLU network the “feature” that is zeroed is an *activation path*, not a single coordinate of $$\mathbf{w}$$, so L1 on raw weights is a blunt instrument. Weight decay (L2) remains the default because it is rotation-friendly in the linear case and cheap.

### Q3. Dropout train vs eval

A unit is dropped with probability $$p=0.5$$ at training time. What must you do at evaluation so that the expected pre-activation matches training? What goes wrong if you forget?

**Hint.** Inverted dropout vs classic dropout scaling.

**Discussion.** Inverted dropout (the usual framework default) scales surviving units by $$1/(1-p)$$ *during training* and uses the full net at eval. Classic dropout leaves training unscaled and multiplies weights by $$1-p$$ at eval. If you drop at train and do nothing at eval, every layer is systematically larger than the network you optimized — calibration and accuracy both suffer. Saying “dropout is an ensemble of $$2^n$$ nets” is a slogan; the operational point is the scaling.

### Q4. BatchNorm at train and at test

Why does BatchNorm store running means, and what breaks if you evaluate a freshly loaded checkpoint in train mode on a batch of size 1?

**Hint.** Train uses batch statistics; eval uses the moving average.

**Discussion.** The stored moments are the test-time estimate of $$\mathbb{E}[\mathbf{h}]$$ and $$\mathrm{Var}(\mathbf{h})$$. A batch of size 1 makes the batch variance undefined or noisy, so train-mode BN is not a drop-in eval. Small-batch training has the same issue — that is why LayerNorm / GroupNorm appear in other chapters’ recipes, but the interview answer for *this* chapter is: know which statistics are in use.

### Q5. Augmentation as a regularizer

You cannot increase $$\lambda$$ because the train loss is already high. Give one data-side regularizer that does not add a penalty term to $$J(\mathbf{w})$$, and one failure mode.

**Hint.** The training distribution becomes a smoothed version of the original.

**Discussion.** Random crops, flips, or color jitter inject invariance you care about without changing the parameter penalty. Failure: augmentations that destroy the label (e.g. a crop that removes the object) add *noise*, not useful invariance, and can raise both train and val error. Early stopping is another penalty-free regularizer; it does not replace a bad augmentation policy.

## Attribution

Kashani, S., and Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Download the PDF from arXiv for the full solved Q&A. This page is original course practice, not a reprint.
