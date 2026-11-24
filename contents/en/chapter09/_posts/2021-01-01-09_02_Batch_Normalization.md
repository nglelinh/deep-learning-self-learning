---
layout: post
title: 09-02 Batch Normalization
chapter: '09'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter09
---

Few ideas in deep learning delivered a bigger immediate payoff than normalizing intermediate layer activations. In just two lines of math—standardize, then scale and shift—Batch Normalization tamed gradient explosion, accelerated convergence by $$5\times\text{–}10\times$$, and unlocked deep architectures like ResNet. This module hub connects the exact train/eval mechanics of BatchNorm with its placement rules and batch-independent successors (LayerNorm, InstanceNorm, GroupNorm).

## Background Knowledge Map: Batch Normalization & Its Family Tree

- **Exact Forward/Eval Mechanics**: Computing mini-batch mean/variance during training, updating EMA running statistics, and freezing them at `model.eval()`.
- **Architectural Placement & `bias=False`**: Understanding Conv–BN–ReLU ordering and why preceding conv biases are mathematically redundant.
- **Choosing the Right Normalizer**: Matching BatchNorm (large-batch CNNs), GroupNorm (small-batch vision), InstanceNorm (style transfer), and LayerNorm/RMSNorm (Transformers).

### Step-by-Step Worked Example (Read Slowly): Verifying Why Conv Bias Cancels Out Before BatchNorm

Suppose a convolutional channel outputs $$z_i = u_i + b$$ across a batch $$i = 1 \dots m$$, where $$b$$ is a constant scalar bias added to every sample in that channel.

1. **Batch mean with bias**:
   $$\mu_z = \frac{1}{m}\sum_{i=1}^m (u_i + b) = \mu_u + b.$$
2. **Centered activations**:
   $$z_i - \mu_z = (u_i + b) - (\mu_u + b) = u_i - \mu_u.$$
3. **Batch variance**:
   $$\sigma_z^2 = \frac{1}{m}\sum_{i=1}^m (u_i - \mu_u)^2 = \sigma_u^2.$$
4. **Conclusion**: Both numerator and denominator of $$\hat{z}_i = (z_i - \mu_z)/\sqrt{\sigma_z^2 + \epsilon}$$ are **100% independent of $$b$$**—proving `bias=False` before BatchNorm loses zero expressivity while saving parameters and memory bandwidth!

Batch normalization:

1. **Mechanics** — what BN is, how it works, implementation, why it helps
2. **Placement, Variants, and Tips** — CNNs, LayerNorm and alternatives, issues, modern practice

## Video tham khảo

- **xem trước** [Why Does Batch Norm Work? (C2W3L06)](https://www.youtube.com/watch?v=nUUqwaxLnWs) — DeepLearningAI, English. Why normalizing a batch changes training, which is this block's question.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — Andrej Karpathy, English. Batch norm implemented inside a real training run, including the activation interaction.
- [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering, English. The lecture placement of batch norm among other training tricks.
