---
layout: post
title: 03-01 Loss Functions
chapter: '03'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

> *Tell a neural network to minimize Mean Squared Error on a dataset with 2% corrupted outliers and it will contort its entire decision surface just to appease those outliers; switch to Huber or Cross-Entropy and the exact same architecture suddenly generalizes. A loss function is the silent specification of what your model truly optimizes.*

## Background Knowledge Map: Designing and Choosing Loss Functions

1. **Statistical Likelihood Correspondence**: Matches each loss to an underlying noise model—MSE to Gaussian noise, MAE to Laplace noise, Binary Cross-Entropy to Bernoulli labels, and Categorical Cross-Entropy to multinomial labels.
2. **Gradient Geometry & Outlier Sensitivity**: Determines how strongly large errors pull the weights—MSE gradients grow linearly without bound ($$2e$$), MAE gradients stay constant ($$\pm 1$$), and cross-entropy gradients scale directly with probability error ($$\hat{p} - y$$).
3. **Differentiability & Convexity in Logits**: Ensures smooth, well-behaved optimization with respect to the network's final linear layer even when the full network is non-convex.
4. **Task & Calibration Alignment**: Connects surrogate training losses (differentiable) to non-differentiable business metrics (accuracy, F1, IoU) while preserving probability calibration.

### Step-by-Step Worked Example (Read Slowly): How One Outlier Pulls MSE vs. MAE Gradients

Suppose a constant predictor $$\hat{y} = c$$ tries to fit three targets $$y = \{1, 2, 100\}$$ (two normal points and one extreme outlier). Compare the optimal $$c^*$$ and gradients at $$c = 2$$ under MSE and MAE.

1. **Under MSE ($$L_{\text{MSE}}(c) = \frac{1}{3}\sum (c - y_i)^2$$)**:
   - Residuals at $$c = 2$$: $$(1, 0, -98)$$.
   - Gradient at $$c = 2$$: $$\frac{2}{3}(1 + 0 - 98) = -\frac{194}{3} \approx -64.67$$ (massive pull toward the outlier!).
   - Minimizer is the **sample mean**: $$c^*_{\text{MSE}} = \frac{1 + 2 + 100}{3} = 34.33$$—terrible for the normal points $$1$$ and $$2$$!
2. **Under MAE ($$L_{\text{MAE}}(c) = \frac{1}{3}\sum \lvert c - y_i\rvert$$)**:
   - Subgradients at $$c = 2$$: $$+1$$ (from $$y=1$$), $$0$$ (from $$y=2$$), $$-1$$ (from $$y=100$$).
   - Average subgradient: $$\frac{1}{3}(+1 + 0 - 1) = 0$$!
   - Minimizer is the **sample median**: $$c^*_{\text{MAE}} = 2.0$$—completely unbothered by whether the outlier is $$100$$ or $$1{,}000{,}000$$.

Loss functions for training neural networks:

1. **Regression Loss Functions** — what is a loss, MSE, MAE, Huber
2. **Classification Losses and Choosing a Loss** — binary/multi-class losses, practical tips, selection guide

## Video tham khảo

- **xem trước** [Lecture 3 | Loss Functions and Optimization](https://www.youtube.com/watch?v=h7iBpEHGVNc) — Stanford University School of Engineering, English. The role of a loss, before the regression and classification splits.
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) — StatQuest with Josh Starmer, English. Logistic regression is the binary classification loss the second lesson derives.
- [But what is cross-entropy? | Compression is Intelligence Part 2](https://www.youtube.com/watch?v=GlYgs6v2YfU) — 3Blue1Brown, English. Cross-entropy is the multi-class and language-model loss the same lesson names.
