---
layout: post
title: 03-04 Practical Training Techniques
chapter: '03'
order: 11
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

> *Two teams can take the exact same neural network architecture and dataset—one gets 94% accuracy in ten minutes, while the other stares at a flat loss curve for hours. The difference almost always lives in the practical training techniques: initialization, normalization, batching, and diagnostic monitoring.*

## Key ideas: Pillars of Stable Neural Training

1. **Symmetry-Breaking & Variance-Preserving Initialization**: Uses Xavier/Glorot (for tanh/sigmoid) or He/Kaiming (for ReLU) scaling so activations and gradients neither vanish nor explode at step 0.
2. **Data Preprocessing & Leakage-Free Splits**: Standardizes inputs using strictly training-set statistics ($$\mu_{\text{train}}, \sigma_{\text{train}}$$) and enforces clean train/validation/test separation.
3. **Diagnostic Monitoring & Checkpointing**: Tracks training vs. validation loss curves, gradient norms, and learning rates to distinguish underfitting, overfitting, and instability in real time.
4. **Early Stopping & Reproducibility**: Halts training when validation performance plateaus, restores the best checkpoint, and pins random seeds for fair ablation experiments.

### Worked example: Why Fitting Scalers on Validation Data Causes Subtle Leakage

Suppose a 1D feature has training values $$X_{\text{train}} = \{10, 20, 30\}$$ and a single held-out test point $$x_{\text{test}} = 100$$.

1. **Correct Pipeline (Fit Scaler on Train Only)**:
   - $$\mu_{\text{train}} = \frac{10 + 20 + 30}{3} = 20$$.
   - $$\sigma_{\text{train}} = \sqrt{\frac{(-10)^2 + 0^2 + 10^2}{3}} = \sqrt{\frac{200}{3}} \approx 8.165$$.
   - Standardized train: $$\{-1.225, 0.000, +1.225\}$$.
   - Standardized test: $$\frac{100 - 20}{8.165} = +9.798$$ (correctly flags the test point as far out-of-distribution!).
2. **Leaky Pipeline (Mistakenly Fit Scaler on Train + Test Combined)**:
   - $$\mu_{\text{all}} = \frac{10 + 20 + 30 + 100}{4} = 40$$, $$\sigma_{\text{all}} \approx 35.355$$.
   - Now every training point shifts downward to $$\{-0.849, -0.566, -0.283\}$$ because future test data leaked into training preprocessing!
   Always fit preprocessing transforms strictly on `train` and only call `.transform()` on `val` and `test`.

Practical techniques for training neural networks:

1. **Initialization, Preprocessing, Batches, and Splits**
2. **Monitoring, Early Stopping, and the Training Loop** — clipping, full loop, hyperparameters

## Video tham khảo

- **xem trước** [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering, English. Initialization, activations, and batching are the practical half of training.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering, English. The follow-on lecture covers the optimization and regularization choices the second lesson monitors.
- [Train, Test, & Validation Sets explained](https://www.youtube.com/watch?v=Zi-0rlM4RDs) — deeplizard, English. Train, validation, and test splits are the data-handling section.
