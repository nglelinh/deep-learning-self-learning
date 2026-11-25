---
layout: post
title: 03-01-01 Regression Loss Functions
chapter: '03'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

> *Predicting house prices, battery lifetimes, or bounding-box coordinates looks simple until a single data-entry typo adds an extra zero to one label. Understanding how MSE, MAE, and Huber loss trade off smooth convergence near zero against robustness to extreme errors is essential for every regression model.*

## Key ideas: Regression Loss Geometry

1. **Mean Squared Error (MSE / $$L_2$$)**: Smooth everywhere with gradient $$2(\hat{y} - y)$$ that shrinks naturally as predictions approach the target, converging cleanly to the conditional mean $$\mathbb{E}[Y\mid X]$$.
2. **Mean Absolute Error (MAE / $$L_1$$)**: Linear penalty converging to the conditional median, robust to heavy-tailed outliers, but its constant-magnitude gradient $$\pm 1$$ can oscillate around the minimum unless the learning rate decays.
3. **Huber Loss (Smooth $$L_1$$ Hybrid)**: Acts quadratically ($$\frac{1}{2}e^2$$) for small errors $$\lvert e\rvert \leq \delta$$ and linearly ($$\delta(\lvert e\rvert - \frac{1}{2}\delta)$$) for large errors $$\lvert e\rvert > \delta$$, combining smooth convergence near $$0$$ with bounded outlier gradients $$\pm\delta$$.
4. **Scale & Metric Alignment**: Reminds practitioners to distinguish training loss (e.g., MSE) from interpretable evaluation units (RMSE in original units, $$R^2$$, or MAPE).

### Worked example: Comparing MSE, MAE, and Huber Loss ($$\delta = 1.0$$) Across Small and Large Errors

Compute loss value and gradient magnitude $$\left\lvert \frac{\partial L}{\partial \hat{y}}\right\rvert$$ for two residuals: a small error $$e_1 = 0.5$$ and an outlier error $$e_2 = 10.0$$ (where $$e = \hat{y} - y$$).

1. **Small Error ($$e_1 = 0.5 \leq \delta = 1.0$$)**:
   - **Half-MSE** ($$\frac{1}{2}e^2$$): loss $$= \frac{1}{2}(0.5)^2 = 0.125$$, gradient $$= 0.5$$.
   - **MAE** ($$\lvert e\rvert$$): loss $$= 0.500$$, gradient $$= 1.0$$ (twice as aggressive near the minimum!).
   - **Huber ($$\delta = 1.0$$)**: since $$\lvert e_1\rvert \leq 1.0$$, uses quadratic branch: loss $$= 0.125$$, gradient $$= 0.5$$.
2. **Outlier Error ($$e_2 = 10.0 > \delta = 1.0$$)**:
   - **Half-MSE**: loss $$= \frac{1}{2}(10)^2 = 50.0$$, gradient $$= 10.0$$ (huge destabilizing kick!).
   - **MAE**: loss $$= 10.0$$, gradient $$= 1.0$$.
   - **Huber ($$\delta = 1.0$$)**: uses linear branch $$\delta(\lvert e\rvert - 0.5\delta) = 1.0(10.0 - 0.5) = 9.5$$, gradient $$= \delta = 1.0$$ (capped!).
3. **Takeaway**:
   Huber matches MSE's gentle vanishing gradient ($$0.5$$) near zero while matching MAE's capped gradient ($$1.0$$) on outliers—which is why object detectors like Faster R-CNN rely on Smooth $$L_1$$ / Huber loss for bounding-box regression.

This lesson covers loss functions (also called cost functions or objective functions), which quantify how well a neural network is performing.

---

## What is a Loss Function?

A **loss function** $$\mathcal{L}$$ measures the discrepancy between the predicted output $$\hat{y}$$ and the true output $$y$$. The goal of training is to find parameters $$\theta$$ (weights and biases) that minimize this loss.

### Single Example Loss

For a single training example:

$$\mathcal{L}(\hat{y}, y)$$

### Cost Function (Total Loss)

For a dataset with $$m$$ examples, the **cost function** $$J$$ is typically the average loss:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

Some formulations also include regularization terms (covered later).

## Loss Functions for Regression

### 1. Mean Squared Error (MSE)

**Formula:**

$$\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{2}(y - \hat{y})^2$$

**Cost function:**

$$J_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2}(y^{(i)} - \hat{y}^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

**Note**: The factor $$\frac{1}{2}$$ is included for mathematical convenience (simplifies derivatives).

**Properties:**
- Always non-negative
- Heavily penalizes large errors (quadratic penalty)
- Sensitive to outliers
- Smooth and differentiable everywhere

**Derivative:**

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}} = \hat{y} - y$$

**Use cases:**
- **Regression tasks**: Predicting continuous values
- When errors are normally distributed
- When all errors should be weighted equally

**Advantages:**
- Simple and intuitive
- Smooth gradients
- Well-understood theoretically

**Disadvantages:**
- Very sensitive to outliers (large errors are heavily penalized)
- Assumes errors are normally distributed

### 2. Mean Absolute Error (MAE)

**Formula:**

$$\mathcal{L}_{\text{MAE}}(\hat{y}, y) = |y - \hat{y}|$$

**Cost function:**

$$J_{\text{MAE}} = \frac{1}{m} \sum_{i=1}^{m} |y^{(i)} - \hat{y}^{(i)}|$$

**Properties:**
- Linear penalty for errors
- More robust to outliers than MSE
- Not differentiable at $$\hat{y} = y$$

**Derivative:**

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{y}} = \begin{cases} 1 & \text{if } \hat{y} > y \\ -1 & \text{if } \hat{y} < y \\ \text{undefined} & \text{if } \hat{y} = y \end{cases}$$

(In practice, we use subgradients or smooth approximations)

**Use cases:**
- Regression with outliers
- When you want equal penalty for all error magnitudes

**Advantages:**
- Robust to outliers
- Intuitive interpretation (average absolute error)

**Disadvantages:**
- Non-differentiable at zero
- Can be slower to converge
- Constant gradient may cause issues near minimum

### 3. Huber Loss

**Formula:**

$$\mathcal{L}_{\text{Huber}}(\hat{y}, y) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

where $$\delta$$ is a threshold parameter.

**Properties:**
- Combines advantages of MSE and MAE
- Quadratic for small errors, linear for large errors
- Smooth and differentiable everywhere

**Use cases:**
- Regression with potential outliers
- When you want smooth gradients but outlier robustness

**Advantages:**
- Less sensitive to outliers than MSE
- Smooth gradients (unlike MAE)
- Configurable via $$\delta$$

**Disadvantages:**
- Requires tuning $$\delta$$ hyperparameter


## Video tham khảo

- **xem trước** [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown, English. The squared difference between prediction and target is the regression cost this lesson defines.
- [Lecture 3 | Loss Functions and Optimization](https://www.youtube.com/watch?v=h7iBpEHGVNc) — Stanford University School of Engineering, English. Places mean squared error among the other losses you will swap in later.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, English. Shows a concrete loss surface being walked, which is what a regression loss looks like once parameters move.
