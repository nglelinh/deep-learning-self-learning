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

