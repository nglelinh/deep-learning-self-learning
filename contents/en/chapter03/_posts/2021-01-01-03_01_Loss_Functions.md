---
layout: post
title: 03-01 Loss Functions
chapter: '03'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter03
lesson_type: required
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

## Loss Functions for Binary Classification

### 1. Binary Cross-Entropy (Log Loss)

**Formula:**

$$\mathcal{L}_{\text{BCE}}(\hat{y}, y) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

where:
- $$y \in \{0, 1\}$$ is the true label
- $$\hat{y} \in (0, 1)$$ is the predicted probability

**Cost function:**

$$J_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

**Intuition:**
- If $$y = 1$$: Loss is $$-\log(\hat{y})$$, minimized when $$\hat{y} \to 1$$
- If $$y = 0$$: Loss is $$-\log(1-\hat{y})$$, minimized when $$\hat{y} \to 0$$

**Properties:**
- Based on maximum likelihood estimation
- Smooth and differentiable
- Well-suited for gradient-based optimization
- Heavily penalizes confident wrong predictions

**Derivative (with sigmoid output):**

For output layer with sigmoid: $$\hat{y} = \sigma(z)$$

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \hat{y} - y$$

This remarkably simple derivative makes training efficient!

**Use cases:**
- **Binary classification**: Is this image a cat or dog?
- Output layer with sigmoid activation

**Advantages:**
- Proper probabilistic interpretation
- Strong gradients for wrong predictions
- Well-suited for sigmoid outputs

**Disadvantages:**
- Can produce very large losses for confident wrong predictions
- Requires predicted probabilities (not logits directly)

### 2. Hinge Loss (SVM Loss)

**Formula:**

$$\mathcal{L}_{\text{Hinge}}(\hat{y}, y) = \max(0, 1 - y \cdot \hat{y})$$

where $$y \in \{-1, +1\}$$ and $$\hat{y}$$ is the raw score (not probability).

**Properties:**
- Used in Support Vector Machines
- Encourages margin maximization
- Not differentiable at $$y \cdot \hat{y} = 1$$

**Use cases:**
- Binary classification with SVM-like objectives
- When you want maximum margin classifiers

## Loss Functions for Multi-Class Classification

### 1. Categorical Cross-Entropy

**Formula:**

$$\mathcal{L}_{\text{CCE}}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

where:
- $$\mathbf{y}$$ is the true label in **one-hot encoding**: $$y_c \in \{0, 1\}$$, $$\sum_c y_c = 1$$
- $$\hat{\mathbf{y}}$$ is the predicted probability distribution: $$\hat{y}_c \in (0, 1)$$, $$\sum_c \hat{y}_c = 1$$
- $$C$$ is the number of classes

**Cost function:**

$$J_{\text{CCE}} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_c^{(i)} \log(\hat{y}_c^{(i)})$$

**Simplified form** (since only one $$y_c = 1$$):

$$\mathcal{L}_{\text{CCE}} = -\log(\hat{y}_{c^*})$$

where $$c^*$$ is the true class.

**Derivative (with softmax output):**

For softmax output: $$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$$

$$\frac{\partial \mathcal{L}_{\text{CCE}}}{\partial z_j} = \hat{y}_j - y_j$$

Again, remarkably simple!

**Use cases:**
- **Multi-class classification**: Digit recognition (MNIST), image classification (ImageNet)
- Output layer with softmax activation

**Advantages:**
- Proper probabilistic framework
- Standard for multi-class problems
- Simple gradients with softmax

**Disadvantages:**
- Requires one-hot encoded labels
- Can be numerically unstable (use log-softmax trick)

### 2. Sparse Categorical Cross-Entropy

**Formula:**

Same as categorical cross-entropy, but accepts integer labels instead of one-hot encoding.

**Input format:**
- $$y \in \{0, 1, \ldots, C-1\}$$ (integer class index)
- $$\hat{\mathbf{y}} \in \mathbb{R}^C$$ (probability distribution)

**Formula:**

$$\mathcal{L}_{\text{SCCE}}(\hat{\mathbf{y}}, y) = -\log(\hat{y}_y)$$

**Use cases:**
- Multi-class classification with integer labels
- Memory-efficient (no need to create one-hot vectors)

### 3. Kullback-Leibler (KL) Divergence

**Formula:**

$$\mathcal{L}_{\text{KL}}(\mathbf{p}, \mathbf{q}) = \sum_{c=1}^{C} p_c \log\left(\frac{p_c}{q_c}\right) = \sum_{c=1}^{C} [p_c \log(p_c) - p_c \log(q_c)]$$

where $$\mathbf{p}$$ is the true distribution and $$\mathbf{q}$$ is the predicted distribution.

**Properties:**
- Measures "distance" between two probability distributions
- Always non-negative
- Not symmetric: $$\text{KL}(p \| q) \neq \text{KL}(q \| p)$$

**Relation to Cross-Entropy:**

$$\text{KL}(p \| q) = H(p, q) - H(p)$$

where $$H(p, q)$$ is cross-entropy and $$H(p)$$ is entropy of $$p$$.

Since $$H(p)$$ is constant (true distribution doesn't change), minimizing KL divergence is equivalent to minimizing cross-entropy.

**Use cases:**
- Variational autoencoders (VAE)
- Distribution matching
- Knowledge distillation

## Practical Considerations

### Numerical Stability

#### Problem: Log of Small Numbers

Computing $$\log(\hat{y})$$ when $$\hat{y}$$ is very close to 0 can cause numerical issues.

#### Solution: Clip Values

```python
epsilon = 1e-7
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
loss = -np.mean(y_true * np.log(y_pred_clipped))
```

#### Better Solution: LogSumExp Trick

For cross-entropy with softmax, compute loss directly from logits:

```python
def softmax_cross_entropy(logits, labels):
    """Numerically stable softmax + cross-entropy"""
    # Log-sum-exp trick
    logits_max = np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = logits_max + np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
    log_softmax = logits - log_sum_exp
    
    # Cross-entropy
    return -np.mean(np.sum(labels * log_softmax, axis=-1))
```

### Implementation Examples

```python
import numpy as np

class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred, epsilon=1e-7):
        """Binary Cross-Entropy"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred, epsilon=1e-7):
        """Categorical Cross-Entropy"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
    
    @staticmethod
    def sparse_categorical_crossentropy(y_true, y_pred, epsilon=1e-7):
        """Sparse Categorical Cross-Entropy
        
        y_true: integer labels (m,)
        y_pred: probabilities (m, C)
        """
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        log_likelihood = -np.log(y_pred[range(m), y_true])
        return np.mean(log_likelihood)
    
    @staticmethod
    def huber(y_true, y_pred, delta=1.0):
        """Huber Loss"""
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# Example usage
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0.1, 0.9, 0.8, 0.3])

print("Binary Cross-Entropy:", LossFunctions.binary_crossentropy(y_true, y_pred))

# Multi-class example
y_true_categorical = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_categorical = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

print("Categorical Cross-Entropy:", LossFunctions.categorical_crossentropy(y_true_categorical, y_pred_categorical))
```

## Choosing the Right Loss Function

### Decision Tree

```
Task Type?
├─ Regression
│  ├─ With outliers? → MAE or Huber Loss
│  └─ Without outliers? → MSE
│
├─ Binary Classification
│  ├─ Probabilistic output? → Binary Cross-Entropy
│  └─ Margin-based? → Hinge Loss
│
└─ Multi-class Classification
   ├─ One-hot labels? → Categorical Cross-Entropy
   └─ Integer labels? → Sparse Categorical Cross-Entropy
```

### Quick Reference

| Task | Loss Function | Output Activation |
|------|--------------|-------------------|
| Regression | MSE, MAE, Huber | Linear |
| Binary Classification | Binary Cross-Entropy | Sigmoid |
| Multi-class Classification | Categorical Cross-Entropy | Softmax |
| Multi-label Classification | Binary Cross-Entropy (per label) | Sigmoid (per label) |

## Summary

- **Loss functions** quantify the difference between predictions and true values
- **Regression losses**: MSE (sensitive to outliers), MAE (robust), Huber (balanced)
- **Classification losses**: Cross-entropy (probabilistic, standard choice)
- **Binary classification**: Binary cross-entropy with sigmoid
- **Multi-class classification**: Categorical cross-entropy with softmax
- **Numerical stability** is crucial when implementing loss functions
- **Choice of loss** should match the task and data characteristics

In the next lesson, we'll learn about gradient descent, the algorithm that uses these loss functions to update network parameters.

