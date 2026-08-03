---
layout: post
title: 09-02-01 Batch Normalization Mechanics
chapter: '09'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter09
---

## What is Batch Normalization?

**Batch Normalization** (BatchNorm or BN), introduced by Ioffe and Szegedy (2015), normalizes the inputs of each layer to have zero mean and unit variance. It has become one of the most important techniques in modern deep learning.

### The Problem: Internal Covariate Shift

As network trains:
- Distribution of layer inputs changes
- Each layer must adapt to new input distribution
- Slows down training significantly
- Makes networks sensitive to initialization

**Batch Norm solution**: Normalize layer inputs to stable distribution.

## How Batch Normalization Works

### Forward Pass (Training)

For a mini-batch of activations $$\mathbf{x} = \{x_1, x_2, \ldots, x_m\}$$:

**Step 1: Compute batch statistics**

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma^2_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**Step 2: Normalize**

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$$

where $$\epsilon$$ (e.g., $$10^{-5}$$) prevents division by zero.

**Step 3: Scale and shift (learnable parameters)**

$$y_i = \gamma \hat{x}_i + \beta$$

where:
- $$\gamma$$: scale parameter (learned)
- $$\beta$$: shift parameter (learned)

### Inference (Testing)

Use population statistics (moving averages from training):

$$\hat{x} = \frac{x - \mu_{\text{pop}}}{\sqrt{\sigma^2_{\text{pop}} + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

## Implementation

```python
import numpy as np

class BatchNorm1D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        num_features: number of features/channels
        eps: small constant for numerical stability
        momentum: for running mean/var updates
        """
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backprop
        self.cache = None
    
    def forward(self, x, training=True):
        """
        x: input of shape (batch_size, num_features)
        training: whether in training mode
        """
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_normalized + self.beta
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
            
            # Cache for backward pass
            self.cache = (x, x_normalized, batch_mean, batch_var)
            
        else:
            # Use running statistics
            x_normalized = (x - self.running_mean) / \
                          np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, dout):
        """
        Backpropagate through batch normalization
        dout: gradient from next layer
        """
        x, x_normalized, mean, var = self.cache
        N, D = x.shape
        
        # Gradients of parameters
        self.dgamma = np.sum(dout * x_normalized, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        
        # Gradient of normalized x
        dx_normalized = dout * self.gamma
        
        # Gradient of variance
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * \
                     (var + self.eps)**(-1.5), axis=0)
        
        # Gradient of mean
        dmean = np.sum(dx_normalized * -1 / np.sqrt(var + self.eps), axis=0) + \
                dvar * np.mean(-2 * (x - mean), axis=0)
        
        # Gradient of x
        dx = dx_normalized / np.sqrt(var + self.eps) + \
             dvar * 2 * (x - mean) / N + \
             dmean / N
        
        return dx

# Example usage
batch_norm = BatchNorm1D(num_features=128)

# Training
x_train = np.random.randn(32, 128)
out_train = batch_norm.forward(x_train, training=True)
print(f"Train output mean: {np.mean(out_train, axis=0)[:5]}")
print(f"Train output std: {np.std(out_train, axis=0)[:5]}")

# Testing
x_test = np.random.randn(10, 128)
out_test = batch_norm.forward(x_test, training=False)
print(f"Test uses running statistics")
```

## Why Batch Normalization Works

### 1. Reduces Internal Covariate Shift
- Stabilizes distribution of layer inputs
- Each layer sees more consistent inputs
- Easier to learn

### 2. Allows Higher Learning Rates
- More stable gradient flow
- Can train 10-100x faster
- Less sensitive to initialization

### 3. Acts as Regularization
- Adds noise to activations (from batch statistics)
- Similar effect to dropout
- Can reduce need for dropout

### 4. Smooths Optimization Landscape
- Makes loss surface smoother
- Gradients more predictable
- Easier optimization

