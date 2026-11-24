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

Before 2015, training a 20-layer neural network felt like balancing a broomstick on your fingertip: choose a learning rate slightly too large or initialize weights slightly off scale, and layer activations either exploded into `NaN` or collapsed into dead ReLUs. Sergey Ioffe and Christian Szegedy's **Batch Normalization** transformed deep learning overnight by standardizing each feature channel to zero mean and unit variance across the mini-batch—then letting two learnable parameters $$(\gamma, \beta)$$ restore any scale or shift the network actually needs. Later work (Santurkar et al., 2018) revealed an even deeper secret: BatchNorm works primarily because it dramatically smooths the loss surface's Lipschitzness, letting SGD take bold, stable steps.

## Background Knowledge Map: The Exact Mechanics of Batch Normalization

- **Per-Channel Standardization**: For a mini-batch of $$m$$ activations $$\{x_1, \dots, x_m\}$$ at one channel/feature, BatchNorm computes $$\mu_B = \frac{1}{m}\sum x_i$$, $$\sigma_B^2 = \frac{1}{m}\sum (x_i - \mu_B)^2$$, and normalizes $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$.
- **Learnable Affine Recovery ($$\gamma, \beta$$)**: Pure standardization would restrict activations to mean 0 and variance 1 (e.g., trapping inputs inside the linear region of a sigmoid). Adding $$y_i = \gamma \hat{x}_i + \beta$$ guarantees BatchNorm can even recover the exact identity function if $$\gamma = \sigma_B, \beta = \mu_B$$.
- **Exponential Moving Average (EMA) for Inference**: At test time, predictions for a single image must be deterministic and independent of other images in the batch—so BatchNorm swaps $$(\mu_B, \sigma_B^2)$$ for running statistics updated during training via $$\mu_{\text{run}} \leftarrow (1 - \alpha)\mu_{\text{run}} + \alpha \mu_B$$.
- **Scale Invariance of Weights**: Scaling the weights of the preceding linear/conv layer by any constant $$c > 0$$ multiplies both $$x$$ and $$\sigma_B$$ by $$c$$, which cancels out in $$\hat{x}$$—making effective step sizes self-stabilizing!

### Step-by-Step Worked Example (Read Slowly): Hand-Computing a Complete BatchNorm Forward Pass

Suppose a single feature channel receives a mini-batch of $$m = 4$$ values $$x = [2.0,\; 4.0,\; 6.0,\; 8.0]$$, with learnable parameters $$\gamma = 1.5$$, $$\beta = 0.5$$, and $$\epsilon = 0$$ (for clean exact arithmetic).

1. **Batch mean $$\mu_B$$**:
   $$\mu_B = \frac{2 + 4 + 6 + 8}{4} = \frac{20}{4} = 5.0.$$
2. **Batch variance $$\sigma_B^2$$ and standard deviation $$\sigma_B$$**:
   $$\sigma_B^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5.0 \implies \sigma_B = \sqrt{5} \approx 2.2361.$$
3. **Standardize each sample $$\hat{x}_i = (x_i - 5.0) / 2.2361$$**:
   $$\hat{x} \approx [-1.3416,\; -0.4472,\; +0.4472,\; +1.3416].$$
   (Check: mean $$= 0.0$$, variance $$= (1.8 + 0.2 + 0.2 + 1.8)/4 = 1.0$$.)
4. **Apply learnable affine transform $$y_i = 1.5 \hat{x}_i + 0.5$$**:
   $$y \approx [-1.5124,\; -0.1708,\; +1.1708,\; +2.5124].$$
5. **Update running mean** (starting from $$\mu_{\text{run}}^{(0)} = 0$$ with PyTorch momentum $$\alpha = 0.1$$):
   $$\mu_{\text{run}}^{(1)} = 0.9(0) + 0.1(5.0) = 0.5.$$

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


## Video tham khảo

- **xem trước** [Why Does Batch Norm Work? (C2W3L06)](https://www.youtube.com/watch?v=nUUqwaxLnWs) — DeepLearningAI, English. The mean-variance normalize-and-rescale formula.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — Andrej Karpathy, English. Those statistics computed in code, so the per-channel reduction is visible.
- [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering, English. The original training-stabilization claim the mechanics are justifying.
