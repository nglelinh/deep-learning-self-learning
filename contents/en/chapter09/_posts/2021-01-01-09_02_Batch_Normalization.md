---
layout: post
title: 09-02 Batch Normalization
chapter: '09'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter09
lesson_type: required
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

## Where to Place Batch Norm

### Before or After Activation?

**Original paper (before activation)**:
```python
x = conv(x)
x = batch_norm(x)
x = relu(x)
```

**Modern practice (after activation)**:
```python
x = conv(x)
x = relu(x)
x = batch_norm(x)
```

Both work, but "after" is more common now.

### Complete Network Example

```python
class ConvNetWithBatchNorm:
    def __init__(self):
        # Convolution + BatchNorm + Activation
        self.conv1 = Conv2D(3, 64, 3, padding=1)
        self.bn1 = BatchNorm2D(64)
        
        self.conv2 = Conv2D(64, 128, 3, padding=1)
        self.bn2 = BatchNorm2D(128)
        
        self.conv3 = Conv2D(128, 256, 3, padding=1)
        self.bn3 = BatchNorm2D(256)
        
        self.fc1 = Linear(256 * 4 * 4, 512)
        self.bn4 = BatchNorm1D(512)
        
        self.fc2 = Linear(512, 10)
    
    def forward(self, x, training=True):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = relu(x)
        x = max_pool(x, 2)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x, training)
        x = relu(x)
        x = max_pool(x, 2)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x, training)
        x = relu(x)
        x = max_pool(x, 2)
        
        # Fully connected
        x = x.flatten()
        x = self.fc1(x)
        x = self.bn4(x, training)
        x = relu(x)
        
        x = self.fc2(x)
        return x
```

## Batch Normalization for CNNs

For convolutional layers, normalize across spatial dimensions:

```python
class BatchNorm2D:
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        
        # Parameters: one per channel
        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))
        
        # Running statistics
        self.running_mean = np.zeros((1, num_channels, 1, 1))
        self.running_var = np.ones((1, num_channels, 1, 1))
    
    def forward(self, x, training=True):
        """
        x: shape (batch, channels, height, width)
        """
        if training:
            # Compute mean and var across batch and spatial dims
            # Keep channel dimension
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * var
            
            return out
        else:
            x_norm = (x - self.running_mean) / \
                    np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta
```

## Variants and Alternatives

### 1. Layer Normalization

Normalize across features instead of batch:

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x: shape (batch, features)
    Normalize each sample independently
    """
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

**Use case**: RNNs, Transformers (where batch size varies or is 1)

### 2. Instance Normalization

Normalize each sample and each channel independently:

```python
def instance_norm(x, gamma, beta, eps=1e-5):
    """
    x: shape (batch, channels, height, width)
    Normalize each instance and channel
    """
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

**Use case**: Style transfer, GANs

### 3. Group Normalization

Compromise between Layer and Instance norm:

```python
def group_norm(x, gamma, beta, num_groups=32, eps=1e-5):
    """
    x: shape (batch, channels, height, width)
    Split channels into groups and normalize
    """
    N, C, H, W = x.shape
    x = x.reshape(N, num_groups, C // num_groups, H, W)
    
    mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    var = np.var(x, axis=(2, 3, 4), keepdims=True)
    
    x_norm = (x - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    
    return gamma * x_norm + beta
```

**Use case**: Small batch sizes, object detection

## Common Issues and Solutions

### Issue 1: Small Batch Sizes

**Problem**: Unreliable batch statistics with small batches

**Solutions**:
- Use Group Normalization or Layer Normalization
- Increase batch size if possible
- Use larger momentum for running statistics

### Issue 2: Train-Test Discrepancy

**Problem**: Different behavior between training and testing

**Solution**: Always remember to set training mode correctly

```python
# Training
model.train()  # or training=True
loss = train_step(data)

# Testing
model.eval()  # or training=False
accuracy = evaluate(test_data)
```

### Issue 3: Batch Norm + Dropout

**Problem**: Can interact poorly

**Solutions**:
- Usually don't need dropout with batch norm
- If using both: dropout after batch norm
- Or use only one of them

## Practical Tips

### 1. Initialization with Batch Norm

Can use larger initial weights:

```python
# Without BatchNorm: careful initialization
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

# With BatchNorm: can be more aggressive
W = np.random.randn(n_in, n_out) * 0.05  # Larger variance OK
```

### 2. Learning Rates

Can use much higher learning rates:

```python
# Without BatchNorm
lr = 0.001

# With BatchNorm
lr = 0.01  # 10x higher!
```

### 3. Momentum for Running Stats

```python
# Fast adaptation (small datasets)
momentum = 0.01

# Stable statistics (large datasets)
momentum = 0.1  # Default

# Very stable (production)
momentum = 0.001
```

## Batch Norm in Modern Architectures

### ResNet

```python
def resnet_block(x):
    identity = x
    
    # Conv -> BN -> ReLU
    x = conv(x)
    x = batch_norm(x)
    x = relu(x)
    
    # Conv -> BN
    x = conv(x)
    x = batch_norm(x)
    
    # Add skip connection before final ReLU
    x = x + identity
    x = relu(x)
    
    return x
```

### MobileNet

```python
def depthwise_separable_block(x):
    # Depthwise conv
    x = depthwise_conv(x)
    x = batch_norm(x)
    x = relu(x)
    
    # Pointwise conv
    x = conv_1x1(x)
    x = batch_norm(x)
    x = relu(x)
    
    return x
```

## Summary

- **Batch Normalization** normalizes layer inputs to stable distribution
- **Reduces internal covariate shift**, enables faster training
- **Allows higher learning rates** (10-100x speedup)
- **Acts as regularization**, reduces need for dropout
- **Variants**: Layer Norm (RNNs), Instance Norm (style transfer), Group Norm (small batches)
- **Modern practice**: Essential in almost all deep networks
- **Key**: Remember training vs. inference modes!

Batch Normalization revolutionized deep learning by making networks much easier to train. It's now a standard component in almost every modern architecture.

**Next**: We'll explore L1/L2 regularization, early stopping, and data augmentation to complete our regularization toolkit!

