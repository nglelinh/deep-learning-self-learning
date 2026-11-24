---
layout: post
title: 09-02-02 Batch Norm Placement, Variants, and Tips
chapter: '09'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter09
---

Batch Normalization conquered ImageNet CNNs trained with batch sizes of 64 to 256—yet the moment you move to high-resolution object detection (batch size 2), style transfer, or variable-length Transformers, normalizing *across the batch axis* breaks down completely: noisy 2-sample variance estimates destabilize training, and padding tokens pollute sequence statistics. That exact limitation gave birth to the normalization family tree—**LayerNorm**, **InstanceNorm**, and **GroupNorm**—each differing solely in *which axes* of the `(N, C, H, W)` tensor are averaged together.

## Background Knowledge Map: The Axis Geometry of Normalization Variants

- **BatchNorm2d — Normalize Over `(N, H, W)` per Channel `C`**: Computes one mean and variance per channel across the entire mini-batch and spatial grid. Excels on medium/large-batch CNNs; degrades sharply when per-GPU batch size drops below ~8.
- **LayerNorm — Normalize Over `(C, H, W)` (or `D`) per Sample `N`**: Completely independent of batch size and sequence length, with identical behavior at train and test time—making it (and RMSNorm) the undisputed standard for Transformers and RNNs.
- **InstanceNorm — Normalize Over `(H, W)` per Sample `N` and Channel `C`**: Strips per-image contrast and color statistics independently for each channel, making it the workhorse of neural style transfer and image-to-image translation.
- **GroupNorm — Normalize Over `(C/G, H, W)` per Sample `N` and Group `G`**: Splits channels into $$G$$ groups (e.g., $$G = 32$$), interpolating smoothly between LayerNorm ($$G=1$$) and InstanceNorm ($$G=C$$) while remaining 100% batch-size independent for detection/segmentation backbones.

### Step-by-Step Worked Example (Read Slowly): Counting Normalization Groups on a `(N=2, C=6, H=4, W=4)` Tensor

Consider a 4D feature tensor of shape `(N=2, C=6, H=4, W=4)` (total $$2 \times 6 \times 16 = 192$$ activations). How many `(mean, var)` pairs are computed, and how many elements are averaged in each?

1. **BatchNorm2d** (axes `(N, H, W)`):
   - $$C = 6$$ `(mean, var)` pairs; each averages $$N \cdot H \cdot W = 2 \times 4 \times 4 = 32$$ elements.
2. **LayerNorm** (axes `(C, H, W)`):
   - $$N = 2$$ `(mean, var)` pairs; each averages $$C \cdot H \cdot W = 6 \times 4 \times 4 = 96$$ elements.
3. **InstanceNorm2d** (axes `(H, W)`):
   - $$N \cdot C = 12$$ `(mean, var)` pairs; each averages $$H \cdot W = 4 \times 4 = 16$$ elements.
4. **GroupNorm with $$G = 2$$ groups** (each group has $$C / G = 3$$ channels):
   - $$N \cdot G = 4$$ `(mean, var)` pairs; each averages $$(C/G) \cdot H \cdot W = 3 \times 4 \times 4 = 48$$ elements.
   Notice the invariant: in every single variant, $$(\text{\# groups}) \times (\text{elements per group}) = 192$$!

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


## Video tham khảo

- **xem trước** [Why Does Batch Norm Work? (C2W3L06)](https://www.youtube.com/watch?v=nUUqwaxLnWs) — DeepLearningAI, English. So placement debates (before or after the activation) have a mechanism under them.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — Andrej Karpathy, English. Where the norm actually sits in a modern block, and what breaks if the batch is tiny.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering, English. Layer-norm style alternatives show up in this second training lecture's modern-architecture discussion.
