---
layout: post
title: 04-02 Pooling Layers and CNN Architectures
chapter: '04'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter04
lesson_type: required
---

## Pooling Layers

**Pooling** (or subsampling) reduces spatial dimensions while retaining important features.

### Max Pooling

Takes the **maximum** value in each region:

```
Input (4×4):          Max Pool 2×2, stride=2:
1  3  2  4            3  4
2  1  5  6      →     
3  2  1  3            3  6
1  0  3  6
```

**Output**: $$\left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

**Properties**:
- **Translation invariance**: Small shifts don't change output
- **No learnable parameters**
- **Preserves dominant features**

### Average Pooling

Takes the **average** value:

```python
def avg_pool2d(input_data, pool_size=2, stride=2):
    """Average pooling"""
    H, W = input_data.shape[:2]
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            output[i, j] = np.mean(input_data[i*stride:i*stride+pool_size, 
                                             j*stride:j*stride+pool_size])
    return output
```

### Global Average Pooling (GAP)

Averages each feature map to a single value:

```
Input (7×7×512) → GAP → Output (512)
```

**Use**: Replace fully connected layers, reduce overfitting.

## Classic CNN Architectures

### LeNet-5 (1998)

First successful CNN for digit recognition:

```
Input (32×32) → Conv(6, 5×5) → AvgPool → Conv(16, 5×5) → AvgPool → FC(120) → FC(84) → FC(10)
```

### AlexNet (2012)

Won ImageNet 2012, started deep learning revolution:

```
Input (227×227×3) → Conv(96, 11×11, s=4) → MaxPool → Conv(256, 5×5) → MaxPool 
→ Conv(384, 3×3) → Conv(384, 3×3) → Conv(256, 3×3) → MaxPool → FC(4096) → FC(4096) → FC(1000)
```

**Key innovations**:
- ReLU activation
- Dropout
- Data augmentation
- GPU training

### VGGNet (2014)

Simple but deep architecture:

```
VGG-16: 13 conv layers (all 3×3) + 3 FC layers
```

**Pattern**: Conv-Conv-Pool repeated, doubling filters each stage (64→128→256→512→512)

**Philosophy**: Deeper is better, small filters are better

### ResNet (2015)

Introduced **residual connections** (skip connections):

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

```python
# Residual block
def residual_block(x):
    identity = x
    
    # Main path
    out = conv(x, filters)
    out = batch_norm(out)
    out = relu(out)
    
    out = conv(out, filters)
    out = batch_norm(out)
    
    # Skip connection
    out = out + identity
    out = relu(out)
    
    return out
```

**Enables**: Training very deep networks (50, 101, 152 layers)

**Why it works**: Easier to learn residual $$F(\mathbf{x})$$ than full mapping

## Complete CNN Example

```python
class SimpleCNN:
    def __init__(self):
        # Conv layers
        self.conv1 = ConvLayer(3, 32, 3, padding=1)    # 32×32×3 → 32×32×32
        self.conv2 = ConvLayer(32, 64, 3, padding=1)   # 16×16×32 → 16×16×64
        self.conv3 = ConvLayer(64, 128, 3, padding=1)  # 8×8×64 → 8×8×128
        
        # FC layers
        self.fc1_size = 4 * 4 * 128  # After 3 pooling layers
        self.fc1 = np.random.randn(256, self.fc1_size) * 0.01
        self.fc2 = np.random.randn(10, 256) * 0.01
    
    def forward(self, x):
        # Block 1
        x = self.conv1.forward(x)
        x = relu(x)
        x = max_pool2d(x, 2, 2)  # 32×32×32 → 16×16×32
        
        # Block 2
        x = self.conv2.forward(x)
        x = relu(x)
        x = max_pool2d(x, 2, 2)  # 16×16×64 → 8×8×64
        
        # Block 3
        x = self.conv3.forward(x)
        x = relu(x)
        x = max_pool2d(x, 2, 2)  # 8×8×128 → 4×4×128
        
        # Flatten
        x = x.reshape(x.shape[0], -1)  # (batch, 4*4*128)
        
        # FC layers
        x = relu(self.fc1 @ x.T)
        x = self.fc2 @ x
        
        return x
```

## Summary

- **Pooling** reduces spatial dimensions, adds translation invariance
- **Max pooling** is most common
- **Classic architectures**: LeNet → AlexNet → VGG → ResNet
- **Key innovations**: Deeper networks, skip connections, batch normalization
- Modern CNNs: ResNet, EfficientNet, Vision Transformers

Next chapter: Recurrent Neural Networks for sequential data!

