---
layout: post
title: 04-01 Convolutional Layers
chapter: '04'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter04
lesson_type: required
---

# Convolutional Layers: Building Blocks of Computer Vision

![CNN Architecture Overview](https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/800px-Typical_cnn.png)
*Hình ảnh: Kiến trúc tổng quan của Convolutional Neural Network (CNN). Nguồn: Wikimedia Commons*

## 1. Concept Overview

**Convolutional layers** are specialized neural network layers designed for processing grid-like data, especially images. Instead of connecting every input to every neuron (fully connected), convolutional layers use small, learnable filters that slide across the input to detect local patterns.

**Why CNNs matter**:
- **Parameter efficiency**: Millions fewer parameters than fully connected
- **Translation invariance**: Detects features regardless of position
- **Hierarchical learning**: Low-level → mid-level → high-level features
- **State-of-the-art**: Best performance on vision tasks

**Key insight**: Images have spatial structure - nearby pixels are related. Convolutional layers exploit this structure through **local connectivity** and **parameter sharing**.

**Analogy**: Think of convolution as sliding a magnifying glass (filter) across an image to find specific patterns (edges, textures, shapes). Each filter specializes in detecting one type of pattern.

## 2. Mathematical Foundation

### 2D Convolution Operation

![Convolution Operation](https://miro.medium.com/v2/resize:fit:1400/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)
*Hình ảnh: Minh họa phép toán Convolution 2D trên ảnh. Nguồn: Medium*

For image $$I$$ (height $$H$$, width $$W$$) and kernel $$K$$ (size $$k \times k$$):

$$(I * K)[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n] \cdot K[m,n]$$

**In deep learning**, we use cross-correlation (same but without kernel flipping):

$$S[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n] \cdot K[m,n]$$

### Multi-Channel Convolution

For RGB image ($$H \times W \times 3$$) and filter ($$k \times k \times 3$$):

$$S[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n, c] \cdot K[m,n,c] + b$$

where:
- $$C_{in}$$: number of input channels (3 for RGB)
- $$b$$: bias term
- Each filter produces one output channel

### Output Dimensions

With:
- Input size: $$n \times n$$
- Kernel size: $$k \times k$$
- Stride: $$s$$
- Padding: $$p$$

**Output size**:

$$n_{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

### Parameter Count

For a convolutional layer:

**Parameters** = $$(k \times k \times C_{in}) \times C_{out} + C_{out}$$

where $$C_{out}$$ is number of output channels (filters).

**Example**: 
- Input: $$32 \times 32 \times 3$$ (RGB image)
- Filter: $$3 \times 3$$, 64 filters
- Parameters: $$(3 \times 3 \times 3) \times 64 + 64 = 1,792$$

**Compare to fully connected**:
- $$(32 \times 32 \times 3) \times 64 = 196,608$$ parameters!
- **110× reduction** with convolution!

## 3. Example / Intuition

### Example 1: Edge Detection

**Vertical edge detector** (3×3):

$$K = \begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$$

**Input image**:
$$I = \begin{bmatrix} 
10 & 10 & 10 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 \\
10 & 10 & 10 & 0 & 0 \\
10 & 10 & 10 & 0 & 0
\end{bmatrix}$$

**Convolution at position (1,1)**:

$$\begin{align}
S[1,1] &= (10×1 + 10×0 + 10×(-1)) \\
&+ (10×1 + 10×0 + 10×(-1)) \\
&+ (10×1 + 10×0 + 10×(-1)) \\
&= 0
\end{align}$$

**At position (1,2)** (on the edge):

$$S[1,2] = (10×1 + 0×0 + 0×(-1)) + \ldots = 30$$

Result: **High activation at vertical edges!**

### Example 2: How CNNs See

**Layer 1 (low-level)**:
- Horizontal edges: $$\begin{bmatrix} 1 & 1 & 1 \\ 0 & 0 & 0 \\ -1 & -1 & -1 \end{bmatrix}$$
- Vertical edges: $$\begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$$
- Diagonal edges, color blobs

**Layer 2-3 (mid-level)**:
- Corners (combination of edges)
- Simple shapes (circles, rectangles)
- Textures (patterns of edges)

**Layer 4-5 (high-level)**:
- Object parts (eyes, wheels, ears)
- Complex patterns

**Final layers**:
- Complete objects (faces, cars, animals)

## 4. Code Snippet

### NumPy Implementation

```python
import numpy as np

def conv2d_simple(image, kernel, stride=1, padding=0):
    """
    Simple 2D convolution (single channel)
    
    image: (H, W)
    kernel: (k, k)
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    
    H, W = image.shape
    k = kernel.shape[0]
    
    # Output dimensions
    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1
    
    output = np.zeros((H_out, W_out))
    
    # Convolution
    for i in range(H_out):
        for j in range(W_out):
            h = i * stride
            w = j * stride
            # Extract region and apply filter
            region = image[h:h+k, w:w+k]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Example: Edge detection
image = np.array([
    [0, 0, 0, 200, 200],
    [0, 0, 0, 200, 200],
    [0, 0, 0, 200, 200],
    [0, 0, 0, 200, 200],
    [0, 0, 0, 200, 200]
])

vertical_edge_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

result = conv2d_simple(image, vertical_edge_kernel)
print("Edge detection result:")
print(result)
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        x: (batch, in_channels, height, width)
        """
        out = self.conv(x)
        out = self.relu(out)
        return out

# Example: RGB to 64 feature maps
conv_layer = ConvLayer(in_channels=3, out_channels=64, 
                       kernel_size=3, padding=1)

# Input: batch of 8 RGB images, 32x32
x = torch.randn(8, 3, 32, 32)
output = conv_layer(x)
print(f"Input shape: {x.shape}")   # (8, 3, 32, 32)
print(f"Output shape: {output.shape}")  # (8, 64, 32, 32)

# Check parameters
num_params = sum(p.numel() for p in conv_layer.parameters())
print(f"Number of parameters: {num_params}")  # 3*3*3*64 + 64 = 1,792
```

### Complete CNN Block

```python
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Example: Standard CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.block1 = CNNBlock(3, 64)      # 32x32x3 → 16x16x64
        self.block2 = CNNBlock(64, 128)    # 16x16x64 → 8x8x128
        self.block3 = CNNBlock(128, 256)   # 8x8x128 → 4x4x256
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Usage
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 32, 32)  # Batch of 4 images
output = model(x)
print(f"Output shape: {output.shape}")  # (4, 10)
```

## 5. Related Concepts

### Fully Connected Layers
- Connect every input to every output
- No spatial structure assumption
- Many more parameters
- Used after conv layers for classification

### Pooling Layers
- Downsample feature maps
- Add translation invariance
- Reduce computation
- No learnable parameters

### Batch Normalization
- Normalize activations
- Stabilize training
- Often used after convolution
- Enables higher learning rates

### Residual Connections (ResNet)
- Skip connections bypass layers
- Help gradients flow
- Enable very deep networks
- $$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

### Depthwise Separable Convolutions
- Factorize standard convolution
- Fewer parameters and computations
- Used in MobileNet, EfficientNet

## 6. Fundamental Papers

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
Introduced LeNet-5, the first successful CNN for digit recognition. Demonstrated that convolutional layers with shared weights can learn hierarchical features, establishing CNNs as the architecture of choice for computer vision.

**["ImageNet Classification with Deep Convolutional Neural Networks" (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)**  
*Authors*: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
AlexNet won ImageNet 2012 by a huge margin using deep CNNs with ReLU, dropout, and GPU training. This victory sparked the deep learning revolution and proved CNNs could scale to real-world vision tasks.

**["Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015)](https://arxiv.org/abs/1409.1556)**  
*Authors*: Karen Simonyan, Andrew Zisserman  
VGGNet showed that network depth is crucial - using small 3×3 filters consistently through 16-19 layers achieved excellent results. Demonstrated that simple, deep architectures with repeated patterns can be very effective.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Authors*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
ResNet introduced skip connections enabling training of 152+ layer networks. Solved degradation problem in very deep networks. Winner of ImageNet 2015, fundamentally changed how we build deep CNNs.

**["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)](https://arxiv.org/abs/1704.04861)**  
*Authors*: Andrew G. Howard et al.  
Introduced depthwise separable convolutions for efficient CNNs on mobile devices. Showed how to maintain accuracy while dramatically reducing computation and parameters, enabling deployment on edge devices.

## Common Pitfalls and Tricks

### ⚠️ Pitfall 1: Not Using Padding
**Issue**: Output shrinks with each layer, losing boundary information  
**Solution**: Use "same" padding to maintain spatial dimensions

```python
# Without padding: 32×32 → 30×30 → 28×28 (shrinks!)
conv1 = nn.Conv2d(3, 64, kernel_size=3)

# With padding: 32×32 → 32×32 → 32×32 (maintains size)
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
```

### ⚠️ Pitfall 2: Too Large Kernel Size
**Issue**: Fewer parameters but worse performance  
**Solution**: Stack multiple 3×3 instead of one large kernel

```python
# Less effective
conv = nn.Conv2d(3, 64, kernel_size=7)  # 7×7 = 49 params per channel

# Better (VGG approach)
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
# Two 3×3 = 18 params, same receptive field as 5×5
```

### ⚠️ Pitfall 3: Forgetting Channel Dimension
**Issue**: Confusing (H, W, C) vs (C, H, W) formats  
**Solution**: PyTorch uses (N, C, H, W), TensorFlow uses (N, H, W, C)

```python
# PyTorch format
x = torch.randn(batch, channels, height, width)

# TensorFlow format
x = tf.random.normal([batch, height, width, channels])
```

### ✅ Trick 1: 1×1 Convolutions
```python
# Reduce channels (dimensionality reduction)
conv_1x1 = nn.Conv2d(256, 64, kernel_size=1)
# Input: 32×32×256 → Output: 32×32×64
# Adds nonlinearity without changing spatial dims
```

### ✅ Trick 2: Strided Convolution for Downsampling
```python
# Instead of pooling, use stride > 1
conv_downsample = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# 32×32×64 → 16×16×128 (learnable downsampling!)
```

### ✅ Trick 3: Receptive Field Calculation
```python
# Receptive field after n layers of 3×3 convs
receptive_field = 1 + 2 * n
# 3 layers: 1 + 2*3 = 7×7 receptive field
```

## Key Takeaways

- **Convolution** = local connectivity + parameter sharing
- **Filters** detect local patterns (edges, textures, shapes)
- **Parameter efficiency**: Millions fewer parameters than FC
- **Translation invariance**: Same filter applied everywhere
- **Hierarchical features**: Low → mid → high level
- **Modern practice**: Small 3×3 filters, deep networks
- **Output size**: Depends on kernel size, stride, padding

Convolutional layers are the foundation of computer vision in deep learning. Mastering them is essential for any vision application!

**Next**: Pooling layers and complete CNN architectures (AlexNet, VGG, ResNet)!
