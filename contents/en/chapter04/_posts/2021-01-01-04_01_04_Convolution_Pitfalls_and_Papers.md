---
layout: post
title: 04-01-04 Convolution Related Concepts, Pitfalls, and Papers
chapter: '04'
order: 8
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

Standard dense convolution launched the deep vision revolution, but deploying CNNs onto real hardware—from smartphone cameras to high-resolution medical segmentation—quickly exposed its cracks: quadratic compute growth with kernel size, checkerboard artifacts, vanishing gradients in deep stacks, and bloated parameter counts. The innovations that followed—depthwise separable convolutions (MobileNet), dilated convolutions (DeepLab), residual connections (ResNet), and careful normalization—are not disconnected tricks; they are targeted mathematical surgeries on the original convolutional operator.

## Key ideas: Architectural Variants and Failure Modes of Convolution

- **Depthwise Separable Factorization**: Standard convolution couples spatial filtering and cross-channel mixing into one $$C_{\text{in}} C_{\text{out}} k^2$$ tensor. Factoring it into a per-channel **depthwise** $$k \times k$$ conv followed by a **pointwise** $$1 \times 1$$ conv decouples space from depth, slashing MACs and weights by roughly $$8\times$$–$$9\times$$ for $$3 \times 3$$ kernels.
- **Dilated (Atrous) Convolutions**: Inserting spacing $$d$$ between kernel elements expands the receptive field exponentially ($$k_{\text{eff}} = d(k-1)+1$$) without losing spatial resolution or adding parameters—critical for dense prediction tasks like semantic segmentation—though uncoordinated dilation rates can cause "gridding" blind spots.
- **Residual Skip Paths & Gradient Highways**: Adding the identity mapping $$y = F(x) + x$$ ensures that the Jacobian $$\partial y / \partial x = I + \partial F / \partial x$$ stays well-conditioned even across 100+ layers, preventing degradation where deeper plain CNNs perform worse on training data than shallower ones.
- **Boundary Artifacts & Receptive Field Mismatch**: Repeated zero-padding leaks border-position information into interior activations, while aggressive early striding can alias fine textures before the network has extracted them.

### Worked example: Standard vs. Depthwise Separable Convolution Cost

Suppose a mobile vision layer transforms a $$28 \times 28$$ feature map from $$C_{\text{in}} = 64$$ channels to $$C_{\text{out}} = 128$$ channels using $$k = 3$$, stride $$s = 1$$, padding $$p = 1$$ (so spatial output remains $$28 \times 28 = 784$$ positions).

1. **Standard 2D convolution**:
   - Parameters (excluding bias):
     $$P_{\text{std}} = C_{\text{in}} \cdot C_{\text{out}} \cdot k^2 = 64 \times 128 \times 9 = 73{,}728.$$
   - Multiply-accumulate operations (MACs):
     $$\text{MACs}_{\text{std}} = 784 \times 73{,}728 = 57{,}802{,}752 \approx 57.8\text{M}.$$
2. **Depthwise separable convolution**:
   - **Stage 1 (Depthwise $$3 \times 3$$, `groups=64`)**: $$64$$ independent $$3 \times 3$$ filters:
     $$P_{\text{dw}} = 64 \times 9 = 576, \quad \text{MACs}_{\text{dw}} = 784 \times 576 = 451{,}584.$$
   - **Stage 2 (Pointwise $$1 \times 1$$)**: mixes $$64 \to 128$$ channels:
     $$P_{\text{pw}} = 64 \times 128 \times 1 = 8{,}192, \quad \text{MACs}_{\text{pw}} = 784 \times 8{,}192 = 6{,}422{,}528.$$
   - **Total**:
     $$P_{\text{sep}} = 576 + 8{,}192 = 8{,}768, \quad \text{MACs}_{\text{sep}} = 6{,}874{,}112 \approx 6.87\text{M}.$$
3. **Exact efficiency ratio**:
   $$\frac{P_{\text{sep}}}{P_{\text{std}}} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2} = \frac{1}{128} + \frac{1}{9} \approx 0.00781 + 0.11111 = 0.1189 \implies 8.41\times \text{ reduction!}$$

## 5. Related Concepts

### CNN vs Regular Neural Networks

![CNN vs Regular Neural Network](/deep-learning-self-learning/img/chapter_img/chapter04/conv7.jpg)
*Why not just use a regular neural network for images? CNNs win because they preserve spatial structure, use shared filters (far fewer parameters), recognize objects regardless of position, and are designed specifically for visual data processing. Source: Analytics Vidhya*

| Aspect | Fully Connected | Convolutional |
|--------|-----------------|---------------|
| Connectivity | All-to-all | Local (kernel size) |
| Parameter sharing | None | Same filter everywhere |
| Parameters (32×32×3 → 64) | 196,672 | 1,792 |
| Translation invariance | No | Yes |
| Preserves spatial structure | No | Yes |

### Fully Connected Layers
- Connect every input neuron to every output neuron
- No assumption about spatial structure
- Used after conv layers for final classification
- Parameters grow quadratically with input size

### Pooling Layers
- Downsample feature maps (reduce spatial dimensions)
- Add translation invariance
- Reduce computation and memory
- No learnable parameters
- Common types: Max pooling, Average pooling

### Batch Normalization
- Normalizes activations to have zero mean and unit variance
- Stabilizes training by reducing internal covariate shift
- Typically placed after convolution, before activation
- Enables higher learning rates and faster convergence

### Residual Connections (Skip Connections)
- Allow gradients to flow directly through the network
- Enable training of very deep networks (100+ layers)
- Output: $$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$
- Key innovation of ResNet (2015)

### Depthwise Separable Convolutions
- Factorize standard convolution into depthwise + pointwise
- Dramatically fewer parameters and computations
- Depthwise: One filter per input channel
- Pointwise: 1×1 conv to combine channels
- Used in MobileNet, EfficientNet for mobile deployment

### Dilated (Atrous) Convolutions
- Insert gaps between kernel elements
- Increase receptive field without more parameters
- Used in semantic segmentation (DeepLab)
- Dilation rate $$d$$: kernel elements spaced $$d$$ pixels apart

---

## 6. Fundamental Papers

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
Introduced LeNet-5, the first successful CNN for digit recognition. Demonstrated that convolutional layers with shared weights can learn hierarchical features from raw pixels. Established the conv-pool-conv-pool-FC pattern that dominated for nearly two decades. This paper laid the foundation for all modern CNNs.

**["ImageNet Classification with Deep Convolutional Neural Networks" (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)**  
*Authors*: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
AlexNet won ImageNet 2012 by a huge margin (top-5 error: 15.3% vs 26.2% runner-up), proving that deep CNNs could scale to real-world vision tasks. Introduced ReLU activation, dropout regularization, and GPU training to deep learning. This paper triggered the deep learning revolution that transformed AI.

**["Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015)](https://arxiv.org/abs/1409.1556)**  
*Authors*: Karen Simonyan, Andrew Zisserman  
VGGNet demonstrated that network depth is crucial for performance. Used exclusively small 3×3 filters stacked to achieve large receptive fields, showing that two 3×3 convs have the same receptive field as one 5×5 but with fewer parameters and more non-linearity. The simple, repeating architecture became a template for future designs.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Authors*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Introduced skip connections that enable training of networks with 152+ layers. Solved the degradation problem where deeper networks had higher training error than shallow ones. ResNet won ImageNet 2015 and fundamentally changed how we design deep networks. The residual learning framework is now used in transformers, diffusion models, and virtually all deep architectures.

**["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)](https://arxiv.org/abs/1704.04861)**  
*Authors*: Andrew G. Howard et al.  
Introduced depthwise separable convolutions that reduce computation by 8-9× while maintaining accuracy. Enabled running CNNs on mobile phones and embedded devices. Pioneered the efficient network design paradigm that led to EfficientNet and mobile-first AI deployment.

**["Rethinking the Inception Architecture for Computer Vision" (2016)](https://arxiv.org/abs/1512.00567)**  
*Authors*: Christian Szegedy et al.  
Inception-v3 introduced factorized convolutions (using 1×n and n×1 instead of n×n) and auxiliary classifiers. Demonstrated that careful architecture design could improve both accuracy and efficiency. Many design principles from this paper influence modern architectures.

---

## 7. Common Pitfalls and Tricks

### Pitfall 1: Not Using Padding

**Issue**: Output shrinks with each layer, losing boundary information

```python
# Without padding: 32×32 → 30×30 → 28×28 → ... (shrinks rapidly!)
conv1 = nn.Conv2d(3, 64, kernel_size=3)  # No padding
conv2 = nn.Conv2d(64, 128, kernel_size=3)

# With padding: 32×32 → 32×32 → 32×32 (maintains size)
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # "Same" padding
conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
```

**Solution**: Use `padding = (kernel_size - 1) // 2` for odd kernel sizes to maintain spatial dimensions.

### Pitfall 2: Forgetting About Channel Dimension Order

**Issue**: Different frameworks use different conventions

```python
# PyTorch: (N, C, H, W) - Channels first
x_pytorch = torch.randn(batch, channels, height, width)

# TensorFlow/Keras: (N, H, W, C) - Channels last
x_tensorflow = tf.random.normal([batch, height, width, channels])

# Converting between them
x_tf_to_pytorch = x_tensorflow.permute(0, 3, 1, 2)  # If using torch
```

### Pitfall 3: Too Large Initial Kernel Size

**Issue**: Large kernels have many parameters but limited receptive field benefit

```python
# Less effective (VGG's insight)
conv = nn.Conv2d(3, 64, kernel_size=7)  # 7×7 = 49 params/channel

# Better: Stack smaller kernels
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
# Three 3×3 = 27 params/channel, same 7×7 receptive field, 3× more nonlinearity
```

### Trick 1: 1×1 Convolutions for Dimensionality Control

```python
# Reduce channels (bottleneck)
bottleneck = nn.Conv2d(512, 64, kernel_size=1)
# 512 → 64 channels with minimal computation

# Increase channels
expand = nn.Conv2d(64, 512, kernel_size=1)
# Mix channel information without spatial operations
```

### Trick 2: Strided Convolution Instead of Pooling

```python
# Traditional: Conv → Pool
conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2, 2)

# Modern: Strided conv (learnable downsampling)
strided_conv = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# Same output size, but the downsampling is learned!
```

### Trick 3: Proper Weight Initialization

```python
# Kaiming initialization for ReLU networks
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

### Trick 4: Calculating Receptive Field

```python
def receptive_field_1d(layers):
    """
    Calculate receptive field for a stack of conv layers.
    
    Each layer is (kernel_size, stride)
    """
    rf = 1  # Start with single pixel
    stride_product = 1
    
    for k, s in layers:
        rf = rf + (k - 1) * stride_product
        stride_product *= s
    
    return rf

# Example: 3 conv layers (k=3, s=1) followed by pool (k=2, s=2)
layers = [(3, 1), (3, 1), (3, 1), (2, 2)]
print(f"Receptive field: {receptive_field_1d(layers)}")  # 14
```

---

## 8. Key Takeaways

1. **Convolution vs Cross-Correlation**: Deep learning uses cross-correlation but calls it convolution. The difference (kernel flipping) doesn't matter because weights are learned.

2. **Local Connectivity + Weight Sharing**: These two properties make CNNs dramatically more parameter-efficient than fully connected networks.

3. **Hierarchical Feature Learning**: CNNs automatically learn edges → textures → parts → objects, mirroring biological vision.

4. **Output Dimensions**: $$n_{out} = \lfloor(n_{in} + 2p - k)/s\rfloor + 1$$ — memorize this formula!

5. **Small Filters Rule**: Stack 3×3 convolutions instead of using larger kernels (more nonlinearity, fewer parameters).

6. **Receptive Field**: Grows with depth. Understanding receptive field is crucial for architecture design.

7. **Modern Best Practices**:
   - Conv → BatchNorm → ReLU
   - Residual connections for deep networks
   - Global average pooling instead of flatten + FC

Convolutional layers are the foundation of computer vision in deep learning. Mastering them is essential for any vision application — from image classification to object detection to image generation.


**Next**: Pooling layers and complete CNN architectures (LeNet, AlexNet, VGG, ResNet)!

## Video tham khảo

- **xem trước** [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, English. The pitfalls that show up as soon as channels and stride interact.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, English. A 2024 statement of which of those pitfalls still matter.
- [C4W2L04 Why ResNets Work](https://www.youtube.com/watch?v=RYth6EbBUqM) — DeepLearningAI, English. Why a plain stack of convolutions degrades, the paper-level pitfall the ResNet note is about.
