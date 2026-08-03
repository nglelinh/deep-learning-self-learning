---
layout: post
title: 04-01-01 Convolution Math and Dimensions
chapter: '04'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

# Convolutional Layers: Building Blocks of Computer Vision

![CNN](/deep-learning-self-learning/img/chapter_img/chapter04/conv1.jpg)
*Overview of Convolutional Neural Network (CNN) architecture. Source: AnalyticsVidhya*

## 1. Concept Overview

**Convolutional layers** are specialized neural network layers designed for processing grid-like data, especially images. Instead of connecting every input to every neuron (fully connected), convolutional layers use small, learnable filters that slide across the input to detect local patterns.

**Why CNNs matter**:
- **Parameter efficiency**: Millions fewer parameters than fully connected networks
- **Translation invariance**: Detects features regardless of their position in the image
- **Hierarchical learning**: Learns low-level → mid-level → high-level features automatically
- **State-of-the-art**: Best performance on virtually all vision tasks

**Key insight**: Images have spatial structure — nearby pixels are related. Convolutional layers exploit this structure through **local connectivity** and **parameter sharing**.

**Analogy**: Think of convolution as sliding a magnifying glass (filter) across an image to find specific patterns (edges, textures, shapes). Each filter specializes in detecting one type of pattern, and you use the same magnifying glass everywhere on the image rather than having a different one for each location.

![CNN](/deep-learning-self-learning/img/chapter_img/chapter04/conv2.jpg)
*Overview of Convolutional Neural Network (CNN) architecture. Source: AnalyticsVidhya*

### The Biological Inspiration

![CNN Inspiration from Human Eye](/deep-learning-self-learning/img/chapter_img/chapter04/conv3.jpg)
*CNNs were directly inspired by how the human brain processes visual information. Your eye detects edges first, then combines them into shapes, then matches patterns to memory. A CNN does the exact same thing. Source: Analytics Vidhya*

In 1959, neurophysiologists David Hubel and Torsten Wiesel discovered that neurons in the visual cortex respond to specific patterns in localized regions of the visual field (called "receptive fields"). Some neurons respond to edges at certain orientations, others to motion in particular directions. This hierarchical processing — from simple features to complex objects — directly inspired the design of convolutional neural networks.

---

## 2. Mathematical Foundation

### Understanding Convolution vs Cross-Correlation

This is a critical distinction that often confuses newcomers. Let's carefully understand both operations.

![Convolution Operation](https://miro.medium.com/v2/resize:fit:1400/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)
*Illustration of 2D convolution operation on an image. Source: Medium*

![The Convolution Filter Sliding](/deep-learning-self-learning/img/chapter_img/chapter04/conv4.jpg)
*A small window called a filter slides across the image, scanning every patch. At each position it asks: "Is this feature here?" Different filters detect different patterns — edges, blur, sharpness. A CNN learns hundreds of these filters automatically during training. Source: Analytics Vidhya*

#### True Convolution (Signal Processing Definition)

In signal processing and mathematics, **true convolution** involves **flipping the kernel** (rotating 180°) before sliding it across the input. For a 2D image $$I$$ and kernel $$K$$:

$$(I * K)[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[m, n] \cdot K[i-m, j-n]$$

For finite, discrete signals with a kernel of size $$k \times k$$:

$$(I * K)[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i-m, j-n] \cdot K[m,n]$$

The key property here is the **subtraction** in the indices ($$i-m, j-n$$), which effectively flips the kernel.

**Why does convolution flip the kernel?** Convolution was originally designed to describe how systems respond to inputs over time. The flipping ensures that convolution is:
1. **Commutative**: $$I * K = K * I$$
2. **Associative**: $$(I * K_1) * K_2 = I * (K_1 * K_2)$$

These properties are essential in signal processing for analyzing linear time-invariant systems.

#### Cross-Correlation (What Deep Learning Actually Uses)

**Cross-correlation** is similar but **without the kernel flip**:

$$S[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n] \cdot K[m,n]$$

Notice the **addition** in the indices ($$i+m, j+n$$) — we simply slide the kernel as-is across the image.

#### Visual Comparison: Convolution vs Cross-Correlation

Let's make this concrete with an example:

**Kernel** $$K$$:
$$K = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

**Flipped kernel** (for true convolution):
$$K_{flipped} = \begin{bmatrix} 9 & 8 & 7 \\ 6 & 5 & 4 \\ 3 & 2 & 1 \end{bmatrix}$$

**For a symmetric kernel** (like many edge detectors):
$$K = \begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$$

Flipping this kernel gives:
$$K_{flipped} = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$$

Notice that the flipped version detects edges in the **opposite direction**!

#### Why Deep Learning Uses Cross-Correlation (and Calls it "Convolution")

Here's the crucial insight: **In deep learning, we don't care about the flip because the kernel weights are learned**.

Consider this reasoning:

1. **If we used true convolution**: The network would learn weights $$W$$, and apply them flipped during the forward pass
2. **If we use cross-correlation**: The network learns weights $$W'$$, applied without flipping

Since the network learns the weights from data anyway, $$W'$$ simply learns to be the flipped version of what $$W$$ would have been. **The network will learn the same function either way**.

**Benefits of using cross-correlation**:
- **Simpler implementation**: No need to flip the kernel
- **More intuitive**: The kernel "template" directly matches the pattern it detects
- **Same learning capacity**: The network can learn any function regardless

**Historical note**: The deep learning community adopted the term "convolution" even though we technically use cross-correlation. This is now standard terminology, but understanding the distinction helps when reading signal processing literature or implementing custom operations.

### Step-by-Step Convolution Example

Let's trace through a complete example to solidify understanding:

**Input image** $$I$$ (5×5):
$$I = \begin{bmatrix} 
1 & 2 & 3 & 0 & 1 \\
0 & 1 & 2 & 3 & 1 \\
1 & 2 & 1 & 0 & 0 \\
0 & 1 & 2 & 3 & 2 \\
2 & 1 & 0 & 1 & 1
\end{bmatrix}$$

**Kernel** $$K$$ (3×3):
$$K = \begin{bmatrix} 
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}$$

**Computing output at position (0,0)** using cross-correlation:

We extract the 3×3 region starting at (0,0):
$$\text{region} = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

Element-wise multiply and sum:
$$\begin{align}
S[0,0] &= (1 \times 1) + (2 \times 0) + (3 \times -1) \\
&+ (0 \times 1) + (1 \times 0) + (2 \times -1) \\
&+ (1 \times 1) + (2 \times 0) + (1 \times -1) \\
&= 1 + 0 - 3 + 0 + 0 - 2 + 1 + 0 - 1 \\
&= -4
\end{align}$$

**Computing output at position (0,1)**:

Region starting at (0,1):
$$\text{region} = \begin{bmatrix} 2 & 3 & 0 \\ 1 & 2 & 3 \\ 2 & 1 & 0 \end{bmatrix}$$

$$\begin{align}
S[0,1] &= (2 \times 1) + (3 \times 0) + (0 \times -1) \\
&+ (1 \times 1) + (2 \times 0) + (3 \times -1) \\
&+ (2 \times 1) + (1 \times 0) + (0 \times -1) \\
&= 2 + 0 + 0 + 1 + 0 - 3 + 2 + 0 + 0 \\
&= 2
\end{align}$$

Continue this process for all valid positions to get the complete output.

### Multi-Channel Convolution

Real images have multiple channels (RGB has 3, intermediate CNN layers can have 64, 128, 256, or more). Here's how convolution extends:

**For an input with $$C_{in}$$ channels** (e.g., $$H \times W \times C_{in}$$) and a **filter of size $$k \times k \times C_{in}$$**:

$$S[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n, c] \cdot K[m,n,c] + b$$

Where:
- $$C_{in}$$: number of input channels (3 for RGB)
- $$b$$: bias term (one scalar per filter)
- Each filter produces **one** output channel

**To produce $$C_{out}$$ output channels**, we need $$C_{out}$$ filters, each of size $$k \times k \times C_{in}$$.

**Visual intuition**: For an RGB image, each filter is actually a 3D block (e.g., 3×3×3). The filter has separate weights for each color channel, and they're all summed together to produce one output value. This allows the filter to learn color-specific patterns (e.g., detecting blue sky vs. green grass).

### Output Dimensions Formula

Understanding how input dimensions transform through convolutional layers is essential for designing architectures.

**Given**:
- Input size: $$H_{in} \times W_{in}$$
- Kernel size: $$k \times k$$
- Stride: $$s$$ (how many pixels to move between positions)
- Padding: $$p$$ (zeros added around the border)

**Output size**:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W_{in} + 2p - k}{s} \right\rfloor + 1$$

**Common configurations**:

| Config | Purpose | Formula Result |
|--------|---------|----------------|
| $$k=3, s=1, p=1$$ | "Same" padding | $$n_{out} = n_{in}$$ |
| $$k=3, s=2, p=1$$ | Downsample by 2 | $$n_{out} = \lceil n_{in}/2 \rceil$$ |
| $$k=1, s=1, p=0$$ | 1×1 convolution | $$n_{out} = n_{in}$$ |
| $$k=7, s=2, p=3$$ | Aggressive downsample | $$n_{out} = \lceil n_{in}/2 \rceil$$ |

### Parameter Count Analysis

For a convolutional layer:

$$\text{Parameters} = (k \times k \times C_{in}) \times C_{out} + C_{out}$$

The first term is the weights (each filter has $$k \times k \times C_{in}$$ weights, and we have $$C_{out}$$ filters). The second term is the biases (one per output channel).

**Example calculation**:
- Input: $$32 \times 32 \times 3$$ (CIFAR-10 RGB image)
- Filter: $$3 \times 3$$, 64 filters
- Parameters: $$(3 \times 3 \times 3) \times 64 + 64 = 27 \times 64 + 64 = 1,792$$

**Compare to fully connected layer**:
- Input: $$32 \times 32 \times 3 = 3,072$$ neurons
- Output: 64 neurons
- Parameters: $$3,072 \times 64 + 64 = 196,672$$

**Reduction factor**: $$196,672 / 1,792 \approx 110\times$$ fewer parameters!

This massive reduction comes from two properties:
1. **Local connectivity**: Each output only connects to a small region of input
2. **Parameter sharing**: Same filter weights used at all spatial locations

---
