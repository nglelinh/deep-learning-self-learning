---
layout: post
title: 04-02-02 Classic CNN Architectures
chapter: '04'
order: 11
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

Reading the history of CNN architectures from 1998 to 2015 is like watching aviation evolve from the Wright Flyer to a commercial jetliner in under two decades. Each landmark paper—LeNet-5, AlexNet, VGGNet, and ResNet—did not merely add more layers; each diagnosed a concrete mathematical bottleneck (saturating activations, bloated FC heads, oversized kernels, or optimization degradation) and invented a reusable architectural principle that still shapes Vision Transformers and ConvNeXt today.

## Background Knowledge Map: The Evolutionary Arc of Landmark CNNs

- **LeNet-5 (1998)**: Established the canonical `Conv -> Pool -> Conv -> Pool -> FC` blueprint and proved end-to-end backpropagation on raw pixels could beat hand-crafted digit features.
- **AlexNet (2012)**: Proved GPU-accelerated deep CNNs with ReLU, Dropout, and data augmentation could scale to million-image datasets (ImageNet).
- **VGGNet (2014)**: Replaced heterogeneous kernel sizes with clean, modular stacks of $$3 \times 3$$ filters, proving depth and architectural regularity drive representation power.
- **ResNet (2015)**: Introduced identity skip connections ($$x + F(x)$$) and $$1\times 1$$ bottlenecks, solving deep optimization degradation and enabling 50–152+ layer networks.

### Step-by-Step Worked Example (Read Slowly): Efficiency Leap Across Four Generations

Compare how efficiently each landmark architecture turns parameters into ImageNet accuracy (approximate canonical published figures):

1. **Parameter count comparison**:
   - **AlexNet (8 layers)**: $$\approx 61.1\text{M}$$ parameters (~$$96\%$$ in FC layers).
   - **VGG-16 (16 layers)**: $$\approx 138.4\text{M}$$ parameters (~$$89\%$$ in FC layers: `25088 x 4096` alone is $$102.8\text{M}$$!).
   - **ResNet-50 (50 layers)**: $$\approx 25.6\text{M}$$ parameters (Global Average Pooling reduces the classifier head to just `2048 x 1000 = 2.05M`).
2. **Parameter ratio**:
   $$\frac{P_{\text{VGG-16}}}{P_{\text{ResNet-50}}} = \frac{138.4\text{M}}{25.6\text{M}} \approx 5.4\times \text{ fewer parameters for ResNet-50}$$
   despite being more than $$3\times$$ deeper and achieving substantially higher top-1 accuracy.
3. **Design Rule**: Depth + residual connections + Global Average Pooling beats shallow width + dense classification heads on both memory and generalization.

Classic CNN architectures:

1. **LeNet and AlexNet** — early pioneers of modern computer vision
2. **VGG, ResNet, and Architecture Evolution** — depth, skip connections, papers, takeaways

## Video tham khảo

- **xem trước** [Lecture 9 | CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y) — Stanford University School of Engineering, English. One lecture that lines LeNet, AlexNet, VGG, and ResNet up.
- [Lecture 1 | Introduction to Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/watch?v=vT1JzLTH4G4) — Stanford University School of Engineering, English. The historical motivation: why ImageNet forced architectures to change.
- [Residual Networks (ResNet) \[Physics Informed Machine Learning\]](https://www.youtube.com/watch?v=w1UsKanMatM) — Steve Brunton, English. The residual step that ends the sequence.
