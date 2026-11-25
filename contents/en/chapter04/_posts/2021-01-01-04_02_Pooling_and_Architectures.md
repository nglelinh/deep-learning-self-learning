---
layout: post
title: 04-02 Pooling Layers and CNN Architectures
chapter: '04'
order: 9
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

A single convolutional layer extracts local textures, but recognizing a whole scene requires coordinating spatial downsampling with multi-stage feature hierarchies. This section pairs **pooling layers**—which compress resolution and introduce local invariance—with the **classic CNN architectures** (LeNet, AlexNet, VGG, ResNet) that assembled convolution, pooling, normalization, and residual paths into complete vision systems.

## Key ideas: From Single Layers to Full Vision Backbones

- **Spatial Pyramids & Channel-Resolution Trade-off**: Classic CNNs follow a disciplined pyramid rule: whenever pooling (or strided conv) halves $$H$$ and $$W$$ (cutting spatial area by $$4\times$$), the next stage doubles the channel count $$C$$, keeping per-layer FLOPs roughly constant across depth.
- **Pooling Semantics (Max, Average, Global Average)**: Local pooling builds shift tolerance across early stages, while Global Average Pooling at the end converts arbitrary $$(C, H, W)$$ grids into compact $$C$$-dimensional semantic embeddings.
- **Macro-Architectural Motifs**: Understanding how LeNet, AlexNet, VGG, and ResNet wire pooling, bottlenecks, and skip connections equips you to read—and design—any modern vision backbone.

### Worked example: Why Doubling Channels When Halving Resolution Keeps FLOPs Constant

Suppose Stage 1 of a CNN operates on feature maps of shape $$(C, H, W) = (64, 56, 56)$$ with a $$3 \times 3$$ conv ($$64 \to 64$$), and Stage 2 follows a $$2 \times 2$$ stride-2 pooling layer with $$(128, 28, 28)$$ and a $$3 \times 3$$ conv ($$128 \to 128$$).

1. **Stage 1 MACs**:
   $$\text{MACs}_1 = H_1 \cdot W_1 \cdot C_1^2 \cdot k^2 = 56 \times 56 \times 64^2 \times 9 = 3{,}136 \times 4{,}096 \times 9 = 115{,}605{,}504.$$
2. **Stage 2 MACs** ($$H_2 = H_1/2, W_2 = W_1/2, C_2 = 2 C_1$$):
   $$\text{MACs}_2 = 28 \times 28 \times 128^2 \times 9 = 784 \times 16{,}384 \times 9 = 115{,}605{,}504.$$
3. **Exact Ratio**:
   $$\frac{\text{MACs}_2}{\text{MACs}_1} = \frac{(H/2)(W/2)(2C)^2}{H W C^2} = \frac{\frac{1}{4} \cdot 4}{1} = 1.00!$$
   That exact algebraic cancellation is the hidden organizing law behind VGG, ResNet, and nearly every hierarchical vision backbone.

This section completes the CNN toolkit: **pooling** for spatial downsampling, and **classic architectures** that show how convolutions and pooling are composed in practice.

1. **Pooling Layers** — max/average/global pooling, math, and implementations
2. **Classic CNN Architectures** — LeNet, AlexNet, VGG, ResNet, papers, and key takeaways

Read the pooling lesson first, then the architecture survey.


## Video tham khảo

- **xem trước** [Max Pooling in Convolutional Neural Networks explained](https://www.youtube.com/watch?v=ZjM_XQa5s6s) — deeplizard, English. Max pooling is the downsampling half of this block.
- [Lecture 9 | CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y) — Stanford University School of Engineering, English. The architecture half: LeNet through ResNet.
- [Residual Networks (ResNet) \[Physics Informed Machine Learning\]](https://www.youtube.com/watch?v=w1UsKanMatM) — Steve Brunton, English. ResNet's skip connection, the last architecture the block reaches.
