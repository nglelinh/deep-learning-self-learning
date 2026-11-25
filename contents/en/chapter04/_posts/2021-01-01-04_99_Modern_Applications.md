---
layout: post
title: 04-99 Modern Applications and Updates (2022–2026)
chapter: '04'
order: 14
owner: Deep Learning Course
lang: en
categories:
- chapter04
lesson_type: optional
---

When Vision Transformers (ViT) arrived in 2020, many predicted the immediate extinction of Convolutional Neural Networks. Yet by 2022, ConvNeXt stunned the field by showing that a pure CNN—modernized with larger $$7 \times 7$$ depthwise kernels, inverted bottlenecks, LayerNorm, GELU, and AdamW training recipes—could match or surpass hierarchical Vision Transformers like Swin on ImageNet, COCO, and ADE20K. Today's frontier vision systems (from ConvNeXt V2 and SAM-style hybrids to edge robotics models) reveal a deeper truth: convolution and attention are complementary operators, not rivals.

## Key ideas: Modernizing CNNs in the Transformer Era

- **Training Recipe vs. Architectural Inductive Bias**: Much of ViT's early advantage came from modern optimization (AdamW, cosine decay, Mixup, CutMix, RandAugment, stochastic depth) rather than self-attention alone; porting those recipes back to ResNet/ConvNeXt closed the gap dramatically.
- **Macro & Micro Modernizations (ConvNeXt Blueprint)**: Replacing $$3 \times 3$$ standard convs with $$7 \times 7$$ depthwise convs expands receptive fields cheaply, while inverted bottlenecks (expand $$4\times$$ then project back) mirror Transformer MLP blocks.
- **Hybrid Convolution–Attention Backbones**: Modern vision foundation models routinely use convolutional stems (fast, inductive-bias-rich local feature extraction at high resolution) feeding global Transformer blocks at low resolution—avoiding the $$O(N^2)$$ cost of self-attention on raw pixels.
- **Edge Deployment & Hardware Determinism**: On mobile NPUs, autonomous vehicles, and high-resolution medical imaging, CNNs still win on latency predictability, memory bandwidth, and small-data sample efficiency.

### Worked example: Why a $$7 \times 7$$ Depthwise Conv Beats a $$3 \times 3$$ Standard Conv

Consider a modern ConvNeXt block operating on $$C = 96$$ channels at resolution $$56 \times 56$$:

1. **Classic ResNet $$3 \times 3$$ standard conv ($$96 \to 96$$)**:
   - Parameters: $$96 \times 96 \times 3 \times 3 = 82{,}944$$.
   - Receptive field of this single layer: $$3 \times 3$$ (9 pixels).
2. **ConvNeXt $$7 \times 7$$ depthwise conv (`groups=96`)**:
   - Parameters: $$96 \times 1 \times 7 \times 7 = 4{,}704$$.
   - Receptive field of this single layer: $$7 \times 7$$ (49 pixels—over $$5.4\times$$ larger area!).
3. **Compare parameter cost**:
   $$\frac{82{,}944}{4{,}704} = 17.63\times \text{ fewer parameters}$$
   for the $$7 \times 7$$ depthwise spatial mixer! The saved parameter budget is then invested into the inverted pointwise MLP ($$96 \to 384 \to 96$$: $$2 \times 96 \times 384 = 73{,}728$$ weights), separating spatial mixing from channel mixing just like a Transformer block.

# Optional: ConvNets after ConvNeXt and vision foundation models

> This lesson is **optional**. It does **not** rewrite convolution math, pooling, or the LeNet–ResNet history. It updates what practitioners actually deploy for vision after 2022.

The convolution $$(\mathbf{x} * \mathbf{k})_{i} = \sum_{u} x_{i+u}\,k_{u}$$ is unchanged. What changed is the **macro architecture** (ConvNeXt) and the **use case** (frozen convolutional or hybrid backbones inside foundation models).

## 1. ConvNeXt: a ConvNet that borrowed Transformer training

[Liu et al., 2022](https://arxiv.org/abs/2201.03545) (“A ConvNet for the 2020s”) showed that a ResNet-style stack, trained with ViT recipes (AdamW, Mixup, stochastic depth, large crops), matches hierarchical Transformers on ImageNet. [ConvNeXt V2](https://arxiv.org/abs/2301.00808) (2023) adds global response normalization (GRN) and a fully convolutional masked autoencoder pretraining stage.

The message for this chapter: **depthwise convolution + inverted bottlenecks** are still the inductive bias; the 2020s gap versus ViT was mostly *training*, not “convolution is obsolete.”

## 2. Concrete applications

### Vision foundation models with convolutional DNA

- **DINOv2** ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)) — self-supervised ViT features used as a drop-in backbone for retrieval, depth, and segmentation. ConvNeXt variants appear in the same benchmark tables.
- **SAM** ([Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)) and **SAM 2** ([Ravi et al., 2024](https://arxiv.org/abs/2408.00714)) — promptable segmentation; the image encoder is a ViT, but production pipelines still pair it with convolutional detectors and lightweight CNN necks.
- Medical imaging and remote sensing still prefer U-Net-style **convolutional decoders** on top of whatever encoder won last year.

### When a CNN is still the right default

High-resolution dense prediction, strict latency on mobile NPUs, and small-data regimes often keep EfficientNetV2 / ConvNeXt-Tiny / YOLO convolutional necks. Transformers win when you can pretrain at web scale.

## 3. Widely used software

- [timm](https://github.com/huggingface/pytorch-image-models) — ConvNeXt, EfficientNetV2, pretrained recipes.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — convolutional detectors (YOLOv8/v11) used in industry.
- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) and [facebookresearch/sam2](https://github.com/facebookresearch/sam2).

## 4. Citations (2022–2026)

- [A ConvNet for the 2020s (Liu et al., 2022)](https://arxiv.org/abs/2201.03545) — ConvNeXt.
- [ConvNeXt V2 (Woo et al., 2023)](https://arxiv.org/abs/2301.00808) — GRN + FCMAE pretraining.
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) — web-scale self-supervised visual features.
- [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643) — promptable segmentation foundation model.

## 5. How this complements the core notes

Keep the convolution-dimension worksheets. This lesson only answers “what replaced VGG/ResNet *in practice* after 2022”: modern ConvNets plus vision foundation models — not a new definition of a filter.

## Video tham khảo

- **xem trước** [ConvNeXt: A ConvNet for the 2020s – Paper Explained (with animations)](https://www.youtube.com/watch?v=QqejV0LNDHA) — AI Coffee Break with Letitia, English. ConvNeXt is the conv net that borrowed transformer training, which is this note's claim.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, English. The modern conv-net lecture those training tricks sit on top of.
- [Residual Networks (ResNet) \[Physics Informed Machine Learning\]](https://www.youtube.com/watch?v=w1UsKanMatM) — Steve Brunton, English. The residual block ConvNeXt did not throw away.
