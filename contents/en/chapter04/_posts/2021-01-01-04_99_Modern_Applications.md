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
