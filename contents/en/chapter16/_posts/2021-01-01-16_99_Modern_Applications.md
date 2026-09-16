---
layout: post
title: 16-99 Modern Applications and Updates (2022–2026)
chapter: '16'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter16
lesson_type: optional
---

# Optional: self-supervised learning after SimCLR — DINOv2, SigLIP, JEPA

> This lesson is **optional**. It does **not** replace contrastive / MLM theory or the SimCLR sketch already in the chapter. It updates the **vision and multimodal** SSL methods that became defaults after 2022.

The chapter already covers SimCLR, CLIP, and MAE. The 2023–2025 production stack moved to **stronger teachers**, **sigmoid language–image losses**, and **predict-in-latent-space** (JEPA) objectives.

## 1. DINOv2: a frozen backbone that just works

[Oquab et al., 2023](https://arxiv.org/abs/2304.07193) combine a DINO/iBOT-style student–teacher with a huge curated data pipeline. The released ViT-g features transfer to retrieval, depth, and segmentation **without** task-specific fine-tune in many demos. That is SSL as a **foundation encoder**, not a pretext homework.

## 2. Concrete applications

### SigLIP / SigLIP 2

[Zhai et al., 2023](https://arxiv.org/abs/2303.15343) replace CLIP’s softmax (which needs a large batch of negatives) with a **sigmoid** pairwise loss

$$\mathcal{L} = -\frac{1}{B}\sum_{i,j}\log\sigma\big(y_{ij}\, t\, x_i^\top y_j\big),$$

where $$y_{ij}=\pm 1$$ marks matched pairs. Open-VLM stacks (PaliGemma, many 2024–2025 VLMs) start from SigLIP vision towers.

### I-JEPA and predict-in-representation

[Assran et al., 2023](https://arxiv.org/abs/2301.08243) (I-JEPA) predict masked patch **embeddings**, not pixels (unlike MAE in Chapter 12’s optional note). [LeCun’s JEPA](https://openreview.net/forum?id=BZ5a1r-kVsf) line is the research bet that SSL should model latent dynamics, not decode RGB.

### What stayed

Masked language modeling and CLIP-style pairing are still the NLP/multimodal workhorses. SimCLR remains the right first contrastive derivation.

## 3. Widely used software

- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
- [google-research/big_vision](https://github.com/google-research/big_vision) / OpenCLIP SigLIP weights on the Hub.
- [facebookresearch/jepa](https://github.com/facebookresearch/jepa).

## 4. Citations (2022–2026)

- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193).
- [Sigmoid Loss for Language Image Pre-Training — SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343).
- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture — I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243).

## 5. How this complements the core notes

Keep the NT-Xent derivation and the BERT MLM formula. This lesson only names the **2023–2025 checkpoints** (DINOv2, SigLIP, JEPA) you will load instead of training SimCLR from scratch.
