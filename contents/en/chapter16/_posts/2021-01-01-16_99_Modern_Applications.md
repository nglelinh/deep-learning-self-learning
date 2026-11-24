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

If SimCLR proved self-supervised vision was possible, the 2023–2026 wave—DINOv2, SigLIP, and I-JEPA—made it the default vision backbone of modern AI. Today you rarely train a vision encoder from scratch: you drop in a frozen DINOv2 ViT for dense geometry (depth, correspondence, segmentation) or a SigLIP encoder for vision-language alignment, and build directly on top of representations learned without a single ImageNet label.

## Background Knowledge Map: Frontier Self-Supervised Learning (2022–2026)

- **Pillar 1 — Self-Distillation Without Negatives (DINOv2)**: Trains a student ViT to match an exponential-moving-average (EMA) teacher across global and local crops plus masked patches (iBOT), yielding emergent object-part segmentation with zero labels.
- **Pillar 2 — Pairwise Sigmoid Contrastive Alignment (SigLIP)**: Replaces CLIP's global batch softmax with independent pairwise binary sigmoid losses, decoupling performance from massive synchronized batch sizes across GPUs.
- **Pillar 3 — Joint-Embedding Predictive Architectures (I-JEPA / V-JEPA)**: Predicts masked target *embeddings* rather than raw pixels, avoiding wasted capacity on unpredictable fine-grained noise like waving leaves.
- **Pillar 4 — Frozen Universal Backbones**: Uses curated, deduplicated self-supervised pretraining so a single frozen backbone powers depth estimation, robotics, and multimodal LLMs simultaneously.

### Step-by-Step Worked Example (Read Slowly): CLIP Softmax vs. SigLIP Pairwise Sigmoid Loss on a $2 \times 2$ Image-Text Batch

Suppose we have 2 image-text pairs with cosine similarity matrix $S = \begin{bmatrix}0.8 & 0.2\\ 0.1 & 0.7\end{bmatrix}$, temperature scale $t = 10$ (so $tS = \begin{bmatrix}8 & 2\\ 1 & 7\end{bmatrix}$), and SigLIP learnable bias $b = -5$.

1. **SigLIP pairwise logits $z_{ij} = t S_{ij} + b$**:
   $$Z = \begin{bmatrix}3 & -3\\ -4 & 2\end{bmatrix}, \qquad \text{labels } Y = \begin{bmatrix}+1 & -1\\ -1 & +1\end{bmatrix}.$$
2. **SigLIP binary logistic loss $\ell_{ij} = \ln(1 + e^{-y_{ij} z_{ij}})$**:
   - Positive $(1,1)$: $\ln(1 + e^{-3}) = \ln(1.0498) = 0.0486$.
   - Negative $(1,2)$: $\ln(1 + e^{-3}) = 0.0486$.
   - Negative $(2,1)$: $\ln(1 + e^{-4}) = \ln(1.0183) = 0.0181$.
   - Positive $(2,2)$: $\ln(1 + e^{-2}) = \ln(1.1353) = 0.1269$.
   - Mean loss per pair: $\frac{0.0486 + 0.0486 + 0.0181 + 0.1269}{4} = 0.0606$.
3. **Takeaway**: Every entry $(i, j)$ in SigLIP computes its own independent sigmoid loss without an all-reduce softmax denominator across GPUs—enabling clean scaling from small batches up to 32k+ pairs.

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

## Video tham khảo

- **xem trước** [DINOv2 from Meta AI - Finally a Foundational Model in Computer Vision?](https://www.youtube.com/watch?v=csEgtSh7jV4) — AI Papers Academy, English. DINOv2 is the frozen vision backbone this note says you can use without fine-tuning everything.
- [DINO: Emerging Properties in Self-Supervised Vision Transformers (Facebook AI Research Explained)](https://www.youtube.com/watch?v=h3ij3F3cPIk) — Yannic Kilcher, English. DINO, the predecessor the note's method extends.
- [Contrastive Learning with SimCLR | Deep Learning Animated](https://www.youtube.com/watch?v=UqJauYELn6c) — Deepia, English. The contrastive objective those self-supervised vision models grew out of.
