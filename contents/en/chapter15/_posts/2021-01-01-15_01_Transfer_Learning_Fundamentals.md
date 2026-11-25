---
layout: post
title: 15-01 Transfer Learning Fundamentals
chapter: '15'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter15
---

Transfer learning is the quiet superpower behind almost every real-world deep learning deployment: almost nobody trains a vision or language backbone from scratch when a strong pretrained checkpoint exists. Pairing the theory of domain shift with concrete PyTorch freezing and fine-tuning recipes gives you a reliable decision matrix for any dataset size and domain distance.

## Key ideas: Transfer Learning Decision Matrix

- **Pillar 1 — The 2×2 Data-vs-Similarity Grid**: Maps target dataset size (small vs. large) and domain similarity (close vs. distant) directly to linear probing, partial unfreezing, or full fine-tuning.
- **Pillar 2 — Representation Depth & Generality**: Connects shallow universal features to deep task-specific features so you know *which* layers to freeze.
- **Pillar 3 — Optimization Hygiene**: Uses LP-FT warm-up, differential learning rates, and weight decay to avoid catastrophic forgetting.
- **Pillar 4 — Empirical Validation**: Benchmarks feature extraction against full fine-tuning on identical data splits to measure true generalization gain.

### Worked example: Memory Savings of Freezing 80% of a Backbone

Consider a $20\text{M}$-parameter model trained with Adam in FP32 ($4$ bytes/param for weights, $4$ for gradients, $8$ for Adam's first and second moments $= 16$ bytes per trainable parameter).

1. **Full Fine-Tuning optimizer + weight + grad memory**:
   $$20\text{M} \times 16\text{ bytes} = 320\text{ MB}.$$
2. **Freezing 80% of parameters ($16\text{M}$ frozen, $4\text{M}$ trainable)**:
   - Frozen weights need only $4$ bytes/param (no gradients, no Adam states): $16\text{M} \times 4 = 64\text{ MB}$.
   - Trainable weights need $16$ bytes/param: $4\text{M} \times 16 = 64\text{ MB}$.
   - Total: $128\text{ MB}$ ($60\%$ memory reduction, plus faster backward passes since frozen layers skip weight-gradient computation!).
3. **Takeaway**: Freezing lower layers is simultaneously a statistical regularizer *and* a major GPU memory and compute accelerator.

Transfer learning:

1. **Theory** — concepts, strategies, intuition
2. **Implementation** — feature extraction vs fine-tuning, papers, pitfalls

## Video tham khảo

- **xem trước** [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, English. Feature extraction versus fine-tuning.
- [Pytorch Transfer Learning and Fine Tuning Tutorial](https://www.youtube.com/watch?v=qaDe0qQZ5AQ) — Aladdin Persson, English. Both strategies in code.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, English. The early visual features those strategies reuse.
