---
layout: post
title: 15-98 Interviews Practice (feature extraction and transfer)
chapter: '15'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter15
lesson_type: optional
---

In senior ML engineering interviews, transfer learning questions rarely ask you to recite definitions—they drop you into a messy production scenario: *"Your linear probe hits 78% on satellite imagery, full fine-tuning overfits to 71%, and your batch size is 4. What do you do?"* Practicing crisp, first-principles answers to these four diagnostic prompts trains you to reason about feature geometry, BatchNorm statistics, and domain shift under pressure.

## Background Knowledge Map: Interview Reasoning on Transfer & Feature Extraction

- **Pillar 1 — Bias–Variance Trade-off in Adaptation**: Diagnoses when frozen features underfit (high bias under domain shift) versus when full fine-tuning overfits (high variance on scarce labels).
- **Pillar 2 — Layer Selection & Feature Extraction Mechanics**: Distinguishes penultimate pooled embeddings from intermediate spatial feature maps depending on whether the task needs semantic class identity or fine-grained localization.
- **Pillar 3 — Domain Shift Diagnostics**: Separates covariate shift (input style/lighting changes), label shift (class priors change), and concept drift (label definitions change).
- **Pillar 4 — Failure Modes in Low-Data Regimes**: Anticipates BatchNorm instability on small batches, head-initialization gradient shock (fixed by LP-FT), and negative transfer.

### Step-by-Step Worked Example (Read Slowly): Diagnosing Feature-Distribution Drift via Cosine Similarity

Suppose a frozen ImageNet encoder maps daytime cat photos to average unit embedding $\bar{u}_S = [0.8,\, 0.6,\, 0.0]^\top$, and infrared thermal cat photos to $\bar{u}_T = [0.2,\, 0.4,\, 0.8944]^\top$ (verify: $0.2^2 + 0.4^2 + 0.8944^2 = 0.04 + 0.16 + 0.80 = 1.00$).

1. **Compute cosine similarity between source and target centroids**:
   $$\cos(\theta) = \bar{u}_S^\top \bar{u}_T = (0.8)(0.2) + (0.6)(0.4) + (0.0)(0.8944) = 0.16 + 0.24 = 0.40 \implies \theta \approx 66.4^\circ.$$
2. **Interpret for an interview decision**:
   A $66.4^\circ$ shift into an orthogonal subspace ($z_3 = 0.8944$) means a frozen linear probe trained on source RGB features will misalign severely. The disciplined engineering response is: **unfreeze at least the top 1–2 residual stages (or add LoRA adapters) and recalibrate BatchNorm/LayerNorm statistics on target data** rather than relying solely on a frozen linear probe.
3. **Takeaway**: Quantifying representation alignment turns hand-wavy interview intuition into an actionable architectural diagnosis.

# Optional: interview practice — feature extraction and transfer

> This lesson is **optional**. It does **not** replace feature-extraction vs fine-tune theory. After those notes, use this drill, then open *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) for solved items on CNN feature extraction. Architecture geometry stays in **04-98**.

**Book themes for this chapter (study in the PDF, not here):** using a trained CNN as a feature extractor; frozen vs trainable trunks.

Hub: **01-98 Deep Learning Interviews practice track**.

## Practice prompts (original)

### Q1. Linear probe vs last-block fine-tune

You have 800 labeled medical images and a ResNet pretrained on ImageNet. Compare (a) freeze the trunk, train a linear head and (b) unfreeze the last residual stage plus the head. What risk does (b) add that (a) does not?

**Hint.** Number of trainable parameters vs how close the source and target statistics are.

**Discussion.** (a) is a linear probe: cheap, hard to destroy pretrained filters, often a strong baseline. (b) can adapt mid/high-level filters to lesions or textures ImageNet never saw, but 800 images can overwrite those filters (catastrophic fine-tuning). A practical compromise is a small learning rate on the trunk and a larger one on the head, or adapters (Chapter 15 optional modern note) — say that as an extension, not as the core answer.

### Q2. When frozen features are the wrong tool

Give a concrete target domain where ImageNet activations are a weak representation even though both source and target are “images,” and name the mismatch.

**Hint.** Think sensors, viewpoint, or label granularity — not “the dataset is small.”

**Discussion.** Examples: grayscale histopathology at 40×, satellite bands beyond RGB, or extreme close-up industrial defects. The mismatch is *low-level statistics and semantics*, not merely sample size. Fine-tuning more of the trunk, or pretraining in-domain (self-supervision, Chapter 16), is more honest than stacking a deeper MLP on frozen RGB-object features.

### Q3. What you actually extract

A teammate “extracts features” by taking the $$7\times7\times2048$$ map of a ResNet-50 and flattening it to train an SVM. Give one reason to global-average-pool first, and one task where you would *keep* the spatial map.

**Hint.** Invariance vs localization.

**Discussion.** GAP to $$2048$$-d yields a translation-tolerant descriptor and a sane SVM dimension. Detection, segmentation, or any task that must know *where* the object is should keep the spatial tensor (or use a feature pyramid). “Feature extraction” is not one vector shape; it is a choice about which invariances you want.

### Q4. Domain shift in one picture

Training images are studio product shots; production images are phone photos in warehouses. You fine-tune the head only and val accuracy looks great (val is also studio). What did you measure, and what experiment would you add before shipping?

**Hint.** i.i.d. val vs the deployment distribution.

**Discussion.** You measured in-domain head quality, not transfer. Add a warehouse-labeled slice (or at least a visually matched proxy) and report that number. Color / crop augmentation that mimics warehouse lighting is a cheap intermediate; it is not a substitute for a true target val set. Interviewers want this humility more than a magic architecture name.

## Attribution

Kashani, S., and Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Download the PDF from arXiv for the full solved Q&A. This page is original course practice, not a reprint.

## Video tham khảo

- **xem trước** [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, English. Feature-extraction versus fine-tuning prompts.
- [Pytorch Transfer Learning and Fine Tuning Tutorial](https://www.youtube.com/watch?v=qaDe0qQZ5AQ) — Aladdin Persson, English. Be able to say which parameters still receive gradients.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, English. Why the early filters are the ones worth keeping.
