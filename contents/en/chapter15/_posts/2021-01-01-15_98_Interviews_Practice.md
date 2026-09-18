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
