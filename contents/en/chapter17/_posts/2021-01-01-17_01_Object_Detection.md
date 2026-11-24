---
layout: post
title: 17-01 Object Detection Fundamentals
chapter: '17'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter17
---

Object detection is where deep learning leaves tidy single-label benchmarks and enters the messy visual world—multiple overlapping objects, dramatic scale changes, and strict real-time latency budgets. Connecting geometric theory (IoU, anchor offsets, RPNs) with torchvision's end-to-end training contract prepares you to read, debug, and customize both two-stage and one-stage detectors.

## Background Knowledge Map: Object Detection Theory-to-Code Bridge

- **Pillar 1 — Two-Stage vs. One-Stage Architectures**: Contrasts proposal-then-refine pipelines (Faster R-CNN) with dense single-shot grids (YOLO, RetinaNet) across accuracy–latency curves.
- **Pillar 2 — Geometric Overlap & Anchor Assignment**: Uses IoU thresholds to match dense anchors to ground-truth objects and regress scale-invariant deltas.
- **Pillar 3 — Post-Processing & Evaluation**: Applies NMS to deduplicate boxes and measures detector quality with precision–recall Area Under Curve (AP / mAP).
- **Pillar 4 — PyTorch Detection Engineering**: Handles ragged multi-box batches, background class indexing, and pretrained FPN backbone fine-tuning.

### Step-by-Step Worked Example (Read Slowly): Computing Precision, Recall, and AP from 4 Ranked Detections

Suppose an image contains $M = 2$ ground-truth dogs, and our detector outputs 4 ranked detections (highest score first) labeled True Positive (TP) or False Positive (FP) at IoU $\ge 0.50$:
$$[\text{TP},\; \text{FP},\; \text{TP},\; \text{FP}].$$

1. **Cumulative Precision & Recall at each rank $k$**:
   - Rank 1 (TP): $\text{Prec} = 1/1 = 1.00,\quad \text{Rec} = 1/2 = 0.50$.
   - Rank 2 (FP): $\text{Prec} = 1/2 = 0.50,\quad \text{Rec} = 1/2 = 0.50$.
   - Rank 3 (TP): $\text{Prec} = 2/3 = 0.667,\quad \text{Rec} = 2/2 = 1.00$.
   - Rank 4 (FP): $\text{Prec} = 2/4 = 0.50,\quad \text{Rec} = 2/2 = 1.00$.
2. **All-point interpolated Average Precision (taking max precision to the right of each recall step)**:
   - Over recall interval $[0.0, 0.5]$: max precision $= 1.00 \implies \text{area} = 0.5 \times 1.00 = 0.500$.
   - Over recall interval $(0.5, 1.0]$: max precision $= 0.667 \implies \text{area} = 0.5 \times 0.667 = 0.333$.
   $$\text{AP@50} = 0.500 + 0.333 = 0.833.$$
3. **Takeaway**: A single high-confidence false positive at rank 2 costs $0.167$ AP points—showing why ranking calibration matters just as much as finding all objects.

Object detection:

1. **Theory** — IoU, boxes, RPN, intuition
2. **Implementation** — code path, papers, pitfalls

## Video tham khảo

- **xem trước** [How YOLO Object Detection Works](https://www.youtube.com/watch?v=svn9-xV7wjk) — DeepBean, English. The grid, the boxes, and the objectness score.
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — Stanford University School of Engineering, English. The family YOLO belongs to, including the two-stage detectors the lesson compares.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, English. What the backbone must already see before a detection head is worth adding.
