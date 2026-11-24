---
layout: post
title: 17-99 Modern Applications and Updates (2022–2026)
chapter: '17'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter17
lesson_type: optional
---

Between 2023 and 2026, computer vision broke free of fixed 80-class COCO label lists. With Segment Anything (SAM / SAM 2), you click a single point on *any* object in an image or video and get a crisp mask zero-shot; with Grounding DINO and RT-DETR, you type an arbitrary text phrase like *"the cracked red mug on the left"* and get end-to-end Transformer bounding boxes without ever running NMS.

## Background Knowledge Map: Post-YOLO & Post-Mask-R-CNN Vision (2022–2026)

- **Pillar 1 — Promptable Segmentation (SAM & SAM 2)**: Decouples heavy image encoding from a lightweight prompt decoder (points, boxes, masks, and video memory banks), enabling real-time interactive segmentation across unseen categories.
- **Pillar 2 — NMS-Free End-to-End Detection (DETR & RT-DETR)**: Uses bipartite Hungarian matching between learned object queries and ground-truth boxes so the Transformer predicts unique objects directly without heuristic NMS.
- **Pillar 3 — Open-Vocabulary Detection (Grounding DINO)**: Fuses text-encoder phrase embeddings with visual multi-scale features via cross-attention, turning fixed-class classification into phrase–region similarity matching.
- **Pillar 4 — Production Edge Detectors (YOLOv8–v11 / YOLO-World)**: Continues to dominate latency-critical industrial inspection, drones, and edge cameras where TensorRT/ONNX throughput is paramount.

### Step-by-Step Worked Example (Read Slowly): Bipartite Hungarian Matching Cost in DETR

Suppose an image has 2 ground-truth objects ($G_1, G_2$) and a DETR decoder emits 2 object queries ($Q_1, Q_2$) with pairwise matching cost matrix $C_{ij} = \mathcal{L}_{\text{cls}}(Q_i, G_j) + \mathcal{L}_{\text{box}}(Q_i, G_j)$:
$$C = \begin{bmatrix} 0.9 & 0.2 \\ 0.3 & 0.8 \end{bmatrix}.$$

1. **Evaluate both possible one-to-one assignments**:
   - Assignment 1 ($Q_1 \to G_1,\; Q_2 \to G_2$): total cost $= 0.9 + 0.8 = 1.7$.
   - Assignment 2 ($Q_1 \to G_2,\; Q_2 \to G_1$): total cost $= 0.2 + 0.3 = 0.5$.
2. **Optimal Hungarian match**: Assignment 2 ($Q_1 \to G_2,\; Q_2 \to G_1$) with minimum cost $0.5$.
3. **Why this eliminates NMS**: Because each ground-truth object is matched to *strictly one* query, every other overlapping query is trained to predict `no-object` ($\varnothing$), teaching the Transformer itself to suppress duplicates during training.
4. **Takeaway**: Hungarian bipartite matching replaces hand-tuned anchor matching and NMS with a clean, global combinatorial assignment.

# Optional: detection and segmentation after YOLO / Mask R-CNN

> This lesson is **optional**. It does **not** replace two-stage vs one-stage detection theory or the YOLO / Faster R-CNN notes. It covers the 2023–2025 models used in products: **SAM/SAM 2**, **RT-DETR**, **Grounding DINO**, and current YOLO lines.

The matching / IoU / NMS ideas in this chapter still decide whether a box is correct:

$$\mathrm{IoU}(A,B) = \frac{|A\cap B|}{|A\cup B|}.$$

The architectures around those metrics changed.

## 1. Promptable segmentation: SAM and SAM 2

[Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) (Segment Anything) treat segmentation as a **promptable** foundation task: points, boxes, or masks in; instance masks out. [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) (SAM 2) add a streaming memory for **video**. Data-science use: label a few points, export masks, train a smaller specialist — not a replacement for U-Net when you already have a pixel-labeled medical set.

## 2. Concrete applications

### Real-time DETR-style detectors

[Zhao et al., 2024](https://arxiv.org/abs/2304.08069) (RT-DETR) make transformer detectors fast enough to displace some YOLO deployments. You still evaluate mAP the way the theory lesson teaches.

### Open-vocabulary: Grounding DINO + SAM

[Liu et al., 2023](https://arxiv.org/abs/2303.05499) (Grounding DINO) take a **text** query (“red helmet”) and return boxes. Combined with SAM (“Grounded SAM”) this is the 2024–2026 annotation and robotics perception pattern. Classic closed-set COCO heads remain for factory SKUs.

### YOLO in production

Ultralytics **YOLOv8 / YOLO11** keep the convolutional one-stage recipe. Use them when latency and export (ONNX, TensorRT, CoreML) matter more than open vocabulary.

## 3. Widely used software

- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) and [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).
- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) and [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR).

## 4. Citations (2022–2026)

- [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643).
- [SAM 2 (Ravi et al., 2024)](https://arxiv.org/abs/2408.00714).
- [Grounding DINO (Liu et al., 2023)](https://arxiv.org/abs/2303.05499).
- [DETRs Beat YOLOs on Real-time Object Detection — RT-DETR (Zhao et al., 2024)](https://arxiv.org/abs/2304.08069).

## 5. How this complements the core notes

Implement the detector matching and the chapter’s R-CNN/YOLO path first. This lesson only adds **promptable and open-vocabulary** heads used in 2023–2026 pipelines.

## Video tham khảo

- **xem trước** [Explaining the Segment Anything Model - Network architecture, Dataset, Training](https://www.youtube.com/watch?v=OhxJkqD1vuE) — Neural Breakdown with AVB, English. The Segment Anything model is the promptable segmenter this note describes.
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — Stanford University School of Engineering, English. The classical detection and segmentation problem SAM generalizes.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, English. The vision backbone underneath that promptable head.
