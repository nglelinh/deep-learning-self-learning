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
