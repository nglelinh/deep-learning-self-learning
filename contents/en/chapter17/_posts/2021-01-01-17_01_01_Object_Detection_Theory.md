---
layout: post
title: 17-01-01 Object Detection Theory
chapter: '17'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter17
---

# Object Detection: Localization and Recognition

![Object Detection Example](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg/800px-Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg)
*Hình ảnh: Object Detection với YOLO - phát hiện và định vị nhiều vật thể. Nguồn: Wikimedia Commons*

## 1. Concept Overview

Object detection extends image classification from answering "what objects are in this image?" to "what objects are present and where are they located?" This seemingly small extension from classification to detection actually requires solving multiple interconnected problems simultaneously: proposing regions that might contain objects (region proposal), classifying what's in each region (recognition), refining the boundaries of detections (localization), and handling multiple objects of different classes at different scales (multi-scale, multi-class detection). The complexity of coordinating these components while maintaining real-time performance has made object detection one of the most challenging and actively researched areas in computer vision.

The evolution of object detection methods reveals a fascinating progression from traditional computer vision to modern deep learning approaches. Classical methods used hand-crafted features (SIFT, HOG) with sliding windows exhaustively searching every possible location and scale, then applying classifiers like SVMs. This was computationally expensive (evaluating millions of windows per image) and limited by feature quality. The deep learning revolution transformed object detection through learned features and end-to-end trainable systems, enabling dramatic improvements in both accuracy and speed.

Modern object detection has branched into two major paradigms. Two-stage detectors like R-CNN, Fast R-CNN, and Faster R-CNN first propose regions likely to contain objects, then classify and refine these proposals. This explicit separation of region proposal and recognition enables high accuracy through focused computation on promising regions. Single-stage detectors like YOLO and SSD directly predict bounding boxes and class probabilities from regular grid positions, enabling real-time performance by avoiding the proposal stage at the cost of slightly lower accuracy on small objects.

Understanding object detection deeply requires grasping several technical innovations that make modern systems work. Region Proposal Networks learn to generate object proposals rather than using hand-crafted rules, making the entire pipeline differentiable. Anchor boxes provide a way to handle objects of different aspect ratios and sizes through predefined box templates. Non-maximum suppression eliminates duplicate detections, addressing the fact that good detectors typically generate multiple overlapping boxes for each object. Feature pyramid networks enable detecting objects at multiple scales by building feature pyramies with rich semantics at all levels. These components, each solving a specific sub-problem, combine into systems that can detect and localize dozens of objects across multiple categories in milliseconds, enabling applications from autonomous driving to medical image analysis to augmented reality.

## 2. Mathematical Foundation

Object detection requires formalizing what we're predicting and how we measure success. An object detection is a tuple $$(\text{class}, x, y, w, h)$$ specifying the object's category and bounding box (center coordinates $$x,y$$ and dimensions $$w,h$$). For an image with $$N$$ objects, the ground truth is a set of such tuples: $$\{(\text{class}_i, x_i, y_i, w_i, h_i)\}_{i=1}^N$$. Our detector must predict this set, which is challenging because $$N$$ varies across images.

### Intersection over Union (IoU)

To measure localization quality, we use Intersection over Union between predicted and ground-truth boxes:

$$\text{IoU}(\text{box}_{\text{pred}}, \text{box}_{\text{gt}}) = \frac{\text{Area}(\text{box}_{\text{pred}} \cap \text{box}_{\text{gt}})}{\text{Area}(\text{box}_{\text{pred}} \cup \text{box}_{\text{gt}})}$$

IoU ranges from 0 (no overlap) to 1 (perfect overlap). Typically, we consider a detection correct if IoU $$\geq 0.5$$ and the predicted class matches ground truth. This threshold balances between requiring precise localization and allowing reasonable bounding box variations.

### Bounding Box Regression

Rather than directly predicting box coordinates, modern detectors predict offsets from anchor boxes (predefined reference boxes). Given anchor box $$(\hat{x}, \hat{y}, \hat{w}, \hat{h})$$ and ground truth $$(\bar{x}, \bar{y}, \bar{w}, \bar{h})$$, we parameterize targets as:

$$t_x = \frac{\bar{x} - \hat{x}}{\hat{w}}, \quad t_y = \frac{\bar{y} - \hat{y}}{\hat{h}}$$

$$t_w = \log\frac{\bar{w}}{\hat{w}}, \quad t_h = \log\frac{\bar{h}}{\hat{h}}$$

The network predicts $$(t_x, t_y, t_w, t_h)$$, and we decode to absolute coordinates:

$$x = \hat{x} + \hat{w} \cdot t_x, \quad y = \hat{y} + \hat{h} \cdot t_y$$

$$w = \hat{w} \cdot \exp(t_w), \quad h = \hat{h} \cdot \exp(t_h)$$

This parameterization is more learnable than direct coordinate prediction because offsets are typically small numbers with similar scales, while absolute coordinates span the entire image with very different scales for small versus large objects.

### Multi-Task Loss

Object detectors optimize combined losses for classification and localization:

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{box}}$$

where $$\mathcal{L}_{\text{cls}}$$ is classification loss (cross-entropy) and $$\mathcal{L}_{\text{box}}$$ is bounding box regression loss (smooth L1 or IoU loss). The weight $$\lambda$$ balances these objectives—too high and the detector focuses on precise localization at the expense of correct classification; too low and classifications are accurate but boxes are poorly localized.

For Faster R-CNN, the classification loss uses cross-entropy over classes plus background:

$$\mathcal{L}_{\text{cls}} = -\log p_{\text{class}}$$

where $$\text{class}$$ is the ground-truth class (or background if IoU < 0.5 with all ground-truth boxes).

The box loss is smooth L1:

$$\mathcal{L}_{\text{box}} = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L1}(t_i - \hat{t}_i)$$

$$\text{smooth}_{L1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

Smooth L1 is less sensitive to outliers than L2 (quadratic becomes linear for large errors) while being differentiable everywhere (unlike pure L1).

### Region Proposal Networks (RPN)

Faster R-CNN introduced RPN, a fully convolutional network that predicts object proposals. At each position in the feature map, RPN predicts:
- Objectness scores: $$k$$ anchors × 2 values (object vs background)
- Box refinements: $$k$$ anchors × 4 coordinates

For a $$H \times W$$ feature map with $$k=9$$ anchors per position, RPN outputs:
- Objectness: $$H \times W \times 9 \times 2$$ scores
- Box deltas: $$H \times W \times 9 \times 4$$ values

Total: $$HW \times 9$$ proposals. Non-maximum suppression filters these to top $$\sim$$2000 based on objectness scores, which then go to the detection head.

The RPN loss combines classification (objectness) and regression:

$$\mathcal{L}_{\text{RPN}} = \frac{1}{N_{\text{cls}}}\sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \frac{\lambda}{N_{\text{box}}}\sum_i p_i^* \mathcal{L}_{\text{box}}(t_i, t_i^*)$$

where $$p_i^* = 1$$ if anchor $$i$$ overlaps ground-truth with IoU > 0.7 (positive), $$p_i^* = 0$$ if IoU < 0.3 (negative), and ignored if in between (to handle ambiguous cases).

## 3. Example / Intuition

Imagine you're trying to find and identify all people in a crowded photograph. Your strategy might be:

1. **Quick scan** for regions likely to contain people (look for head shapes, body outlines)
2. **Closer examination** of promising regions (is this actually a person or a statue? Which person is it?)
3. **Refinement** of boundaries (exactly where does this person's bounding box start/end?)

This three-stage process mirrors two-stage object detection. The Region Proposal Network does the quick scan, proposing ~2000 regions that might contain objects (people, cars, dogs, anything). The detection head examines each proposal, classifying what's there and refining the bounding box. Non-maximum suppression eliminates duplicates (multiple overlapping boxes for the same person).

Consider detecting cars in a street scene. The image might contain:
- 3 cars at different distances (different sizes)
- 2 pedestrians
- 1 traffic sign
- Complex background (buildings, trees)

A single-stage detector like YOLO divides the image into a grid (say 13×13). Each grid cell predicts:
- Multiple bounding boxes (say 3, with different aspect ratios: tall, wide, square)
- Class probabilities for each box
- Confidence scores (is there an object here?)

For a grid cell at position (5, 8) near a car, it might predict:
- Box 1: class=car, confidence=0.95, coordinates offset from cell center
- Box 2: class=background, confidence=0.05
- Box 3: class=background, confidence=0.02

After processing all 13×13 cells, we have 13×13×3 = 507 predictions. Most are background (confidence near 0). NMS keeps only high-confidence, non-overlapping boxes:
- Car 1: confidence=0.95, box=[120, 200, 60, 40]
- Car 2: confidence=0.89, box=[300, 180, 80, 50]
- Car 3: confidence=0.76, box=[450, 220, 40, 25] (far car, smaller)
- Person 1: confidence=0.92, box=[200, 150, 30, 80]

The detector has identified all objects, classified them, and localized them with bounding boxes—exactly what object detection requires.

