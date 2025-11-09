---
layout: post
title: 17-01 Object Detection Fundamentals
chapter: '17'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter17
lesson_type: required
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

## 4. Code Snippet

Complete Faster R-CNN style implementation:

```python
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

print("="*70)
print("Object Detection with Faster R-CNN")
print("="*70)

# Load pre-trained Faster R-CNN
print("\n1. Loading Pre-trained Faster R-CNN")
print("-" * 70)

model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

print("Faster R-CNN architecture:")
print("  Backbone: ResNet-50 + Feature Pyramid Network")
print("  Region Proposal Network (RPN): Generates object proposals")
print("  RoI Pooling: Extracts features from proposals")
print("  Detection Head: Classifies and refines boxes")

# Adapt for custom dataset (e.g., 10 classes + background)
num_classes = 11  # 10 object classes + background

# Replace the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

print(f"\nAdapted for {num_classes-1} object classes + background")

# 2. Prepare custom dataset format
print("\n2. Dataset Format for Object Detection")
print("-" * 70)

class CustomDetectionDataset(torch.utils.data.Dataset):
    """
    Object detection dataset format.
    
    Each sample returns:
    - image: (3, H, W) tensor
    - target: dictionary with:
        - boxes: (N, 4) tensor of [x1, y1, x2, y2] coordinates
        - labels: (N,) tensor of class indices
        - (optional) masks, keypoints, etc.
    """
    
    def __init__(self, images, annotations, transform=None):
        self.images = images
        self.annotations = annotations
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image (simulated here)
        img = self.images[idx]
        
        # Get annotations for this image
        boxes = self.annotations[idx]['boxes']  # [[x1,y1,x2,y2], ...]
        labels = self.annotations[idx]['labels']  # [class1, class2, ...]
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

# Create dummy data for demonstration
print("Creating simulated detection dataset...")

# Simulate 100 images with random objects
n_images = 100
dummy_images = [torch.rand(3, 600, 800) for _ in range(n_images)]

# Simulate annotations (random boxes and classes)
dummy_annotations = []
for _ in range(n_images):
    n_objects = torch.randint(1, 5, (1,)).item()  # 1-4 objects per image
    
    # Random boxes (x1, y1, x2, y2)
    boxes = []
    labels = []
    for _ in range(n_objects):
        x1 = torch.randint(0, 700, (1,)).item()
        y1 = torch.randint(0, 500, (1,)).item()
        x2 = x1 + torch.randint(50, 200, (1,)).item()
        y2 = y1 + torch.randint(50, 200, (1,)).item()
        
        boxes.append([x1, y1, min(x2, 800), min(y2, 600)])
        labels.append(torch.randint(1, 11, (1,)).item())  # Classes 1-10
    
    dummy_annotations.append({'boxes': boxes, 'labels': labels})

dataset = CustomDetectionDataset(dummy_images, dummy_annotations)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, 
                                         shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

print(f"Dataset: {len(dataset)} images")
print(f"Sample annotation: {dummy_annotations[0]}")

# 3. Training
print("\n3. Training Object Detector")
print("-" * 70)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

model.train()
num_epochs = 3

print(f"Training for {num_epochs} epochs...")
print("(Using dummy data - in practice, use real annotated images)")

for epoch in range(num_epochs):
    epoch_loss = 0
    
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass returns loss dict during training
        loss_dict = model(images, targets)
        
        # Combine losses
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Loss = {epoch_loss/len(dataloader):.4f}")

print("\nTraining complete!")

# 4. Inference
print("\n4. Running Inference")
print("-" * 70)

model.eval()

# Test on one image
test_img = torch.rand(3, 600, 800).to(device)

with torch.no_grad():
    predictions = model([test_img])

pred = predictions[0]
print(f"Predictions for test image:")
print(f"  Boxes shape: {pred['boxes'].shape}")
print(f"  Labels shape: {pred['labels'].shape}")
print(f"  Scores shape: {pred['scores'].shape}")

# Filter by confidence threshold
confidence_threshold = 0.5
keep = pred['scores'] > confidence_threshold

print(f"\nDetections with confidence > {confidence_threshold}:")
print(f"  {keep.sum().item()} objects detected")

for i in range(keep.sum().item()):
    box = pred['boxes'][keep][i].cpu().numpy()
    label = pred['labels'][keep][i].item()
    score = pred['scores'][keep][i].item()
    print(f"    Class {label}: box={box.round()}, confidence={score:.3f}")
```

Implement YOLO-style single-stage detector:

```python
print("\n" + "="*70)
print("Single-Stage Detection (YOLO-style)")
print("="*70)

class SimpleSingleStageDetector(nn.Module):
    """
    Simplified YOLO-style detector for educational purposes.
    
    Architecture:
    - Backbone CNN extracts features
    - Detection head predicts boxes + classes for grid cells
    - Each cell predicts B boxes with (x, y, w, h, confidence, class_probs)
    """
    
    def __init__(self, num_classes=10, num_boxes=3, grid_size=13):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.grid_size = grid_size
        
        # Backbone (simplified - use any CNN)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
        )
        
        # Detection head
        # Each grid cell outputs: B boxes × (5 + num_classes)
        # 5 = (x, y, w, h, confidence)
        output_channels = num_boxes * (5 + num_classes)
        
        self.detection_head = nn.Conv2d(512, output_channels, 1)
    
    def forward(self, x):
        """
        x: (B, 3, H, W) images
        
        Returns: (B, grid_size, grid_size, num_boxes, 5+num_classes)
        """
        # Extract features
        features = self.backbone(x)  # (B, 512, grid_size, grid_size)
        
        # Predict detections
        detections = self.detection_head(features)
        
        # Reshape to (B, grid, grid, boxes, 5+classes)
        B = x.size(0)
        detections = detections.view(
            B, self.num_boxes, 5 + self.num_classes, 
            self.grid_size, self.grid_size
        )
        detections = detections.permute(0, 3, 4, 1, 2)  # (B, grid, grid, boxes, ...)
        
        return detections

detector = SimpleSingleStageDetector(num_classes=10, num_boxes=3, grid_size=13)

print("Single-stage detector architecture:")
print(f"  Grid size: {13}×{13}")
print(f"  Boxes per cell: 3")
print(f"  Classes: 10")
print(f"  Total predictions: 13×13×3 = 507 boxes per image")

# Forward pass
test_input = torch.rand(2, 3, 416, 416)  # Batch of 2, 416×416 images
output = detector(test_input)

print(f"\nOutput shape: {output.shape}")  # (2, 13, 13, 3, 15)
print("Dimensions: (batch, grid_y, grid_x, boxes, 5+classes)")
print("\nEach box prediction has:")
print("  - 4 coordinates (x, y, w, h)")
print("  - 1 confidence score")
print("  - 10 class probabilities")
print("\nSingle forward pass produces all detections - very fast!")
```

## 5. Related Concepts

Object detection connects to image classification through its foundation in convolutional feature extraction. The backbone networks (ResNet, VGG, MobileNet) are typically classification networks adapted for detection by removing final classification layers and adding detection heads. The features learned for classification—edges, textures, object parts—transfer naturally to detection because recognizing "this is a car" (classification) and "there's a car at this location" (detection) both require understanding car appearance. However, detection requires additional capabilities: localizing precisely where objects are, handling multiple objects and scales, and distinguishing object from background. Understanding this connection helps appreciate why ImageNet pre-trained classifiers provide good starting points for detection but require architectural extensions (FPN for multi-scale, RPN for proposals) to reach full detection capability.

The relationship to semantic segmentation illuminates different granularities of visual understanding. Classification assigns one label per image. Detection assigns labels and boxes to multiple objects per image. Semantic segmentation assigns labels to every pixel, delineating object boundaries exactly. Instance segmentation combines detection and segmentation, providing pixel-perfect masks for each object instance. This progression from coarse (image-level) to fine (pixel-level) understanding reflects different application requirements and computational tradeoffs. Detection provides a balance: more informative than classification (where are objects?) without the computational cost of pixel-level segmentation.

Object detection's connection to attention mechanisms is increasingly important in modern architectures. Transformers are replacing traditional detection heads through DETR (Detection Transformer), which treats object detection as set prediction using attention to directly predict all objects in parallel without anchors or NMS. The attention mechanism learns to attend to object locations and extents, providing an elegant end-to-end trainable alternative to the complex pipelines of traditional detectors. Understanding how attention can replace hand-crafted components like anchors and NMS helps appreciate Transformers' generality beyond NLP.

The evolution from R-CNN to Fast R-CNN to Faster R-CNN demonstrates systematic optimization of computational bottlenecks. R-CNN ran CNN feature extraction separately for each proposal (2000 forward passes per image—very slow). Fast R-CNN extracted features once for the whole image, then used RoI pooling to get proposal features (single forward pass—much faster). Faster R-CNN made proposal generation part of the network through RPN (fully differentiable, end-to-end trainable). Each innovation addressed a specific inefficiency while maintaining accuracy, showing how systems evolve through targeted improvements rather than complete redesigns.

## 6. Fundamental Papers

**["Rich feature hierarchies for accurate object detection and semantic segmentation" (2014)](https://arxiv.org/abs/1311.2524)**  
*Authors*: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik  
R-CNN revolutionized object detection by applying CNNs, previously successful for classification, to detection through region proposals. The approach was conceptually simple: use selective search to propose ~2000 regions per image, extract CNN features from each (forward passing each through AlexNet), then classify regions with SVMs and refine boxes with regression. While computationally expensive (2000 forward passes per image), R-CNN achieved dramatic improvements over traditional methods, demonstrating that learned features vastly outperform hand-crafted features for detection. The paper established the region-based detection paradigm and showed transfer learning (ImageNet pre-training for detection) was highly effective. R-CNN's success sparked the deep learning revolution in object detection, leading to numerous improvements addressing its computational limitations while maintaining its core insight: detection can be solved through region classification with learned features.

**["Fast R-CNN" (2015)](https://arxiv.org/abs/1504.08083)**  
*Author*: Ross Girshick  
Fast R-CNN addressed R-CNN's computational bottleneck by sharing CNN computation across proposals through RoI (Region of Interest) pooling. Instead of running CNN separately for each proposal, extract features once for the whole image, then use RoI pooling to extract fixed-size feature vectors for each proposal from the shared feature map. This reduced forward passes from 2000 per image to 1, achieving ~10× speedup while improving accuracy through joint training of feature extraction, classification, and bounding box regression. The paper introduced multi-task loss combining classification and localization, showing that joint training improves both tasks compared to training separately. Fast R-CNN demonstrated that systematic analysis of computational bottlenecks and clever architectural innovations could dramatically improve efficiency without sacrificing accuracy, establishing principles for designing practical detection systems.

**["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2016)](https://arxiv.org/abs/1506.01497)**  
*Authors*: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun  
Faster R-CNN completed the evolution to fully differentiable detection by replacing selective search with Region Proposal Networks, making the entire detection pipeline trainable end-to-end. RPN uses learned convolutional filters to predict objectness and box coordinates at every position in the feature map, generating proposals through learned mechanisms rather than hand-crafted algorithms. This innovation enabled sharing computation between proposal generation and detection, improved proposal quality through supervised training, and achieved near real-time speeds (5 FPS). The anchor box mechanism—predicting offsets from predefined boxes of different aspect ratios and scales—became standard in subsequent detectors. Faster R-CNN set the template for two-stage detection and remained state-of-the-art for years, demonstrating that careful end-to-end design outperforms pipelined approaches with non-differentiable components.

**["You Only Look Once: Unified, Real-Time Object Detection" (2016)](https://arxiv.org/abs/1506.02640)**  
*Authors*: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi  
YOLO fundamentally changed object detection by framing it as regression from image pixels directly to bounding box coordinates and class probabilities, enabling real-time detection (45 FPS) through a single forward pass. By dividing images into grids and having each cell predict boxes, YOLO eliminated region proposals and their associated computational cost. While initial accuracy was lower than Faster R-CNN (particularly for small objects), YOLO's speed enabled real-time applications like autonomous driving and robotics. The paper showed that detection need not follow the two-stage paradigm, inspiring numerous single-stage detectors. YOLO's end-to-end design philosophy—predicting everything in one shot—demonstrated that simple, unified approaches could compete with complex pipelines when properly designed.

**["Feature Pyramid Networks for Object Detection" (2017)](https://arxiv.org/abs/1612.03144)**  
*Authors*: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie  
FPN addressed multi-scale detection by building feature pyramids with strong semantics at all scales, combining high-resolution but semantically weak low-level features with low-resolution but semantically strong high-level features through top-down pathways and lateral connections. This enables detecting large objects using high-level features and small objects using low-level features enriched with semantic information from higher layers. FPN dramatically improved detection of objects at different scales, particularly small objects which previous methods struggled with. The architectural pattern—building pyramids with both bottom-up (standard CNN) and top-down pathways—has been widely adopted beyond detection to segmentation and other dense prediction tasks. FPN demonstrated that careful multi-scale architecture design addresses fundamental challenges in visual recognition.

## Common Pitfalls and Tricks

Anchor box design significantly affects detection performance but is often overlooked. Anchors should match typical object aspect ratios and sizes in your dataset. For pedestrian detection (tall, narrow objects), use anchors like 1:3 and 1:4 aspect ratios. For cars (wider), use 2:1 or 3:2. The k-means clustering on training set bounding boxes can discover good anchor dimensions automatically. Having too many anchors wastes computation without improving accuracy, while too few miss important object types. Typical YOLO uses 9 anchors (3 scales × 3 aspect ratios), Faster R-CNN uses 9 (3 scales × 3 ratios) per position.

Non-maximum suppression threshold selection involves precision-recall tradeoffs. Lower IoU threshold (0.3) suppresses more boxes, reducing duplicates but potentially eliminating valid detections of nearby objects. Higher threshold (0.7) keeps more boxes, detecting nearby objects well but producing duplicates. For crowded scenes (many nearby objects), use higher threshold. For sparse scenes, lower threshold. Understanding that NMS threshold controls this tradeoff allows tuning for specific applications.

Class imbalance in object detection is severe and requires careful handling. Most anchor boxes are background (no object), creating extreme imbalance between background and object classes (often 1000:1 or more). Without handling, the detector learns to predict background for everything (trivial solution achieving 99.9% accuracy). Solutions include hard negative mining (train on hard background examples, ignore easy ones), focal loss (weight loss by difficulty, downweighting easy classifications), and balanced sampling (ensure batches contain similar numbers of object and background examples).

## Key Takeaways

Object detection extends classification to localizing and recognizing multiple objects per image, requiring simultaneous region proposal, classification, and bounding box regression. Two-stage detectors separate proposal generation from detection, using Region Proposal Networks to generate candidates and detection heads to classify and refine, achieving high accuracy through focused computation on object regions. Single-stage detectors directly predict boxes and classes from grid cells, enabling real-time performance through single forward pass at slight accuracy cost. Intersection over Union measures localization quality, with detections considered correct when IoU with ground-truth exceeds threshold (typically 0.5) and class matches. Anchor boxes provide reference boxes at different scales and aspect ratios, with networks predicting offsets rather than absolute coordinates, improving training stability. Feature pyramids enable multi-scale detection by combining high-level semantic features with high-resolution spatial features. Non-maximum suppression eliminates duplicate detections by keeping highest-confidence boxes and suppressing overlapping boxes. Multi-task training jointly optimizes classification and localization, with losses balanced to achieve both accurate recognition and precise localization. Understanding object detection requires appreciating how these components coordinate to handle variable numbers of objects at different scales, positions, and classes while maintaining real-time or near-real-time performance for practical applications.

