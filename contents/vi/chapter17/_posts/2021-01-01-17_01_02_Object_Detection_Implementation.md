---
layout: post
title: 17-01-02 Cài đặt Phát hiện Đối tượng
chapter: '17'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter17
---

## 4. Mã minh họa

Ví dụ cài đặt đầy đủ theo kiến trúc Faster R-CNN:

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

Ví dụ cài đặt bộ phát hiện một giai đoạn kiểu YOLO:

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

## 5. Khái niệm liên quan

Phát hiện đối tượng gắn bó mật thiết với phân loại ảnh thông qua mạng trích xuất đặc trưng tích chập. Các mạng xương sống (*backbone* như ResNet, VGG, MobileNet) thường là mạng phân loại được điều chỉnh bằng cách bỏ các tầng phân loại cuối và gắn thêm đầu phát hiện. Những đặc trưng học được cho bài toán phân loại — như cạnh, kết cấu và bộ phận đối tượng — chuyển giao tự nhiên sang phát hiện đối tượng vì cả hai tác vụ đều cần nhận diện hình dáng đối tượng. Tuy nhiên, phát hiện đối tượng còn đòi hỏi định vị chính xác, xử lý nhiều đối tượng ở nhiều tỷ lệ kích thước và phân biệt đối tượng với nền. Vì vậy, bộ phân loại tiền huấn luyện trên ImageNet là điểm khởi đầu phù hợp, nhưng cần bổ sung các thành phần kiến trúc như FPN (cho đa tỷ lệ) và RPN (cho đề xuất vùng).

So sánh với phân đoạn ngữ nghĩa (*semantic segmentation*) giúp thấy rõ các mức độ phân tích ảnh khác nhau. Phân loại gán một nhãn cho toàn bộ ảnh; phát hiện gán nhãn cùng hộp bao cho từng đối tượng; phân đoạn ngữ nghĩa gán nhãn cho từng pixel để xác định chính xác đường biên; còn phân đoạn thể hiện (*instance segmentation*) kết hợp cả hai để tạo mặt nạ ở mức pixel cho từng thể hiện riêng lẻ. Lộ trình từ mức ảnh đến mức pixel phản ánh sự đánh đổi giữa độ chi tiết thông tin và chi phí tính toán. Phát hiện đối tượng nằm ở điểm cân bằng: cung cấp vị trí đối tượng mà không đòi hỏi chi phí gán nhãn và suy luận ở cấp pixel.

Mối liên hệ giữa phát hiện đối tượng và cơ chế *attention* ngày càng rõ nét trong các kiến trúc hiện đại. DETR (*Detection Transformer*) sử dụng Transformer để coi phát hiện đối tượng như bài toán dự đoán tập hợp (*set prediction*), dự đoán song song tất cả các đối tượng mà không cần hộp neo hay NMS. Cơ chế *attention* học cách tập trung vào vị trí và phạm vi của từng đối tượng, cho phép huấn luyện đầu–cuối gọn gàng thay cho quy trình nhiều bước thủ công của các bộ phát hiện cổ điển.

Quá trình phát triển từ R-CNN sang Fast R-CNN rồi Faster R-CNN là ví dụ điển hình về việc tối ưu hóa từng nút thắt tính toán. R-CNN chạy CNN riêng lẻ cho từng vùng đề xuất (khoảng 2000 lượt lan truyền xuôi mỗi ảnh, rất chậm). Fast R-CNN trích xuất đặc trưng một lần cho toàn ảnh rồi dùng RoI pooling để lấy đặc trưng cho từng đề xuất, giảm xuống còn một lượt lan truyền xuôi. Faster R-CNN đưa luôn bước sinh đề xuất vào trong mạng qua RPN, tạo thành hệ thống khả vi và huấn luyện đầu–cuối. Mỗi bước cải tiến đều xử lý một điểm nghẽn cụ thể trong khi vẫn giữ hoặc nâng cao độ chính xác.

## 6. Các bài báo nền tảng

**["Rich feature hierarchies for accurate object detection and semantic segmentation" (2014)](https://arxiv.org/abs/1311.2524)**  
*Tác giả*: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik  
R-CNN đưa CNN vào bài toán phát hiện đối tượng thông qua đề xuất vùng: dùng thuật toán *selective search* sinh khoảng 2000 vùng đề xuất mỗi ảnh, đưa từng vùng qua AlexNet để trích xuất đặc trưng, rồi phân loại bằng SVM và tinh chỉnh hộp bao bằng hồi quy. Dù tốn nhiều chi phí tính toán, R-CNN đạt bước tiến lớn so với các phương pháp đặc trưng thủ công, khẳng định hiệu quả của học chuyển giao từ ImageNet sang phát hiện đối tượng và mở đường cho họ mô hình phát hiện dựa trên vùng.

**["Fast R-CNN" (2015)](https://arxiv.org/abs/1504.08083)**  
*Tác giả*: Ross Girshick  
Fast R-CNN khắc phục nút thắt tốc độ của R-CNN bằng cách chia sẻ tính toán tích chập giữa các vùng đề xuất thông qua lớp *RoI (Region of Interest) pooling*. Ảnh chỉ cần đi qua CNN một lần để tạo bản đồ đặc trưng dùng chung, sau đó RoI pooling trích ra vector đặc trưng kích thước cố định cho từng đề xuất. Nhờ vậy, số lượt lan truyền xuôi giảm từ 2000 xuống còn 1 (nhanh hơn khoảng 10 lần), đồng thời độ chính xác tăng nhờ huấn luyện chung bộ trích đặc trưng, phân loại và hồi quy hộp bao với hàm mất mát đa tác vụ.

**["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2016)](https://arxiv.org/abs/1506.01497)**  
*Tác giả*: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun  
Faster R-CNN thay thế *selective search* bằng Mạng đề xuất vùng (RPN), giúp toàn bộ hệ thống khả vi và huấn luyện đầu–cuối. RPN dùng các lớp tích chập để dự đoán điểm *objectness* và độ lệch hộp bao tại mọi vị trí trên bản đồ đặc trưng, đồng thời chia sẻ đặc trưng với phần phát hiện, đạt tốc độ gần thời gian thực (5 FPS). Cơ chế hộp neo được giới thiệu trong bài báo này trở thành chuẩn mực cho nhiều bộ phát hiện sau đó.

**["You Only Look Once: Unified, Real-Time Object Detection" (2016)](https://arxiv.org/abs/1506.02640)**  
*Tác giả*: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi  
YOLO đặt bài toán phát hiện đối tượng thành bài toán hồi quy trực tiếp từ pixel ảnh sang tọa độ hộp bao và xác suất lớp trong một lượt lan truyền xuôi duy nhất, đạt tốc độ thời gian thực (45 FPS). Bằng cách chia ảnh thành lưới và để mỗi ô dự đoán trực tiếp, YOLO loại bỏ hoàn toàn giai đoạn đề xuất vùng. Tuy độ chính xác ban đầu thấp hơn Faster R-CNN (nhất là với đối tượng nhỏ), tốc độ của YOLO mở ra khả năng ứng dụng trong xe tự lái và robot, đặt nền móng cho dòng bộ phát hiện một giai đoạn.

**["Feature Pyramid Networks for Object Detection" (2017)](https://arxiv.org/abs/1612.03144)**  
*Tác giả*: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie  
FPN giải quyết bài toán phát hiện đa tỷ lệ bằng cách xây dựng kim tự tháp đặc trưng giàu ngữ nghĩa ở mọi độ phân giải, kết hợp đặc trưng tầng thấp (độ phân giải không gian cao nhưng ngữ nghĩa yếu) với đặc trưng tầng cao (độ phân giải thấp nhưng ngữ nghĩa mạnh) qua nhánh từ trên xuống (*top-down*) và các kết nối ngang. Nhờ đó, mô hình phát hiện tốt cả đối tượng lớn lẫn đối tượng nhỏ. Mẫu kiến trúc kết hợp hai chiều *bottom-up* và *top-down* này cũng được áp dụng rộng rãi cho phân đoạn ảnh và các tác vụ dự đoán dày đặc khác.

## Bẫy thường gặp và mẹo thực hành

Thiết kế hộp neo ảnh hưởng đáng kể đến độ chính xác nhưng thường bị bỏ qua. Hộp neo cần phản ánh tỷ lệ khung hình và kích thước đặc trưng của đối tượng trong tập dữ liệu: chẳng hạn, với người đi bộ (cao, hẹp) nên dùng tỷ lệ 1:3 hoặc 1:4; với ô tô (rộng hơn) nên dùng 2:1 hoặc 3:2. Có thể dùng phân cụm *k-means* trên các hộp bao trong tập huấn luyện để chọn kích thước hộp neo phù hợp. Quá nhiều hộp neo gây tốn tính toán mà không tăng độ chính xác, còn quá ít dễ bỏ sót các dạng đối tượng quan trọng. Thông thường, YOLO và Faster R-CNN dùng 9 hộp neo (3 tỷ lệ kích thước × 3 tỷ lệ khung hình) tại mỗi vị trí.

Việc chọn ngưỡng NMS quyết định sự đánh đổi giữa *precision* và *recall*. Ngưỡng IoU thấp (ví dụ 0.3) loại bỏ nhiều hộp chồng lấn hơn, giảm dự đoán trùng lặp nhưng có thể vô tình xóa nhầm đối tượng nằm sát nhau. Ngưỡng cao (ví dụ 0.7) giữ lại nhiều hộp hơn, phù hợp với cảnh đông đúc nhưng dễ để lọt dự đoán trùng. Vì vậy, nên chọn ngưỡng cao hơn cho cảnh đông và ngưỡng thấp hơn cho cảnh thưa.

Mất cân bằng lớp là vấn đề lớn trong phát hiện đối tượng: phần lớn hộp neo rơi vào nền, tạo tỷ lệ chênh lệch giữa nền và đối tượng thường từ 1000:1 trở lên. Nếu không xử lý, mô hình dễ rơi vào lời giải tầm thường là dự đoán toàn bộ là nền (đạt độ chính xác tổng thể tới 99.9% nhưng vô dụng). Các kỹ thuật khắc phục phổ biến gồm *hard negative mining* (tập trung huấn luyện trên các mẫu nền khó, bỏ qua mẫu dễ), *focal loss* (giảm trọng số của các mẫu phân loại dễ) và lấy mẫu cân bằng giữa mẫu đối tượng và mẫu nền trong mỗi batch.

## Điểm chính cần nhớ

Phát hiện đối tượng mở rộng phân loại ảnh sang định vị và nhận dạng nhiều đối tượng trong cùng một ảnh, kết hợp đề xuất vùng, phân loại và hồi quy hộp bao. Bộ phát hiện hai giai đoạn tách bước sinh đề xuất (qua RPN) khỏi bước phân loại và tinh chỉnh, ưu tiên độ chính xác nhờ tập trung tính toán vào vùng tiềm năng. Bộ phát hiện một giai đoạn dự đoán trực tiếp trên lưới trong một lượt lan truyền xuôi, ưu tiên tốc độ thời gian thực với mức giảm nhẹ về độ chính xác. IoU đánh giá độ chính xác định vị; một phát hiện thường được tính là đúng khi IoU đạt ngưỡng (phổ biến là 0.5) và đúng lớp. Hộp neo cho phép mạng dự đoán độ lệch tương đối thay vì tọa độ tuyệt đối, giúp huấn luyện ổn định hơn. Kim tự tháp đặc trưng (FPN) hỗ trợ phát hiện đa tỷ lệ bằng cách kết hợp thông tin ngữ nghĩa tầng cao với độ phân giải không gian tầng thấp. NMS loại bỏ các hộp trùng lặp bằng cách giữ hộp có độ tin cậy cao nhất và ức chế các hộp chồng lấn, trong khi hàm mất mát đa tác vụ cân bằng đồng thời giữa phân loại và định vị.

## Video tham khảo

- **Xem trước** [How YOLO Object Detection Works](https://www.youtube.com/watch?v=svn9-xV7wjk) — DeepBean, tiếng Anh. Giúp đối chiếu cấu trúc tensor trong đoạn mã cài đặt với đầu ra của một bộ phát hiện thực tế.
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — Stanford University School of Engineering, tiếng Anh. Giải thích các thành phần mất mát (phân loại, hộp bao, *objectness*) cần cộng gộp khi cài đặt.
- [PyTorch Tutorial 14 - Convolutional Neural Network (CNN)](https://www.youtube.com/watch?v=pDdP0TFzsoQ) — Patrick Loeber, tiếng Anh. Hướng dẫn cài đặt mạng xương sống tích chập trong PyTorch trước khi gắn đầu phát hiện.
