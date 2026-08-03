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
Cài đặt kiểu Faster R-CNN đầy đủ:

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

Cài đặt bộ phát hiện một giai đoạn kiểu YOLO:

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
Phát hiện đối tượng gắn với phân loại ảnh thông qua nền tảng trích xuất đặc trưng tích chập. Các mạng xương sống (*backbone* — ResNet, VGG, MobileNet) thường là mạng phân loại được thích ứng cho phát hiện bằng cách bỏ các tầng phân loại cuối và thêm đầu phát hiện. Các đặc trưng học cho phân loại — cạnh, kết cấu, bộ phận đối tượng — chuyển giao tự nhiên sang phát hiện vì nhận ra “đây là xe” (phân loại) và “có xe tại vị trí này” (phát hiện) đều đòi hỏi hiểu ngoại hình xe. Tuy nhiên, phát hiện cần thêm khả năng: định vị chính xác nơi đối tượng nằm, xử lý nhiều đối tượng và nhiều tỷ lệ, phân biệt đối tượng với nền. Hiểu mối liên hệ này giúp thấy vì sao bộ phân loại tiền huấn luyện trên ImageNet là điểm khởi đầu tốt cho phát hiện nhưng cần mở rộng kiến trúc (FPN cho đa tỷ lệ, RPN cho đề xuất) để đạt đủ năng lực phát hiện.

Mối quan hệ với phân đoạn ngữ nghĩa (*semantic segmentation*) làm sáng tỏ các mức độ hiểu hình ảnh khác nhau. Phân loại gán một nhãn cho mỗi ảnh. Phát hiện gán nhãn và hộp cho nhiều đối tượng mỗi ảnh. Phân đoạn ngữ nghĩa gán nhãn cho mọi pixel, phác đúng biên đối tượng. Phân đoạn thể hiện (*instance segmentation*) kết hợp phát hiện và phân đoạn, cung cấp mặt nạ pixel-perfect cho từng thể hiện đối tượng. Lộ trình từ thô (cấp ảnh) đến mịn (cấp pixel) phản ánh yêu cầu ứng dụng và đánh đổi tính toán khác nhau. Phát hiện mang lại sự cân bằng: giàu thông tin hơn phân loại (đối tượng ở đâu?) mà không tốn chi phí phân đoạn cấp pixel.

Mối liên hệ của phát hiện đối tượng với cơ chế attention ngày càng quan trọng trong kiến trúc hiện đại. Transformer đang thay thế các đầu phát hiện truyền thống qua DETR (*Detection Transformer*), coi phát hiện đối tượng như dự đoán tập (*set prediction*) dùng attention để dự đoán song song mọi đối tượng mà không cần neo hay NMS. Cơ chế attention học chú ý đến vị trí và phạm vi đối tượng, mang lại giải pháp huấn luyện đầu–cuối thanh lịch thay cho đường ống phức tạp của bộ phát hiện cổ điển. Hiểu cách attention có thể thay các thành phần thủ công như neo và NMS giúp đánh giá tính tổng quát của Transformer vượt ngoài NLP.

Sự tiến hóa từ R-CNN đến Fast R-CNN rồi Faster R-CNN minh họa tối ưu hóa có hệ thống các nút thắt tính toán. R-CNN chạy trích xuất đặc trưng CNN riêng cho mỗi đề xuất (2000 lượt forward mỗi ảnh — rất chậm). Fast R-CNN trích đặc trưng một lần cho toàn ảnh, rồi dùng RoI pooling lấy đặc trưng đề xuất (một lượt forward — nhanh hơn nhiều). Faster R-CNN đưa việc sinh đề xuất vào mạng qua RPN (khả vi đầy đủ, huấn luyện đầu–cuối). Mỗi đổi mới giải một điểm kém hiệu quả cụ thể trong khi giữ độ chính xác, cho thấy hệ thống tiến hóa qua cải tiến có mục tiêu chứ không phải thiết kế lại toàn bộ.

## 6. Các Bài báo Nền tảng

**["Rich feature hierarchies for accurate object detection and semantic segmentation" (2014)](https://arxiv.org/abs/1311.2524)**  
*Tác giả*: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik  
R-CNN cách mạng hóa phát hiện đối tượng bằng cách áp dụng CNN — vốn đã thành công trong phân loại — vào phát hiện thông qua đề xuất vùng. Cách tiếp cận về khái niệm đơn giản: dùng selective search đề xuất ~2000 vùng mỗi ảnh, trích đặc trưng CNN từ mỗi vùng (forward qua AlexNet), rồi phân loại bằng SVM và tinh chỉnh hộp bằng hồi quy. Dù tốn tính toán (2000 lượt forward mỗi ảnh), R-CNN đạt cải thiện lớn so với phương pháp truyền thống, chứng tỏ đặc trưng học được vượt trội đặc trưng thủ công cho phát hiện. Bài báo thiết lập mô hình phát hiện dựa trên vùng và cho thấy học chuyển giao (tiền huấn luyện ImageNet cho phát hiện) rất hiệu quả. Thành công của R-CNN khơi mào cuộc cách mạng học sâu trong phát hiện đối tượng, dẫn đến nhiều cải tiến giải quyết hạn chế tính toán trong khi giữ nguyên ý tưởng cốt lõi: phát hiện có thể giải qua phân loại vùng với đặc trưng học được.

**["Fast R-CNN" (2015)](https://arxiv.org/abs/1504.08083)**  
*Tác giả*: Ross Girshick  
Fast R-CNN giải quyết nút thắt tính toán của R-CNN bằng cách chia sẻ tính toán CNN giữa các đề xuất thông qua RoI (*Region of Interest*) pooling. Thay vì chạy CNN riêng cho mỗi đề xuất, trích đặc trưng một lần cho toàn ảnh, rồi dùng RoI pooling lấy vectơ đặc trưng kích thước cố định cho mỗi đề xuất từ bản đồ đặc trưng dùng chung. Điều này giảm lượt forward từ 2000 mỗi ảnh xuống 1, đạt tăng tốc ~10× đồng thời cải thiện độ chính xác nhờ huấn luyện chung trích đặc trưng, phân loại và hồi quy hộp. Bài báo giới thiệu mất đa tác vụ kết hợp phân loại và định vị, cho thấy huấn luyện chung cải thiện cả hai so với huấn luyện riêng. Fast R-CNN chứng minh rằng phân tích có hệ thống các nút thắt tính toán và đổi mới kiến trúc khéo léo có thể cải thiện mạnh hiệu quả mà không hy sinh độ chính xác.

**["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2016)](https://arxiv.org/abs/1506.01497)**  
*Tác giả*: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun  
Faster R-CNN hoàn tất tiến hóa sang phát hiện khả vi đầy đủ bằng cách thay selective search bằng Mạng Đề xuất Vùng, khiến toàn bộ đường ống huấn luyện được đầu–cuối. RPN dùng bộ lọc tích chập học được để dự đoán objectness và tọa độ hộp tại mọi vị trí trên bản đồ đặc trưng, sinh đề xuất qua cơ chế học thay vì thuật toán thủ công. Đổi mới này cho phép chia sẻ tính toán giữa sinh đề xuất và phát hiện, cải thiện chất lượng đề xuất qua huấn luyện có giám sát, và đạt tốc độ gần thời gian thực (5 FPS). Cơ chế hộp neo — dự đoán độ lệch so với hộp định sẵn ở các tỷ lệ khung hình và kích thước khác nhau — trở thành chuẩn trong các bộ phát hiện sau này. Faster R-CNN định hình mẫu cho phát hiện hai giai đoạn và duy trì vị thế tiên tiến trong nhiều năm.

**["You Only Look Once: Unified, Real-Time Object Detection" (2016)](https://arxiv.org/abs/1506.02640)**  
*Tác giả*: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi  
YOLO thay đổi căn bản phát hiện đối tượng bằng cách đặt bài toán như hồi quy từ pixel ảnh trực tiếp ra tọa độ hộp bao và xác suất lớp, cho phép phát hiện thời gian thực (45 FPS) qua một lượt forward duy nhất. Bằng cách chia ảnh thành lưới và để mỗi ô dự đoán hộp, YOLO loại bỏ đề xuất vùng cùng chi phí tính toán liên quan. Dù độ chính xác ban đầu thấp hơn Faster R-CNN (đặc biệt với đối tượng nhỏ), tốc độ của YOLO mở ra các ứng dụng thời gian thực như xe tự lái và robot. Bài báo cho thấy phát hiện không nhất thiết theo mô hình hai giai đoạn, truyền cảm hứng cho nhiều bộ phát hiện một giai đoạn. Triết lý thiết kế đầu–cuối của YOLO — dự đoán mọi thứ trong một lần — chứng minh rằng cách tiếp cận đơn giản, thống nhất có thể cạnh tranh với đường ống phức tạp khi được thiết kế đúng.

**["Feature Pyramid Networks for Object Detection" (2017)](https://arxiv.org/abs/1612.03144)**  
*Tác giả*: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie  
FPN giải quyết phát hiện đa tỷ lệ bằng cách xây kim tự tháp đặc trưng giàu ngữ nghĩa ở mọi tỷ lệ, kết hợp đặc trưng cấp thấp độ phân giải cao nhưng ngữ nghĩa yếu với đặc trưng cấp cao độ phân giải thấp nhưng ngữ nghĩa mạnh thông qua đường top-down và kết nối ngang. Điều này cho phép phát hiện đối tượng lớn bằng đặc trưng cấp cao và đối tượng nhỏ bằng đặc trưng cấp thấp đã được làm giàu ngữ nghĩa từ tầng trên. FPN cải thiện mạnh phát hiện đối tượng ở nhiều tỷ lệ, đặc biệt đối tượng nhỏ mà phương pháp trước gặp khó. Mẫu kiến trúc — xây kim tự tháp với cả đường bottom-up (CNN chuẩn) và top-down — được áp dụng rộng rãi ngoài phát hiện sang phân đoạn và các tác vụ dự đoán dày. FPN chứng minh rằng thiết kế kiến trúc đa tỷ lệ cẩn thận giải quyết các thách thức cơ bản trong nhận dạng hình ảnh.

## Bẫy Thường gặp và Mẹo

Thiết kế hộp neo ảnh hưởng mạnh hiệu năng phát hiện nhưng thường bị bỏ qua. Neo cần khớp tỷ lệ khung hình và kích thước điển hình của đối tượng trong tập dữ liệu. Với phát hiện người đi bộ (đối tượng cao, hẹp), dùng neo tỷ lệ 1:3 và 1:4. Với xe (rộng hơn), dùng 2:1 hoặc 3:2. Phân cụm k-means trên hộp bao tập huấn luyện có thể tự động tìm kích thước neo tốt. Quá nhiều neo lãng phí tính toán mà không cải thiện độ chính xác; quá ít thì bỏ sót loại đối tượng quan trọng. YOLO điển hình dùng 9 neo (3 tỷ lệ × 3 khung hình); Faster R-CNN dùng 9 (3 tỷ lệ × 3 khung) mỗi vị trí.

Chọn ngưỡng NMS liên quan đánh đổi precision–recall. Ngưỡng IoU thấp (0.3) ức chế nhiều hộp hơn, giảm trùng lặp nhưng có thể loại bỏ phát hiện hợp lệ của đối tượng gần nhau. Ngưỡng cao (0.7) giữ nhiều hộp hơn, phát hiện tốt đối tượng gần nhau nhưng sinh trùng lặp. Với cảnh đông (nhiều đối tượng gần), dùng ngưỡng cao. Với cảnh thưa, dùng ngưỡng thấp. Hiểu rằng ngưỡng NMS điều khiển đánh đổi này cho phép tinh chỉnh theo ứng dụng cụ thể.

Mất cân bằng lớp trong phát hiện đối tượng rất nghiêm trọng và cần xử lý cẩn thận. Hầu hết hộp neo là nền (không có đối tượng), tạo mất cân bằng cực đoan giữa nền và lớp đối tượng (thường 1000:1 hoặc hơn). Nếu không xử lý, bộ phát hiện học dự đoán nền cho mọi thứ (lời giải tầm thường đạt 99.9% độ chính xác). Các giải pháp gồm hard negative mining (huấn luyện trên mẫu nền khó, bỏ qua mẫu dễ), focal loss (cân trọng mất mát theo độ khó, giảm trọng các phân loại dễ), và lấy mẫu cân bằng (đảm bảo batch chứa số lượng tương đương mẫu đối tượng và nền).

## Điểm Chính Cần Nhớ

Phát hiện đối tượng mở rộng phân loại sang định vị và nhận dạng nhiều đối tượng mỗi ảnh, đòi hỏi đồng thời đề xuất vùng, phân loại và hồi quy hộp bao. Bộ phát hiện hai giai đoạn tách sinh đề xuất khỏi phát hiện, dùng RPN sinh ứng viên và đầu phát hiện phân loại/tinh chỉnh, đạt độ chính xác cao nhờ tính toán tập trung trên vùng đối tượng. Bộ phát hiện một giai đoạn dự đoán trực tiếp hộp và lớp từ ô lưới, đạt hiệu năng thời gian thực qua một lượt forward với đánh đổi độ chính xác nhẹ. IoU đo chất lượng định vị; phát hiện được coi là đúng khi IoU với nhãn gốc vượt ngưỡng (thường 0.5) và lớp khớp. Hộp neo cung cấp hộp tham chiếu ở nhiều tỷ lệ và khung hình, mạng dự đoán độ lệch thay vì tọa độ tuyệt đối, cải thiện ổn định huấn luyện. Kim tự tháp đặc trưng cho phép phát hiện đa tỷ lệ bằng cách kết hợp đặc trưng ngữ nghĩa cấp cao với đặc trưng không gian độ phân giải cao. NMS loại bỏ phát hiện trùng bằng cách giữ hộp tin cậy cao nhất và ức chế các hộp chồng lấn. Huấn luyện đa tác vụ tối ưu đồng thời phân loại và định vị, với mất mát được cân bằng để vừa nhận dạng đúng vừa định vị chính xác. Hiểu phát hiện đối tượng đòi hỏi nắm cách các thành phần phối hợp xử lý số lượng đối tượng biến thiên ở nhiều tỷ lệ, vị trí và lớp, đồng thời duy trì hiệu năng thời gian thực hoặc gần thời gian thực cho ứng dụng thực tế.
