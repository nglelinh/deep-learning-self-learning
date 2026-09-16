---
layout: post
title: 04-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '04'
order: 14
owner: Deep Learning Course
lang: vi
categories:
- chapter04
lesson_type: optional
---

# Tùy chọn: ConvNet sau ConvNeXt và mô hình nền tảng thị giác

> Bài này là **tùy chọn**. Nó **không** viết lại toán tích chập, pooling, hay lịch sử LeNet–ResNet. Nó cập nhật những gì người thực hành thực sự triển khai cho thị giác sau 2022.

Tích chập $$(\mathbf{x} * \mathbf{k})_{i} = \sum_{u} x_{i+u}\,k_{u}$$ không đổi. Cái đổi là **kiến trúc vĩ mô** (ConvNeXt) và **cách dùng** (backbone tích chập hoặc lai, đóng băng, trong foundation model).

## 1. ConvNeXt: ConvNet mượn công thức huấn luyện Transformer

[Liu et al., 2022](https://arxiv.org/abs/2201.03545) (“A ConvNet for the 2020s”) cho thấy stack kiểu ResNet, huấn luyện theo công thức ViT (AdamW, Mixup, stochastic depth, crop lớn), bắt kịp Transformer phân cấp trên ImageNet. [ConvNeXt V2](https://arxiv.org/abs/2301.00808) (2023) thêm global response normalization (GRN) và giai đoạn pretrain autoencoder che masked, thuần tích chập.

Thông điệp cho chương này: **depthwise convolution + inverted bottleneck** vẫn là inductive bias; khoảng cách với ViT những năm 2020 chủ yếu là *cách huấn luyện*, không phải “tích chập đã lỗi thời.”

## 2. Ứng dụng cụ thể

### Foundation model thị giác vẫn mang DNA tích chập

- **DINOv2** ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)) — đặc trưng ViT tự giám sát dùng như backbone cho retrieval, depth, segmentation. Biến thể ConvNeXt nằm cùng bảng benchmark.
- **SAM** ([Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)) và **SAM 2** ([Ravi et al., 2024](https://arxiv.org/abs/2408.00714)) — phân đoạn theo prompt; encoder ảnh là ViT, nhưng pipeline production vẫn ghép detector tích chập và cổ CNN nhẹ.
- Y tế và viễn thám vẫn ưa **decoder tích chập** kiểu U-Net trên encoder vừa thắng năm trước.

### Khi CNN vẫn là mặc định đúng

Dự đoán dày ở độ phân giải cao, latency nghiêm trên NPU di động, và dữ liệu nhỏ thường giữ EfficientNetV2 / ConvNeXt-Tiny / cổ YOLO. Transformer thắng khi pretrain được ở quy mô web.

## 3. Phần mềm phổ biến

- [timm](https://github.com/huggingface/pytorch-image-models) — ConvNeXt, EfficientNetV2, công thức pretrained.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — detector tích chập (YOLOv8/v11) trong công nghiệp.
- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) và [facebookresearch/sam2](https://github.com/facebookresearch/sam2).

## 4. Trích dẫn (2022–2026)

- [A ConvNet for the 2020s (Liu et al., 2022)](https://arxiv.org/abs/2201.03545) — ConvNeXt.
- [ConvNeXt V2 (Woo et al., 2023)](https://arxiv.org/abs/2301.00808) — GRN + pretrain FCMAE.
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) — đặc trưng thị giác tự giám sát quy mô web.
- [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643) — foundation model phân đoạn theo prompt.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Giữ worksheet chiều tích chập. Bài này chỉ trả lời “thực tế cái gì thay VGG/ResNet sau 2022”: ConvNet hiện đại cộng foundation model thị giác — không phải định nghĩa mới của bộ lọc.
