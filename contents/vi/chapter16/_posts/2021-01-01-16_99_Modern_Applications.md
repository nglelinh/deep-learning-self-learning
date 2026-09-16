---
layout: post
title: 16-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '16'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter16
lesson_type: optional
---

# Tùy chọn: học tự giám sát sau SimCLR — DINOv2, SigLIP, JEPA

> Bài này là **tùy chọn**. Nó **không** thay lý thuyết contrastive / MLM hay phác thảo SimCLR đã có trong chương. Nó cập nhật các phương pháp SSL **thị giác và đa phương thức** thành mặc định sau 2022.

Chương đã phủ SimCLR, CLIP, và MAE. Stack production 2023–2025 chuyển sang **giáo viên mạnh hơn**, **loss sigmoid ngôn ngữ–ảnh**, và mục tiêu **dự đoán trong không gian ẩn** (JEPA).

## 1. DINOv2: backbone đóng băng dùng được ngay

[Oquab et al., 2023](https://arxiv.org/abs/2304.07193) kết hợp student–teacher kiểu DINO/iBOT với pipeline dữ liệu chọn lọc lớn. Đặc trưng ViT-g chuyển tới retrieval, depth, segmentation **không** fine-tune theo tác vụ trong nhiều demo. Đó là SSL như **encoder nền tảng**, không phải bài pretext.

## 2. Ứng dụng cụ thể

### SigLIP / SigLIP 2

[Zhai et al., 2023](https://arxiv.org/abs/2303.15343) thay softmax của CLIP (cần batch âm lớn) bằng loss **sigmoid** từng cặp

$$\mathcal{L} = -\frac{1}{B}\sum_{i,j}\log\sigma\big(y_{ij}\, t\, x_i^\top y_j\big),$$

trong đó $$y_{ij}=\pm 1$$ đánh dấu cặp khớp. Stack VLM mở (PaliGemma, nhiều VLM 2024–2025) bắt đầu từ tháp thị giác SigLIP.

### I-JEPA và dự đoán trong biểu diễn

[Assran et al., 2023](https://arxiv.org/abs/2301.08243) (I-JEPA) dự đoán **embedding** patch bị che, không phải pixel (khác MAE trong bài tùy chọn Chương 12). Dòng JEPA của LeCun là cược nghiên cứu: SSL nên mô hình động lực ẩn, không giải mã RGB.

### Cái gì còn lại

MLM và ghép cặp kiểu CLIP vẫn là trụ NLP/đa phương thức. SimCLR vẫn là suy diễn contrastive đầu đúng.

## 3. Phần mềm phổ biến

- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
- Trọng số SigLIP OpenCLIP trên Hub.
- [facebookresearch/jepa](https://github.com/facebookresearch/jepa).

## 4. Trích dẫn (2022–2026)

- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193).
- [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343).
- [I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Giữ suy diễn NT-Xent và công thức MLM của BERT. Bài này chỉ đặt tên **checkpoint 2023–2025** (DINOv2, SigLIP, JEPA) bạn sẽ tải thay vì huấn luyện SimCLR từ đầu.
