---
layout: post
title: 17-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '17'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter17
lesson_type: optional
---

# Tùy chọn: phát hiện và phân đoạn sau YOLO / Mask R-CNN

> Bài này là **tùy chọn**. Nó **không** thay lý thuyết detector một/hai giai đoạn hay ghi chú YOLO / Faster R-CNN. Nó phủ các mô hình 2023–2025 trong sản phẩm: **SAM/SAM 2**, **RT-DETR**, **Grounding DINO**, và dòng YOLO hiện tại.

Ý IoU / NMS / ghép cặp của chương vẫn quyết định hộp có đúng không. Kiến trúc quanh chúng đã đổi.

## 1. Phân đoạn theo prompt: SAM và SAM 2

[Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) (Segment Anything) coi phân đoạn là tác vụ nền tảng **theo prompt**: điểm, hộp, hoặc mask vào; mask instance ra. [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) (SAM 2) thêm bộ nhớ luồng cho **video**. Dùng trong DS: gắn vài điểm, xuất mask, huấn luyện chuyên gia nhỏ — không thay U-Net khi đã có tập y tế gắn nhãn pixel.

## 2. Ứng dụng cụ thể

### Detector DETR thời gian thực

[Zhao et al., 2024](https://arxiv.org/abs/2304.08069) (RT-DETR) khiến detector Transformer đủ nhanh để thay một số triển khai YOLO. Vẫn đánh giá mAP như bài lý thuyết.

### Từ vựng mở: Grounding DINO + SAM

[Liu et al., 2023](https://arxiv.org/abs/2303.05499) (Grounding DINO) nhận truy vấn **văn bản** (“mũ bảo hiểm đỏ”) và trả hộp. Ghép SAM (“Grounded SAM”) là mẫu gắn nhãn và tri giác robot 2024–2026. Đầu COCO tập đóng vẫn cho SKU nhà máy.

### YOLO trong production

Ultralytics **YOLOv8 / YOLO11** giữ công thức một giai đoạn tích chập. Dùng khi latency và xuất (ONNX, TensorRT, CoreML) quan trọng hơn từ vựng mở.

## 3. Phần mềm phổ biến

- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) và Segment Anything.
- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) và [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR).

## 4. Trích dẫn (2022–2026)

- [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643).
- [SAM 2 (Ravi et al., 2024)](https://arxiv.org/abs/2408.00714).
- [Grounding DINO (Liu et al., 2023)](https://arxiv.org/abs/2303.05499).
- [RT-DETR (Zhao et al., 2024)](https://arxiv.org/abs/2304.08069).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Cài ghép cặp detector và đường R-CNN/YOLO của chương trước. Bài này chỉ thêm đầu **theo prompt và từ vựng mở** dùng trong pipeline 2023–2026.
