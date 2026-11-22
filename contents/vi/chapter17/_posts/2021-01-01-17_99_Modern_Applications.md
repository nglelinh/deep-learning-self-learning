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

# Tùy chọn: Phát hiện và phân đoạn sau YOLO / Mask R-CNN

> Bài này là **tùy chọn**, **không** thay thế phần lý thuyết về bộ phát hiện một/hai giai đoạn hay các ghi chú về YOLO và Faster R-CNN. Nội dung bổ sung các mô hình được sử dụng rộng rãi trong thực tế giai đoạn 2023–2025: **SAM/SAM 2**, **RT-DETR**, **Grounding DINO** và các phiên bản YOLO mới.

Các khái niệm cốt lõi của chương như IoU, NMS và cơ chế ghép cặp dự đoán vẫn là tiêu chuẩn quyết định một hộp bao có chính xác hay không:

$$\mathrm{IoU}(A,B) = \frac{|A\cap B|}{|A\cup B|}.$$

Tuy nhiên, các kiến trúc xoay quanh những chỉ số này đã thay đổi đáng kể.

## 1. Phân đoạn theo prompt: SAM và SAM 2

[Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) (Segment Anything) mô hình hóa bài toán phân đoạn như một tác vụ nền tảng **theo prompt**: đầu vào là điểm, hộp bao hoặc mặt nạ sơ bộ, đầu ra là mặt nạ của từng thể hiện đối tượng. [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) (SAM 2) bổ sung bộ nhớ theo luồng thời gian để xử lý **video**. Trong quy trình khoa học dữ liệu thực tế, các mô hình này thường dùng để hỗ trợ gán nhãn nhanh (chấm vài điểm, xuất mặt nạ rồi huấn luyện một mô hình chuyên biệt nhỏ hơn), chứ không thay thế U-Net khi đã có tập dữ liệu y tế được gán nhãn chi tiết ở mức pixel.

## 2. Ứng dụng cụ thể

### Bộ phát hiện DETR thời gian thực

[Zhao et al., 2024](https://arxiv.org/abs/2304.08069) (RT-DETR) cải thiện tốc độ cho dòng bộ phát hiện dựa trên Transformer đến mức có thể thay thế YOLO trong một số hệ thống thực tế, trong khi vẫn đánh giá bằng chỉ số mAP như phần lý thuyết đã trình bày.

### Phát hiện từ vựng mở: Grounding DINO + SAM

[Liu et al., 2023](https://arxiv.org/abs/2303.05499) (Grounding DINO) nhận truy vấn **văn bản** (ví dụ “mũ bảo hiểm đỏ”) và trả về hộp bao tương ứng. Kết hợp Grounding DINO với SAM (“Grounded SAM”) là mẫu quy trình phổ biến trong gán nhãn dữ liệu và tri giác robot giai đoạn 2024–2026. Dù vậy, với các bài toán có tập nhãn cố định như kiểm đếm mã sản phẩm (SKU) trong nhà máy, bộ phát hiện tập đóng kiểu COCO vẫn là lựa chọn phù hợp.

### YOLO trong triển khai thực tế

Các phiên bản Ultralytics **YOLOv8 / YOLO11** vẫn giữ thiết kế tích chập một giai đoạn, phù hợp khi yêu cầu về độ trễ thấp và khả năng xuất sang các định dạng triển khai (ONNX, TensorRT, CoreML) quan trọng hơn khả năng nhận dạng từ vựng mở.

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

Trước hết cần nắm vững cơ chế ghép cặp dự đoán và quy trình R-CNN/YOLO trong các bài cốt lõi. Bài này bổ sung các đầu phát hiện và phân đoạn **theo prompt hoặc từ vựng mở** được dùng phổ biến trong các hệ thống giai đoạn 2023–2026.

## Video tham khảo

- **Xem trước** [Explaining the Segment Anything Model - Network architecture, Dataset, Training](https://www.youtube.com/watch?v=OhxJkqD1vuE) — Neural Breakdown with AVB, tiếng Anh. Giải thích kiến trúc, dữ liệu và cách huấn luyện của Segment Anything — mô hình phân đoạn theo prompt trong bài.
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — Stanford University School of Engineering, tiếng Anh. Ôn lại bài toán phát hiện và phân đoạn cổ điển mà SAM tổng quát hóa.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, tiếng Anh. Nền tảng về mạng xương sống thị giác bên dưới các mô hình phân đoạn theo prompt.
