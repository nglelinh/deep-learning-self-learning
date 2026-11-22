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

# Tùy chọn: Học tự giám sát sau SimCLR — DINOv2, SigLIP và JEPA

> Bài này là **tùy chọn**, **không** thay thế phần lý thuyết học đối chiếu / MLM hay bản phác thảo SimCLR trong chương. Nội dung cập nhật các phương pháp học tự giám sát (SSL) cho **thị giác và đa phương thức** được dùng rộng rãi từ sau năm 2022.

Các bài trước trong chương đã giới thiệu SimCLR, CLIP và MAE. Trong giai đoạn 2023–2025, các hệ thống thực tế chuyển dần sang **mô hình thầy–trò mạnh hơn**, **hàm mất mát sigmoid cho cặp ảnh–văn bản**, và mục tiêu **dự đoán trong không gian biểu diễn ẩn** (JEPA).

## 1. DINOv2: Mạng xương sống đóng băng dùng trực tiếp

[Oquab et al., 2023](https://arxiv.org/abs/2304.07193) kết hợp cơ chế thầy–trò (*student–teacher*) kiểu DINO/iBOT với quy trình chọn lọc dữ liệu quy mô lớn. Đặc trưng từ ViT-g có thể dùng trực tiếp cho truy xuất ảnh, ước lượng độ sâu và phân đoạn mà **không** cần tinh chỉnh theo từng tác vụ trong nhiều bài toán thử nghiệm. Cách tiếp cận này xem học tự giám sát như phương pháp xây dựng **bộ mã hóa nền tảng**, thay vì chỉ dừng ở một tác vụ tiền huấn luyện đơn lẻ.

## 2. Ứng dụng cụ thể

### SigLIP và SigLIP 2

[Zhai et al., 2023](https://arxiv.org/abs/2303.15343) thay hàm mất mát softmax của CLIP (vốn đòi hỏi batch mẫu âm lớn) bằng hàm mất mát **sigmoid** tính trên từng cặp:

$$\mathcal{L} = -\frac{1}{B}\sum_{i,j}\log\sigma\big(y_{ij}\, t\, x_i^\top y_j\big),$$

trong đó $$y_{ij}=\pm 1$$ cho biết cặp $$(i, j)$$ có khớp nhau hay không. Nhiều mô hình thị giác–ngôn ngữ mở (như PaliGemma và các VLM giai đoạn 2024–2025) sử dụng nhánh mã hóa thị giác SigLIP làm thành phần khởi đầu.

### I-JEPA và dự đoán trong không gian biểu diễn

[Assran et al., 2023](https://arxiv.org/abs/2301.08243) (I-JEPA) dự đoán **vector nhúng** (*embedding*) của các mảng ảnh (*patch*) bị che thay vì tái tạo trực tiếp giá trị pixel như MAE (đã nhắc ở bài tùy chọn Chương 12). Hướng nghiên cứu JEPA của Yann LeCun đặt giả thuyết rằng học tự giám sát nên mô hình hóa cấu trúc và biến đổi trong không gian ẩn thay vì giải mã chi tiết từng pixel RGB.

### Những nền tảng vẫn giữ nguyên giá trị

MLM và cơ chế ghép cặp ảnh–văn bản kiểu CLIP vẫn là nền tảng trong NLP và mô hình đa phương thức, còn SimCLR vẫn là bước dẫn nhập cơ bản để hiểu cơ chế học đối chiếu.

## 3. Phần mềm phổ biến

- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
- Trọng số SigLIP trong OpenCLIP trên Hugging Face Hub.
- [facebookresearch/jepa](https://github.com/facebookresearch/jepa).

## 4. Trích dẫn (2022–2026)

- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193).
- [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343).
- [I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Phần suy diễn NT-Xent và công thức MLM của BERT trong các bài cốt lõi vẫn giữ nguyên giá trị. Bài này bổ sung các **mô hình tiền huấn luyện giai đoạn 2023–2025** (DINOv2, SigLIP, JEPA) thường được tải về sử dụng trực tiếp thay vì huấn luyện SimCLR từ đầu.

## Video tham khảo

- **Xem trước** [DINOv2 from Meta AI - Finally a Foundational Model in Computer Vision?](https://www.youtube.com/watch?v=csEgtSh7jV4) — AI Papers Academy, tiếng Anh. Giới thiệu DINOv2, mạng xương sống thị giác đóng băng có thể dùng cho nhiều tác vụ mà không cần tinh chỉnh toàn bộ.
- [DINO: Emerging Properties in Self-Supervised Vision Transformers (Facebook AI Research Explained)](https://www.youtube.com/watch?v=h3ij3F3cPIk) — Yannic Kilcher, tiếng Anh. Trình bày DINO, nền tảng tiền thân mà DINOv2 mở rộng.
- [Contrastive Learning with SimCLR | Deep Learning Animated](https://www.youtube.com/watch?v=UqJauYELn6c) — Deepia, tiếng Anh. Minh họa mục tiêu học đối chiếu, tiền đề của nhiều mô hình thị giác tự giám sát hiện đại.
