---
layout: post
title: 17-01 Cơ sở Phát hiện Đối tượng
chapter: '17'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter17
---

Chủ đề phát hiện đối tượng (*object detection*) được chia thành hai phần:

1. **Lý thuyết** — IoU, hồi quy hộp bao (*bounding box*), RPN và trực giác hoạt động.
2. **Cài đặt** — mã nguồn minh họa, các bài báo nền tảng và những lỗi thường gặp.

## Video tham khảo

- **Xem trước** [How YOLO Object Detection Works](https://www.youtube.com/watch?v=svn9-xV7wjk) — DeepBean, tiếng Anh. Trình bày cách chia lưới, dự đoán hộp bao và điểm tin cậy đối tượng (*objectness*).
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — Stanford University School of Engineering, tiếng Anh. Đặt YOLO trong bức tranh chung cùng các bộ phát hiện hai giai đoạn được so sánh trong bài.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, tiếng Anh. Làm rõ loại đặc trưng mà mạng xương sống tích chập cần trích xuất trước khi gắn thêm đầu phát hiện.
