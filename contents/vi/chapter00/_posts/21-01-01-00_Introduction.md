---
layout: post
title: 00 Giới thiệu
chapter: '00'
order: 1
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Tối ưu hóa giữ vai trò trung tâm trong khoa học dữ liệu. Dù bạn đang huấn luyện mạng nơ-ron, cực tiểu hóa sai số trong mô hình hồi quy hay phân bổ tài nguyên trong hệ thống gợi ý, về bản chất bạn đều đang giải bài toán tìm nghiệm tốt nhất trong một không gian khả năng rất lớn. Để làm việc hiệu quả với các bài toán này, bạn cần nắm vững ngôn ngữ toán học nền tảng. Chương này ôn tập các khái niệm cốt lõi từ đại số tuyến tính, lý thuyết tập hợp, giải tích và xác suất, giúp bạn xử lý tốt gradient, ma trận, ràng buộc và yếu tố bất định trong các tác vụ tối ưu hóa.

**Đọc thêm (tùy chọn, 2022–2026):** Sau khi học xong phần toán nền tảng, bạn có thể xem bài **00-99 Ứng dụng và cập nhật hiện đại** để tìm hiểu về trình biên dịch vi phân tự động (JAX, `torch.compile`) cũng như cách gradient, ma trận và hàm hợp lý xuất hiện trong học máy khoa học (*scientific ML*) và huấn luyện theo quy luật mở rộng (*scaling law*).

**Luyện phỏng vấn (tùy chọn):** Sau các bài về xác suất và giải tích, bạn có thể xem **00-98 Luyện phỏng vấn** với các câu hỏi gốc về entropy, phân kỳ KL và vi phân tự động; bộ ngân hàng câu hỏi có lời giải đầy đủ được cung cấp miễn phí dưới dạng PDF trong *Deep Learning Interviews* ([arXiv:2201.00650](https://arxiv.org/abs/2201.00650)).

## Video tham khảo

- **Xem trước:** [MaDL - Introduction](https://www.youtube.com/watch?v=gq0m_2Glz0g) — Tübingen Machine Learning, tiếng Anh. Video giới thiệu ngắn gọn các kiến thức đại số tuyến tính, giải tích và xác suất mà chương này giả định trước khi đi vào mạng nơ-ron.
- [Vectors | Chapter 1, Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs) — 3Blue1Brown, tiếng Anh. Cách tiếp cận vector trong video này cũng là cách các bài sau biểu diễn trọng số, giá trị kích hoạt và điểm dữ liệu.
- [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, tiếng Anh. Ba tiên đề xác suất trong bài giảng này là nền tảng xuyên suốt nửa sau của chương.
