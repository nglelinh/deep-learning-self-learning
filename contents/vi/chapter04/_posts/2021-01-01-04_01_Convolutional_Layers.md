---
layout: post
title: 04-01 Tầng Tích chập
chapter: '04'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

Bài này trình bày các tầng tích chập (*convolutional layers*) — khối xây dựng cốt lõi của CNN — được tổ chức thành các phần tập trung:

1. **Toán học Tích chập và Kích thước** — tổng quan khái niệm, nền tảng toán, tích chập đa kênh, kích thước đầu ra và số tham số
2. **Trực giác Tích chập và Ví dụ** — phát hiện cạnh, đặc trưng phân cấp, và sự tăng của trường tiếp nhận (*receptive field*)
3. **Cài đặt Tích chập** — NumPy từ đầu, API PyTorch, và một khối CNN hoàn chỉnh
4. **Khái niệm Liên quan, Cạm bẫy và Bài báo** — liên hệ với các tầng khác, lỗi thường gặp, mẹo thực tiễn, và các bài báo nền tảng

Hãy đọc lần lượt từng phần trước khi chuyển sang gộp (*pooling*) và các kiến trúc kinh điển.
