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

## Video tham khảo

- **xem trước** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, tiếng Anh. Trước khi các bài toán, trực giác và mã tách một phép toán ra.
- [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, tiếng Anh. Bài giảng đưa phép toán đó thành một lớp có kênh.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Một ví dụ ảnh làm trọn của cùng lớp đó.
