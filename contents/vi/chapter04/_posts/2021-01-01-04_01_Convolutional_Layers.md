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

Phần này trình bày chi tiết về các tầng tích chập (*convolutional layers*) — thành phần cốt lõi của CNN — theo bốn mục chính:

1. **Toán học Tích chập và Kích thước** — khái niệm nền tảng, tích chập đa kênh, công thức tính kích thước đầu ra và số lượng tham số
2. **Trực giác Tích chập và Ví dụ** — phát hiện cạnh, hệ thống đặc trưng phân cấp và sự mở rộng của trường tiếp nhận (*receptive field*)
3. **Cài đặt Tích chập** — cài đặt từ đầu bằng NumPy, sử dụng API PyTorch và xây dựng khối CNN hoàn chỉnh
4. **Khái niệm Liên quan, Cạm bẫy và Bài báo** — mối liên hệ với các tầng khác, lỗi thường gặp, mẹo thực hành và các công trình nền tảng

Bạn nên đọc tuần tự từng mục trước khi chuyển sang phần tầng gộp (*pooling*) và các kiến trúc CNN kinh điển.

## Video tham khảo

- **Xem trước:** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, tiếng Anh. Giới thiệu trực giác chung về phép tích chập trước khi đi sâu vào công thức và mã nguồn.
- [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, tiếng Anh. Mở rộng phép tích chập thành tầng nhiều kênh trong mạng CNN.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trọn vẹn cách một tầng tích chập xử lý ảnh cụ thể.
