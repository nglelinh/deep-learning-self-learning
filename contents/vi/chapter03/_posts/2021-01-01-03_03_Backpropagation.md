---
layout: post
title: 03-03 Thuật toán Lan truyền Ngược
chapter: '03'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

Phần này trình bày thuật toán lan truyền ngược (*backpropagation*) — cơ chế tính gradient cốt lõi của học sâu:

1. **Lý thuyết và ví dụ tính toán** — khái niệm, nền tảng toán học và ví dụ số từng bước
2. **Triển khai** — cài đặt từ đầu bằng NumPy và bằng PyTorch, các bài báo nền tảng, lỗi thường gặp và điểm then chốt

## Video tham khảo

- **Xem trước:** [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown, tiếng Anh. Diễn giải trực quan cách mỗi mẫu dữ liệu điều chỉnh từng trọng số trong mạng.
- [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown, tiếng Anh. Phép tính quy tắc dây chuyền đứng sau quá trình cập nhật trọng số.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, tiếng Anh. Cài đặt hoàn chỉnh thuật toán lan truyền ngược bằng mã nguồn, tương ứng với bài thứ hai trong phần này.
