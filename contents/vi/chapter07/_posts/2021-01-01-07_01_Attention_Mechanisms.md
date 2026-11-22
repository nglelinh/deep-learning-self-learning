---
layout: post
title: 07-02 Cơ chế Chú ý Đi sâu
chapter: '07'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter07
---

# 07-02 Cơ chế Chú ý Đi sâu

Phần này mở rộng sâu hơn so với bài cơ sở (07-01). Sau khi đã hiểu lý do tồn tại của attention và các công thức nền tảng, ta sẽ hệ thống hóa chặt chẽ phần toán học rồi cài đặt đầy đủ từng thành phần.

## Nội dung chính

1. **Toán học Attention**  
   Thống nhất các hàm chấm điểm, dạng ma trận QKV theo batch, self-attention như phép tương tác $$L\times L$$, đại số multi-head, lý thuyết padding/causal mask trước softmax, độ phức tạp tính toán và lập luận về độ dài đường truyền thông tin.

2. **Cài đặt Attention**  
   Cài đặt từ đầu attention Bahdanau và scaled dot-product attention, xây dựng module multi-head, minh họa masking, phác thảo mô hình seq2seq có attention, điểm qua các bài báo nền tảng và các lỗi thường gặp trong thực tế.

## Lộ trình đọc

| Thứ tự | Bài | Mục tiêu đạt được |
|--------|-----|------------------|
| 1 | **07-02-01 Toán học Attention** | Nắm chắc và tự viết lại được toàn bộ các công thức cốt lõi |
| 2 | **07-02-02 Cài đặt** (tổng quan) | Đối chiếu từ công thức toán sang cấu trúc các module mã nguồn |
| 3 | **07-02-02-01 Cài đặt cốt lõi** | Chạy và kiểm chứng được mã nguồn Bahdanau và self-attention |
| 4 | **07-02-02-02 Multi-Head, Papers, Cạm bẫy** | Nắm vững multi-head, causal mask, tài liệu nền tảng và quy trình gỡ lỗi |

Nếu một biểu thức trong bài **07-02-01** còn trừu tượng, bạn có thể xem trước đoạn mã minh họa tương ứng ở **07-02-02-01** rồi quay lại đối chiếu.

## Liên kết trong khóa học

- **Xem lại bài 07-01** nếu cần ôn lại trực giác và bối cảnh lịch sử của attention.
- **Tiếp nối sang Chương 08 (Transformer)** để kết hợp self-attention với mạng feed-forward, kết nối phần dư và mã hóa vị trí.

Khi đã sẵn sàng, hãy bắt đầu với bài **Toán học Attention**.

## Video tham khảo

- **Xem trước:** [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Trình bày chi tiết scaled dot-product attention, trọng tâm của phần đi sâu này.
- [Attention for Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=PSs6nxngL6k) — StatQuest with Josh Starmer (tiếng Anh). Minh họa dạng attention cho mô hình chuỗi để đối chiếu với self-attention.
- [A Dive Into Multihead Attention, Self-Attention and Cross-Attention](https://www.youtube.com/watch?v=mmzRYGCfTzc) — Machine Learning Studio (tiếng Anh). Phân tách rõ ba dạng multi-head, self-attention và cross-attention đúng theo cấu trúc phần này.
