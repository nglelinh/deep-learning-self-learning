---
layout: post
title: 07-02-02 Cài đặt Attention
chapter: '07'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter07
---

# 07-02-02 Cài đặt Attention

Phần toán học xác định rõ attention tính những đại lượng nào, còn phần cài đặt đảm bảo phép toán đó được hiện thực hóa chính xác — từ kích thước tensor, mặt nạ, độ ổn định số học cho tới cách dùng API.

## Các bài trong mục này

1. **Cài đặt cốt lõi**  
   Cài đặt attention Bahdanau và scaled self-attention từ đầu, chạy ví dụ minh họa kiểu dịch máy và đối chiếu phiên bản NumPy với PyTorch.

2. **Multi-Head, Papers và Cạm bẫy**  
   Xây dựng module multi-head attention, cài đặt causal mask, phác thảo mô hình seq2seq có attention, điểm qua các bài báo nền tảng và danh mục kiểm tra gỡ lỗi thực tế.

## Danh mục kiểm tra khi cài đặt

| Hạng mục kiểm tra | Lý do quan trọng |
|-------------------|------------------|
| Softmax trên trục **key** | Trọng số phải tạo thành phân phối xác suất trên các vị trí bộ nhớ |
| Chia tỉ lệ cho $$\sqrt{d_k}$$ | Tránh bão hòa softmax khi số chiều lớn |
| Áp mask **trước** softmax bằng số âm lớn / $$-\infty$$ | Đảm bảo trọng số bằng 0 và gradient chính xác |
| Causal mask cho decoder tự hồi quy | Ngăn rò rỉ thông tin từ các token tương lai |
| Padding mask khi huấn luyện theo batch | Tránh phân bổ trọng số vào các token `PAD` |
| Reshape multi-head: `(B,L,H,d)` ↔ `(B,H,L,d)` | Hoán vị sai chiều làm trộn lẫn đặc trưng dù chương trình vẫn chạy không báo lỗi |

Hãy bắt đầu từ bài **Cài đặt cốt lõi**, sau đó tiếp tục sang phần multi-head và các cạm bẫy thường gặp.

## Video tham khảo

- **Xem trước:** [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Ôn nhanh công thức toán học trước khi chuyển sang phần cài đặt.
- [A Dive Into Multihead Attention, Self-Attention and Cross-Attention](https://www.youtube.com/watch?v=mmzRYGCfTzc) — Machine Learning Studio (tiếng Anh). Sơ đồ phân biệt các module self-attention, cross-attention và multi-head.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Cài đặt trực tiếp cơ chế attention bằng Python, sát với mục tiêu thực hành của mục này.
