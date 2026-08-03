---
layout: post
title: 08-01 Kiến trúc Transformer
chapter: '08'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter08
---

# 08-01 Kiến trúc Transformer

Phần này là trái tim của chương: cách các mảnh attention từ Chương 07 được lắp ghép thành một mô hình đầy đủ.

## Các bài học

1. **Lý thuyết Transformer**  
   Động lực so với RNN, self-attention QKV trong mô hình, đóng gói multi-head, mã hóa vị trí (*positional encoding*), tầng encoder, tầng decoder (masked self-attn + cross-attn + FFN), residual/LayerNorm, các họ chỉ-encoder / chỉ-decoder / encoder–decoder.

2. **Cài đặt Transformer**  
   Các module viết từ đầu, kiểm tra khói (*smoke test*), demo huấn luyện nhỏ, bài báo, và các bẫy sản xuất (quy ước mask, sai sót PE, KV-cache, chi phí $$O(L^2)$$).

## Lộ trình gợi ý

| Bước | Bài học | Kết quả |
|------|--------|---------|
| 1 | **08-01-01 Lý thuyết** | Viết được phương trình tầng mà không cần ghi chú |
| 2 | **08-01-02 Tổng quan cài đặt** | Bản đồ module + checklist |
| 3 | **08-01-02-01 Mã cốt lõi** | Các khối PyTorch chạy được |
| 4 | **08-01-02-02 Demo & bẫy** | Huấn luyện mô hình toy; gỡ lỗi tự tin |

Nếu lý thuyết còn trừu tượng, mở class tương ứng trong Mã cốt lõi, chạy forward trên token ngẫu nhiên, rồi quay lại các phương trình.

Tiếp tục với **Lý thuyết Transformer**.
