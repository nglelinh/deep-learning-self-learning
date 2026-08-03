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

Toán học cho biết attention *tính* cái gì. Cài đặt cho biết ta có thực sự dựng đúng đối tượng đó hay không—gồm shape, mask, ổn định số và API.

## Các bài trong mục này

1. **Cài đặt cốt lõi**  
   Attention Bahdanau từ đầu, scaled self-attention, demo kiểu dịch máy, và đường đi rõ ràng từ NumPy → PyTorch.

2. **Multi-Head, Papers và Cạm bẫy**  
   Module multi-head, causal masking, phác thảo seq2seq-with-attention tối giản, papers chuẩn mực, và checklist debug dùng trong huấn luyện thực tế.

## Danh mục kiểm tra khi cài đặt
| Kiểm tra | Vì sao quan trọng |
|----------|-------------------|
| Softmax trên trục **key** | Trọng số phải là phân phối trên các khe nhớ |
| Scale bởi $$\sqrt{d_k}$$ | Tránh attention bão hòa khi chiều lớn |
| Mask **trước** softmax bằng số âm lớn | Zero đúng + gradient đúng |
| Causal mask cho decoder tự hồi quy | Ngăn rò rỉ tương lai |
| Padding mask khi huấn luyện theo batch | Ngăn chú ý tới `PAD` |
| Reshape multi-head: `(B,L,H,d)` ↔ `(B,H,L,d)` | Transpose sai làm xáo trộn đặc trưng mà vẫn “chạy” |

Bắt đầu với **Cài đặt cốt lõi**, rồi tiếp tục multi-head và các cạm bẫy.
