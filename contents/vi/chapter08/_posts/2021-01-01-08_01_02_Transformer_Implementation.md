---
layout: post
title: 08-01-02 Cài đặt Transformer
chapter: '08'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter08
---

# 08-01-02 Cài đặt Transformer

Lý thuyết định nghĩa các ánh xạ. Cài đặt phải tôn trọng **shape, mask, thứ tự residual, và ổn định số**. Các bài này xây một chồng Transformer nhỏ nhưng trung thực.

## Các bài học

1. **Cài đặt cốt lõi**  
   Attention có scale, multi-head, mã hóa vị trí, FFN, tầng/chồng encoder—viết từ đầu trong PyTorch.

2. **Demo, bài báo và bẫy**  
   Kiểm tra khói forward, quan sát attention, một tác vụ chuỗi nhỏ, tài liệu, và checklist gỡ lỗi theo hướng sản xuất.

## Bản đồ module

```text
Token ids
  → Embedding + PositionalEncoding
  → N × EncoderLayer
        MultiHeadSelfAttention (+ residual/norm)
        PositionwiseFFN        (+ residual/norm)
  → (tùy chọn) chồng DecoderLayer với causal + cross-attn
  → Đầu Linear lên từ vựng
```

## Danh mục kiểm tra trước khi mở rộng quy mô
| Hạng mục | Kiểm tra |
|------|------|
| Mask nhân quả | Trọng số tương lai ≈ 0 |
| Mask đệm | Key pad nhận trọng số 0 |
| Đường residual | Ablation: bỏ residual sẽ hại độ sâu |
| PE cộng một lần ở đầu vào (cổ điển) | Không phải mỗi tầng trừ khi bạn chọn thiết kế đó |
| Nhất quán `batch_first` | Mọi module thống nhất `(B,L,d)` so với `(L,B,d)` |
| Dropout chỉ ở chế độ train | `model.eval()` khi suy luận |

Bắt đầu với **Cài đặt cốt lõi**.
