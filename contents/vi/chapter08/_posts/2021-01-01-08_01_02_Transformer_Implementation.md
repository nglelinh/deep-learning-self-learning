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

## Video tham khảo

- **xem trước** [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, tiếng Anh. Bản đồ module (attention, MLP, khối, đầu mô hình ngôn ngữ) được dựng trên video.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, tiếng Anh. Các phương trình các module đó phải khớp.
- [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY) — StatQuest with Josh Starmer, tiếng Anh. Một cách gắn nhãn thứ hai cho khối nếu một lỗi kích thước khó khoanh vùng.
