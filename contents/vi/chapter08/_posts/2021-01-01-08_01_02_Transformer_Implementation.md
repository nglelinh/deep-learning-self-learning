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

Phần lý thuyết mô tả các phép biến đổi toán học, còn phần cài đặt đảm bảo mô hình tuân thủ chặt chẽ **kích thước tensor, quy ước mặt nạ, thứ tự kết nối phần dư và độ ổn định số học**. Hai bài trong mục này hướng dẫn xây dựng một mô hình Transformer nhỏ gọn nhưng đầy đủ.

## Các bài học trong mục này

1. **Cài đặt cốt lõi**  
   Cài đặt từ đầu bằng PyTorch các khối scaled dot-product attention, multi-head attention, mã hóa vị trí, FFN và khối encoder.

2. **Demo, bài báo và bẫy thường gặp**  
   Chạy kiểm thử nhanh lượt truyền xuôi, quan sát trọng số attention, huấn luyện trên một tác vụ chuỗi nhỏ, điểm qua các bài báo nền tảng và danh mục gỡ lỗi thực tế.

## Sơ đồ các module

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

| Hạng mục | Tiêu chí kiểm tra |
|------|------|
| Mặt nạ nhân quả | Trọng số ở các vị trí tương lai xấp xỉ 0 |
| Mặt nạ đệm | Các vị trí `PAD` có trọng số bằng 0 |
| Kết nối phần dư | Thử nghiệm cắt giảm: bỏ residual sẽ làm giảm mạnh khả năng huấn luyện mạng sâu |
| Mã hóa vị trí (PE) | Với bản cổ điển, cộng một lần ở đầu vào (trừ khi dùng thiết kế PE tương đối/RoPE theo từng tầng) |
| Thống nhất `batch_first` | Mọi module dùng chung quy ước `(B, L, d)` hoặc `(L, B, d)` |
| Chế độ `train` / `eval` | Đảm bảo gọi `model.eval()` khi suy luận để tắt dropout |

Hãy bắt đầu với bài **Cài đặt cốt lõi**.

## Video tham khảo

- **Xem trước:** [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Hướng dẫn xây dựng lần lượt từng module (attention, MLP, khối Transformer, đầu mô hình ngôn ngữ) khớp với sơ đồ bài này.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown (tiếng Anh). Đối chiếu các phương trình toán học tương ứng với từng module.
- [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY) — StatQuest with Josh Starmer (tiếng Anh). Góc nhìn trực quan bổ sung giúp khoanh vùng nhanh khi gặp lỗi kích thước tensor.
