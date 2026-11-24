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

Tự tay lắp ráp một Transformer hoạt động trơn tru bằng PyTorch không dài như nhiều người tưởng — toàn bộ lõi từ Positional Encoding, Multi-Head Attention, FFN đến một Encoder–Decoder hoàn chỉnh chỉ gói gọn trong chưa đầy 200 dòng code rõ ràng. Bí quyết nằm ở việc chia mô-đun rành mạch và thiết lập các bài kiểm tra khói (smoke tests) ngay trước khi bước vào vòng huấn luyện.

## Bản đồ kiến thức nền: Ba nguyên tắc vàng khi lập trình kiến trúc Transformer

1. **Phân tách mô-đun sạch theo sơ đồ dữ liệu**:
   Xây dựng tuần tự từ dưới lên: `PositionalEncoding` $$\to$$ `MultiHeadAttention` $$\to$$ `PositionwiseFFN` $$\to$$ `EncoderLayer` / `DecoderLayer` $$\to$$ `Transformer`. Mỗi mô-đun có hợp đồng đầu vào/đầu ra `(B, L, d_model)` thống nhất.
2. **Quản lý hai loại mặt nạ độc lập (Padding Mask và Causal Mask)**:
   Kết hợp bằng phép giao logic (chỉ cho phép chú ý những vị trí **vừa** không phải `<PAD>`, **vừa** không nằm ở tương lai) với hình dạng broadcast `(B, 1, L_q, L_k)` để áp dụng cho mọi head.
3. **Danh mục kiểm tra trước khi huấn luyện quy mô lớn**:
   Kiểm tra tính nhân quả (*causal leak test*: đổi token cuối xem logit các token trước có đổi 0,0 hay không), kiểm tra dòng gradient không bị `NaN`, và thử nghiệm overfit hoàn hảo trên 1 batch nhỏ (10 mẫu) để xác minh không có lỗi logic ngầm.

### Khung giải thích (nên đọc chậm): Tính tay phép kết hợp Padding Mask và Causal Mask cho chuỗi dài $$L = 4$$

Giả sử trong một batch, câu có độ dài tối đa $$L = 4$$, nhưng câu này chỉ có 3 từ thật và vị trí cuối (`idx=3`) là `<PAD>`. Quy ước `1 = được phép chú ý`, `0 = bị che`:

- **Causal Mask** (tam giác dưới $$4 \times 4$$):
  $$\begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix}$$
- **Key Padding Mask** (`[1, 1, 1, 0]` phát sóng theo mọi hàng):
  $$\begin{bmatrix} 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \end{bmatrix}$$
- **Mặt nạ kết hợp (`causal & padding`)**:
  $$\begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 0 \end{bmatrix}$$

Tại hàng `idx=2` (từ thật thứ 3), mô hình nhìn thấy các vị trí `0, 1, 2` và không bao giờ nhìn thấy `<PAD>` ở cột `3` — sạch sẽ tuyệt đối!

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
