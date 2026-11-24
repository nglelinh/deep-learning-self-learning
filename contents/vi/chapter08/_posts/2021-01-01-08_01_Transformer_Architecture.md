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

Nhiều người mới học cảm thấy sơ đồ kiến trúc Transformer trong bài báo gốc năm 2017 trông như một mê cung đầy mũi tên rẽ nhánh. Nhưng khi tách nó thành hai góc nhìn — **lý thuyết từng khối con** (tại sao cần FFN mở rộng $$4\times$$, tại sao Pre-LN ổn định hơn Post-LN) và **cài đặt mã nguồn gọn sạch** — bạn sẽ nhận ra Transformer thực chất là một trong những kiến trúc tối giản và đối xứng nhất lịch sử học sâu.

## Bản đồ kiến thức nền: Ba trụ cột học phần Kiến trúc Transformer

1. **Giải phẫu toán học từng khối con**:
   Token Embedding (nhân $$\sqrt{d_{\text{model}}}$$), Positional Encoding, Multi-Head Attention, Position-wise FFN ($$\text{Linear} \to \text{Activation} \to \text{Linear}$$), Residual Connection và LayerNorm.
2. **Thiết kế ba họ mô hình từ cùng một bộ khối**:
   Chỉ bằng cách bật/tắt mặt nạ nhân quả (causal mask) và giữ hay bỏ khối cross-attention, cùng một khối lõi sinh ra cả BERT, GPT lẫn T5.
3. **Kiểm chứng từ code tới thực nghiệm**:
   Tự lắp ráp một Transformer mini hoàn chỉnh bằng PyTorch, kiểm tra bằng smoke tests (shape, gradient, causal leak) và hiểu rõ các cạm bẫy khi mở rộng quy mô (warmup, weight tying, KV-cache).

### Khung giải thích (nên đọc chậm): Tính tay tỷ lệ phân bổ tham số giữa Attention và FFN trong 1 tầng Transformer

Với $$d_{\text{model}} = 512$$ và chiều ẩn của FFN chuẩn là $$d_{\text{ff}} = 4 \, d_{\text{model}} = 2{,}048$$ (bỏ qua bias và LayerNorm nhỏ):

1. **Khối Multi-Head Self-Attention** (4 ma trận $$W^Q, W^K, W^V, W^O$$ kích thước $$512 \times 512$$):
   $$4 \times 512^2 = \mathbf{1{,}048{,}576}$$ tham số (**33,3%**).
2. **Khối Feed-Forward Network (FFN)** (hai ma trận $$W_1 \in \mathbb{R}^{512 \times 2048}$$ và $$W_2 \in \mathbb{R}^{2048 \times 512}$$):
   $$2 \times 512 \times 2{,}048 = 8 \times 512^2 = \mathbf{2{,}097{,}152}$$ tham số (**66,7%**).
3. **Tổng 1 tầng Transformer Encoder/Decoder-only**: $$12 \, d_{\text{model}}^2 = \mathbf{3{,}145{,}728}$$ tham số!

Một sự thật hay gây bất ngờ: **hai phần ba** số tham số của một mô hình ngôn ngữ Transformer không nằm ở Attention, mà nằm ở các tầng **FFN** — nơi mô hình lưu trữ phần lớn tri thức sự kiện (*factual knowledge*) như một bộ nhớ khóa–giá trị khổng lồ!

# 08-01 Kiến trúc Transformer

Đây là phần trọng tâm của Chương 08, trình bày cách ghép nối các thành phần attention từ Chương 07 thành kiến trúc Transformer hoàn chỉnh.

## Nội dung chính

1. **Lý thuyết Transformer**  
   Động lực thay thế RNN, self-attention QKV trong mô hình hoàn chỉnh, multi-head attention, mã hóa vị trí (*positional encoding*), cấu trúc tầng encoder và decoder (masked self-attention, cross-attention, FFN), kết nối phần dư và LayerNorm, cùng ba dòng mô hình chỉ-encoder, chỉ-decoder và encoder–decoder.

2. **Cài đặt Transformer**  
   Xây dựng các module từ đầu bằng PyTorch, chạy kiểm thử nhanh (*smoke test*), huấn luyện ví dụ nhỏ, điểm qua các bài báo nền tảng và tổng hợp bẫy vận hành (quy ước mặt nạ, lỗi PE, KV-cache, chi phí $$O(L^2)$$).

## Lộ trình gợi ý

| Bước | Bài học | Mục tiêu đạt được |
|------|--------|---------|
| 1 | **08-01-01 Lý thuyết** | Tự viết lại được phương trình của một tầng Transformer hoàn chỉnh |
| 2 | **08-01-02 Tổng quan cài đặt** | Nắm rõ sơ đồ module và danh mục kiểm tra |
| 3 | **08-01-02-01 Mã cốt lõi** | Cài đặt và kiểm chứng các khối PyTorch chạy được |
| 4 | **08-01-02-02 Demo & bẫy** | Huấn luyện thành công bài toán nhỏ và nắm quy trình gỡ lỗi |

Nếu phần phương trình ban đầu còn trừu tượng, bạn có thể mở lớp tương ứng trong bài **Mã cốt lõi**, chạy thử trên dữ liệu ngẫu nhiên rồi quay lại đối chiếu với lý thuyết.

Hãy bắt đầu với bài **Lý thuyết Transformer**.

## Video tham khảo

- **Xem trước:** [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown (tiếng Anh). Trình bày luồng dữ liệu xuyên suốt cả phần lý thuyết và phần cài đặt.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Đích thực hành của phần cài đặt: xây dựng mô hình GPT từ các khối cơ bản.
- [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY) — StatQuest with Josh Starmer (tiếng Anh). Giải thích chi tiết từng tensor trước khi đọc mã nguồn.
