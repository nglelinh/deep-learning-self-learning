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
