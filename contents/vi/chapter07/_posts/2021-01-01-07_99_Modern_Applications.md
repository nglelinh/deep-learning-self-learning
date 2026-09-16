---
layout: post
title: 07-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '07'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter07
lesson_type: optional
---

# Tùy chọn: kernel attention, serving, và biến thể multi-query

> Bài này là **tùy chọn**. Nó **không** thay toán scaled-dot-product hay multi-head. Nó phủ các hệ thống và biến thể kiến trúc khiến attention triển khai được ở quy mô LLM.

Điểm số vẫn là

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V.$$

Từ 2022 đến 2026 cuộc chiến chuyển sang **IO**, **bố cục KV-cache**, và **ít head key/value hơn**.

## 1. FlashAttention: attention đúng, lưu lượng bộ nhớ tốt hơn

[Dao et al., 2022](https://arxiv.org/abs/2205.14135) chia ô $$Q,K,V$$ để ma trận $$T\times T$$ không nằm trên HBM. [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023) cải thiện song song; [FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024) nhắm Hopper (bất đồng bộ + FP8). `scaled_dot_product_attention` của PyTorch chuyển tới các kernel này khi shape cho phép.

Cùng softmax như bài lý thuyết. Cái mới là **cài đặt thuật toán**.

## 2. Ứng dụng cụ thể

### PagedAttention và serving LLM

[Kwon et al., 2023](https://arxiv.org/abs/2309.06180) (vLLM) lưu KV cache theo khối phân trang để nhiều chuỗi decode chia sẻ GPU mà không phí chỗ đặt trước. Đó là *serving* attention, không phải hàm điểm số mới.

### GQA / MQA

Multi-query ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) và grouped-query attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245), GQA) cho nhiều query head dùng chung một hoặc vài KV head. Llama 2/3 và Mistral dùng GQA để cache

$$\text{KV bytes} \propto T \cdot n_{\text{kv}} \cdot d, \quad n_{\text{kv}} \ll n_{\text{heads}}.$$

### Sliding-window và linear attention

Attention **cửa sổ trượt** kiểu Mistral và các bài linear-attention đổi pattern đầy đủ $$T\times T$$ lấy chi phí $$O(T)$$. Dùng khi cảnh báo bậc hai của Chương 07 trở thành ràng buộc sản phẩm.

## 3. Phần mềm phổ biến

- PyTorch `F.scaled_dot_product_attention` (SDPA) / [FlashAttention](https://github.com/Dao-AILab/flash-attention).
- [vLLM](https://github.com/vllm-project/vllm) — serving PagedAttention.
- [xFormers](https://github.com/facebookresearch/xformers) — toán tử attention tiết kiệm bộ nhớ.

## 4. Trích dẫn (2022–2026)

- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135).
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691).
- [FlashAttention-3 (Shah et al., 2024)](https://arxiv.org/abs/2407.08608).
- [PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180).
- [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Viết attention NumPy và bài pitfalls multi-head trước. Bài này chỉ thêm **kernel, KV-cache, và GQA** — thứ bạn gặp trong mọi codebase LLM.
