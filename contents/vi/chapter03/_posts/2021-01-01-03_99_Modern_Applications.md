---
layout: post
title: 03-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '03'
order: 14
owner: Deep Learning Course
lang: vi
categories:
- chapter03
lesson_type: optional
---

# Tùy chọn: lan truyền ngược như bài toán compiler

> Bài này là **tùy chọn**. Nó **không** thay lý thuyết hàm mất mát, gradient descent hay lan truyền ngược. Nó ánh xạ các thuật toán ấy sang compiler và kernel dùng để huấn luyện sau 2022.

Backprop vẫn là autodiff chiều ngược của loss vô hướng $$\mathcal{L}$$. Câu chuyện 2022–2026: **băng được biên dịch** — toán tử được fuse, activation được tính lại, mixed precision là mặc định.

## 1. Cùng quy tắc chuỗi, bộ lập lịch khác

Nếu lớp tính $$\mathbf{y} = f(\mathbf{x}; \theta)$$, chiều ngược cần $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$$ và $$\frac{\partial \mathcal{L}}{\partial \theta}$$. Compiler (`torch.compile`, XLA, Triton) quyết định:

- trung gian nào **giữ** hay **tính lại** (activation checkpointing),
- phép từng phần tử nào **fuse** thành một kernel,
- có chạy BF16/FP8 với tích lũy FP32 hay không.

Toán Chương 03 không đổi; thời gian tường và bộ nhớ thì có.

## 2. Ứng dụng cụ thể

### `torch.compile` và JAX `jit`

[Ansel et al., 2024](https://arxiv.org/abs/2404.14294) mô tả đường Dynamo + AOTAutograd + inductor của PyTorch 2: bytecode Python được bắt thành đồ thị FX rồi hạ xuống. JAX làm cùng ý tưởng với `jit`/`grad`. Cả hai là câu trả lời production cho “tôi viết backprop bằng NumPy; ngành chạy nó thế nào?”

### Mixed precision và loss scaling

Một bước cập nhật điển hình:

$$\theta \leftarrow \theta - \eta \cdot \mathrm{Cast}_{fp32}(\widehat{\nabla \mathcal{L}}),$$

trong đó $$\widehat{\nabla \mathcal{L}}$$ tính bằng BF16/FP16. Tràn số được tránh bằng cách nhân loss trước chiều ngược (AMP trong PyTorch). Cùng bước SGD/Adam như lý thuyết, thêm chính sách định dạng số.

### Backward mức kernel cho attention

[Dao et al., 2022](https://arxiv.org/abs/2205.14135) (FlashAttention) suy ra **backward nhận thức IO** không hiện ma trận attention $$N\times N$$. Gradient softmax-attention giống công thức Chương 07; cài đặt là kernel backward đã fuse. Nhắc ở đây vì đó là kết quả “kỹ thuật backprop” nổi tiếng nhất của giai đoạn.

## 3. Phần mềm phổ biến

- [PyTorch 2](https://pytorch.org/) — `torch.compile`, `torch.func`, `torch.utils.checkpoint`.
- [JAX](https://github.com/google/jax) — `grad`, `value_and_grad`, XLA.
- [Triton](https://github.com/triton-lang/triton) — viết kernel backward khi compiler chưa đủ.

## 4. Trích dẫn (2022–2026)

- [PyTorch 2 (Ansel et al., 2024)](https://arxiv.org/abs/2404.14294) — bắt đồ thị động + autograd biên dịch.
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — gradient attention đúng, backward theo ô, tiết kiệm IO.
- [Online softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867) — softmax luồng trong các kernel đó (vẫn là xương sống số trong mã 2024–2026).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Làm worksheet backprop và trainer NumPy trước. Bài này chỉ trả lời “ngành thêm gì quanh băng”: compiler, AMP, backward đã fuse — không phải quy tắc học mới.
