---
layout: post
title: 02-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '02'
order: 10
owner: Deep Learning Course
lang: vi
categories:
- chapter02
lesson_type: optional
---

# Tùy chọn: khối MLP hiện đại

> Bài này là **tùy chọn**. Nó **không** viết lại lý thuyết perceptron, kiến trúc, hàm kích hoạt hay lan truyền thuận. Nó cho thấy các khối ấy được đóng gói thế nào trong foundation model 2022–2026.

Neuron vẫn là $$z = \mathbf{w}^\top \mathbf{x} + b$$ rồi phi tuyến. Cái đổi là **khối**: đường residual, MLP có cổng, và chuẩn hóa RMS trở thành “layer” mặc định để chồng.

## 1. Từ một neuron tới MLP của Transformer

Hầu hết LLM dùng khối feed-forward có cổng (SwiGLU / GeGLU), không phải `Linear → ReLU → Linear`. Với đầu vào $$\mathbf{x} \in \mathbb{R}^{d}$$,

$$\mathrm{SwiGLU}(\mathbf{x}) = \big(\mathrm{SiLU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)\big) W_3,$$

trong đó $$\mathrm{SiLU}(z) = z\,\sigma(z)$$. Cùng đại số lan truyền thuận của Chương 02, thêm một cổng (tích Hadamard). [Shazeer, 2020](https://arxiv.org/abs/2002.05202) giới thiệu biến thể GLU; chúng thành chuẩn trong PaLM, LLaMA và các mô hình mở sau này.

## 2. Ứng dụng cụ thể

### Pre-norm + RMSNorm thay BatchNorm

LLM hầu như không dùng BatchNorm. Chúng dùng **RMSNorm**

$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \odot \boldsymbol{\gamma}$$

và đặt trước nhánh residual (pre-norm). Đó là câu chuyện tỷ lệ kích hoạt từ bài vanishing gradient, áp ở quy mô 7B–400B.

### MLP-Mixer và thị giác “all-MLP”

[Tolstikhin et al., 2021](https://arxiv.org/abs/2105.01601) (Mixer) cho thấy MLP trộn token + trộn kênh có thể phân loại ImageNet không cần tích chập. Di sản 2022–2024: Mixer là baseline rẻ cạnh ConvNeXt và ViT, không thay CNN.

### Kolmogorov–Arnold Networks (KAN)

[Liu et al., 2024](https://arxiv.org/abs/2404.19756) thay hàm kích hoạt cố định trên nút bằng hàm một biến học được trên **cạnh**. Hãy coi đây là thí nghiệm cho câu “neuron là gì?” — không phải mặc định production cho thị giác hay NLP.

## 3. Phần mềm phổ biến

- `torch.nn.SiLU`, `RMSNorm`, và SwiGLU trong [LLaMA](https://github.com/meta-llama/llama) / [Hugging Face transformers](https://github.com/huggingface/transformers).
- [timm](https://github.com/huggingface/pytorch-image-models) — khối MLP và ConvNeXt hiện đại cho thị giác.

## 4. Trích dẫn (2022–2026)

- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) — MLP có cổng trong mã LM 2024–2026.
- [RMSNorm (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) — mặc định họ LLaMA.
- [KAN (Liu et al., 2024)](https://arxiv.org/abs/2404.19756) — kích hoạt học được trên cạnh; nghiên cứu bổ sung, không thay MLP.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Dùng bài perceptron và hàm kích hoạt để hiểu $$z$$ và $$\sigma$$. Bài này chỉ ghi **cách đóng gói** (SwiGLU, RMSNorm, residual) khi bạn mở checkpoint LLM 2025 hoặc model card `timm`.
