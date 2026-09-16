---
layout: post
title: 09-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '09'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter09
lesson_type: optional
---

# Tùy chọn: chính quy hóa sau dropout — SAM, DropPath, và công thức LLM

> Bài này là **tùy chọn**. Nó **không** thay lý thuyết dropout hay BatchNorm. Nó phủ các regularizer thực sự xuất hiện trong công thức huấn luyện 2022–2026.

Dropout

$$\tilde{\mathbf{h}} = \frac{\mathbf{m} \odot \mathbf{h}}{1-p}, \quad m_i \sim \mathrm{Bernoulli}(1-p)$$

vẫn là mô hình tư duy đúng. Transformer lớn, tuy nhiên, thường bỏ **nhánh residual** (stochastic depth / DropPath) chứ không phải đơn vị ẩn, và dựa **weight decay + quy mô dữ liệu** nhiều hơn dropout cổ điển.

## 1. Sharpness-Aware Minimization (SAM)

[Foret et al., 2021](https://arxiv.org/abs/2010.01412) (được thị giác dùng rộng sau 2022) tìm tham số vẫn tốt trong một lân cận:

$$\min_\theta \max_{\|\boldsymbol{\varepsilon}\|_2 \le \rho} \mathcal{L}(\theta + \boldsymbol{\varepsilon}).$$

Bước thực tế là cập nhật hai lần forward: leo tới $$\theta + \rho \frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|}$$, rồi bước bằng gradient “xấu nhất” đó. Đây là chính quy hóa **quỹ đạo optimizer**, bổ sung cho nhiễu dropout trên activation.

## 2. Ứng dụng cụ thể

### Stochastic depth / DropPath

Công thức ConvNeXt và ViT ngẫu nhiên bỏ cả khối residual lúc huấn luyện (mặc định trong `timm` 2022–2026). Đó là dropout trên **đường**, không phải đơn vị — ensemble ẩn của mạng nông hơn.

### LLM thực sự dùng gì

Huấn luyện kiểu LLaMA: **weight decay** (AdamW), ít dropout mức token, đôi khi dropout attention chỉ trên mô hình nhỏ. Regularizer quan trọng ở 7B+ là **hỗn hợp dữ liệu + packing + decay**, cộng RLHF/DPO sau đó (Chương 21). Đừng kỳ vọng $$p=0.5$$ của bài tập trong trainer Llama.

### Biến thể chuẩn hóa như regularizer

LayerNorm / RMSNorm thay BatchNorm trong mô hình chuỗi (đã có trong biến thể BN của Chương 09). Residual + norm là một phần câu chuyện regularizer: chúng ổn định tỷ lệ để huấn luyện lâu hơn mà feature không nổ.

## 3. Phần mềm phổ biến

- `DropPath` trong `timm` và SAM trong [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer) / [sam](https://github.com/davda54/sam).
- PyTorch `torch.optim.AdamW` + `weight_decay` (xem Chương 10).
- Công thức Hugging Face (`Trainer`, `trl`) — mặc định weight decay, ít hidden dropout.

## 4. Trích dẫn (2022–2026)

- [SAM (Foret et al., 2021)](https://arxiv.org/abs/2010.01412) — mục tiêu SAM dùng xuyên suốt thị giác 2022–2026.
- [ConvNeXt (Liu et al., 2022)](https://arxiv.org/abs/2201.03545) — stochastic depth là nguyên liệu công thức hạng nhất.
- [AdamW (Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) — decay tách; vẫn mặc định LLM năm 2026.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Giữ suy diễn inverted-dropout và ghi chú vị trí BatchNorm. Bài này chỉ thêm **SAM** và **DropPath**, và cảnh báo công thức foundation model đã rời dropout đơn vị nặng.
