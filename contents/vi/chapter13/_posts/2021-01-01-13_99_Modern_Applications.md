---
layout: post
title: 13-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '13'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter13
lesson_type: optional
---

# Tùy chọn: VAE như latent cho khuếch tán, video, và nén

> Bài này là **tùy chọn**. Nó **không** thay ELBO, tái tham số hóa, hay cài đặt VAE. Nó cho thấy cùng autoencoder biến phân trở thành **codec** trong hệ sinh 2022–2026.

ELBO

$$\mathcal{L}(\theta,\phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x)\,\|\,p(z))$$

vẫn là mục tiêu đúng. Sản phẩm ít khi lấy mẫu “khuôn mặt từ $$z\sim\mathcal{N}(0,I)$$”; chúng huấn luyện VAE để mô hình **thứ hai** (khuếch tán hoặc Transformer) làm việc trong không gian $$z$$.

## 1. VAE của Stable Diffusion

[Rombach et al., 2022](https://arxiv.org/abs/2112.10752) huấn luyện autoencoder chính quy KL (thường gọi `AutoencoderKL`) với latent là lưới không gian giảm mẫu, không phải một vector. Hạng KL là regularizer Chương 13; hạng tri giác + đối kháng giữ tái tạo đủ sắc cho bộ khử nhiễu. Hầu hết pipeline text-to-image mở vẫn giao mẫu này (SD, SDXL, nhiều stack lớp SD3).

## 2. Ứng dụng cụ thể

### Latent video và 3D

Stable Video Diffusion và các mô hình video sau tái sử dụng VAE ảnh hoặc huấn luyện VAE **thời gian** để bộ khử nhiễu đắt thấy ít token hơn. Toán vẫn encoder–decoder + KL; kỹ thuật là conv 3D/nhân quả.

### Consistency và generator một bước trong latent

[Song et al., 2023](https://arxiv.org/abs/2303.01469) (consistency models) và latent consistency (LCM) chưng cất ODE khuếch tán trong latent VAE. VAE đóng băng; việc mới là bộ lấy mẫu. Đó là lý do Chương 13 vẫn cần sau khi khuếch tán “thắng.”

### Học biểu diễn

Tách $$\beta$$-VAE kém thời thượng hơn 2018, nhưng nút thắt **biến phân** vẫn xuất hiện trong world model (DreamerV3, bài tùy chọn Chương 21) và nén.

## 3. Phần mềm phổ biến

- [diffusers](https://github.com/huggingface/diffusers) `AutoencoderKL` / `AutoencoderTiny` (TAESD).
- Checkpoint VAE Stability-AI trên Hub.
- [openai/consistency_models](https://github.com/openai/consistency_models).

## 4. Trích dẫn (2022–2026)

- [Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752).
- [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01469).
- [SD3 (Esser et al., 2024)](https://arxiv.org/abs/2403.03206) — vẫn autoencoder ẩn + flow.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Suy ra ELBO và mẹo tái tham số trước. Bài này chỉ cho thấy **$$z$$ được tiêu thụ ở đâu** trong sản phẩm 2022–2026 — không phải cận dưới chứng cứ mới.
