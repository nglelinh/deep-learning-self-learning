---
layout: post
title: 11-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '11'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter11
lesson_type: optional
---

# Tùy chọn: mô hình sinh sau GAN — khuếch tán và flow matching

> Bài này là **tùy chọn**. Nó **không** thay nền tảng hợp lý / MLE / đánh giá của chương, và **không** lấy chỗ các chương VAE hay GAN. Nó ghi nhận bước chuyển 2022–2026: **khuếch tán theo score** và **flow matching** trở thành họ sinh mặc định cho ảnh (rồi video/âm thanh).

Ý tưởng cốt lõi vẫn đúng: mô hình sinh là phân phối $$p_\theta(\mathbf{x})$$ bạn lấy mẫu được và, lý tưởng, chấm điểm được. Cái đổi là *họ* $$p_\theta$$ nào được giao hàng.

## 1. Khuếch tán khử nhiễu (bức tranh toàn cảnh)

[Ho et al., 2020](https://arxiv.org/abs/2006.11239) (DDPM) và làn sóng latent diffusion 2022 ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752), Stable Diffusion) huấn luyện bộ khử nhiễu $$\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$$ trên quá trình nhiễu hóa thuận. Loss rút gọn thường là

$$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\varepsilon}}\big\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\big\|^2, \quad
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\varepsilon}.$$

[Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748) (DiT) đặt bộ khử nhiễu trên Transformer; [Esser et al., 2024](https://arxiv.org/abs/2403.03206) (SD3) chuyển dần sang mục tiêu flow-matching / rectifier trong không gian ẩn.

## 2. Flow matching

[Lipman et al., 2023](https://arxiv.org/abs/2210.02747) huấn luyện trường vector $$v_\theta(\mathbf{x}, t)$$ dọc đường xác suất giữa nhiễu và dữ liệu. Đường thẳng $$\mathbf{x}_t = (1-t)\mathbf{z} + t\mathbf{x}$$ cho hồi quy

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,\mathbf{x},\mathbf{z}}\big\| v_\theta(\mathbf{x}_t, t) - (\mathbf{x} - \mathbf{z}) \big\|^2.$$

Đây là câu chuyện “ODE sạch hơn” 2023–2025 mà nhiều mô hình ảnh/video mới quảng bá (Stable Diffusion 3, hệ lớp Flux).

## 3. Ứng dụng cụ thể

- **Text-to-image / video**: Stable Diffusion, SDXL, SD3, khuếch tán video thương mại.
- **Khoa học dữ liệu**: tăng cường ảnh/bảng tổng hợp khi riêng tư cấm mẫu thật (vẫn đánh giá bằng ý FID / mật độ của chương này).
- **Đừng bỏ AE/GAN**: autoencoder (Chương 12) trở thành *latent* của khuếch tán; GAN (Chương 14) còn cho khuôn mặt, inversion, và generator một bước nhanh.

## 4. Phần mềm phổ biến

- [Hugging Face diffusers](https://github.com/huggingface/diffusers) — pipeline DDPM, LDM, flow-matching.
- [Stability-AI / SD3](https://github.com/Stability-AI/sd3) và [black-forest-labs/flux](https://github.com/black-forest-labs/flux).
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion) — trainer tham chiếu 2022.

## 5. Trích dẫn (2022–2026)

- [Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752).
- [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748).
- [Flow Matching (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747).
- [SD3 (Esser et al., 2024)](https://arxiv.org/abs/2403.03206).

## 6. Bài này bổ sung gì cho ghi chú cốt lõi

Học MLE, mật độ hiện/ẩn, và GAN đồ chơi trong chương này trước. Khuếch tán và flow matching là **cập nhật cây họ**, không thay các định nghĩa đó. Chi tiết VAE và GAN ở lại Chương 13–14.
