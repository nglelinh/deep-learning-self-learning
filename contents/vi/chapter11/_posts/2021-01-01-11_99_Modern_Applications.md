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

# Tùy chọn: Mô hình sinh sau GAN — Khuếch tán và Flow Matching

> Bài này mang tính **tùy chọn**, **không** thay thế phần nền tảng về hàm hợp lý, MLE và đánh giá mô hình của chương, cũng **không** thay thế các chương về VAE hay GAN. Nội dung ghi nhận sự chuyển dịch lớn giai đoạn 2022–2026: **mô hình khuếch tán dựa trên score** và **flow matching** đã trở thành lựa chọn mặc định cho sinh ảnh, sau đó mở rộng sang video và âm thanh.

Nguyên lý cốt lõi vẫn giữ nguyên: một mô hình sinh biểu diễn phân phối $$p_\theta(\mathbf{x})$$ cho phép lấy mẫu và (lý tưởng nhất) đánh giá xác suất. Điểm thay đổi nằm ở *họ* mô hình $$p_\theta$$ được triển khai rộng rãi trong thực tế.

## 1. Mô hình khuếch tán khử nhiễu (tổng quan)

[Ho et al., 2020](https://arxiv.org/abs/2006.11239) (DDPM) và làn sóng khuếch tán không gian ẩn năm 2022 ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752), Stable Diffusion) huấn luyện mạng khử nhiễu $$\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)$$ đảo ngược một quá trình thêm nhiễu thuận. Hàm mất mát rút gọn thường có dạng:

$$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\varepsilon}}\big\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\big\|^2, \quad
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\varepsilon}.$$

[Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748) (DiT) thay mạng khử nhiễu bằng kiến trúc Transformer, còn [Esser et al., 2024](https://arxiv.org/abs/2403.03206) (SD3) chuyển dần sang mục tiêu flow matching / rectified flow trong không gian ẩn.

## 2. Flow Matching

[Lipman et al., 2023](https://arxiv.org/abs/2210.02747) huấn luyện trường vector vận tốc $$v_\theta(\mathbf{x}, t)$$ dọc theo đường xác suất nối giữa phân phối nhiễu và phân phối dữ liệu. Với quỹ đạo nội suy thẳng $$\mathbf{x}_t = (1-t)\mathbf{z} + t\mathbf{x}$$, mục tiêu hồi quy có dạng:

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,\mathbf{x},\mathbf{z}}\big\| v_\theta(\mathbf{x}_t, t) - (\mathbf{x} - \mathbf{z}) \big\|^2.$$

Hướng tiếp cận theo phương trình vi phân thường (ODE) với quỹ đạo thẳng hơn này là nền tảng của nhiều hệ thống sinh ảnh và video giai đoạn 2023–2025 như Stable Diffusion 3 và dòng mô hình Flux.

## 3. Ứng dụng tiêu biểu

- **Sinh ảnh và video từ văn bản:** Stable Diffusion, SDXL, SD3 và các hệ thống khuếch tán video thương mại.
- **Khoa học dữ liệu:** tạo ảnh hoặc dữ liệu bảng tổng hợp khi quy định bảo mật không cho phép dùng trực tiếp dữ liệu thật (vẫn đánh giá bằng các chỉ số như FID và mật độ đã trình bày trong chương).
- **Vai trò tiếp nối của AE và GAN:** autoencoder (Chương 12) đóng vai trò tạo không gian *ẩn* cho mô hình khuếch tán, còn GAN (Chương 14) vẫn hữu ích cho tổng hợp khuôn mặt, ánh xạ ngược (*inversion*) và các bộ sinh một bước tốc độ cao.

## 4. Thư viện và mã nguồn phổ biến

- [Hugging Face diffusers](https://github.com/huggingface/diffusers) — cung cấp pipeline cho DDPM, LDM và flow matching.
- [Stability-AI / SD3](https://github.com/Stability-AI/sd3) và [black-forest-labs/flux](https://github.com/black-forest-labs/flux).
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion) — mã nguồn huấn luyện tham chiếu giai đoạn 2022.

## 5. Tài liệu tham khảo (2022–2026)

- [Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752).
- [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748).
- [Flow Matching (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747).
- [SD3 (Esser et al., 2024)](https://arxiv.org/abs/2403.03206).

## 6. Phần bổ sung so với nội dung cốt lõi

Bạn nên học vững MLE, sự khác biệt giữa mật độ tường minh và mật độ ẩn, cũng như ví dụ GAN đơn giản trong chương này trước. Khuếch tán và flow matching là **bản cập nhật các họ mô hình hiện đại**, không thay thế các định nghĩa cơ bản đó; chi tiết về VAE và GAN vẫn nằm ở Chương 13 và 14.

## Video tham khảo

- **Xem trước:** [Flow-Matching vs Diffusion Models explained side by side](https://www.youtube.com/watch?v=firXjwZ_6KI) — AI Coffee Break with Letitia, tiếng Anh. So sánh trực tiếp flow matching và khuếch tán khử nhiễu, khớp với bức tranh giai đoạn 2022–2026 trong bài.
- [But how do AI images and videos actually work? | Guest video by Welch Labs](https://www.youtube.com/watch?v=iv-5mZ_9CPY) — 3Blue1Brown, tiếng Anh. Giải thích cách mô hình khuếch tán biến nhiễu thành ảnh, bao gồm cả kỹ thuật classifier-free guidance.
- [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) — AI Coffee Break with Letitia, tiếng Anh. Trình bày mô hình khuếch tán trong không gian ẩn — dạng triển khai thực tế phổ biến nhất của các ý tưởng này.
