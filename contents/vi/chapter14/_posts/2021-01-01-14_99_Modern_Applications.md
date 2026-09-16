---
layout: post
title: 14-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '14'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter14
lesson_type: optional
---

# Tùy chọn: GAN sau khuếch tán — khuôn mặt, inversion, và generator một bước

> Bài này là **tùy chọn**. Nó **không** thay loss GAN, lý thuyết DCGAN/StyleGAN, hay ghi chú pitfalls huấn luyện. Nó ghi nhận GAN vẫn *dùng để làm gì* sau khi khuếch tán thành bộ lấy mẫu mặc định.

Cặp đối kháng

$$\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]$$

không đổi. Thị phần đổi: text-to-image chủ yếu khuếch tán/flow; **GAN còn** khi cần lấy mẫu tức thì, inversion, hoặc prior khuôn mặt đã nghiên cứu kỹ.

## 1. StyleGAN3 và stack khuôn mặt 2022–2023

[Karras et al., 2021](https://arxiv.org/abs/2106.12423) (StyleGAN3, dùng suốt 2022–2024) bỏ dính texture nhờ conv không alias. Công nghiệp vẫn dùng StyleGAN2/3 cho danh tính, tuổi tác, chỉnh sửa vì latent **W / W+** khả nghịch và chỉnh được. Inversion khuếch tán tồn tại, nhưng inversion StyleGAN rẻ và chín hơn.

## 2. Ứng dụng cụ thể

### GigaGAN và GAN text-to-image quy mô lớn

[Kang et al., 2023](https://arxiv.org/abs/2303.05511) (GigaGAN) cho thấy GAN có thể đạt chất lượng text-to-image gần khuếch tán với **một lượt thuận**. Bài báo là bằng chứng huấn luyện đối kháng không chết; hầu hết sản phẩm mở vẫn chọn khuếch tán vì ổn định.

### Chưng cất: GAN làm học sinh của khuếch tán

Nhiều generator một bước (SD-Turbo, chưng cất kiểu LCM-LoRA, họ UFOGen) thêm đầu **đối kháng** để học sinh nhỏ khớp giáo viên chậm. Đó là discriminator Chương 14 dùng như công cụ *chưng cất*, không phải mô hình sinh duy nhất.

### Khi nào chọn GAN năm 2026

- Avatar thời gian thực, talking-head, sprite game.
- Cân bằng lại tập khi đã fit StyleGAN.
- Không phải lựa chọn đầu cho “một khối đỏ trên mặt trăng” mở từ vựng.

## 3. Phần mềm phổ biến

- [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) và StyleGAN2-ADA.
- [mingukkang/GigaGAN](https://github.com/mingukkang/GigaGAN).
- [diffusers](https://github.com/huggingface/diffusers) — một số checkpoint turbo/đối kháng.

## 4. Trích dẫn (2022–2026)

- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423).
- [GigaGAN (Kang et al., 2023)](https://arxiv.org/abs/2303.05511).
- [Adversarial Diffusion Distillation (Sauer et al., 2023)](https://arxiv.org/abs/2311.17042) — SD-Turbo.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Huấn luyện GAN đồ chơi và đọc ghi chú mode-collapse trước. Bài này chỉ trả lời “vì sao StyleGAN vẫn trong production trong khi demo lớp DALL·E dùng khuếch tán?”
