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

Có một sự thật thú vị ít người để ý: dù Stable Diffusion, Sora hay Veo nổi tiếng nhờ mô hình khuếch tán và Transformer, nhưng bức ảnh hay thước phim cuối cùng bạn nhìn thấy trên màn hình lại luôn được "vẽ ra" bởi một **VAE**! Thay vì cạnh tranh trực tiếp với Diffusion về độ sắc nét, VAE hiện đại đã tìm thấy vị trí không thể thay thế trong vai trò bộ mã hóa – giải mã không gian ẩn cho ảnh độ phân giải cao, video 3D và các bộ sinh một bước.

## Kiến thức cần nhớ: Vai trò chiến lược của VAE trong hệ sinh thái tạo sinh 2022–2026

- **KL-VAE + Perceptual/GAN Loss trong Stable Diffusion:** Kết hợp trọng số KL rất nhỏ ($$\sim 10^{-6}$$) với hàm mất mát cảm nhận (LPIPS) và bộ phân biệt PatchGAN để xóa bỏ nhược điểm ảnh mờ của VAE cổ điển.
- **Causal 3D VAE cho mô hình sinh video:** Mở rộng nén theo cả không gian lẫn thời gian (ví dụ $$4\times$$ thời gian và $$8\times 8$$ không gian) giúp nén 1 đoạn video thành chuỗi token gọn nhẹ cho DiT.
- **VQ-VAE và mã hóa rời rạc (Discrete Tokenizers):** Ánh xạ vector ẩn vào sổ mã hữu hạn (codebook) để ghép nối trực tiếp ảnh/video với mô hình ngôn ngữ tự hồi quy.
- **Chưng cất khuếch tán và Consistency Models trong không gian ẩn:** Thực hiện sinh ảnh trong 1–4 bước ngay trên không gian ẩn của VAE, giảm độ trễ từ vài giây xuống dưới 100 mili-giây.

### Giải thích: Tính hệ số nén của Causal 3D VAE trên một đoạn video ngắn

Xét một đoạn video 4 giây ở tốc độ 24 khung hình/giây ($$T = 96$$ khung hình), độ phân giải $$512 \times 512$$ với 3 kênh màu RGB.

1. **Số giá trị pixel thô:**
   $$96 \times 512 \times 512 \times 3 = 75{,}497{,}472 \text{ giá trị (}\approx 75{,}5\text{ triệu số).}$$
2. **Qua Causal 3D VAE** với hệ số nén $$4\times$$ theo thời gian, $$8\times$$ theo chiều cao, $$8\times$$ theo chiều rộng và $$C_z = 16$$ kênh ẩn:
   - Kích thước tensor ẩn: $$(96/4) \times (512/8) \times (512/8) \times 16 = 24 \times 64 \times 64 \times 16 = 1{,}572{,}864$$ giá trị.
3. **Tỷ lệ nén tổng thể:**
   $$\frac{75{,}497{,}472}{1{,}572{,}864} = 48\times.$$
   Quan trọng hơn, số vị trí không-thời gian giảm từ $$96 \times 512 \times 512$$ xuống $$24 \times 64 \times 64 = 98{,}304$$ (giảm $$256\times$$) — biến việc huấn luyện Diffusion Transformer trên video từ bất khả thi thành khả thi trên cụm GPU hiện đại.

# Tùy chọn: VAE trong vai trò không gian ẩn cho khuếch tán, video và nén

> Bài này mang tính **tùy chọn** và **không** thay thế phần ELBO, thủ thuật tái tham số hóa hay cài đặt VAE cốt lõi. Nội dung cho thấy cùng kiến trúc autoencoder biến phân đã trở thành **bộ nén (codec)** chuẩn trong các hệ thống sinh giai đoạn 2022–2026 như thế nào.

Mục tiêu ELBO:

$$\mathcal{L}(\theta,\phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x)\,\|\,p(z))$$

vẫn giữ nguyên vai trò nền tảng. Trong các hệ thống thực tế hiện nay, VAE ít được dùng để lấy mẫu trực tiếp từ $$z\sim\mathcal{N}(0,I)$$ mà thường đóng vai trò nén dữ liệu sang không gian ẩn $$z$$ cho một mô hình **thứ hai** (khuếch tán hoặc Transformer) hoạt động hiệu quả hơn.

## 1. VAE trong Stable Diffusion

[Rombach et al., 2022](https://arxiv.org/abs/2112.10752) huấn luyện một autoencoder chính quy hóa KL (thường gọi là `AutoencoderKL`) với đầu ra ẩn là một lưới không gian giảm kích thước thay vì vector 1D. Số hạng KL tương ứng trực tiếp với chính quy hóa trong Chương 13, còn mất mát cảm nhận và đối kháng giúp giữ độ sắc nét tái tạo cho mạng khử nhiễu hoạt động ở phía sau. Phần lớn các pipeline sinh ảnh từ văn bản mã nguồn mở (SD, SDXL và nhiều hệ thống thế hệ SD3) đều áp dụng thiết kế này.

## 2. Một số ứng dụng tiêu biểu

### Không gian ẩn cho video và dữ liệu 3D

Stable Video Diffusion và các mô hình sinh video sau này tái sử dụng VAE ảnh hoặc huấn luyện VAE **không gian–thời gian** (dùng tích chập 3D hoặc tích chập nhân quả) để nén chuỗi khung hình thành ít token hơn trước khi đưa vào bộ khử nhiễu.

### Mô hình nhất quán (*consistency models*) và bộ sinh một bước trong không gian ẩn

[Song et al., 2023](https://arxiv.org/abs/2303.01469) (consistency models) và Latent Consistency Models (LCM) chưng cất quá trình giải ODE khuếch tán ngay trong không gian ẩn của VAE. Trong các hệ thống này, VAE được giữ cố định và phần cải tiến tập trung vào bộ lấy mẫu — cho thấy VAE vẫn là thành phần cốt lõi ngay cả khi khuếch tán chiếm ưu thế.

### Học biểu diễn

Hướng tách nhân tố bằng $$\beta$$-VAE không còn là tâm điểm như giai đoạn 2018, nhưng nút thắt **biến phân** vẫn xuất hiện đều đặn trong các mô hình thế giới (*world models* như DreamerV3, xem bài tùy chọn Chương 21) và bài toán nén dữ liệu học được.

## 3. Thư viện và mã nguồn phổ biến

- [diffusers](https://github.com/huggingface/diffusers) với `AutoencoderKL` và `AutoencoderTiny` (TAESD).
- Các checkpoint VAE của Stability-AI trên Hugging Face Hub.
- [openai/consistency_models](https://github.com/openai/consistency_models).

## 4. Tài liệu tham khảo (2022–2026)

- [Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752).
- [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01469).
- [SD3 (Esser et al., 2024)](https://arxiv.org/abs/2403.03206) — tiếp tục kết hợp autoencoder ẩn với flow matching.

## 5. Phần bổ sung so với nội dung cốt lõi

Bạn nên tự suy ra ELBO và nắm vững thủ thuật tái tham số hóa trước. Bài này chỉ minh họa **không gian ẩn $$z$$ được khai thác ở đâu** trong các hệ thống giai đoạn 2022–2026 chứ không thay đổi cận dưới ELBO cơ bản.

## Video tham khảo

- **Xem trước:** [How does Stable Diffusion work? – Latent Diffusion Models EXPLAINED](https://www.youtube.com/watch?v=J87hffSMB60) — AI Coffee Break with Letitia, tiếng Anh. Giải thích vai trò của VAE trong Stable Diffusion như phần trình bày của bài.
- [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — Arxiv Insights, tiếng Anh. Ôn lại mục tiêu ELBO mà VAE vẫn tối ưu khi làm bộ nén cho khuếch tán.
- [But how do AI images and videos actually work? | Guest video by Welch Labs](https://www.youtube.com/watch?v=iv-5mZ_9CPY) — 3Blue1Brown, tiếng Anh. Minh họa cách mô hình khuếch tán thao tác trên không gian ẩn của VAE.
