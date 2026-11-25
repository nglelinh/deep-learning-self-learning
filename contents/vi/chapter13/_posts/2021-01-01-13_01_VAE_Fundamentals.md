---
layout: post
title: 13-01 Variational Autoencoder
chapter: '13'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter13
---

Nếu Autoencoder là chiếc máy ảnh kỹ thuật số chỉ biết nén và giải nén lại đúng bức ảnh bạn chụp, thì Variational Autoencoder giống như một nhà điêu khắc nắm được toàn bộ giải phẫu học: chỉ cần một tọa độ ngẫu nhiên từ phân phối chuẩn, nó có thể tạc nên một tác phẩm mới chưa từng tồn tại. Mục tổng quan này kết nối trực giác xác suất, nền tảng lý thuyết ELBO và bài cài đặt PyTorch thành một mạch học thống nhất.

## Kiến thức cần nhớ: Bốn mảnh ghép cốt lõi của bài học VAE

- **Sự khác biệt cấu trúc với Autoencoder:** Chuyển từ mã ẩn xác định $$\mathbf{z} = f(\mathbf{x})$$ sang tham số phân phối $$(\boldsymbol{\mu}(\mathbf{x}), \log\boldsymbol{\sigma}^2(\mathbf{x}))$$.
- **Cân bằng sinh mẫu và tái tạo:** Nắm vững vai trò của hai số hạng trong ELBO và cách tham số $$\beta$$ điều chỉnh sự đánh đổi giữa độ nét tái tạo và độ mượt không gian ẩn.
- **Liên thông lý thuyết – thực hành:** Đi từ chứng minh ELBO & Reparameterization (**13-01-01**) sang cài đặt PyTorch và trực quan hóa đa tạp 2D (**13-01-02**).
- **Lý do ảnh VAE thường hơi mờ:** Hiểu tác động của phân phối đầu ra độc lập theo pixel (factorized Gaussian/Bernoulli) lên chi tiết tần số cao, mở đường sang GAN và Diffusion.

### Giải thích: Lấy mẫu sinh ảnh mới từ phân phối tiên nghiệm trên VAE 2 chiều

Sau khi huấn luyện xong VAE với không gian ẩn $$d = 2$$, ta **bỏ hoàn toàn bộ mã hóa** và chỉ giữ bộ giải mã $$p_\theta(\mathbf{x}\mid\mathbf{z})$$. Để vẽ lưới $$3 \times 3$$ ảnh đại diện cho toàn bộ phân phối chuẩn $$\mathcal{N}(\mathbf{0}, \mathbf{I}_2)$$:

1. Chọn 3 mức xác suất tích lũy đều nhau: $$p \in \{0{,}10;\; 0{,}50;\; 0{,}90\}$$.
2. Tra hàm phân vị ngược (PPF) của phân phối chuẩn tắc $$\Phi^{-1}(p)$$:
   - $$\Phi^{-1}(0{,}10) \approx -1{,}282$$, $$\Phi^{-1}(0{,}50) = 0{,}000$$, $$\Phi^{-1}(0{,}90) \approx +1{,}282$$.
3. Ghép thành 9 tọa độ $$(z_1, z_2) \in \{-1{,}282, 0, +1{,}282\}^2$$ và truyền qua bộ giải mã. Vì mọi điểm đều nằm trong vùng xác suất dày đặc mà phân kỳ KL đã kéo các đám mây huấn luyện về, cả 9 ô ảnh đều cho ra chữ số hợp lệ và biến hình liên tục theo hai trục!

Nội dung về Variational Autoencoder gồm hai phần:

1. **Lý thuyết** — mục tiêu ELBO, thủ thuật tái tham số hóa và trực giác không gian ẩn.
2. **Cài đặt** — mã huấn luyện PyTorch, lấy mẫu, nội suy, bài báo nền tảng và các cạm bẫy thường gặp.

## Video tham khảo

- **Xem trước:** [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) — Arxiv Insights, tiếng Anh. Tổng quan ngắn gọn trước khi đi vào phần lý thuyết và cài đặt chi tiết.
- [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, tiếng Anh. Trình bày cùng mô hình xác suất trong bối cảnh khóa học của MIT.
- [MaDL - Kullback-Leibler Divergence](https://www.youtube.com/watch?v=vPvnpC0PNVQ) — Tübingen Machine Learning, tiếng Anh. Giải thích số hạng phân kỳ KL giữ vai trò then chốt trong cả hai bài học.
