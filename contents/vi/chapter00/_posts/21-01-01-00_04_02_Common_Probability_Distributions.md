---
layout: post
title: 00-04-02 Các Phân Phối Xác Suất Thông Dụng
chapter: '00'
order: 19
owner: AI Assistant
lang: vi
categories:
- chapter00
---

Trong tự nhiên và dữ liệu số, sự ngẫu nhiên không hề hỗn loạn vô định—chúng thường lặp lại theo một số ít "khuôn mẫu toán học" kinh điển gọi là các **phân phối xác suất**. Việc nhận diện đúng bài toán của bạn thuộc khuôn mẫu nào (nhị phân Bernoulli, đếm sự kiện Poisson, đo lường liên tục Gaussian hay vector tương quan Multivariate Normal) quyết định trực tiếp việc chọn hàm kích hoạt đầu ra, hàm mất mát và kiến trúc mô hình sinh. Bài tổng quan này phác thảo bản đồ ba nhóm phân phối bạn sẽ gặp đi gặp lại xuyên suốt khóa học.

## Bản đồ kiến thức nền: Ba họ Phân phối Xác suất thiết yếu trong Học sâu

1. **Họ phân phối Rời rạc (Bernoulli, Binomial, Categorical, Poisson)**:
   Mô hình hóa nhãn lớp, token ngôn ngữ, mặt nạ Dropout (Bernoulli) và số lần xuất hiện sự kiện; trực tiếp sinh ra Binary Cross-Entropy và Categorical Cross-Entropy.

2. **Họ phân phối Liên tục một biến (Uniform, Normal/Gaussian, Exponential)**:
   Cơ sở cho khởi tạo trọng số (Uniform/Normal), mô hình hóa nhiễu đo lường trong bài toán hồi quy (Gaussian $$\to$$ MSE), và lấy mẫu bước thời gian/nhiễu trong Diffusion Models.

3. **Phân phối Đa biến (Multivariate Normal & Covariance Matrix)**:
   Nắm bắt mối tương quan đồng thời giữa nhiều đặc trưng, làm nền cho không gian ẩn (*latent space*) của Variational Autoencoder (VAE), Gaussian Mixture Models và phép làm trắng dữ liệu (*whitening*).

### Khung giải thích (nên đọc chậm): Ánh xạ nhanh 4 bài toán Deep Learning sang đúng phân phối và hàm mất mát

| Bài toán Deep Learning | Đầu ra cần dự đoán | Phân phối xác suất phù hợp | Hàm kích hoạt cuối & Loss tương ứng |
| :--- | :--- | :--- | :--- |
| **Lọc email rác** | Nhãn $$y \in \{0, 1\}$$ | Bernoulli$$(p)$$ | Sigmoid + Binary Cross-Entropy |
| **Nhận diện 10 chữ số MNIST** | Nhãn $$y \in \{0,\dots,9\}$$ | Categorical$$(\mathbf{p})$$ | Softmax + Categorical Cross-Entropy |
| **Dự báo giá nhà** | Số thực $$y \in \mathbb{R}$$ | Gaussian$$(\mu, \sigma^2)$$ | Linear + MSE (Mean Squared Error) |
| **Sinh ảnh bằng VAE** | Vector ẩn $$\mathbf{z} \in \mathbb{R}^d$$ | Multivariate Normal$$(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$$ | KL-Divergence + Reconstruction Loss |

Mỗi khi thiết kế một mạng nơ-ron mới, hãy tự hỏi: *"Dữ liệu đầu ra của mình sinh ra từ phân phối nào?"*—câu trả lời sẽ tự động chỉ ra kiến trúc tầng cuối và hàm mất mát chuẩn xác nhất.

Bài học này khảo sát các phân phối xác suất thường dùng trong học sâu, chia thành ba phần:

1. **Phân phối rời rạc**
2. **Phân phối liên tục**
3. **Phân phối đa biến và ứng dụng**

## Video tham khảo

- **Xem trước:** [The Main Ideas behind Probability Distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc) — StatQuest with Josh Starmer, tiếng Anh. Giới thiệu khái niệm phân phối xác suất trước khi đi vào các nhóm phân phối rời rạc, liên tục và đa biến.
- [MaDL - Common Probability Distributions](https://www.youtube.com/watch?v=NbzVwvE_syQ) — Tübingen Machine Learning, tiếng Anh. Điểm qua các họ phân phối thông dụng sẽ được trình bày chi tiết trong từng bài sau.
- [The Normal Distribution, Clearly Explained!!!](https://www.youtube.com/watch?v=rzFX5NWojp0) — StatQuest with Josh Starmer, tiếng Anh. Giải thích phân phối chuẩn (Gauss) — phân phối liên tục xuất hiện xuyên suốt các bài tiếp theo.
