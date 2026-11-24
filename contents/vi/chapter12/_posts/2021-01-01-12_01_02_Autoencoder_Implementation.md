---
layout: post
title: 12-01-02 Cài đặt Autoencoder
chapter: '12'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter12
---

Hiểu một Autoencoder trên giấy chỉ là nửa chặng đường; nửa còn lại thú vị hơn nhiều: nhìn một bức ảnh nhiễu hạt biến thành chữ số sắc nét ngay trên màn hình, rồi tự tay xoay vòng trong không gian ẩn 2D để xem máy tính tự sắp xếp 10 chữ số ra sao. Mục cài đặt này gom gọn hai phần thực hành trọng tâm: xây dựng các biến thể Autoencoder cốt lõi và khảo sát cấu trúc không gian ẩn.

## Bản đồ kiến thức nền: Bản đồ thực hành cài đặt Autoencoder

- **Phần lõi kiến trúc (12-01-02-01):** Triển khai đầy đủ Vanilla, Denoising và Convolutional Autoencoder trên MNIST với vòng lặp huấn luyện sạch và so sánh sai số tái tạo.
- **Phần phân tích không gian ẩn (12-01-02-02):** Thực hiện nội suy tuyến tính, vẽ bản đồ mã ẩn 2D theo màu nhãn, và chẩn đoán các bẫy kỹ thuật.
- **Chuẩn hóa dữ liệu và hàm mất mát:** Đối chiếu trực tiếp sự phù hợp giữa thang giá trị ảnh (`[0, 1]` vs `[-1, 1]`), hàm kích hoạt cuối và hàm loss (BCE vs MSE).
- **Chẩn đoán lỗi nhanh:** Quy trình kiểm tra nhanh kích thước tensor qua từng tầng tích chập/giải tích chập và phát hiện hiện tượng học thuộc.

### Khung giải thích (nên đọc chậm): Kiểm tra nhanh sự khác biệt giữa BCE và MSE trên một điểm ảnh xám

Giả sử điểm ảnh thật là $$x = 0{,}9$$ và bộ giải mã dự đoán $$\hat{x} = 0{,}2$$ (sai khá nặng).

1. **Với hàm MSE:** $$\mathcal{L}_{\text{MSE}} = (0{,}9 - 0{,}2)^2 = 0{,}49$$, đạo hàm theo $$\hat{x}$$ là $$2(\hat{x} - x) = -1{,}4$$.
   Nếu đi qua hàm Sigmoid $$\hat{x} = \sigma(u)$$ có $$\sigma'(u) = \hat{x}(1 - \hat{x}) = 0{,}2 \times 0{,}8 = 0{,}16$$, thì gradient theo logit $$u$$ bị co lại chỉ còn: $$-1{,}4 \times 0{,}16 = -0{,}224$$.
2. **Với hàm BCE:** $$\mathcal{L}_{\text{BCE}} = -[0{,}9\ln(0{,}2) + 0{,}1\ln(0{,}8)] \approx 1{,}471$$, nhưng điều kỳ diệu nằm ở đạo hàm theo logit $$u$$ (triệt tiêu nhân tử $$\hat{x}(1-\hat{x})$$):
   $$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial u} = \hat{x} - x = 0{,}2 - 0{,}9 = -0{,}700.$$
3. **Bài học thực hành:** Gradient của BCE mạnh hơn gấp **3 lần** MSE ở vùng bão hòa Sigmoid, giúp Autoencoder hội tụ nhanh và sắc nét hơn rõ rệt trên dữ liệu ảnh chuẩn hóa về $$[0, 1]$$.

Phần cài đặt autoencoder gồm hai bài:

1. **Cài đặt Cốt lõi** — xây dựng, huấn luyện và đánh giá tái tạo autoencoder chuẩn và khử nhiễu trên MNIST.
2. **Không gian Ẩn, Bài báo và Bẫy thường gặp** — nội suy mã ẩn, trực quan hóa không gian ẩn 2D, autoencoder tích chập, khái niệm liên quan và điểm then chốt.

## Video tham khảo

- **Xem trước:** [What are Autoencoders?](https://www.youtube.com/watch?v=qiUEgSCyY5o) — IBM Technology, tiếng Anh. Ôn lại kiến trúc encoder–decoder tương ứng với phần mã cài đặt.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, tiếng Anh. Giải thích hàm mất mát tái tạo và cách trực quan hóa không gian ẩn trong các bài thực hành.
- [PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) — Patrick Loeber, tiếng Anh. Cấu trúc vòng lặp huấn luyện PyTorch áp dụng cho bộ mã hóa và bộ giải mã.
