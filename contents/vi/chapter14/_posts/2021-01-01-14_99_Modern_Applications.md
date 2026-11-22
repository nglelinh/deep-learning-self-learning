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

# Tùy chọn: GAN sau làn sóng khuếch tán — Khuôn mặt, Inversion và Bộ sinh một bước

> Bài này mang tính **tùy chọn** và **không** thay thế phần hàm mất mát GAN, lý thuyết DCGAN/StyleGAN hay các lưu ý huấn luyện trong chương. Nội dung tổng hợp những vai trò mà GAN vẫn đảm nhiệm tốt sau khi mô hình khuếch tán trở thành bộ lấy mẫu mặc định.

Công thức đối kháng cốt lõi:

$$\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]$$

không thay đổi; điều thay đổi là phạm vi ứng dụng. Sinh ảnh từ văn bản mở hiện chủ yếu dựa trên khuếch tán và flow matching, còn **GAN vẫn giữ ưu thế** khi cần sinh ảnh tức thì trong một bước, thực hiện ánh xạ ngược (*inversion*) hoặc khai thác không gian khuôn mặt đã được nghiên cứu sâu.

## 1. StyleGAN3 và các hệ thống tổng hợp khuôn mặt giai đoạn 2022–2023

[Karras et al., 2021](https://arxiv.org/abs/2106.12423) (StyleGAN3, ứng dụng rộng rãi giai đoạn 2022–2024) khắc phục hiện tượng kết cấu bị dính vào tọa độ ảnh (*texture sticking*) nhờ phép tích chập chống răng cưa (*alias-free*). Trong thực tế công nghiệp, StyleGAN2/3 vẫn được dùng cho các tác vụ chỉnh sửa khuôn mặt, thay đổi tuổi tác và bảo toàn danh tính vì không gian ẩn **W / W+** hỗ trợ ánh xạ ngược và điều chỉnh thuộc tính nhanh, ổn định hơn so với inversion trên mô hình khuếch tán.

## 2. Một số ứng dụng tiêu biểu

### GigaGAN và GAN sinh ảnh từ văn bản quy mô lớn

[Kang et al., 2023](https://arxiv.org/abs/2303.05511) (GigaGAN) chứng minh GAN vẫn có thể đạt chất lượng sinh ảnh từ văn bản tiệm cận mô hình khuếch tán chỉ với **một lượt truyền xuôi**. Kết quả này cho thấy huấn luyện đối kháng vẫn có tiềm năng mở rộng quy mô, dù phần lớn hệ thống mã nguồn mở ưu tiên khuếch tán nhờ tính ổn định cao hơn.

### Chưng cất mô hình: GAN làm thành phần huấn luyện cho mô hình khuếch tán một bước

Nhiều bộ sinh một bước hoặc vài bước (như SD-Turbo, các phương pháp chưng cất kết hợp LCM-LoRA, họ UFOGen) bổ sung đầu **đối kháng** giúp mô hình học sinh nhỏ khớp tốt với mô hình giáo viên khuếch tán nhiều bước. Ở đây, bộ phân biệt của Chương 14 đóng vai trò công cụ *chưng cất*, không phải mô hình sinh độc lập.

### Khi nào nên chọn GAN trong giai đoạn hiện nay

- Ứng dụng cần tốc độ thời gian thực như avatar, *talking-head* hoặc đồ họa game (*sprite*).
- Bổ sung dữ liệu cân bằng lớp trên miền hẹp đã có sẵn mô hình StyleGAN phù hợp.
- Không phải lựa chọn ưu tiên cho bài toán sinh ảnh từ văn bản tự do nhiều khái niệm mở.

## 3. Thư viện và mã nguồn phổ biến

- [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) và StyleGAN2-ADA.
- [mingukkang/GigaGAN](https://github.com/mingukkang/GigaGAN).
- [diffusers](https://github.com/huggingface/diffusers) — cung cấp một số checkpoint turbo sử dụng chưng cất đối kháng.

## 4. Tài liệu tham khảo (2022–2026)

- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423).
- [GigaGAN (Kang et al., 2023)](https://arxiv.org/abs/2303.05511).
- [Adversarial Diffusion Distillation (Sauer et al., 2023)](https://arxiv.org/abs/2311.17042) — nền tảng của SD-Turbo.

## 5. Phần bổ sung so với nội dung cốt lõi

Bạn nên tự huấn luyện mô hình GAN cơ bản và nắm rõ hiện tượng sụp đổ mode trước. Bài này chỉ giải thích vì sao StyleGAN và huấn luyện đối kháng vẫn tiếp tục hiện diện trong thực tế sản xuất ngay cả khi các hệ thống sinh ảnh tổng quát đã chuyển sang khuếch tán.

## Video tham khảo

- **Xem trước:** [AI generated faces - StyleGAN explained!](https://www.youtube.com/watch?v=4LNO8nLxF4Y) — AI Bites, tiếng Anh. Giải thích kiến trúc StyleGAN tương ứng với phần ứng dụng tổng hợp khuôn mặt trong bài.
- [A Friendly Introduction to Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8) — Luis Serrano Academy, tiếng Anh. Ôn lại cơ chế trò chơi hai người mà StyleGAN vẫn kế thừa.
- [What are GANs (Generative Adversarial Networks)?](https://www.youtube.com/watch?v=TpMIssRdhco) — IBM Technology, tiếng Anh. Tóm tắt nhanh giúp đối chiếu điểm khác biệt giữa không gian phong cách của StyleGAN và GAN chuẩn.
