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

Liệu mô hình khuếch tán đã "khai tử" hoàn toàn GAN? Nếu bạn mở mã nguồn của Stable Diffusion XL Turbo, SDXL-Lightning hay hệ thống nén ảnh/video thế hệ mới, bạn sẽ ngạc nhiên khi thấy **bộ phân biệt đối kháng (Discriminator)** vẫn đang âm thầm làm việc ở tuyến đầu! Bài cập nhật này giải thích sự chuyển mình của GAN giai đoạn 2022–2026: từ vị trí bộ sinh độc lập sang "vũ khí bí mật" cho tổng hợp khuôn mặt thời gian thực, GAN Inversion và chưng cất mô hình khuếch tán một bước.

## Kiến thức cần nhớ: Ba mặt trận hiện đại nơi tư duy GAN vẫn tỏa sáng

- **StyleGAN3 và bất biến dịch chuyển/xoay:** Loại bỏ hiện tượng "dính kết cấu" (texture sticking) bằng xử lý tín hiệu chống răng cưa, phục vụ chỉnh sửa khuôn mặt và nhân vật ảo thời gian thực.
- **GAN Inversion và chỉnh sửa ngữ nghĩa trong không gian $$\mathcal{W}+$$:** Tìm ngược mã ẩn $$\mathbf{w}^*$$ tái tạo chính xác một bức ảnh thật rồi trượt theo các hướng thuộc tính (tuổi tác, ánh sáng, biểu cảm).
- **GigaGAN và bộ sinh quy mô tỷ tham số:** Chứng minh GAN có thể mở rộng lên sinh ảnh từ văn bản với tốc độ dưới 0,2 giây/ảnh.
- **Chưng cất đối kháng (ADD / DMD / SDXL-Turbo):** Dùng bộ phân biệt đối kháng ép mô hình khuếch tán sinh ảnh sắc nét chỉ trong **1 đến 4 bước** thay vì 50 bước.

### Giải thích: So sánh độ trễ suy luận: Diffusion 50 bước vs. Diffusion chưng cất GAN 1–4 bước

Giả sử một lượt suy luận mạng UNet/DiT trên ảnh $$1024 \times 1024$$ tốn $$45\text{ ms}$$ trên một GPU tầm trung, và bước giải mã VAE cuối cùng tốn $$25\text{ ms}$$.

1. **Diffusion chuẩn (50 bước, có Classifier-Free Guidance chạy 2 nhánh/bước):**
   - Thời gian khử nhiễu: $$50 \times 2 \times 45\text{ ms} = 4{,}500\text{ ms}$$.
   - Tổng thời gian sinh 1 ảnh: $$4{,}500 + 25 = 4{,}525\text{ ms} \approx 4{,}53\text{ giây}$$.
2. **Mô hình chưng cất đối kháng (kiểu Adversarial Diffusion Distillation — 2 bước, không cần CFG):**
   - Thời gian khử nhiễu: $$2 \times 45\text{ ms} = 90\text{ ms}$$.
   - Tổng thời gian sinh 1 ảnh: $$90 + 25 = 115\text{ ms} \approx 0{,}115\text{ giây}$$.
3. **Hiệu quả tăng tốc:** $$4{,}525 / 115 \approx 39{,}3\times$$ — biến trải nghiệm tạo ảnh từ "chờ gần 5 giây" thành **tương tác tức thời theo từng nét vẽ**, nhờ chính hàm loss đối kháng của GAN đảm bảo độ sắc nét ngay ở bước 1–2!

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
