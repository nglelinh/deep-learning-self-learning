---
layout: post
title: 12-01 Nền tảng Autoencoder
chapter: '12'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter12
---

Trong thế giới dữ liệu khổng lồ hôm nay, phần lớn ảnh, âm thanh và văn bản đều không có nhãn. Làm thế nào để máy tính tự học được bản chất của thế giới mà không cần con người ngồi gắn nhãn từng mẫu một? Autoencoder là câu trả lời kinh điển và thanh lịch nhất: biến chính dữ liệu đầu vào thành đích đến, dùng áp lực nén để chắt lọc tri thức.

## Bản đồ kiến thức nền: Khung tổng quan nền tảng Autoencoder

- **Tư duy nén có cấu trúc:** Phân biệt nén dữ liệu tổng quát (như ZIP/JPEG) với nén học được theo miền dữ liệu (Autoencoder chỉ nén xuất sắc đúng loại dữ liệu nó đã thấy).
- **Phổ các biến thể cốt lõi:** Nắm vững vai trò riêng của Undercomplete, Denoising, Sparse và Convolutional Autoencoder.
- **Đánh giá biểu diễn không giám sát:** Đánh giá không chỉ qua sai số tái tạo mà còn qua khả năng tách cụm tuyến tính (linear probing) và phát hiện bất thường.
- **Điểm dừng của Autoencoder xác định:** Nhận rõ ranh giới giữa *học biểu diễn* (Autoencoder làm rất tốt) và *sinh dữ liệu mới* (cần nâng cấp lên VAE ở Chương 13).

### Khung giải thích (nên đọc chậm): So sánh tỷ lệ nén của Autoencoder MNIST ở ba mức kích thước nút cổ chai

Một ảnh MNIST có $$784$$ số thực (32-bit float = $$3{,}136\text{ byte}$$). Xét 3 cấu hình nút cổ chai $$d$$:

1. **$$d = 64$$:** Tỷ lệ nén $$784 / 64 = 12{,}25\times$$. Bộ nhớ mỗi mã ẩn chỉ $$256\text{ byte}$$; ảnh tái tạo gần như không phân biệt được với ảnh gốc, phù hợp cho khử nhiễu hoặc rút trích đặc trưng.
2. **$$d = 16$$:** Tỷ lệ nén $$784 / 16 = 49\times$$. Ảnh tái tạo giữ tốt hình dáng chữ số nhưng bắt đầu mất các nét rung tay nhỏ — điểm cân bằng tốt để phát hiện bất thường.
3. **$$d = 2$$:** Tỷ lệ nén $$784 / 2 = 392\times$$. Mọi ảnh chỉ còn đúng một tọa độ $$(z_1, z_2)$$ vẽ được trực tiếp lên mặt phẳng 2D, nhưng ảnh tái tạo sẽ khá mờ vì 2 số thực không đủ mã hóa độ nghiêng, độ dày và kiểu chữ của 10 lớp cùng lúc.

Nội dung nền tảng về autoencoder gồm hai phần:

1. **Lý thuyết** — tổng quan khái niệm, cơ sở toán học và ví dụ trực quan.
2. **Cài đặt** — mã nguồn PyTorch, trực quan hóa không gian ẩn, bài báo nền tảng và các cạm bẫy thường gặp.

## Video tham khảo

- **Xem trước:** [What are Autoencoders?](https://www.youtube.com/watch?v=qiUEgSCyY5o) — IBM Technology, tiếng Anh. Trình bày trực quan ba thành phần: bộ mã hóa, nút thắt và bộ giải mã.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, tiếng Anh. Giải thích hàm mất mát tái tạo dùng để huấn luyện kiến trúc này.
- [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, tiếng Anh. Làm rõ vì sao nút thắt tất định chưa đủ để tạo thành một mô hình sinh hoàn chỉnh, dẫn sang chương sau.
