---
layout: post
title: 11-01-02 Cài đặt Mô hình Sinh
chapter: '11'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter11
---

Đọc công thức MLE hay minimax thường mang lại cảm giác mọi thứ đều trơn tru, cho đến khi bạn bấm chạy vòng lặp huấn luyện đầu tiên và thấy đường loss nhảy múa không ngừng. Trang tổng hợp cài đặt này đóng vai trò cầu nối giữa hai thế giới: biến lý thuyết mô hình sinh thành mã nguồn chạy được trên bộ dữ liệu 8-Gaussian kinh điển để bạn trực tiếp so sánh mô hình tự hồi quy với GAN.

## Kiến thức cần nhớ: Lộ trình từ mô hình toán học sang thí nghiệm sinh dữ liệu

- **Thiết kế tập dữ liệu chuẩn đoán (Toy Benchmark):** Dùng hỗn hợp 8 phân phối chuẩn 2D đặt trên vòng tròn giúp trực quan hóa tức thì độ phủ mode, độ sắc nét và mật độ tràn mà không cần GPU lớn.
- **Nhánh 1 — Cài đặt tự hồi quy:** Thực hành tách biến, tối ưu NLL chính xác và sinh mẫu tuần tự để thấy ưu thế ổn định của mật độ tường minh.
- **Nhánh 2 — Cài đặt GAN và kiểm tra bẫy:** Thực hành vòng lặp đối kháng, đo độ phủ mode tự động và nhận diện lỗi huấn luyện phổ biến.
- **Đánh giá thực nghiệm đa góc nhìn:** Kết hợp cả thước đo định lượng (log-likelihood, số mode phủ được) lẫn trực quan hóa phân bố mẫu sinh.

### Giải thích: Cách tính nhanh chỉ số độ phủ mode (Mode Coverage) trên 8 cụm Gaussian

Giả sử 8 tâm cụm nằm trên vòng tròn bán kính $$R = 2{,}0$$ tại các góc $$\theta_k = \frac{2\pi k}{8}$$ ($$k = 0,\dots,7$$), mỗi cụm có độ lệch chuẩn $$\sigma = 0{,}1$$. Ta rút $$N = 800$$ mẫu từ mô hình và gán mỗi mẫu vào tâm gần nhất nếu khoảng cách Euclid $$d < 3\sigma = 0{,}3$$.

1. **Kỳ vọng lý tưởng:** Mỗi cụm nhận khoảng $$800 / 8 = 100$$ mẫu ($$12{,}5\%$$), tỷ lệ mẫu hợp lệ (nằm trong bán kính $$0{,}3$$ quanh một tâm) đạt gần $$99\%$$.
2. **Kịch bản GAN bị sụp đổ mode:** Giả sử đếm được số mẫu ở 8 cụm là `[390, 0, 0, 385, 0, 0, 0, 0]` và $$25$$ mẫu rơi ngoài bán kính $$0{,}3$$.
   - **Độ phủ mode:** chỉ $$2/8 = 25\%$$ (thất bại nghiêm trọng dù $$775/800 = 96{,}9\%$$ mẫu trông "rất thật"!).
3. **Kịch bản tự hồi quy đơn đỉnh:** Phủ được cả $$8/8$$ hướng nhưng có tới $$320/800 = 40\%$$ mẫu rơi vào khoảng trống giữa các cụm (chất lượng cao về độ phủ nhưng thấp về độ sắc nét). Con số đơn giản này cho thấy vì sao một chỉ số duy nhất không bao giờ đủ để đánh giá mô hình sinh.

Phần cài đặt mô hình sinh gồm hai bài:

1. **Mô hình tự hồi quy (Autoregressive Model)** — tạo tập dữ liệu hỗn hợp 2D đơn giản và huấn luyện mô hình tự hồi quy.
2. **GAN đơn giản, Bài báo và Bẫy thường gặp** — cài đặt GAN để so sánh trực tiếp, giới thiệu các chỉ số đánh giá, bài báo nền tảng và điểm then chốt.

## Video tham khảo

- **Xem trước:** [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy, tiếng Anh. Minh họa trực tiếp nhánh cài đặt mô hình tự hồi quy.
- [A Friendly Introduction to Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8) — Luis Serrano Academy, tiếng Anh. Giải thích cơ chế mô hình GAN đơn giản được huấn luyện ở bài thứ hai.
- [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, tiếng Anh. Giúp phân biệt rõ mục tiêu và hàm mất mát của từng họ mô hình khi cài đặt.
