---
layout: post
title: 14-01 Mạng Sinh Đối kháng
chapter: '14'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter14
---

Giữa ba đại gia mô hình sinh cổ điển — Autoregressive, VAE và GAN — GAN luôn là cái tên gây ấn tượng thị giác mạnh nhất ngay từ cái nhìn đầu tiên nhờ độ sắc nét vượt trội. Trang nền tảng này kết nối lý thuyết đối kháng (**14-01-01**) với thực hành cài đặt PyTorch (**14-01-02**), giúp bạn nắm chắc khi nào nên dùng GAN và cách đọc vị đường cong loss đầy biến động của nó.

## Bản đồ kiến thức nền: Khung tư duy cốt lõi về Mạng Sinh Đối kháng

- **Tư duy học hàm mất mát thay vì cố định hàm mất mát:** Bộ phân biệt $$D$$ thực chất đóng vai trò một *hàm mất mát học được* tự thích nghi theo từng lỗi tinh vi nhất của ảnh sinh.
- **Nhịp độ cân bằng giữa $$G$$ và $$D$$:** Hiểu vì sao loss của $$D$$ và $$G$$ không giảm đơn điệu về 0 mà dao động quanh ngưỡng cân bằng.
- **Các biến thể nền tảng:** Từ Vanilla GAN, DCGAN (tích chập), Conditional GAN (sinh theo nhãn lớp) đến WGAN-GP (ổn định hình học).
- **Tiêu chuẩn chẩn đoán trực quan:** Luôn theo dõi lưới ảnh từ `fixed_noise` song song với đường cong loss để phát hiện sớm hiện tượng sụp đổ mode.

### Khung giải thích (nên đọc chậm): Đọc vị 3 kịch bản đường cong loss của $$D$$ và $$G$$ khi huấn luyện GAN

Nhớ rằng tại điểm cân bằng lý tưởng $$D(x) = D(G(z)) = 0{,}5$$, ta có:
- $$\mathcal{L}_D = -\ln(0{,}5) - \ln(0{,}5) = 2\ln 2 \approx 1{,}386$$ (hoặc $$0{,}693$$ nếu lấy trung bình hai phần).
- $$\mathcal{L}_G = -\ln(0{,}5) = \ln 2 \approx 0{,}693$$.

Đối chiếu 3 dấu hiệu chẩn đoán thực tế trên bảng log:
1. **$$\mathcal{L}_D \to 0{,}01$$, $$\mathcal{L}_G \to 5{,}0$$:** Bộ phân biệt thắng áp đảo, gradient của $$G$$ bị bão hòa — cần giảm learning rate của $$D$$, thêm label smoothing hoặc kiểm tra lại chuẩn hóa dữ liệu.
2. **$$\mathcal{L}_D \approx 1{,}38$$, $$\mathcal{L}_G \approx 0{,}70$$ nhưng lưới ảnh lặp đúng 1 chữ số:** Dấu hiệu *mode collapse* ẩn — $$G$$ đang lừa được $$D$$ bằng một mẫu an toàn duy nhất.
3. **$$\mathcal{L}_D$$ dao động quanh $$1{,}0\text{–}1{,}4$$, $$\mathcal{L}_G$$ quanh $$0{,}7\text{–}1{,}2$$ và ảnh `fixed_noise` sắc nét dần qua từng epoch:** Quá trình huấn luyện lành mạnh!

Nội dung về Mạng Sinh Đối kháng gồm hai phần:

1. **Lý thuyết** — công thức trò chơi minimax, cân bằng tối ưu và trực giác đối kháng.
2. **Cài đặt** — vòng lặp huấn luyện PyTorch, các bài báo nền tảng và cạm bẫy thường gặp.

## Video tham khảo

- **Xem trước:** [A Friendly Introduction to Generative Adversarial Networks (GANs)](https://www.youtube.com/watch?v=8L11aMN5KY8) — Luis Serrano Academy, tiếng Anh. Tổng quan ngắn gọn trước khi đi vào phần lý thuyết và mã cài đặt.
- [The Math Behind Generative Adversarial Networks Clearly Explained!](https://www.youtube.com/watch?v=Gib_kiXgnvA) — Normalized Nerd, tiếng Anh. Trình bày hàm giá trị minimax xuyên suốt cả hai bài học.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, tiếng Anh. Phần GAN trong bài giảng CS231n, bao gồm các chế độ thất bại thường gặp.
