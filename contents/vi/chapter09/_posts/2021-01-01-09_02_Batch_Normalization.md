---
layout: post
title: 09-02 Chuẩn hóa Batch
chapter: '09'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter09
---

Trước năm 2015, huấn luyện một mạng CNN trên 10 tầng giống như việc cân bằng một chồng đĩa sứ cao ngất: chỉ cần chỉnh nhích learning rate lên một chút hoặc khởi tạo trọng số hơi lệch, các tầng sâu lập tức nhận đầu vào trôi dạt vào vùng bão hòa và toàn bộ quá trình huấn luyện đổ vỡ. Sự ra đời của **Batch Normalization** (Ioffe & Szegedy, 2015) đã thay đổi cuộc chơi mãi mãi — biến việc huấn luyện mạng cực sâu trở nên ổn định, nhanh gấp nhiều lần và mở đường cho cả họ chuẩn hóa hiện đại (LayerNorm, GroupNorm, RMSNorm).

## Bản đồ kiến thức nền: Ba trụ cột của Chuẩn hóa Batch và Họ Chuẩn hóa

1. **Cơ chế chuẩn hóa 4 bước kèm tham số khôi phục $$(\gamma, \beta)$$**:
   Tính trung bình $$\mu_B$$ và phương sai $$\sigma_B^2$$ trên mini-batch, chuẩn hóa về phân phối chuẩn $$(0, 1)$$, rồi cho phép mạng tự học lại tỷ lệ $$\gamma$$ và độ dịch $$\beta$$ để không làm mất sức biểu đạt.
2. **Hành vi khác biệt giữa Train (thống kê batch) và Eval (Running Mean/Var)**:
   Lúc suy luận (đôi khi chỉ có 1 ảnh duy nhất!), ta không thể tính phương sai của batch 1 mẫu; thay vào đó BatchNorm dùng trung bình trượt hàm mũ (*running statistics*) đã tích lũy suốt quá trình huấn luyện.
3. **Bản đồ chọn lớp Chuẩn hóa theo miền dữ liệu**:
   BatchNorm cho CNN với batch đủ lớn ($$\ge 16$$), GroupNorm cho phát hiện vật thể/phân đoạn có batch rất nhỏ (1–4 ảnh), InstanceNorm cho chuyển đổi phong cách ảnh, và LayerNorm/RMSNorm cho Transformer/NLP.

### Khung giải thích (nên đọc chậm): Phân biệt trục tính trung bình & phương sai của 4 họ Chuẩn hóa trên tensor `(N, C, H, W)`

Giả sử ta có một batch ảnh kích thước `(N=8, C=64, H=16, W=16)`:

| Lớp chuẩn hóa | Tính $$\mu, \sigma^2$$ gộp trên các trục nào? | Số cặp $$(\mu, \sigma^2)$$ tính ra mỗi lượt | Phụ thuộc kích thước batch $$N$$? |
| :--- | :---: | :---: | :---: |
| **BatchNorm** | `(N, H, W)` — riêng từng kênh $$C$$ | $$C = 64$$ | **Có** (hỏng khi $$N$$ rất nhỏ) |
| **LayerNorm** | `(C, H, W)` — riêng từng mẫu $$N$$ | $$N = 8$$ | **Không** (chuẩn của Transformer) |
| **InstanceNorm** | `(H, W)` — riêng từng mẫu và từng kênh | $$N \times C = 512$$ | **Không** (chuẩn Style Transfer) |
| **GroupNorm** ($$G=8$$ nhóm) | `(C/G, H, W)` — từng nhóm kênh của mỗi mẫu | $$N \times G = 64$$ | **Không** (tốt cho Detection batch nhỏ) |

Nắm chắc bảng 4 dòng này giúp bạn chọn đúng chuẩn hóa cho mọi kiến trúc Deep Learning chỉ trong 3 giây!

Mục này trình bày kỹ thuật chuẩn hóa theo lô (*Batch Normalization*) qua hai bài:

1. **Cơ chế** — khái niệm, công thức lượt xuôi/lượt ngược, cài đặt bằng NumPy và lý do kỹ thuật này ổn định quá trình huấn luyện.
2. **Vị trí đặt, Biến thể và Mẹo** — BatchNorm2D cho CNN, các biến thể thay thế (LayerNorm, InstanceNorm, GroupNorm), vấn đề thường gặp và kinh nghiệm thực hành.

## Video tham khảo

- **Xem trước:** [Why Does Batch Norm Work? (C2W3L06)](https://www.youtube.com/watch?v=nUUqwaxLnWs) — DeepLearningAI (tiếng Anh). Giải thích lý do chuẩn hóa theo mini-batch cải thiện đáng kể quá trình huấn luyện.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — Andrej Karpathy (tiếng Anh). Cài đặt thực tế BatchNorm trong vòng huấn luyện và phân tích tương tác với hàm kích hoạt.
- [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering (tiếng Anh). Đặt BatchNorm trong hệ thống các kỹ thuật tối ưu và huấn luyện mạng sâu.
