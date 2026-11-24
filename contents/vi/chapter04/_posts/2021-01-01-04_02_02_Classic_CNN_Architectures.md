---
layout: post
title: 04-02-02 Kiến trúc CNN Kinh điển
chapter: '04'
order: 11
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

Năm 2012, tại cuộc thi thị giác máy tính danh giá ImageNet, một mô hình CNN mang tên AlexNet bất ngờ đánh bại toàn bộ các phương pháp thị giác cổ điển với khoảng cách sai số hơn 10 điểm phần trăm — một cú địa chấn làm thay đổi hoàn toàn lịch sử trí tuệ nhân tạo. Từ cột mốc đó đến năm 2015, các kiến trúc kinh điển nối tiếp nhau ra đời, mỗi kiến trúc đều giải quyết một nút thắt kỹ thuật sống còn của thế hệ trước.

## Bản đồ kiến thức nền: Bốn mốc tiến hóa của Kiến trúc CNN Kinh điển

1. **LeNet-5 (1998) — Bản thiết kế sơ khai hoàn chỉnh**:
   Yann LeCun chứng minh chuỗi `Conv -> Pool -> Conv -> Pool -> FC` có thể đọc chữ số viết tay trên hàng triệu tờ séc ngân hàng.
2. **AlexNet (2012) — Mở khóa quy mô bằng GPU, ReLU và Dropout**:
   Thay `tanh/sigmoid` bằng ReLU giúp huấn luyện nhanh gấp 6 lần, kết hợp tăng cường dữ liệu (data augmentation) và Dropout để chống overfitting trên 1,2 triệu ảnh.
3. **VGGNet (2014) — Triết lý đồng nhất bằng bộ lọc $$3 \times 3$$**:
   Chứng minh rằng xếp chồng nhiều bộ lọc nhỏ $$3 \times 3$$ tốt hơn hẳn việc dùng bộ lọc lớn $$7 \times 7$$ hay $$11 \times 11$$: ít tham số hơn nhưng có nhiều tầng phi tuyến hơn.
4. **ResNet (2015) — Phá vỡ trần chiều sâu bằng kết nối tắt**:
   Giải quyết nghịch lý suy thoái (mạng sâu hơn lại có sai số huấn luyện cao hơn) bằng cách học phần dư $$F(x) = H(x) - x$$ thay vì học trực tiếp ánh xạ $$H(x)$$.

### Khung giải thích (nên đọc chậm): Bảng đối chiếu quy mô và bước nhảy vọt hiệu năng trên ImageNet

| Kiến trúc | Năm | Độ sâu có trọng số | Top-5 Error (ImageNet) | Đột phá cốt lõi |
| :--- | :---: | :---: | :---: | :--- |
| **LeNet-5** | 1998 | 5 tầng | — (MNIST ~0,95%) | Kết hợp Conv + Pooling + Backprop đầu tiên |
| **AlexNet** | 2012 | 8 tầng | **16,4%** | ReLU, Dropout, huấn luyện trên 2 GPU |
| **VGG-16** | 2014 | 16 tầng | **7,3%** | Chuẩn hóa hoàn toàn bằng nhân nhỏ $$3 \times 3$$ |
| **ResNet-152** | 2015 | 152 tầng | **3,57%** | Kết nối tắt $$y = F(x) + x$$, vượt chuẩn người (~5%) |

Chỉ trong 3 năm (2012–2015), sai số Top-5 giảm gần **5 lần** trong khi độ sâu tăng gần **20 lần** — tất cả xoay quanh hai câu hỏi cốt lõi: *dùng bộ lọc cỡ nào* và *giữ gradient sống sót ra sao*.

Phần này giới thiệu các kiến trúc CNN kinh điển theo tiến trình phát triển:

1. **LeNet và AlexNet** — hai cột mốc mở đầu của thị giác máy tính dùng học sâu
2. **VGG, ResNet và Tiến hóa Kiến trúc** — mở rộng độ sâu, kết nối tắt (*skip connections*), các bài báo nền tảng và tổng kết điểm chính

## Video tham khảo

- **Xem trước:** [Lecture 9 | CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y) — Stanford University School of Engineering, tiếng Anh. Bài giảng trình bày liên mạch các kiến trúc LeNet, AlexNet, VGG và ResNet.
- [Lecture 1 | Introduction to Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/watch?v=vT1JzLTH4G4) — Stanford University School of Engineering, tiếng Anh. Bối cảnh lịch sử lý giải vì sao bài toán ImageNet thúc đẩy sự đổi mới kiến trúc.
- [Residual Networks (ResNet) \[Physics Informed Machine Learning\]](https://www.youtube.com/watch?v=w1UsKanMatM) — Steve Brunton, tiếng Anh. Phân tích cơ chế kết nối dư khép lại chuỗi kiến trúc kinh điển này.
