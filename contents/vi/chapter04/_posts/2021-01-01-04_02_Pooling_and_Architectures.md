---
layout: post
title: 04-02 Tầng Gộp và Kiến trúc CNN
chapter: '04'
order: 9
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

Nếu tầng tích chập là đôi mắt tinh tường phát hiện từng chi tiết nhỏ, thì **tầng gộp (pooling)** và **kiến trúc tổng thể** chính là cách bộ não chắt lọc thông tin: bỏ qua những xê dịch nhỏ không đáng kể, nén dần không gian để tập trung vào ngữ nghĩa, và tổ chức hàng chục đến hàng trăm tầng sao cho tín hiệu học không bao giờ bị nghẽn.

## Kiến thức cần nhớ: Ba trụ cột của Tầng Gộp và Thiết kế Kiến trúc CNN

1. **Tầng Gộp (Max, Average, Global Average Pooling)**:
   Đóng vai trò giảm độ phân giải không gian, tiết kiệm bộ nhớ/FLOPs, tăng trường tiếp nhận và mang lại tính bất biến cục bộ trước các dịch chuyển nhỏ vài pixel.
2. **Thay thế tầng Dense khổng lồ bằng Global Average Pooling (GAP)**:
   Thay vì duỗi phẳng (flatten) bản đồ đặc trưng rồi nối vào tầng FC hàng chục triệu tham số dễ overfitting, GAP lấy trung bình toàn bộ mỗi kênh thành đúng 1 con số duy nhất.
3. **Quy luật tiến hóa của các kiến trúc CNN kinh điển**:
   Đi từ những khối xếp chồng tuần tự (LeNet, AlexNet, VGG) sang các thiết kế đa nhánh (Inception) và đường cao tốc gradient bằng kết nối tắt (ResNet).

### Giải thích: Tính tay lượng tham số tiết kiệm khi thay Flatten + FC bằng Global Average Pooling

Giả sử trước bộ phân loại 1.000 lớp (ImageNet), bản đồ đặc trưng cuối cùng có kích thước $$7 \times 7 \times 512$$:

- **Kiểu cổ điển (VGG — Flatten + tầng FC 4096 ẩn)**:
  Số phần tử sau khi duỗi phẳng: $$7 \times 7 \times 512 = 25{,}088$$.
  Riêng tầng FC đầu tiên ánh xạ $$25{,}088 \to 4{,}096$$ đã tốn:
  $$25{,}088 \times 4{,}096 + 4{,}096 = \mathbf{102{,}764{,}544}$$ tham số (hơn 102 triệu!).
- **Kiểu hiện đại (ResNet — GAP + 1 tầng Linear 1000)**:
  GAP nén mỗi mặt phẳng $$7 \times 7$$ thành 1 số trung bình $$\Rightarrow$$ vector chỉ còn **512** chiều (0 tham số học).
  Tầng phân loại cuối ánh xạ $$512 \to 1{,}000$$ chỉ tốn:
  $$512 \times 1{,}000 + 1{,}000 = \mathbf{513{,}000}$$ tham số.

Chiến lược GAP cắt giảm hơn **200 lần** số tham số ở phần đầu ra, đồng thời giúp mạng chấp nhận ảnh đầu vào với nhiều độ phân giải khác nhau.

Phần này hoàn thiện các thành phần cốt lõi của CNN: **tầng gộp** (*pooling*) dùng để giảm kích thước không gian và **các kiến trúc kinh điển** minh họa cách kết hợp tầng tích chập cùng tầng gộp trong thực tế.

1. **Tầng Gộp** — max pooling, average pooling, global average pooling, nền tảng toán học và cài đặt
2. **Kiến trúc CNN Kinh điển** — LeNet, AlexNet, VGG, ResNet, các bài báo nền tảng và tổng kết

Bạn nên đọc bài về tầng gộp trước rồi chuyển sang phần khảo sát các kiến trúc.

## Video tham khảo

- **Xem trước:** [Max Pooling in Convolutional Neural Networks explained](https://www.youtube.com/watch?v=ZjM_XQa5s6s) — deeplizard, tiếng Anh. Trình bày max pooling — thành phần giảm mẫu cốt lõi của phần đầu.
- [Lecture 9 | CNN Architectures](https://www.youtube.com/watch?v=DAOcjicFr1Y) — Stanford University School of Engineering, tiếng Anh. Tổng quan các kiến trúc kinh điển từ LeNet đến ResNet.
- [Residual Networks (ResNet) \[Physics Informed Machine Learning\]](https://www.youtube.com/watch?v=w1UsKanMatM) — Steve Brunton, tiếng Anh. Phân tích kết nối tắt trong ResNet — kiến trúc trọng tâm ở cuối phần này.
