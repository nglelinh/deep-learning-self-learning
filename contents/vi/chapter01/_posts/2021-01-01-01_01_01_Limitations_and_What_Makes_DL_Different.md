---
layout: post
title: 01-01-01 Hạn chế của học máy truyền thống và điều làm học sâu khác biệt
chapter: '01'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter01
---

Trước năm 2012, một nhóm nghiên cứu muốn xây dựng hệ thống nhận diện khuôn mặt phải dành hàng năm trời thiết kế các bộ lọc toán học thủ công để dò sống mũi, khóe mắt, viền môi—nhưng chỉ cần đối tượng đeo kính râm hoặc xoay nghiêng đầu 30 độ, cả hệ thống lại sụp đổ. Đó là "bức tường kính" của Học máy truyền thống. Bài học này phân tích trực diện bốn hạn chế cố hữu của các mô hình cổ điển—từ thiết kế đặc trưng thủ công đến lời nguyền chiều cao (*curse of dimensionality*)—và chỉ ra cách Học sâu phá vỡ từng bức tường đó.

## Bản đồ kiến thức nền: Bốn bức tường của Học máy Cổ điển và cách Học sâu vượt qua

1. **Nút thắt Thiết kế Đặc trưng (Feature Engineering Bottleneck)**:
   Học máy truyền thống tách rời hai khâu *trích xuất đặc trưng* (con người làm) và *phân loại* (máy học). Học sâu hợp nhất cả hai vào cùng một mạng khả vi được tối ưu từ đầu đến cuối.

2. **Lời nguyền Chiều cao (Curse of Dimensionality)**:
   Trong không gian hàng trăm nghìn chiều của điểm ảnh, các phương pháp dựa trên khoảng cách trực tiếp (như k-NN hay RBF-SVM) thất bại vì mọi điểm đều cách đều nhau một cách kỳ lạ; mạng sâu giải quyết bằng cách chiếu dữ liệu xuống các đa tạp (*manifold*) chiều thấp giàu ngữ nghĩa.

3. **Đường cong Hiệu năng theo Quy mô Dữ liệu (Data Scaling)**:
   Khi dữ liệu tăng từ $$10^4$$ lên $$10^7$$ mẫu, mô hình tuyến tính hay cây nông sớm bão hòa do dung lượng biểu diễn hữu hạn; mạng sâu có thể mở rộng dung lượng mượt mà để tiếp tục hấp thụ tri thức mới.

4. **Học Chuyển giao (Transfer Learning)**:
   Đặc trưng học được ở các tầng đầu của mạng sâu trên tập ảnh lớn có tính phổ quát cao, cho phép tái sử dụng cho bài toán y tế hay công nghiệp chỉ có vài trăm ảnh—điều gần như bất khả thi với cây quyết định.

### Khung giải thích (nên đọc chậm): Tính tay "Lời nguyền Chiều cao" trong không gian lưới đơn vị

Giả sử bạn muốn phủ kín một không gian đặc trưng chuẩn hóa $$[0, 1]^d$$ bằng các ô lưới có cạnh dài $$0.1$$ (tức chia mỗi trục thành 10 khoảng) để đảm bảo mọi điểm test đều có ít nhất một điểm train nằm gần:

- Với **$$d = 1$$ chiều** (1 đặc trưng): cần $$10^1 = 10$$ mẫu.
- Với **$$d = 2$$ chiều**: cần $$10^2 = 100$$ mẫu.
- Với **$$d = 10$$ chiều**: cần $$10^{10} = 10\text{ tỷ}$$ mẫu!
- Với một ảnh nhỏ **$$28 \times 28 = 784$$ chiều**: số ô lưới là $$10^{784}$$ (lớn hơn tổng số nguyên tử trong vũ trụ quan sát được $$\approx 10^{80}$$ gấp vô số lần!).

Con số khủng khiếp này chứng minh một sự thật sâu sắc: **Học sâu thành công không phải vì nó phủ kín không gian $$\mathbb{R}^{784}$$**, mà vì ảnh thật chỉ nằm trên một bề mặt cong (manifold) cực mảnh có cấu trúc cục bộ và đối xứng mà mạng nhiều tầng khai thác được!

## Lợi thế của học sâu

Học sâu trở thành hướng tiếp cận chủ đạo trong nhiều bài toán trí tuệ nhân tạo vì nó giải quyết được những hạn chế cốt lõi của học máy truyền thống.

## Hạn chế của học máy truyền thống

### 1. Thiết kế đặc trưng thủ công

**Cách tiếp cận truyền thống:**
```python
# Trích xuất đặc trưng thủ công cho phân loại ảnh
features = []
features.append(calculate_histogram(image))
features.append(detect_edges(image))
features.append(extract_textures(image))
features.append(compute_color_moments(image))

# Sau đó huấn luyện bộ phân loại trên các đặc trưng này
model = train_svm(features, labels)
```

**Hạn chế:**
- Đòi hỏi kiến thức chuyên sâu về từng miền bài toán.
- Tốn nhiều thời gian thiết kế và tinh chỉnh.
- Dễ bỏ sót các mẫu thông tin quan trọng.
- Khó tái sử dụng hoặc mở rộng sang các miền bài toán khác.

**Cách tiếp cận học sâu:**
```python
# Học đầu-cuối
model = build_cnn()
model.train(images, labels)  # Tự động học đặc trưng!
```

### 2. Biểu diễn cố định

Học máy truyền thống phụ thuộc vào các đặc trưng thiết kế sẵn, vốn có các nhược điểm:
- Không tự thích nghi theo phân phối dữ liệu.
- Có thể không tối ưu cho mục tiêu dự đoán cụ thể.
- Phải thiết kế lại gần như từ đầu khi chuyển sang bài toán mới.

Ngược lại, học sâu **tự học biểu diễn phù hợp** cho từng tác vụ trực tiếp từ dữ liệu.

### 3. Hạn chế về khả năng mở rộng

Trong nhiều bài toán phức tạp, **học máy truyền thống** thường sớm bão hòa khi lượng dữ liệu tăng lên, còn **học sâu** tiếp tục cải thiện hiệu năng khi có thêm dữ liệu và năng lực tính toán:

```
Traditional ML:  _______________  (bão hòa)
                     /
Deep Learning:     /  (tiếp tục cải thiện)
                  /
                 |
              Performance
```

## Điều gì làm học sâu khác biệt?

### 1. Học đặc trưng phân cấp

Mạng nơ-ron sâu học đặc trưng ở nhiều mức trừu tượng khác nhau. Chẳng hạn, với bài toán **nhận diện khuôn mặt**:

```
Layer 1 (Low-level):    Cạnh, màu, mẫu đơn giản
         ↓
Layer 2 (Mid-level):    Mắt, mũi, bộ phận miệng
         ↓
Layer 3 (High-level):   Khuôn mặt hoàn chỉnh, biểu cảm
         ↓
Output:                  Danh tính người
```

Cấu trúc phân cấp này tương đồng với cách con người tiếp nhận thông tin thị giác — đi từ các chi tiết đơn giản đến khái niệm hoàn chỉnh.

### 2. Học từ đầu đến cuối (*end-to-end learning*)

**Quy trình truyền thống:**
```
Raw Data → Preprocessing → Feature Extraction → Feature Selection → Model → Output
         (Thủ công)       (Thủ công)           (Thủ công)
```

**Quy trình học sâu:**
```
Raw Data → Neural Network → Output
         (Tất cả được học tự động)
```

### 3. Bộ xấp xỉ hàm phổ quát

Theo **định lý xấp xỉ phổ quát**, một mạng nơ-ron chỉ cần một tầng ẩn với đủ số lượng nơ-ron có thể xấp xỉ bất kỳ hàm liên tục nào trên một miền đóng và bị chặn với độ chính xác tùy ý. Nhờ khả năng biểu diễn linh hoạt này, mạng sâu có thể học xấp xỉ nhiều dạng ánh xạ phức tạp như:
- Phép biến đổi trên ảnh.
- Cấu trúc và quy luật ngôn ngữ.
- Chiến lược trong trò chơi.
- Mô phỏng các hệ thống vật lý.
- Ranh giới quyết định phi tuyến phức tạp.

## Video tham khảo

- **Xem trước:** [Machine Learning vs Deep Learning](https://www.youtube.com/watch?v=q6kJ71tEYqM) — IBM Technology, tiếng Anh. Phân tích sự khác biệt giữa thiết kế đặc trưng thủ công và học biểu diễn tự động.
- [The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) — StatQuest with Josh Starmer, tiếng Anh. Minh họa ví dụ dữ liệu mà mô hình tuyến tính không khớp được nhưng một mạng nơ-ron nhỏ có thể giải quyết, làm rõ hạn chế được nêu trong bài.
- [MIT Introduction to Deep Learning (2024) | 6.S191](https://www.youtube.com/watch?v=ErnWZxJovaM) — Alexander Amini, tiếng Anh. Đặt sự khác biệt giữa học máy cổ điển và học sâu vào bức tranh tổng thể về dữ liệu, tính toán và độ sâu mô hình.
