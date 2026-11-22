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
