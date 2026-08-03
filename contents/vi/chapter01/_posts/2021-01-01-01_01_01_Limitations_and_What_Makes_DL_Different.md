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

## Sức mạnh của học sâu

Học sâu đã trở thành cách tiếp cận thống trị trong trí tuệ nhân tạo vì nó giải quyết các hạn chế căn bản của học máy truyền thống.

## Hạn chế của học máy truyền thống

### 1. Thiết kế đặc trưng thủ công

**Cách tiếp cận truyền thống**:
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

**Vấn đề**:
- Đòi hỏi chuyên môn miền
- Tốn thời gian
- Có thể bỏ sót mẫu quan trọng
- Khó mở rộng qua các miền

**Giải pháp học sâu**:
```python
# Học đầu-cuối
model = build_cnn()
model.train(images, labels)  # Tự động học đặc trưng!
```

### 2. Biểu diễn cố định

Học máy truyền thống dùng đặc trưng thủ công, vốn:
- Không thích nghi với dữ liệu
- Có thể không tối ưu cho tác vụ
- Cần thiết kế lại cho bài toán mới

Học sâu **học biểu diễn tối ưu** cho từng tác vụ cụ thể.

### 3. Hạn chế về khả năng mở rộng

**Học máy truyền thống**: thường bão hòa khi có thêm dữ liệu

**Học sâu**: hiệu năng cải thiện theo quy mô

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

Mạng sâu học đặc trưng ở nhiều mức:

**Ví dụ: nhận diện khuôn mặt**

```
Layer 1 (Low-level):    Cạnh, màu, mẫu đơn giản
         ↓
Layer 2 (Mid-level):    Mắt, mũi, bộ phận miệng
         ↓
Layer 3 (High-level):   Khuôn mặt hoàn chỉnh, biểu cảm
         ↓
Output:                  Danh tính người
```

Cấu trúc này phản ánh cách con người nhận thức — từ khái niệm đơn giản đến phức tạp.

### 2. Học đầu-cuối

**Pipeline truyền thống**:
```
Raw Data → Preprocessing → Feature Extraction → Feature Selection → Model → Output
         (Thủ công)       (Thủ công)           (Thủ công)
```

**Học sâu**:
```
Raw Data → Neural Network → Output
         (Tất cả được học tự động)
```

### 3. Bộ xấp xỉ hàm phổ quát

**Định lý xấp xỉ phổ quát**: Mạng nơ-ron chỉ với một tầng ẩn, nếu có đủ số nơ-ron, có thể xấp xỉ bất kỳ hàm liên tục nào.

Mạng sâu có thể học xấp xỉ:
- Phép biến đổi ảnh
- Mẫu ngôn ngữ
- Chiến lược game
- Mô phỏng vật lý
- Ranh giới quyết định phức tạp
