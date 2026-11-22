---
layout: post
title: 04-01-04 Khái niệm Liên quan, Cạm bẫy và Bài báo về Tích chập
chapter: '04'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

## 5. Khái niệm Liên quan

### CNN so với Mạng Neuron Liên kết Đầy đủ

![CNN vs Regular Neural Network](/deep-learning-self-learning/img/chapter_img/chapter04/conv7.jpg)
*So sánh CNN và mạng neuron thông thường trên dữ liệu ảnh: CNN bảo toàn cấu trúc không gian, chia sẻ trọng số bộ lọc để giảm mạnh số tham số và nhận diện đặc trưng không phụ thuộc vị trí. Nguồn: Analytics Vidhya*

| Khía cạnh | Liên kết đầy đủ | Tích chập |
|--------|-----------------|---------------|
| Kết nối | Mọi đầu vào với mọi đầu ra | Cục bộ (theo kích thước nhân) |
| Chia sẻ tham số | Không | Dùng chung bộ lọc ở mọi vị trí |
| Tham số (32×32×3 → 64) | 196,672 | 1,792 |
| Bất biến tịnh tiến | Không | Có |
| Bảo toàn cấu trúc không gian | Không | Có |

### Tầng Liên kết Đầy đủ
- Nối mọi neuron đầu vào với mọi neuron đầu ra.
- Không có giả định về cấu trúc không gian.
- Thường đặt sau các tầng tích chập để thực hiện bước phân loại cuối cùng.
- Số tham số tăng nhanh theo kích thước đầu vào.

### Tầng Gộp (*Pooling*)
- Giảm kích thước không gian của bản đồ đặc trưng (*downsampling*).
- Tăng độ ổn định trước các dịch chuyển nhỏ.
- Giảm chi phí tính toán và bộ nhớ cho các tầng phía sau.
- Không chứa tham số học được.
- Các dạng phổ biến: Max pooling, Average pooling.

### Chuẩn hóa Batch (*Batch Normalization*)
- Chuẩn hóa giá trị kích hoạt về trung bình 0 và phương sai 1 theo từng mini-batch.
- Giúp quá trình huấn luyện ổn định hơn.
- Thường đặt sau tầng tích chập và trước hàm kích hoạt.
- Cho phép sử dụng tốc độ học cao hơn và hội tụ nhanh hơn.

### Kết nối Dư (*Residual / Skip Connections*)
- Tạo đường dẫn trực tiếp cho tín hiệu và gradient đi qua nhiều tầng.
- Cho phép huấn luyện hiệu quả các mạng rất sâu (trên 100 tầng).
- Biểu thức đầu ra: $$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$.
- Đóng góp cốt lõi của kiến trúc ResNet (2015).

### Tích chập Tách theo Chiều sâu (*Depthwise Separable Convolutions*)
- Tách phép tích chập chuẩn thành hai bước: *depthwise* và *pointwise*.
- Giảm đáng kể số tham số và số phép tính.
- *Depthwise*: áp dụng một bộ lọc riêng cho từng kênh đầu vào.
- *Pointwise*: dùng tích chập 1×1 để kết hợp thông tin giữa các kênh.
- Được sử dụng rộng rãi trong MobileNet và EfficientNet cho thiết bị di động.

### Tích chập Giãn (*Dilated / Atrous Convolutions*)
- Chèn khoảng trống giữa các phần tử của nhân tích chập.
- Mở rộng trường tiếp nhận mà không tăng số lượng tham số.
- Thường dùng trong bài toán phân đoạn ngữ nghĩa (như DeepLab).
- Với tỷ lệ giãn $$d$$, các phần tử của nhân cách nhau $$d$$ điểm ảnh.

---

## 6. Các Bài báo Nền tảng

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Tác giả*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
Giới thiệu LeNet-5 — mạng CNN đầu tiên thành công trong bài toán nhận diện chữ số viết tay. Bài báo chứng minh các tầng tích chập dùng chung trọng số có thể tự học đặc trưng phân cấp từ điểm ảnh thô và định hình khuôn mẫu `conv → pool → conv → pool → FC` được sử dụng suốt gần hai thập kỷ.

**["ImageNet Classification with Deep Convolutional Neural Networks" (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)**  
*Tác giả*: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
AlexNet giành chiến thắng tại ImageNet 2012 với cách biệt lớn (lỗi top-5 đạt 15,3% so với 26,2% của đội xếp thứ hai), khẳng định khả năng mở rộng của CNN sâu trên bài toán thị giác quy mô lớn. Công trình cũng đưa ReLU, dropout và huấn luyện trên GPU vào dòng chính của học sâu.

**["Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015)](https://arxiv.org/abs/1409.1556)**  
*Tác giả*: Karen Simonyan, Andrew Zisserman  
VGGNet nhấn mạnh vai trò của độ sâu kiến trúc và chỉ sử dụng các bộ lọc nhỏ 3×3 xếp chồng liên tiếp để đạt trường tiếp nhận lớn. Thiết kế đồng nhất, dễ lặp lại này trở thành nền tảng tham chiếu cho nhiều kiến trúc về sau.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Tác giả*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Đề xuất kết nối tắt (*skip connections*) giải quyết hiện tượng suy giảm hiệu năng khi mạng quá sâu, cho phép huấn luyện thành công mạng hơn 150 tầng. ResNet giành vị trí dẫn đầu tại ImageNet 2015 và nguyên lý học phần dư của nó hiện được dùng rộng rãi trong cả Transformer lẫn mô hình khuếch tán.

**["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)](https://arxiv.org/abs/1704.04861)**  
*Tác giả*: Andrew G. Howard và cộng sự  
Giới thiệu tích chập tách theo chiều sâu giúp giảm 8–9 lần khối lượng tính toán mà vẫn giữ độ chính xác tốt, mở đường cho việc triển khai CNN trên điện thoại và thiết bị nhúng.

**["Rethinking the Inception Architecture for Computer Vision" (2016)](https://arxiv.org/abs/1512.00567)**  
*Tác giả*: Christian Szegedy và cộng sự  
Inception-v3 đề xuất phân rã nhân tích chập (dùng cặp $$1 \times n$$ và $$n \times 1$$ thay cho $$n \times n$$) cùng các nhánh phân loại phụ, cho thấy thiết kế kiến trúc cẩn thận có thể cải thiện đồng thời độ chính xác và chi phí tính toán.

---

## 7. Cạm bẫy Thường gặp và Mẹo Thực hành

### Cạm bẫy 1: Không Dùng Padding

**Vấn đề:** Kích thước bản đồ đặc trưng thu nhỏ dần sau mỗi tầng và thông tin ở vùng biên bị mất nhanh chóng.

```python
# Không padding: 32×32 → 30×30 → 28×28 → ... (co nhanh!)
conv1 = nn.Conv2d(3, 64, kernel_size=3)  # Không padding
conv2 = nn.Conv2d(64, 128, kernel_size=3)

# Có padding: 32×32 → 32×32 → 32×32 (giữ kích thước)
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Đệm "same"
conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
```

**Cách khắc phục:** Với nhân kích thước lẻ và `stride=1`, đặt `padding = (kernel_size - 1) // 2` để giữ nguyên kích thước không gian.

### Cạm bẫy 2: Nhầm Thứ tự Chiều Kênh

**Vấn đề:** Các framework sử dụng quy ước thứ tự chiều tensor khác nhau.

```python
# PyTorch: (N, C, H, W) - Kênh trước
x_pytorch = torch.randn(batch, channels, height, width)

# TensorFlow/Keras: (N, H, W, C) - Kênh sau
x_tensorflow = tf.random.normal([batch, height, width, channels])

# Chuyển đổi giữa chúng
x_tf_to_pytorch = x_tensorflow.permute(0, 3, 1, 2)  # Nếu dùng torch
```

### Cạm bẫy 3: Dùng Nhân Quá Lớn ở Các Tầng Đầu

**Vấn đề:** Nhân kích thước lớn tốn nhiều tham số trong khi chỉ có một lớp phi tuyến.

```python
# Kém hiệu quả hơn (hiểu biết của VGG)
conv = nn.Conv2d(3, 64, kernel_size=7)  # 7×7 = 49 params/kênh

# Tốt hơn: Xếp chồng kernel nhỏ
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
# Ba 3×3 = 27 params/kênh, cùng trường tiếp nhận 7×7, gấp 3 phi tuyến
```

### Mẹo 1: Dùng Tích chập 1×1 để Điều chỉnh Số Kênh

```python
# Giảm kênh (nút thắt / bottleneck)
bottleneck = nn.Conv2d(512, 64, kernel_size=1)
# 512 → 64 kênh với tính toán tối thiểu

# Tăng kênh
expand = nn.Conv2d(64, 512, kernel_size=1)
# Trộn thông tin kênh mà không thao tác không gian
```

### Mẹo 2: Dùng Tích chập có Stride thay cho Pooling

```python
# Truyền thống: Conv → Pool
conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2, 2)

# Hiện đại: Conv stride (giảm mẫu học được)
strided_conv = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# Cùng kích thước đầu ra, nhưng giảm mẫu được học!
```

### Mẹo 3: Khởi tạo Trọng số Phù hợp

```python
# Khởi tạo Kaiming cho mạng ReLU
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

### Mẹo 4: Tính Trường Tiếp nhận

```python
def receptive_field_1d(layers):
    """
    Tính trường tiếp nhận cho một chồng tầng conv.
    
    Mỗi tầng là (kernel_size, stride)
    """
    rf = 1  # Bắt đầu với một điểm ảnh
    stride_product = 1
    
    for k, s in layers:
        rf = rf + (k - 1) * stride_product
        stride_product *= s
    
    return rf

# Ví dụ: 3 tầng conv (k=3, s=1) rồi pool (k=2, s=2)
layers = [(3, 1), (3, 1), (3, 1), (2, 2)]
print(f"Trường tiếp nhận: {receptive_field_1d(layers)}")  # 8
```

---

## 8. Tóm tắt các Điểm chính

1. **Tích chập và Tương quan chéo:** Học sâu thường cài đặt tương quan chéo nhưng gọi chung là tích chập; việc có lật nhân hay không không làm thay đổi khả năng học vì trọng số được tối ưu từ dữ liệu.
2. **Kết nối Cục bộ và Chia sẻ Trọng số:** Hai cơ chế này giúp CNN tiết kiệm tham số hơn rất nhiều so với mạng liên kết đầy đủ.
3. **Học Đặc trưng Phân cấp:** CNN tự động xây dựng biểu diễn từ cạnh, kết cấu đến bộ phận và đối tượng hoàn chỉnh.
4. **Công thức Kích thước Đầu ra:** $$n_{out} = \lfloor(n_{in} + 2p - k)/s\rfloor + 1$$.
5. **Xếp chồng Bộ lọc Nhỏ:** Dùng nhiều tầng 3×3 liên tiếp thường hiệu quả hơn một tầng nhân lớn (ít tham số hơn và nhiều lớp phi tuyến hơn).
6. **Trường Tiếp nhận:** Mở rộng dần theo độ sâu, là đại lượng quan trọng cần theo dõi khi thiết kế kiến trúc.
7. **Mẫu thiết kế phổ biến:**
   - Chuỗi `Conv → BatchNorm → ReLU`
   - Kết nối dư cho các mạng sâu
   - Global average pooling thay cho `flatten + FC` nhiều tham số

Tầng tích chập là thành phần trung tâm của thị giác máy tính trong học sâu, phục vụ từ phân loại ảnh, phát hiện đối tượng cho đến các mô hình sinh ảnh.

**Tiếp theo:** Tầng gộp và các kiến trúc CNN kinh điển (LeNet, AlexNet, VGG, ResNet).

## Video tham khảo

- **Xem trước:** [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, tiếng Anh. Phân tích rõ các lỗi kích thước dễ gặp khi kết hợp số kênh, padding và stride.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, tiếng Anh. Cập nhật góc nhìn hiện đại về các lựa chọn thiết kế và cạm bẫy trong CNN.
- [C4W2L04 Why ResNets Work](https://www.youtube.com/watch?v=RYth6EbBUqM) — DeepLearningAI, tiếng Anh. Giải thích nguyên nhân mạng tích chập thuần bị suy giảm hiệu năng khi tăng độ sâu và cách ResNet khắc phục.
