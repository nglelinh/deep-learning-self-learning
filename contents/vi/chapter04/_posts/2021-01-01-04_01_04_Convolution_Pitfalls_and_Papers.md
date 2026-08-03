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

## 5. Khái niệm liên quan
### CNN so với Mạng Neuron Thường

![CNN vs Regular Neural Network](/deep-learning-self-learning/img/chapter_img/chapter04/conv7.jpg)
*Vì sao không chỉ dùng mạng neuron thường cho ảnh? CNN thắng vì giữ cấu trúc không gian, dùng bộ lọc chia sẻ (ít tham số hơn nhiều), nhận diện đối tượng bất kể vị trí, và được thiết kế riêng cho xử lý dữ liệu thị giác. Nguồn: Analytics Vidhya*

| Khía cạnh | Kết nối đầy đủ | Tích chập |
|--------|-----------------|---------------|
| Kết nối | Tất cả-với-tất cả | Cục bộ (kích thước nhân) |
| Chia sẻ tham số | Không | Cùng bộ lọc mọi nơi |
| Tham số (32×32×3 → 64) | 196,672 | 1,792 |
| Bất biến tịnh tiến | Không | Có |
| Giữ cấu trúc không gian | Không | Có |

### Tầng Kết nối Đầy đủ
- Nối mọi neuron đầu vào với mọi neuron đầu ra
- Không giả định về cấu trúc không gian
- Dùng sau các tầng conv cho phân loại cuối
- Tham số tăng bậc hai theo kích thước đầu vào

### Tầng Gộp (*Pooling*)
- Giảm mẫu bản đồ đặc trưng (giảm kích thước không gian)
- Tăng bất biến tịnh tiến
- Giảm tính toán và bộ nhớ
- Không có tham số học được
- Các loại phổ biến: Max pooling, Average pooling

### Chuẩn hóa Batch (*Batch Normalization*)
- Chuẩn hóa kích hoạt về trung bình zero và phương sai đơn vị
- Ổn định huấn luyện bằng cách giảm dịch chuyển hiệp biến nội (*internal covariate shift*)
- Thường đặt sau tích chập, trước kích hoạt
- Cho phép learning rate cao hơn và hội tụ nhanh hơn

### Kết nối Dư (*Residual / Skip Connections*)
- Cho phép gradient chảy trực tiếp qua mạng
- Cho phép huấn luyện mạng rất sâu (100+ tầng)
- Đầu ra: $$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$
- Đổi mới then chốt của ResNet (2015)

### Tích chập Tách theo Chiều sâu (*Depthwise Separable Convolutions*)
- Phân rã tích chập chuẩn thành depthwise + pointwise
- Ít tham số và phép tính hơn rất nhiều
- Depthwise: Một bộ lọc cho mỗi kênh đầu vào
- Pointwise: Conv 1×1 để kết hợp kênh
- Dùng trong MobileNet, EfficientNet cho triển khai di động

### Tích chập Giãn (*Dilated / Atrous Convolutions*)
- Chèn khoảng trống giữa các phần tử nhân
- Tăng trường tiếp nhận mà không thêm tham số
- Dùng trong phân đoạn ngữ nghĩa (DeepLab)
- Tỷ lệ giãn $$d$$: các phần tử nhân cách nhau $$d$$ điểm ảnh

---

## 6. Bài báo Nền tảng

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Tác giả*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
Giới thiệu LeNet-5, CNN thành công đầu tiên cho nhận diện chữ số. Chứng minh rằng các tầng tích chập với trọng số chia sẻ có thể học đặc trưng phân cấp từ điểm ảnh thô. Thiết lập mẫu conv-pool-conv-pool-FC thống trị gần hai thập kỷ. Bài báo này đặt nền móng cho mọi CNN hiện đại.

**["ImageNet Classification with Deep Convolutional Neural Networks" (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)**  
*Tác giả*: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
AlexNet thắng ImageNet 2012 với khoảng cách lớn (lỗi top-5: 15,3% so với 26,2% của hạng nhì), chứng minh CNN sâu có thể mở rộng cho tác vụ thị giác thực tế. Giới thiệu kích hoạt ReLU, chính quy hóa dropout, và huấn luyện GPU vào học sâu. Bài báo này khơi mào cuộc cách mạng học sâu đã biến đổi AI.

**["Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015)](https://arxiv.org/abs/1409.1556)**  
*Tác giả*: Karen Simonyan, Andrew Zisserman  
VGGNet chứng minh độ sâu mạng là then chốt cho hiệu năng. Chỉ dùng bộ lọc 3×3 nhỏ xếp chồng để đạt trường tiếp nhận lớn, cho thấy hai conv 3×3 có cùng trường tiếp nhận như một 5×5 nhưng ít tham số hơn và nhiều phi tuyến hơn. Kiến trúc đơn giản, lặp lại trở thành khuôn mẫu cho các thiết kế sau.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Tác giả*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Giới thiệu kết nối bỏ qua cho phép huấn luyện mạng 152+ tầng. Giải quyết bài toán suy giảm trong đó mạng sâu hơn có lỗi huấn luyện cao hơn mạng nông. ResNet thắng ImageNet 2015 và thay đổi căn bản cách ta thiết kế mạng sâu. Khung học dư giờ được dùng trong transformer, mô hình khuếch tán, và hầu như mọi kiến trúc sâu.

**["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)](https://arxiv.org/abs/1704.04861)**  
*Tác giả*: Andrew G. Howard và cộng sự  
Giới thiệu tích chập tách theo chiều sâu giảm tính toán 8–9× trong khi giữ độ chính xác. Cho phép chạy CNN trên điện thoại và thiết bị nhúng. Mở đường cho mô hình hiệu quả dẫn đến EfficientNet và triển khai AI ưu tiên di động.

**["Rethinking the Inception Architecture for Computer Vision" (2016)](https://arxiv.org/abs/1512.00567)**  
*Tác giả*: Christian Szegedy và cộng sự  
Inception-v3 giới thiệu tích chập phân rã (dùng 1×n và n×1 thay vì n×n) và bộ phân loại phụ. Chứng minh thiết kế kiến trúc cẩn thận có thể cải thiện cả độ chính xác lẫn hiệu quả. Nhiều nguyên tắc thiết kế từ bài này ảnh hưởng kiến trúc hiện đại.

---

## 7. Cạm bẫy Thường gặp và Mẹo

### Cạm bẫy 1: Không Dùng Padding

**Vấn đề**: Đầu ra co lại mỗi tầng, mất thông tin biên

```python
# Không padding: 32×32 → 30×30 → 28×28 → ... (co nhanh!)
conv1 = nn.Conv2d(3, 64, kernel_size=3)  # Không padding
conv2 = nn.Conv2d(64, 128, kernel_size=3)

# Có padding: 32×32 → 32×32 → 32×32 (giữ kích thước)
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Đệm "same"
conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
```

**Giải pháp**: Dùng `padding = (kernel_size - 1) // 2` với kích thước nhân lẻ để giữ kích thước không gian.

### Cạm bẫy 2: Quên Thứ tự Chiều Kênh

**Vấn đề**: Các framework dùng quy ước khác nhau

```python
# PyTorch: (N, C, H, W) - Kênh trước
x_pytorch = torch.randn(batch, channels, height, width)

# TensorFlow/Keras: (N, H, W, C) - Kênh sau
x_tensorflow = tf.random.normal([batch, height, width, channels])

# Chuyển đổi giữa chúng
x_tf_to_pytorch = x_tensorflow.permute(0, 3, 1, 2)  # Nếu dùng torch
```

### Cạm bẫy 3: Kernel Ban đầu Quá Lớn

**Vấn đề**: Kernel lớn có nhiều tham số nhưng lợi ích trường tiếp nhận hạn chế

```python
# Kém hiệu quả hơn (hiểu biết của VGG)
conv = nn.Conv2d(3, 64, kernel_size=7)  # 7×7 = 49 params/kênh

# Tốt hơn: Xếp chồng kernel nhỏ
conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
# Ba 3×3 = 27 params/kênh, cùng trường tiếp nhận 7×7, gấp 3 phi tuyến
```

### Mẹo 1: Tích chập 1×1 để Kiểm soát Số chiều

```python
# Giảm kênh (nút thắt / bottleneck)
bottleneck = nn.Conv2d(512, 64, kernel_size=1)
# 512 → 64 kênh với tính toán tối thiểu

# Tăng kênh
expand = nn.Conv2d(64, 512, kernel_size=1)
# Trộn thông tin kênh mà không thao tác không gian
```

### Mẹo 2: Tích chập Stride thay cho Pooling

```python
# Truyền thống: Conv → Pool
conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2, 2)

# Hiện đại: Conv stride (giảm mẫu học được)
strided_conv = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# Cùng kích thước đầu ra, nhưng giảm mẫu được học!
```

### Mẹo 3: Khởi tạo Trọng số Đúng

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
print(f"Trường tiếp nhận: {receptive_field_1d(layers)}")  # 14
```

---

## 8. Tóm tắt các điểm chính
1. **Tích chập so với Tương quan chéo**: Học sâu dùng tương quan chéo nhưng gọi là tích chập. Sự khác biệt (lật nhân) không quan trọng vì trọng số được học.

2. **Kết nối Cục bộ + Chia sẻ Trọng số**: Hai tính chất này làm CNN hiệu quả tham số hơn rất nhiều so với mạng kết nối đầy đủ.

3. **Học Đặc trưng Phân cấp**: CNN tự động học cạnh → kết cấu → bộ phận → đối tượng, phản ánh thị giác sinh học.

4. **Kích thước Đầu ra**: $$n_{out} = \lfloor(n_{in} + 2p - k)/s\rfloor + 1$$ — hãy nhớ công thức này!

5. **Bộ lọc Nhỏ Thắng**: Xếp chồng tích chập 3×3 thay vì dùng kernel lớn (nhiều phi tuyến hơn, ít tham số hơn).

6. **Trường Tiếp nhận**: Tăng theo độ sâu. Hiểu trường tiếp nhận là then chốt khi thiết kế kiến trúc.

7. **Thực hành Tốt nhất Hiện đại**:
   - Conv → BatchNorm → ReLU
   - Kết nối dư cho mạng sâu
   - Global average pooling thay vì flatten + FC

Tầng tích chập là nền tảng của thị giác máy tính trong học sâu. Nắm vững chúng là thiết yếu cho mọi ứng dụng thị giác — từ phân loại ảnh đến phát hiện đối tượng đến sinh ảnh.


**Tiếp theo**: Tầng gộp và các kiến trúc CNN hoàn chỉnh (LeNet, AlexNet, VGG, ResNet)!
