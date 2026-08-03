---
layout: post
title: 09-02-02 Vị trí BN, Biến thể và Mẹo
chapter: '09'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter09
---

## Đặt Batch Norm ở Đâu

### Trước hay Sau Kích hoạt?

**Bài báo gốc (trước kích hoạt)**:
```python
x = conv(x)
x = batch_norm(x)
x = relu(x)
```

**Thực hành hiện đại (sau kích hoạt)**:
```python
x = conv(x)
x = relu(x)
x = batch_norm(x)
```

Cả hai đều hoạt động, nhưng “sau” phổ biến hơn hiện nay.

### Ví dụ Mạng Hoàn chỉnh

```python
class ConvNetWithBatchNorm:
    def __init__(self):
        # Convolution + BatchNorm + Activation
        self.conv1 = Conv2D(3, 64, 3, padding=1)
        self.bn1 = BatchNorm2D(64)
        
        self.conv2 = Conv2D(64, 128, 3, padding=1)
        self.bn2 = BatchNorm2D(128)
        
        self.conv3 = Conv2D(128, 256, 3, padding=1)
        self.bn3 = BatchNorm2D(256)
        
        self.fc1 = Linear(256 * 4 * 4, 512)
        self.bn4 = BatchNorm1D(512)
        
        self.fc2 = Linear(512, 10)
    
    def forward(self, x, training=True):
        # Khối 1
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = relu(x)
        x = max_pool(x, 2)
        
        # Khối 2
        x = self.conv2(x)
        x = self.bn2(x, training)
        x = relu(x)
        x = max_pool(x, 2)
        
        # Khối 3
        x = self.conv3(x)
        x = self.bn3(x, training)
        x = relu(x)
        x = max_pool(x, 2)
        
        # Kết nối đầy đủ
        x = x.flatten()
        x = self.fc1(x)
        x = self.bn4(x, training)
        x = relu(x)
        
        x = self.fc2(x)
        return x
```

## Chuẩn hóa Batch cho CNN

Với tầng tích chập, chuẩn hóa trên các chiều không gian:

```python
class BatchNorm2D:
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        
        # Tham số: một cho mỗi kênh
        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))
        
        # Thống kê chạy
        self.running_mean = np.zeros((1, num_channels, 1, 1))
        self.running_var = np.ones((1, num_channels, 1, 1))
    
    def forward(self, x, training=True):
        """
        x: dạng (batch, channels, height, width)
        """
        if training:
            # Tính mean và var trên batch và các chiều không gian
            # Giữ chiều kênh
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            # Chuẩn hóa
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Scale và dịch
            out = self.gamma * x_norm + self.beta
            
            # Cập nhật thống kê chạy
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * var
            
            return out
        else:
            x_norm = (x - self.running_mean) / \
                    np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta
```

## Biến thể và Lựa chọn Thay thế

### 1. Chuẩn hóa Tầng (*Layer Normalization*)

Chuẩn hóa trên các đặc trưng thay vì batch:

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x: dạng (batch, features)
    Chuẩn hóa mỗi mẫu độc lập
    """
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

**Tác vụ dùng**: RNN, Transformer (nơi batch size thay đổi hoặc bằng 1)

### 2. Chuẩn hóa Instance (*Instance Normalization*)

Chuẩn hóa mỗi mẫu và mỗi kênh độc lập:

```python
def instance_norm(x, gamma, beta, eps=1e-5):
    """
    x: dạng (batch, channels, height, width)
    Chuẩn hóa mỗi instance và kênh
    """
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

**Tác vụ dùng**: Chuyển phong cách (*style transfer*), GAN

### 3. Chuẩn hóa Nhóm (*Group Normalization*)

Thỏa hiệp giữa Layer và Instance norm:

```python
def group_norm(x, gamma, beta, num_groups=32, eps=1e-5):
    """
    x: dạng (batch, channels, height, width)
    Chia kênh thành nhóm và chuẩn hóa
    """
    N, C, H, W = x.shape
    x = x.reshape(N, num_groups, C // num_groups, H, W)
    
    mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    var = np.var(x, axis=(2, 3, 4), keepdims=True)
    
    x_norm = (x - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    
    return gamma * x_norm + beta
```

**Tác vụ dùng**: Batch size nhỏ, phát hiện đối tượng

## Vấn đề Thường gặp và Giải pháp

### Vấn đề 1: Batch Size Nhỏ

**Vấn đề**: Thống kê batch không tin cậy với batch nhỏ

**Giải pháp**:
- Dùng Group Normalization hoặc Layer Normalization
- Tăng batch size nếu có thể
- Dùng momentum lớn hơn cho thống kê chạy

### Vấn đề 2: Lệch Train–Test

**Vấn đề**: Hành vi khác nhau giữa huấn luyện và kiểm tra

**Giải pháp**: Luôn nhớ đặt đúng chế độ huấn luyện

```python
# Huấn luyện
model.train()  # hoặc training=True
loss = train_step(data)

# Kiểm tra
model.eval()  # hoặc training=False
accuracy = evaluate(test_data)
```

### Vấn đề 3: Batch Norm + Dropout

**Vấn đề**: Có thể tương tác kém

**Giải pháp**:
- Thường không cần dropout khi có batch norm
- Nếu dùng cả hai: dropout sau batch norm
- Hoặc chỉ dùng một trong hai

## Mẹo Thực tiễn

### 1. Khởi tạo với Batch Norm

Có thể dùng trọng số ban đầu lớn hơn:

```python
# Không BatchNorm: khởi tạo cẩn thận
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

# Có BatchNorm: có thể mạnh tay hơn
W = np.random.randn(n_in, n_out) * 0.05  # Phương sai lớn hơn được
```

### 2. Learning Rate

Có thể dùng learning rate cao hơn nhiều:

```python
# Không BatchNorm
lr = 0.001

# Có BatchNorm
lr = 0.01  # Cao hơn 10×!
```

### 3. Momentum cho Thống kê Chạy

```python
# Thích nghi nhanh (tập dữ liệu nhỏ)
momentum = 0.01

# Thống kê ổn định (tập dữ liệu lớn)
momentum = 0.1  # Mặc định

# Rất ổn định (sản xuất)
momentum = 0.001
```

## Batch Norm trong Kiến trúc Hiện đại

### ResNet

```python
def resnet_block(x):
    identity = x
    
    # Conv -> BN -> ReLU
    x = conv(x)
    x = batch_norm(x)
    x = relu(x)
    
    # Conv -> BN
    x = conv(x)
    x = batch_norm(x)
    
    # Cộng kết nối bỏ qua trước ReLU cuối
    x = x + identity
    x = relu(x)
    
    return x
```

### MobileNet

```python
def depthwise_separable_block(x):
    # Depthwise conv
    x = depthwise_conv(x)
    x = batch_norm(x)
    x = relu(x)
    
    # Pointwise conv
    x = conv_1x1(x)
    x = batch_norm(x)
    x = relu(x)
    
    return x
```

## Tóm tắt

- **Chuẩn hóa Batch** chuẩn hóa đầu vào tầng về phân phối ổn định
- **Giảm dịch chuyển hiệp biến nội**, cho phép huấn luyện nhanh hơn
- **Cho phép learning rate cao hơn** (tăng tốc 10–100×)
- **Hoạt động như chính quy hóa**, giảm nhu cầu dropout
- **Biến thể**: Layer Norm (RNN), Instance Norm (chuyển phong cách), Group Norm (batch nhỏ)
- **Thực hành hiện đại**: Thiết yếu trong hầu hết mạng sâu
- **Then chốt**: Nhớ chế độ huấn luyện so với suy luận!

Chuẩn hóa Batch đã cách mạng hóa học sâu bằng cách làm mạng dễ huấn luyện hơn rất nhiều. Nó giờ là thành phần chuẩn trong hầu hết kiến trúc hiện đại.

**Tiếp theo**: Ta sẽ khám phá chính quy hóa L1/L2, dừng sớm, và tăng cường dữ liệu để hoàn thiện bộ công cụ chính quy hóa!
