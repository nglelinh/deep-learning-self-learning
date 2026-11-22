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

## Vị trí đặt BatchNorm

### Trước hay sau hàm kích hoạt?

**Cách bố trí trong bài báo gốc (trước hàm kích hoạt):**
```python
x = conv(x)
x = batch_norm(x)
x = relu(x)
```

**Cách bố trí thay thế (sau hàm kích hoạt):**
```python
x = conv(x)
x = relu(x)
x = batch_norm(x)
```

Cả hai cách đều hoạt động tốt tùy kiến trúc; trong các mạng tích chập chuẩn như ResNet hay MobileNet, thứ tự `Conv → BN → ReLU` vẫn rất phổ biến, trong khi một số thực nghiệm lại chọn đặt sau hàm kích hoạt.

### Ví dụ mạng CNN kết hợp BatchNorm

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

## Chuẩn hóa Batch cho CNN (BatchNorm2D)

Với tầng tích chập có đầu vào dạng `(batch, channels, height, width)`, thống kê trung bình và phương sai được tính trên chiều batch cùng hai chiều không gian, giữ riêng cho từng kênh:

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

## Các biến thể chuẩn hóa thay thế

### 1. Chuẩn hóa lớp (*Layer Normalization*)

Chuẩn hóa dọc theo chiều đặc trưng của từng mẫu độc lập thay vì dọc theo batch:

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

**Ứng dụng tiêu biểu:** RNN và Transformer (nơi độ dài chuỗi thay đổi hoặc batch size nhỏ).

### 2. Chuẩn hóa theo từng mẫu và kênh (*Instance Normalization*)

Chuẩn hóa độc lập theo hai chiều không gian cho từng mẫu và từng kênh:

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

**Ứng dụng tiêu biểu:** chuyển đổi phong cách ảnh (*style transfer*) và GAN.

### 3. Chuẩn hóa theo nhóm (*Group Normalization*)

Chia các kênh thành nhiều nhóm rồi chuẩn hóa trong từng nhóm, nằm giữa LayerNorm và InstanceNorm:

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

**Ứng dụng tiêu biểu:** bài toán phát hiện đối tượng và các hệ thống bị giới hạn batch size nhỏ.

## Các vấn đề thường gặp và hướng xử lý

### Vấn đề 1: Batch size quá nhỏ

Khi kích thước mini-batch nhỏ, ước lượng trung bình và phương sai theo batch trở nên nhiễu và kém tin cậy.

**Hướng khắc phục:**
- Chuyển sang Group Normalization hoặc Layer Normalization.
- Tăng batch size nếu bộ nhớ cho phép.
- Giảm `momentum` (theo quy ước PyTorch `running = (1 - m) * running + m * batch`) để thống kê chạy cập nhật mượt hơn qua nhiều batch.

### Vấn đề 2: Quên chuyển chế độ huấn luyện và kiểm tra

BatchNorm sử dụng thống kê của batch hiện tại khi huấn luyện nhưng dùng thống kê chạy khi suy luận, nên bắt buộc phải đặt đúng chế độ:

```python
# Huấn luyện
model.train()  # hoặc training=True
loss = train_step(data)

# Kiểm tra
model.eval()  # hoặc training=False
accuracy = evaluate(test_data)
```

### Vấn đề 3: Kết hợp BatchNorm và Dropout

Do cả hai cùng tác động lên phân phối và phương sai kích hoạt, việc kết hợp tùy tiện có thể làm giảm hiệu năng. Thông thường chỉ cần dùng BatchNorm ở các tầng tích chập; nếu vẫn muốn thêm dropout, nên đặt dropout sau BatchNorm và hàm kích hoạt.

## Mẹo thực hành

### 1. Độ nhạy với khởi tạo trọng số

BatchNorm giúp mạng ít nhạy cảm hơn với thang độ lớn của trọng số khởi tạo:

```python
# Không BatchNorm: khởi tạo cẩn thận
W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

# Có BatchNorm: ít nhạy cảm hơn với thang khởi tạo
W = np.random.randn(n_in, n_out) * 0.05
```

### 2. Tốc độ học (*Learning Rate*)

Nhờ bề mặt tối ưu ổn định hơn, mạng có BatchNorm thường cho phép dùng tốc độ học lớn hơn:

```python
# Không BatchNorm
lr = 0.001

# Có BatchNorm
lr = 0.01  # Có thể thử cao hơn đáng kể
```

### 3. Hệ số `momentum` cho thống kê chạy

Với quy ước `running = (1 - momentum) * running + momentum * batch`:

```python
# Thích nghi nhanh theo batch mới (mặc định PyTorch)
momentum = 0.1

# Mượt hơn, giảm nhiễu khi batch nhỏ
momentum = 0.01

# Rất ổn định khi cập nhật trên luồng dữ liệu dài
momentum = 0.001
```

## BatchNorm trong các kiến trúc phổ biến

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

- **Batch Normalization** chuẩn hóa kích hoạt theo mini-batch để ổn định phân phối giữa các tầng và hỗ trợ tốc độ học lớn hơn.
- **Hiệu ứng chính quy hóa nhẹ** giúp giảm nhu cầu dùng dropout trong nhiều mạng CNN.
- **Các biến thể thay thế** gồm LayerNorm (RNN, Transformer), InstanceNorm (chuyển phong cách, GAN) và GroupNorm (batch size nhỏ).
- **Lưu ý quan trọng**: luôn chuyển đúng chế độ `model.train()` khi huấn luyện và `model.eval()` khi đánh giá.

Các kỹ thuật chính quy hóa khác như L1/L2, dừng sớm và tăng cường dữ liệu kết hợp cùng dropout và chuẩn hóa tạo thành bộ công cụ hoàn chỉnh để cải thiện khả năng tổng quát hóa.

## Video tham khảo

- **Xem trước:** [Why Does Batch Norm Work? (C2W3L06)](https://www.youtube.com/watch?v=nUUqwaxLnWs) — DeepLearningAI (tiếng Anh). Cơ chế nền tảng giúp đánh giá việc đặt BatchNorm trước hay sau hàm kích hoạt.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — Andrej Karpathy (tiếng Anh). Trình bày vị trí đặt chuẩn hóa trong khối mạng hiện đại và hiện tượng suy giảm khi batch quá nhỏ.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering (tiếng Anh). Giới thiệu LayerNorm và các biến thể chuẩn hóa trong phần kiến trúc mạng hiện đại.
