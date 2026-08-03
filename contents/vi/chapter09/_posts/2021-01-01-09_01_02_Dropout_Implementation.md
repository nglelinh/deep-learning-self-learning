---
layout: post
title: 09-01-02 Cài đặt Dropout và Cạm bẫy
chapter: '09'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter09
---

## 4. Đoạn Mã

### Cài đặt PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPWithDropout(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10,
                 dropout_rate=0.5):
        super(MLPWithDropout, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Tầng 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)  # Dropout sau kích hoạt
        
        # Tầng 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Tầng đầu ra (không dropout)
        x = self.fc3(x)
        return x

# Huấn luyện
model = MLPWithDropout(dropout_rate=0.5)
model.train()  # Bật dropout

x_train = torch.randn(32, 784)  # Batch 32
output_train = model(x_train)
print(f"Shape đầu ra huấn luyện: {output_train.shape}")  # (32, 10)

# Kiểm tra
model.eval()  # Tắt dropout

x_test = torch.randn(10, 784)
output_test = model(x_test)
print(f"Shape đầu ra kiểm tra: {output_test.shape}")  # (10, 10)

# Kiểm tra hiệu ứng dropout
model.train()
outputs = []
for _ in range(5):
    out = model(x_train[0:1])  # Cùng đầu vào
    outputs.append(out)

print("Huấn luyện (có dropout) - đầu ra biến thiên:")
print(torch.stack(outputs).std(dim=0).mean().item())

model.eval()
outputs_test = []
for _ in range(5):
    out = model(x_train[0:1])  # Cùng đầu vào
    outputs_test.append(out)

print("Kiểm tra (không dropout) - đầu ra giống hệt:")
print(torch.stack(outputs_test).std(dim=0).mean().item())
```

### Cài đặt Dropout Thủ công

```python
class DropoutManual:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, X, training=True):
        """
        X: (batch, features)
        """
        if training:
            # Sinh mặt nạ: 1 với xác suất (1-p), 0 với xác suất p
            keep_prob = 1 - self.dropout_rate
            self.mask = np.random.binomial(1, keep_prob, size=X.shape)
            
            # Áp dụng mặt nạ và scale
            return X * self.mask / keep_prob
        else:
            return X
    
    def backward(self, dout):
        """Gradient chỉ chảy qua các neuron được giữ"""
        keep_prob = 1 - self.dropout_rate
        return dout * self.mask / keep_prob

# Ví dụ
dropout = DropoutManual(dropout_rate=0.5)
X = np.random.randn(4, 100)

# Huấn luyện
X_train = dropout.forward(X, training=True)
print(f"Đơn vị bị loại: {np.sum(X_train == 0)}/{X_train.size}")

# Kiểm tra  
X_test = dropout.forward(X, training=False)
print(f"Đơn vị bị loại lúc test: {np.sum(X_test == 0)}/{X_test.size}")
```

### Spatial Dropout cho CNN

```python
class SpatialDropout2D(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        x: (batch, channels, height, width)
        Loại bỏ toàn bộ bản đồ đặc trưng
        """
        if not self.training:
            return x
        
        # Shape mặt nạ: (batch, channels, 1, 1)
        # Cùng mặt nạ cho mọi vị trí không gian trong một kênh
        batch, channels = x.shape[:2]
        mask = torch.bernoulli(torch.ones(batch, channels, 1, 1) * 
                              (1 - self.dropout_rate))
        
        return x * mask / (1 - self.dropout_rate)

# Ví dụ
spatial_dropout = SpatialDropout2D(dropout_rate=0.3)
x = torch.randn(2, 64, 32, 32)  # Batch=2, 64 kênh, 32×32
out = spatial_dropout(x)
print(f"Toàn bộ kênh bị loại: {(out.sum(dim=(2,3)) == 0).sum().item()}")
```

## 5. Khái niệm liên quan
### Chuẩn hóa Batch (*Batch Normalization*)
- Cũng cung cấp chính quy hóa
- Thường giảm nhu cầu dùng dropout
- Kiến trúc hiện đại: BN thay dropout trong nhiều trường hợp
- Nếu dùng cả hai: dropout sau batch norm

### Tăng cường Dữ liệu (*Data Augmentation*)
- Một dạng chính quy hóa khác
- Thêm nhiễu/biến thiên vào dữ liệu huấn luyện
- Bổ trợ cho dropout

### Dừng Sớm (*Early Stopping*)
- Dừng huấn luyện khi loss validation tăng
- Ngăn overfitting
- Dùng cùng với dropout

### Phương pháp Ensemble
- Huấn luyện nhiều mô hình độc lập
- Trung bình các dự đoán
- Dropout xấp xỉ điều này với một mô hình

### DropConnect
- Loại bỏ kết nối thay vì neuron
- Hiệu ứng tương tự, cài đặt khác
- Ít phổ biến hơn

## 6. Bài báo Nền tảng

**["Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)](https://jmlr.org/papers/v15/srivastava14a.html)**  
*Tác giả*: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov  
**Bài báo nền tảng về dropout**. Giới thiệu dropout và phân tích lý thuyết cho thấy nó ngăn đồng thích nghi của neuron. Chứng minh cải thiện rõ rệt trên nhiều benchmark gồm MNIST, CIFAR và ImageNet. Trở thành kỹ thuật chính quy hóa chuẩn trong học sâu.

**["Improving neural networks by preventing co-adaptation of feature detectors" (2012)](https://arxiv.org/abs/1207.0580)**  
*Tác giả*: Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, và cộng sự  
Bài dropout sớm giới thiệu khái niệm. Giải thích cách dropout tạo ensemble của số mũ các mạng mỏng chia sẻ tham số. Cho thấy cải thiện thực nghiệm và cung cấp trực giác vì sao vô hiệu hóa ngẫu nhiên giúp.

**["Regularization of Neural Networks using DropConnect" (2013)](http://proceedings.mlr.press/v28/wan13.html)**  
*Tác giả*: Li Wan, Matthew Zeiler, Sixin Zhang, Yann LeCun, Rob Fergus  
Giới thiệu DropConnect — loại bỏ kết nối thay vì neuron. Cho thấy biến thể này đôi khi vượt dropout. Tổng quát hóa khái niệm loại bỏ ngẫu nhiên ra ngoài đơn vị riêng lẻ.

**["Spatial Dropout" (2015) - một phần của "Efficient Object Localization Using CNNs"](https://arxiv.org/abs/1411.4280)**  
*Tác giả*: Jonathan Tompson, Ross Goroshin, Arjun Jain, Yann LeCun, Christoph Bregler  
Giới thiệu spatial dropout cho tầng tích chập — loại bỏ toàn bộ bản đồ đặc trưng thay vì từng kích hoạt. Cho thấy hiệu quả hơn với CNN nơi các điểm ảnh gần nhau tương quan.

**["A Theoretically Grounded Application of Dropout in RNNs" (2016)](https://arxiv.org/abs/1512.05287)**  
*Tác giả*: Yarin Gal, Zoubin Ghahramani  
Phân tích dropout trong RNN và giới thiệu variational dropout — dùng cùng mặt nạ qua các bước thời gian. Cung cấp nền tảng lý thuyết nối dropout với suy luận Bayesian, cho thấy dropout xấp xỉ ước lượng bất định.

## Cạm bẫy Thường gặp và Mẹo

### ⚠️ Cạm bẫy 1: Quên Tắt khi Kiểm tra
**Vấn đề**: Dropout vẫn bật lúc test → dự đoán không nhất quán  
**Giải pháp**: Luôn đặt `model.eval()`

```python
# Sai
output = model(test_data)  # Dropout vẫn bật nếu model ở chế độ train!

# Đúng
model.eval()
with torch.no_grad():
    output = model(test_data)
```

### ⚠️ Cạm bẫy 2: Tỷ lệ Dropout Quá Cao
**Vấn đề**: Mạng mất quá nhiều dung lượng  
**Giải pháp**: Bắt đầu với 0,5, giảm nếu hiệu năng tụt

```python
# Quá mạnh - có thể hại hiệu năng
dropout = nn.Dropout(0.9)  # Loại 90%!

# Điểm xuất phát tốt hơn
dropout_fc = nn.Dropout(0.5)      # Kết nối đầy đủ
dropout_conv = nn.Dropout(0.2)    # Tích chập
dropout_input = nn.Dropout(0.1)   # Tầng đầu vào
```

### ⚠️ Cạm bẫy 3: Dùng cùng Batch Normalization
**Vấn đề**: Cả hai đều chính quy hóa, có thể tương tác kém  
**Giải pháp**: Thường chọn một trong hai

```python
# Thực hành hiện đại: Dùng BN, bỏ dropout
x = conv(x)
x = batch_norm(x)
x = relu(x)
# Không cần dropout!

# Nếu dùng cả hai: dropout sau BN
x = conv(x)
x = batch_norm(x)
x = relu(x)
x = dropout(x)  # Sau kích hoạt
```

### ✅ Mẹo 1: Tỷ lệ Khác nhau cho Tầng Khác nhau
```python
class SmartDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_input = nn.Dropout(0.2)  # Bảo thủ
        self.dropout_hidden = nn.Dropout(0.5)  # Chuẩn
        self.dropout_deep = nn.Dropout(0.3)    # Thấp hơn cho tầng sâu
```

### ✅ Mẹo 2: Monte Carlo Dropout (Ước lượng Bất định)
```python
def mc_dropout_predict(model, x, num_samples=10):
    """
    Nhiều lượt xuôi ngẫu nhiên để ước lượng bất định
    """
    model.train()  # Giữ dropout BẬT
    predictions = []
    
    for _ in range(num_samples):
        with torch.no_grad():
            pred = model(x)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)
    
    return mean_pred, uncertainty

# Lấy dự đoán kèm bất định
mean, std = mc_dropout_predict(model, x_test)
print(f"Dự đoán: {mean}, Bất định: {std}")
```

### ✅ Mẹo 3: Dropout Lịch trình
```python
class ScheduledDropout:
    """Tăng dần tỷ lệ dropout trong quá trình huấn luyện"""
    def __init__(self, initial_rate=0.1, final_rate=0.5, total_epochs=100):
        self.initial = initial_rate
        self.final = final_rate
        self.total = total_epochs
    
    def get_rate(self, epoch):
        progress = min(epoch / self.total, 1.0)
        return self.initial + (self.final - self.initial) * progress
```

## Điểm then chốt

- **Dropout** vô hiệu hóa ngẫu nhiên neuron khi huấn luyện
- **Tỷ lệ**: 0,5 cho tầng FC, 0,1–0,3 cho conv, 0,2 cho đầu vào
- **Scale**: Nhân với $$\frac{1}{1-p}$$ khi huấn luyện
- **Suy luận**: Dùng tất cả neuron, không dropout
- **Hiệu ứng**: Học ensemble, ngăn đồng thích nghi
- **Dùng hiện đại**: Ít phổ biến hơn khi có chuẩn hóa batch
- **Then chốt**: Nhớ chế độ train so với eval!

Dropout vẫn là một trong những kỹ thuật chính quy hóa đơn giản nhất nhưng hiệu quả nhất trong học sâu!

**Tiếp theo**: Chuẩn hóa Batch — kỹ thuật mạnh khác đã cách mạng hóa huấn luyện!
