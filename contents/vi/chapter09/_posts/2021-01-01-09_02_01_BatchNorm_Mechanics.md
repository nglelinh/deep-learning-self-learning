---
layout: post
title: 09-02-01 Cơ chế Chuẩn hóa Batch
chapter: '09'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter09
---

## Chuẩn hóa Batch là gì?

**Chuẩn hóa Batch** (*Batch Normalization* — BatchNorm hay BN), được giới thiệu bởi Ioffe và Szegedy (2015), chuẩn hóa đầu vào của mỗi tầng về trung bình zero và phương sai đơn vị. Nó đã trở thành một trong những kỹ thuật quan trọng nhất trong học sâu hiện đại.

### Vấn đề: Dịch chuyển Hiệp biến Nội

Khi mạng huấn luyện:
- Phân phối đầu vào tầng thay đổi
- Mỗi tầng phải thích nghi với phân phối đầu vào mới
- Làm chậm huấn luyện đáng kể
- Làm mạng nhạy với khởi tạo

**Giải pháp Batch Norm**: Chuẩn hóa đầu vào tầng về phân phối ổn định.

## Chuẩn hóa Batch Hoạt động như thế nào

### Lượt Xuôi (Huấn luyện)

Với mini-batch kích hoạt $$\mathbf{x} = \{x_1, x_2, \ldots, x_m\}$$:

**Bước 1: Tính thống kê batch**

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma^2_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**Bước 2: Chuẩn hóa**

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$$

trong đó $$\epsilon$$ (ví dụ $$10^{-5}$$) ngăn chia cho zero.

**Bước 3: Scale và dịch (tham số học được)**

$$y_i = \gamma \hat{x}_i + \beta$$

trong đó:
- $$\gamma$$: tham số scale (học được)
- $$\beta$$: tham số dịch (học được)

### Suy luận (Kiểm tra)

Dùng thống kê quần thể (trung bình trượt từ huấn luyện):

$$\hat{x} = \frac{x - \mu_{\text{pop}}}{\sqrt{\sigma^2_{\text{pop}} + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

## Cài đặt

```python
import numpy as np

class BatchNorm1D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        num_features: số đặc trưng/kênh
        eps: hằng số nhỏ cho ổn định số
        momentum: cho cập nhật trung bình/phương sai chạy
        """
        self.eps = eps
        self.momentum = momentum
        
        # Tham số học được
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Thống kê chạy (cho suy luận)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache cho backprop
        self.cache = None
    
    def forward(self, x, training=True):
        """
        x: đầu vào dạng (batch_size, num_features)
        training: có đang ở chế độ huấn luyện không
        """
        if training:
            # Tính thống kê batch
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Chuẩn hóa
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Scale và dịch
            out = self.gamma * x_normalized + self.beta
            
            # Cập nhật thống kê chạy
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
            
            # Cache cho lượt ngược
            self.cache = (x, x_normalized, batch_mean, batch_var)
            
        else:
            # Dùng thống kê chạy
            x_normalized = (x - self.running_mean) / \
                          np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, dout):
        """
        Lan truyền ngược qua chuẩn hóa batch
        dout: gradient từ tầng sau
        """
        x, x_normalized, mean, var = self.cache
        N, D = x.shape
        
        # Gradient của tham số
        self.dgamma = np.sum(dout * x_normalized, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        
        # Gradient của x đã chuẩn hóa
        dx_normalized = dout * self.gamma
        
        # Gradient của phương sai
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * \
                     (var + self.eps)**(-1.5), axis=0)
        
        # Gradient của trung bình
        dmean = np.sum(dx_normalized * -1 / np.sqrt(var + self.eps), axis=0) + \
                dvar * np.mean(-2 * (x - mean), axis=0)
        
        # Gradient của x
        dx = dx_normalized / np.sqrt(var + self.eps) + \
             dvar * 2 * (x - mean) / N + \
             dmean / N
        
        return dx

# Ví dụ sử dụng
batch_norm = BatchNorm1D(num_features=128)

# Huấn luyện
x_train = np.random.randn(32, 128)
out_train = batch_norm.forward(x_train, training=True)
print(f"Trung bình đầu ra train: {np.mean(out_train, axis=0)[:5]}")
print(f"Độ lệch chuẩn đầu ra train: {np.std(out_train, axis=0)[:5]}")

# Kiểm tra
x_test = np.random.randn(10, 128)
out_test = batch_norm.forward(x_test, training=False)
print(f"Test dùng thống kê chạy")
```

## Vì sao Chuẩn hóa Batch Hoạt động

### 1. Giảm Dịch chuyển Hiệp biến Nội
- Ổn định phân phối đầu vào tầng
- Mỗi tầng thấy đầu vào nhất quán hơn
- Dễ học hơn

### 2. Cho phép Learning Rate Cao hơn
- Dòng gradient ổn định hơn
- Có thể huấn luyện nhanh hơn 10–100×
- Ít nhạy với khởi tạo

### 3. Hoạt động như Chính quy hóa
- Thêm nhiễu vào kích hoạt (từ thống kê batch)
- Hiệu ứng tương tự dropout
- Có thể giảm nhu cầu dùng dropout

### 4. Làm Mượt Bề mặt Tối ưu
- Làm mặt loss mượt hơn
- Gradient dễ dự đoán hơn
- Tối ưu dễ dàng hơn
