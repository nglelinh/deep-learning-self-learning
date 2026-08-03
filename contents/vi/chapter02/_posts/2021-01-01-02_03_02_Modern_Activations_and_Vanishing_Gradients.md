---
layout: post
title: 02-03-02 Hàm Kích hoạt Hiện đại và Gradient Biến mất
chapter: '02'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

## Chọn Hàm Kích hoạt Phù hợp

### Cho Lớp Ẩn

**Khuyến nghị mặc định: ReLU**
- Bắt đầu với ReLU cho hầu hết ứng dụng
- Hiệu quả tính toán
- Hoạt động tốt trong thực tế

**Nếu gặp vấn đề ReLU chết:**
- Thử Leaky ReLU hoặc ELU
- Kiểm tra learning rate và khởi tạo

**Cho mô hình hiện đại / quy mô lớn:**
- GELU cho transformer và NLP
- Swish cho mô hình ảnh khi hiệu năng then chốt

**Cho mạng rất sâu:**
- Cân nhắc ELU hoặc SELU
- Có thể cần kỹ thuật chuẩn hóa (trình bày sau)

### Cho Lớp Đầu ra

**Phân loại nhị phân:**
- **Sigmoid**: Cho ra xác suất của lớp dương

**Phân loại đa lớp:**
- **Softmax**: Cho ra phân phối xác suất trên các lớp

**Hồi quy:**
- **Tuyến tính (đồng nhất)**: Cho đầu ra không bị chặn
- **ReLU**: Cho đầu ra không âm (ví dụ: giá, số đếm)
- **Sigmoid/tanh**: Cho đầu ra bị chặn

**Phân loại đa nhãn:**
- **Sigmoid**: Xác suất độc lập cho mỗi nhãn

## Triển khai Thực tế

```python
import numpy as np

class Activations:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(z):
        s = Activations.sigmoid(z)
        return s * (1 - s)
    
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z)**2
    
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)
    
    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        return np.where(z > 0, 1, alpha)
    
    @staticmethod
    def elu(z, alpha=1.0):
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))
    
    @staticmethod
    def elu_derivative(z, alpha=1.0):
        return np.where(z > 0, 1, Activations.elu(z, alpha) + alpha)
    
    @staticmethod
    def softmax(z):
        # Ổn định số: trừ max
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    @staticmethod
    def swish(z):
        return z * Activations.sigmoid(z)
    
    @staticmethod
    def gelu(z):
        # Xấp xỉ
        return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

# Ví dụ sử dụng
z = np.array([-2, -1, 0, 1, 2])
print("ReLU:", Activations.relu(z))
print("Leaky ReLU:", Activations.leaky_relu(z))
print("Sigmoid:", Activations.sigmoid(z))
print("Tanh:", Activations.tanh(z))
```

## Bài toán Gradient Biến mất

### Vì sao Điều này Quan trọng

Trong lan truyền ngược, gradient được nhân qua các lớp:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \cdot \frac{\partial \mathbf{a}^{[L]}}{\partial \mathbf{z}^{[L]}} \cdot \ldots \cdot \frac{\partial \mathbf{z}^{[2]}}{\partial \mathbf{a}^{[1]}} \cdot \frac{\partial \mathbf{a}^{[1]}}{\partial \mathbf{z}^{[1]}} \cdot \frac{\partial \mathbf{z}^{[1]}}{\partial \mathbf{W}^{[1]}}$$

### Vấn đề với Sigmoid/Tanh

- Đạo hàm cực đại: $$\sigma'(z) = 0.25$$ (sigmoid), $$\tanh'(z) = 1$$ (tanh tại $$z=0$$)
- Đạo hàm điển hình: Nhỏ hơn nhiều ($$< 0.25$$ với sigmoid)
- Sau nhiều lớp: $$0.25^{10} \approx 9.5 \times 10^{-7}$$ (cực kỳ nhỏ!)

**Kết quả**: Gradient biến mất, các lớp đầu học rất chậm hoặc không học.

### ReLU Cứu Cánh

- Đạo hàm bằng 1 với đầu vào dương (không biến mất)
- Gradient chảy không đổi qua các đơn vị ReLU đang hoạt động
- Cho phép huấn luyện mạng sâu hơn nhiều

### Bài toán ReLU Chết

- Nếu $$z < 0$$ luôn, gradient bằng 0, không có học
- Có thể xảy ra với:
  - Khởi tạo kém
  - Learning rate cao
  - Cập nhật không may

**Giải pháp:**
- Dùng Leaky ReLU, ELU, hoặc các biến thể khác
- Khởi tạo đúng (He initialization cho ReLU)
- Learning rate hợp lý
- Batch normalization (trình bày sau)

## Tóm tắt

- **Hàm kích hoạt** đưa vào phi tuyến tính, cho phép mạng học các mẫu phức tạp
- **ReLU** là lựa chọn mặc định cho lớp ẩn trong học sâu hiện đại
- **Sigmoid** dùng cho đầu ra phân loại nhị phân
- **Softmax** dùng cho đầu ra phân loại đa lớp
- **Hàm kích hoạt nâng cao** (ELU, Swish, GELU) có thể cải thiện hiệu năng
- **Gradient biến mất** là vấn đề lớn với sigmoid/tanh trong mạng sâu
- **ReLU** giảm gradient biến mất nhưng đưa vào bài toán ReLU chết
- Lựa chọn hàm kích hoạt ảnh hưởng đáng kể đến huấn luyện và hiệu năng

Ở bài tiếp theo, ta sẽ xét lan truyền xuôi chi tiết với các ví dụ cụ thể.
