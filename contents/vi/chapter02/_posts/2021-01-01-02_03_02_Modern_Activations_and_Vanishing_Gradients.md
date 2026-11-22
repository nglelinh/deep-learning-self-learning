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

**Lựa chọn mặc định: ReLU**
- Nên bắt đầu với ReLU cho hầu hết các bài toán
- Tính toán nhanh và đơn giản
- Hoạt động ổn định trong thực tế

**Khi gặp hiện tượng ReLU chết:**
- Thử chuyển sang Leaky ReLU hoặc ELU
- Kiểm tra lại tốc độ học (*learning rate*) và cách khởi tạo trọng số

**Cho các mô hình hiện đại hoặc quy mô lớn:**
- GELU thường dùng cho kiến trúc Transformer và NLP
- Swish thường dùng cho các mô hình thị giác khi ưu tiên độ chính xác

**Cho mạng rất sâu:**
- Có thể cân nhắc ELU hoặc SELU
- Thường kết hợp thêm các kỹ thuật chuẩn hóa (sẽ học ở các chương sau)

### Cho Lớp Đầu ra

**Phân loại nhị phân:**
- **Sigmoid:** Xuất ra xác suất thuộc lớp dương

**Phân loại đa lớp:**
- **Softmax:** Xuất ra phân phối xác suất trên tất cả các lớp

**Hồi quy:**
- **Tuyến tính (hàm đồng nhất):** Khi đầu ra không bị giới hạn miền giá trị
- **ReLU:** Khi đầu ra phải không âm (ví dụ: giá cả, số đếm)
- **Sigmoid/tanh:** Khi đầu ra nằm trong một khoảng bị chặn

**Phân loại đa nhãn:**
- **Sigmoid:** Xuất xác suất độc lập cho từng nhãn

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

Trong quá trình lan truyền ngược, gradient được tính bằng cách nhân liên tiếp các đạo hàm qua từng lớp:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \cdot \frac{\partial \mathbf{a}^{[L]}}{\partial \mathbf{z}^{[L]}} \cdot \ldots \cdot \frac{\partial \mathbf{z}^{[2]}}{\partial \mathbf{a}^{[1]}} \cdot \frac{\partial \mathbf{a}^{[1]}}{\partial \mathbf{z}^{[1]}} \cdot \frac{\partial \mathbf{z}^{[1]}}{\partial \mathbf{W}^{[1]}}$$

### Hạn chế của Sigmoid và Tanh

- Giá trị đạo hàm cực đại: $$\sigma'(z) = 0.25$$ (với sigmoid tại $$z=0$$), $$\tanh'(z) = 1$$ (với tanh tại $$z=0$$)
- Đạo hàm thực tế thường nhỏ hơn nhiều ($$< 0.25$$ với sigmoid khi $$z \neq 0$$)
- Khi nhân qua nhiều lớp: $$0.25^{10} \approx 9.5 \times 10^{-7}$$ (rất nhỏ)

**Hệ quả:** Gradient suy giảm nhanh về 0, khiến các lớp đầu gần như không cập nhật được trọng số.

### Vai trò của ReLU

- Đạo hàm bằng 1 ở miền dương nên không bị suy giảm qua phép nhân
- Gradient truyền ổn định qua các neuron ReLU đang hoạt động
- Cho phép huấn luyện các mạng sâu hơn đáng kể

### Hiện tượng ReLU Chết

- Nếu giá trị $$z < 0$$ trên mọi mẫu, đạo hàm luôn bằng 0 và neuron ngừng học
- Nguyên nhân thường gặp:
  - Khởi tạo trọng số chưa hợp lý
  - Tốc độ học quá cao
  - Bước cập nhật lớn đẩy trọng số sang miền âm

**Hướng khắc phục:**
- Dùng Leaky ReLU, ELU hoặc các biến thể tương tự
- Áp dụng khởi tạo phù hợp (khởi tạo He cho ReLU)
- Chọn tốc độ học hợp lý
- Sử dụng batch normalization (trình bày ở chương sau)

## Tóm tắt

- **Hàm kích hoạt** bổ sung tính phi tuyến, giúp mạng học các mẫu dữ liệu phức tạp.
- **ReLU** là lựa chọn mặc định cho các lớp ẩn trong hầu hết kiến trúc học sâu.
- **Sigmoid** phù hợp cho lớp đầu ra của bài toán phân loại nhị phân.
- **Softmax** phù hợp cho lớp đầu ra của bài toán phân loại đa lớp.
- **Các hàm kích hoạt hiện đại** (ELU, Swish, GELU) có thể cải thiện hiệu năng trên nhiều kiến trúc.
- **Gradient biến mất** là rào cản lớn khi dùng sigmoid hoặc tanh trong mạng sâu.
- **ReLU** giảm mạnh hiện tượng gradient biến mất nhưng có thể gặp tình trạng ReLU chết.
- Lựa chọn hàm kích hoạt ảnh hưởng trực tiếp tới tốc độ hội tụ và chất lượng mô hình.

Ở bài tiếp theo, ta sẽ tìm hiểu chi tiết quá trình lan truyền xuôi kèm các ví dụ tính toán cụ thể.

## Video tham khảo

- **Xem trước:** [The GELU, SiLU and SwiGLU activation functions, clearly explained!!!](https://www.youtube.com/watch?v=2FaI2Fen1mQ) — StatQuest with Josh Starmer, tiếng Anh. Giải thích rõ các hàm kích hoạt hiện đại (GELU, SiLU, SwiGLU) được dùng phổ biến sau ReLU.
- [Vanishing/Exploding Gradients (C2W1L10)](https://www.youtube.com/watch?v=qhXZsFVxGKo) — DeepLearningAI, tiếng Anh. Phân tích hiện tượng gradient biến mất và bùng nổ — lý do chính thúc đẩy việc cải tiến hàm kích hoạt.
- [Neural Networks Pt. 3: ReLU In Action!!!](https://www.youtube.com/watch?v=68BZ5f7P94E) — StatQuest with Josh Starmer, tiếng Anh. Ôn lại cách hoạt động của ReLU làm cơ sở so sánh với các hàm kích hoạt mới hơn.
