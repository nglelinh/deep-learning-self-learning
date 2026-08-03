---
layout: post
title: 02-02-02 Độ sâu, Độ rộng và Mẫu Thiết kế
chapter: '02'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

## Độ sâu và Độ rộng của Mạng

### Độ rộng

**Độ rộng** (*width*) của một lớp là số neuron trong lớp đó.

- **Mạng rộng hơn**: Nhiều neuron hơn trên mỗi lớp
  - Khả năng lớn hơn để học các mẫu phức tạp trong một lớp đơn
  - Nhiều tham số hơn (có thể dẫn đến quá khớp — *overfitting*)
  - Chi phí tính toán lớn hơn trên mỗi lớp

### Độ sâu

**Độ sâu** (*depth*) của mạng là số lớp.

- **Mạng sâu hơn**: Nhiều lớp hơn
  - Có thể học biểu diễn phân cấp (*hierarchical representations*)
  - Biểu đạt mạnh hơn (biểu diễn được các hàm phức tạp hơn)
  - Có thể khó huấn luyện hơn (gradient biến mất/bùng nổ — *vanishing/exploding gradients*)
  - Thuật ngữ "học sâu" xuất phát từ việc dùng mạng sâu

### Định lý Xấp xỉ Phổ quát

**Định lý**: Một mạng neuron feedforward với:
- Một lớp ẩn đơn
- Số neuron hữu hạn
- Hàm kích hoạt thích hợp (ví dụ: sigmoid, ReLU)

có thể xấp xỉ bất kỳ hàm liên tục nào trên một tập compact của $$\mathbb{R}^n$$ với độ chính xác tùy ý.

**Lưu ý quan trọng:**
- Đây là **định lý tồn tại**, không phải hướng dẫn thực tế
- Định lý không chỉ ra cần bao nhiêu neuron (có thể theo cấp số nhân)
- Mạng sâu hơn thường xấp xỉ hàm với ít tham số hơn nhiều
- Mạng sâu hơn có xu hướng học đặc trưng phân cấp một cách tự nhiên

**Hình dung độ sâu/độ rộng qua bài xấp xỉ 1D:** cùng dữ liệu gợn sóng — dung lượng thấp → khớp kém; tăng hidden + huấn luyện → khớp tốt (xem chi tiết kèm ảnh ở bài Lý thuyết Kiến trúc).

![Khớp kém khi dung lượng thấp](/deep-learning-self-learning/img/chapter_img/chapter02/nn_shallow_poor_fit.jpg)
*Hình: Mạng quá “mỏng”/nông không đủ sức xấp xỉ. (Minh họa từ video về bản chất mạng nơ-ron)*

![Khớp tốt khi đủ dung lượng](/deep-learning-self-learning/img/chapter_img/chapter02/nn_deeper_good_fit.jpg)
*Hình: Đủ độ sâu/rộng + train → xấp xỉ hàm gần hoàn hảo. (Minh họa từ video về bản chất mạng nơ-ron)*

## Các Mẫu Thiết kế Phổ biến

### Độ rộng Giảm dần

Một mẫu phổ biến là giảm dần độ rộng lớp:

```
Input (784) → 512 → 256 → 128 → 64 → Output (10)
```

**Lý do**: Nén dần thông tin thành các trừu tượng mức cao hơn.

### Kiến trúc Đồng hồ cát / Nghẽn (*Hourglass/Bottleneck*)

Giảm rồi tăng độ rộng:

```
Input (784) → 256 → 64 → 256 → Output (784)
```

**Trường hợp dùng**: Autoencoder cho giảm chiều và tái tạo.

### Độ rộng Đồng nhất

Giữ mọi lớp ẩn cùng kích thước:

```
Input (784) → 256 → 256 → 256 → Output (10)
```

**Lý do**: Đơn giản và dễ tinh chỉnh siêu tham số hơn.

## Hàm Kích hoạt theo Lớp

Các lớp khác nhau có thể dùng các hàm kích hoạt khác nhau:

**Cấu hình điển hình:**
- **Lớp ẩn**: ReLU (hoặc biến thể như Leaky ReLU, ELU)
  - Hiệu quả tính toán
  - Giảm nhẹ gradient biến mất
  
- **Lớp đầu ra**: Phụ thuộc tác vụ
  - Phân loại nhị phân: Sigmoid
  - Phân loại đa lớp: Softmax
  - Hồi quy: Tuyến tính (hàm đồng nhất)

## Lớp Liên kết Đầy đủ so với Các Kiến trúc Khác

### Lớp Liên kết Đầy đủ (*Fully Connected / Dense*)

Mọi neuron ở lớp $$l$$ được kết nối với mọi neuron ở lớp $$l-1$$.

**Ưu điểm:**
- Linh hoạt tối đa
- Có thể học bất kỳ mẫu nào (với đủ neuron)

**Nhược điểm:**
- Nhiều tham số ($$n^{[l]} \times n^{[l-1]}$$)
- Không có giả định sẵn về cấu trúc đầu vào
- Không hiệu quả với dữ liệu có cấu trúc (ảnh, chuỗi)

### Các Kiến trúc Chuyên biệt

Với các kiểu dữ liệu cụ thể, kiến trúc chuyên biệt hiệu quả hơn:

- **Lớp tích chập** (*convolutional layers*): Cho ảnh (cấu trúc không gian)
- **Lớp hồi quy** (*recurrent layers*): Cho chuỗi (cấu trúc thời gian)
- **Cơ chế attention**: Xử lý phụ thuộc tầm xa

Các kiến trúc này sẽ được trình bày ở các chương sau.

## Biểu diễn Mạng

### Biểu diễn Đồ thị

Mạng thường được minh họa dưới dạng đồ thị có hướng không chu trình (DAG):

```
      Input Layer    Hidden Layer 1   Hidden Layer 2   Output Layer
         (3)             (4)              (4)              (2)
    
    x₁  ○────────────────●──────────────────●──────────────────●  ŷ₁
                        ╱│╲              ╱│╲              ╱│
    x₂  ○──────────────●─●─●────────────●─●─●────────────●─●
                        ╲│╱              ╲│╱              ╲│
    x₃  ○────────────────●──────────────────●──────────────────●  ŷ₂
```

### Biểu diễn Ma trận

Vì hiệu quả tính toán, ta biểu diễn các phép toán dưới dạng nhân ma trận:

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

trong đó:
- $$\mathbf{A}^{[l-1]}$$: ma trận kích hoạt (mỗi cột là một mẫu)
- $$\mathbf{W}^{[l]}$$: ma trận trọng số
- $$\mathbf{b}^{[l]}$$: vectơ độ lệch (được phát sóng — *broadcast* — qua các mẫu)

## Triển khai Thực tế

### Ví dụ: Mạng Neuron Đơn giản bằng Python

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: danh sách kích thước các lớp, gồm đầu vào và đầu ra
        Ví dụ: [784, 128, 64, 10] cho MNIST
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        
        # Khởi tạo trọng số và độ lệch
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # Khởi tạo He cho mạng dùng ReLU
            w = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2.0 / layer_sizes[i-1])
            b = np.zeros((layer_sizes[i], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward(self, x):
        """
        x: đầu vào có dạng (input_size, num_examples)
        Trả về: đầu ra có dạng (output_size, num_examples)
        """
        a = x
        activations = [x]
        zs = []
        
        # Lan truyền xuôi qua các lớp ẩn
        for i in range(self.num_layers - 2):
            z = self.weights[i] @ a + self.biases[i]
            a = self.relu(z)
            zs.append(z)
            activations.append(a)
        
        # Lớp đầu ra (softmax)
        z = self.weights[-1] @ a + self.biases[-1]
        a = self.softmax(z)
        zs.append(z)
        activations.append(a)
        
        return a, activations, zs
    
    def predict(self, x):
        """Trả về dự đoán lớp"""
        output, _, _ = self.forward(x)
        return np.argmax(output, axis=0)

# Ví dụ sử dụng
network = NeuralNetwork([784, 128, 64, 10])
x = np.random.randn(784, 5)  # 5 mẫu
output, _, _ = network.forward(x)
print(f"Output shape: {output.shape}")  # (10, 5)
print(f"Predictions: {network.predict(x)}")
```

## Các Cân nhắc Thiết kế

### Số Lớp

- **1–2 lớp ẩn**: Bài toán đơn giản, tập dữ liệu nhỏ
- **3–5 lớp ẩn**: Độ phức tạp vừa phải
- **5+ lớp ẩn**: Bài toán phức tạp, tập dữ liệu lớn, học "sâu"

### Số Neuron trên mỗi Lớp

Quy tắc thực nghiệm:
- Bắt đầu với kích thước lớp nằm giữa kích thước đầu vào và đầu ra
- Kích thước phổ biến: 32, 64, 128, 256, 512
- Nhiều neuron hơn = dung lượng lớn hơn nhưng rủi ro quá khớp cao hơn
- Dùng hiệu năng trên tập kiểm định để định hướng lựa chọn

### Tìm kiếm Kiến trúc

Việc tìm kiến trúc tối ưu thường thực hiện qua:
- **Thử nghiệm thủ công**: Thử các cấu hình khác nhau
- **Grid search**: Thử hệ thống các tổ hợp
- **Random search**: Thường hiệu quả hơn grid search
- **Neural Architecture Search (NAS)**: Phương pháp tự động (chủ đề nâng cao)

## Tóm tắt

- **Mạng neuron** gồm các lớp neuron được tổ chức thành lớp đầu vào, lớp ẩn và lớp đầu ra
- **Mạng feedforward** (MLP) là kiến trúc đơn giản nhất, trong đó thông tin chảy theo một chiều
- **Lan truyền xuôi** tính đầu ra bằng cách truyền đầu vào qua các lớp liên tiếp
- **Độ sâu mạng** (số lớp) và **độ rộng** (số neuron trên lớp) quyết định dung lượng
- **Định lý Xấp xỉ Phổ quát** cho thấy mạng có thể xấp xỉ bất kỳ hàm nào, nhưng không đảm bảo hiệu quả
- **Lớp liên kết đầy đủ** kết nối mọi neuron với mọi neuron ở các lớp kề
- **Thiết kế kiến trúc hợp lý** phụ thuộc bài toán, dữ liệu và tài nguyên tính toán

Ở bài tiếp theo, ta sẽ xét hàm kích hoạt chi tiết hơn và vai trò then chốt của chúng trong quá trình học.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Mạng nơ-ron là bộ xấp xỉ hàm](https://www.facebook.com/reel/720970114372332)
