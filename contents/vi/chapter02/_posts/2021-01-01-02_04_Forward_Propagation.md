---
layout: post
title: 02-04 Lan truyền Xuôi
chapter: '02'
order: 9
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

Bài này cung cấp hiểu biết toàn diện về lan truyền xuôi (*forward propagation*), quá trình mạng neuron đưa ra dự đoán.

---

## Lan truyền Xuôi là gì?

**Lan truyền xuôi** là quá trình tính đầu ra của mạng neuron khi cho trước một đầu vào. Dữ liệu "chảy xuôi" qua mạng từ lớp đầu vào đến lớp đầu ra, đi qua tất cả các lớp ẩn theo thứ tự.

Đây là pha **suy luận** (*inference*) hoặc **dự đoán** của mạng neuron.

### Trực giác hình học của một bước forward

Với một lớp ẩn ReLU và đầu ra qua $$\sigma$$, forward có thể viết gọn:

$$\mathbf{h} = (W_h \mathbf{x} + b_h)^+,\qquad \hat{\mathbf{y}} = \sigma(W_y \mathbf{h} + b_y)$$

![Forward: hidden ReLU rồi output σ](/deep-learning-self-learning/img/chapter_img/chapter02/mlp_hidden_relu_forward.jpg)
*Hình: Pipeline $$x \to h \to \hat{y}$$ với ReLU ở lớp ẩn và $$\sigma$$ ở đầu ra. (Minh họa từ video nhập môn neural network)*

Mỗi lớp **affine** (nhân $$W$$, cộng $$b$$) biến đổi không gian; ReLU “gấp / cắt” nửa không gian âm — đó là nguồn phi tuyến khiến nhiều lớp mạnh hơn một lớp tuyến tính.

![ReLU gấp không gian đặc trưng](/deep-learning-self-learning/img/chapter_img/chapter02/feature_space_folding_relu.jpg)
*Hình: Sau $$h=(W_h x+b_h)^+$$, đám điểm bị gấp/biến dạng phi tuyến trong không gian feature. (Minh họa từ video nhập môn neural network)*

![Biểu diễn sau biến đổi lớp](/deep-learning-self-learning/img/chapter_img/chapter02/feature_space_after_transform.jpg)
*Hình: Cùng dữ liệu nhìn trong không gian sau các lớp — forward là chuỗi biến đổi hình học, không chỉ “nhân ma trận khô”. (Minh họa từ video nhập môn neural network)*

**Forward = “vẽ” hàm.** Với hồi quy 1D, mỗi lần đẩy một $$x$$ qua mạng sinh một điểm $$(x,\hat{y})$$; lặp trên nhiều $$x$$ sẽ tái tạo đường cong đã học. Neuron ẩn sáng/tối khác nhau tùy vùng input.

![Dự đoán nhiều điểm trên đường cong](/deep-learning-self-learning/img/chapter_img/chapter02/nn_predictions_complex_curve.jpg)
*Hình: Lan truyền xuôi lặp lại “vẽ” hàm xấp xỉ. (Minh họa từ video về bản chất mạng nơ-ron)*

## Bước Lan truyền Xuôi: Từng Bước

Xét một mạng 3 lớp đơn giản:
- **Lớp đầu vào**: $$n^{[0]} = 3$$ đặc trưng
- **Lớp ẩn 1**: $$n^{[1]} = 4$$ neuron với ReLU
- **Lớp ẩn 2**: $$n^{[2]} = 4$$ neuron với ReLU  
- **Lớp đầu ra**: $$n^{[3]} = 1$$ neuron với sigmoid (phân loại nhị phân)

### Lớp 0: Đầu vào

$$\mathbf{a}^{[0]} = \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$$

### Lớp 1: Lớp Ẩn Thứ nhất

**Biến đổi tuyến tính:**

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{a}^{[0]} + \mathbf{b}^{[1]}$$

Trong đó:
- $$\mathbf{W}^{[1]} \in \mathbb{R}^{4 \times 3}$$ (4 neuron, mỗi neuron 3 đầu vào)
- $$\mathbf{b}^{[1]} \in \mathbb{R}^{4}$$ (4 độ lệch)
- $$\mathbf{z}^{[1]} \in \mathbb{R}^{4}$$ (giá trị tiền kích hoạt)

**Kích hoạt:**

$$\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \max(0, \mathbf{z}^{[1]})$$

$$\mathbf{a}^{[1]} \in \mathbb{R}^{4}$$ (kích hoạt/đầu ra của lớp 1)

### Lớp 2: Lớp Ẩn Thứ hai

**Biến đổi tuyến tính:**

$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$

Trong đó:
- $$\mathbf{W}^{[2]} \in \mathbb{R}^{4 \times 4}$$
- $$\mathbf{b}^{[2]} \in \mathbb{R}^{4}$$

**Kích hoạt:**

$$\mathbf{a}^{[2]} = \text{ReLU}(\mathbf{z}^{[2]})$$

### Lớp 3: Lớp Đầu ra

**Biến đổi tuyến tính:**

$$\mathbf{z}^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]}$$

Trong đó:
- $$\mathbf{W}^{[3]} \in \mathbb{R}^{1 \times 4}$$
- $$\mathbf{b}^{[3]} \in \mathbb{R}^{1}$$

**Kích hoạt (đầu ra):**

$$\hat{y} = \mathbf{a}^{[3]} = \sigma(\mathbf{z}^{[3]}) = \frac{1}{1 + e^{-\mathbf{z}^{[3]}}}$$

Kết quả là xác suất dự đoán của lớp dương.

## Lan truyền Xuôi Dạng Vectơ

Vì hiệu quả tính toán, ta xử lý **nhiều mẫu đồng thời** bằng **vectorization**.

### Xử lý theo Batch

Thay vì xử lý từng mẫu một, ta tổ chức $$m$$ mẫu thành một ma trận:

$$\mathbf{X} = \begin{bmatrix} | & | & & | \\ \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)} \\ | & | & & | \end{bmatrix} \in \mathbb{R}^{n^{[0]} \times m}$$

Mỗi cột là một mẫu huấn luyện.

### Tính toán Dạng Vectơ

Với lớp $$l$$:

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

Trong đó:
- $$\mathbf{A}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$$ (mỗi cột là kích hoạt của một mẫu)
- $$\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$$
- $$\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$$
- $$\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$$ (được phát sóng qua tất cả $$m$$ mẫu)

### Broadcasting

Python/NumPy tự động phát sóng $$\mathbf{b}^{[l]}$$ qua mọi mẫu:

$$\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]} \times 1} \rightarrow \mathbb{R}^{n^{[l]} \times m}$$

Mỗi cột của kết quả được cộng cùng một vectơ độ lệch.

## Ví dụ Cụ thể với Số

Ta xét một ví dụ nhỏ với số thực.

### Thiết lập Mạng

- Đầu vào: 2 đặc trưng ($$n^{[0]} = 2$$)
- Lớp ẩn: 3 neuron với ReLU ($$n^{[1]} = 3$$)
- Đầu ra: 1 neuron với sigmoid ($$n^{[2]} = 1$$)
- Kích thước batch: 2 mẫu ($$m = 2$$)

### Tham số

$$\mathbf{W}^{[1]} = \begin{bmatrix} 0.5 & -0.3 \\ 0.2 & 0.8 \\ -0.4 & 0.6 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.3 \end{bmatrix}$$

$$\mathbf{W}^{[2]} = \begin{bmatrix} 1.0 & -0.5 & 0.7 \end{bmatrix}, \quad \mathbf{b}^{[2]} = \begin{bmatrix} 0.5 \end{bmatrix}$$

### Dữ liệu Đầu vào

$$\mathbf{X} = \mathbf{A}^{[0]} = \begin{bmatrix} 1.0 & 0.5 \\ 2.0 & 1.5 \end{bmatrix}$$

Mẫu 1: $$\mathbf{x}^{(1)} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$, Mẫu 2: $$\mathbf{x}^{(2)} = \begin{bmatrix} 0.5 \\ 1.5 \end{bmatrix}$$

### Lan truyền Xuôi: Lớp 1

**Tính $$\mathbf{Z}^{[1]}$$:**

$$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]} \mathbf{A}^{[0]} + \mathbf{b}^{[1]}$$

$$= \begin{bmatrix} 0.5 & -0.3 \\ 0.2 & 0.8 \\ -0.4 & 0.6 \end{bmatrix} \begin{bmatrix} 1.0 & 0.5 \\ 2.0 & 1.5 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \\ 0.3 \end{bmatrix}$$

**Nhân ma trận:**

Cột thứ nhất: $$\begin{bmatrix} 0.5(1.0) + (-0.3)(2.0) \\ 0.2(1.0) + 0.8(2.0) \\ -0.4(1.0) + 0.6(2.0) \end{bmatrix} = \begin{bmatrix} -0.1 \\ 1.8 \\ 0.8 \end{bmatrix}$$

Cột thứ hai: $$\begin{bmatrix} 0.5(0.5) + (-0.3)(1.5) \\ 0.2(0.5) + 0.8(1.5) \\ -0.4(0.5) + 0.6(1.5) \end{bmatrix} = \begin{bmatrix} -0.2 \\ 1.3 \\ 0.7 \end{bmatrix}$$

Sau khi cộng độ lệch:

$$\mathbf{Z}^{[1]} = \begin{bmatrix} -0.1+0.1 & -0.2+0.1 \\ 1.8-0.2 & 1.3-0.2 \\ 0.8+0.3 & 0.7+0.3 \end{bmatrix} = \begin{bmatrix} 0.0 & -0.1 \\ 1.6 & 1.1 \\ 1.1 & 1.0 \end{bmatrix}$$

**Áp dụng kích hoạt ReLU:**

$$\mathbf{A}^{[1]} = \text{ReLU}(\mathbf{Z}^{[1]}) = \begin{bmatrix} 0.0 & 0.0 \\ 1.6 & 1.1 \\ 1.1 & 1.0 \end{bmatrix}$$

### Lan truyền Xuôi: Lớp 2 (Đầu ra)

**Tính $$\mathbf{Z}^{[2]}$$:**

$$\mathbf{Z}^{[2]} = \mathbf{W}^{[2]} \mathbf{A}^{[1]} + \mathbf{b}^{[2]}$$

$$= \begin{bmatrix} 1.0 & -0.5 & 0.7 \end{bmatrix} \begin{bmatrix} 0.0 & 0.0 \\ 1.6 & 1.1 \\ 1.1 & 1.0 \end{bmatrix} + \begin{bmatrix} 0.5 \end{bmatrix}$$

$$= \begin{bmatrix} 1.0(0.0) + (-0.5)(1.6) + 0.7(1.1) + 0.5 & 1.0(0.0) + (-0.5)(1.1) + 0.7(1.0) + 0.5 \end{bmatrix}$$

$$= \begin{bmatrix} 0.47 & 0.65 \end{bmatrix}$$

**Áp dụng kích hoạt sigmoid:**

$$\mathbf{A}^{[2]} = \sigma(\mathbf{Z}^{[2]}) = \begin{bmatrix} \frac{1}{1+e^{-0.47}} & \frac{1}{1+e^{-0.65}} \end{bmatrix} \approx \begin{bmatrix} 0.615 & 0.657 \end{bmatrix}$$

### Dự đoán Cuối cùng

- Mẫu 1: $$\hat{y}^{(1)} = 0.615$$ (xác suất 61.5% thuộc lớp dương)
- Mẫu 2: $$\hat{y}^{(2)} = 0.657$$ (xác suất 65.7% thuộc lớp dương)

## Triển khai bằng Python

### Triển khai Cơ bản

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def forward_propagation(X, parameters):
    """
    Arguments:
    X -- dữ liệu đầu vào có dạng (n_x, m)
    parameters -- từ điển Python chứa W1, b1, W2, b2, W3, b3, ...
    
    Returns:
    AL -- giá trị hậu kích hoạt cuối (dự đoán)
    caches -- danh sách cache chứa (A_prev, W, b, Z) cho mỗi lớp
    """
    caches = []
    A = X
    L = len(parameters) // 2  # số lớp
    
    # Lan truyền xuôi qua lớp ẩn (kích hoạt ReLU)
    for l in range(1, L):
        A_prev = A
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        
        Z = np.dot(W, A_prev) + b
        A = relu(Z)
        
        cache = (A_prev, W, b, Z)
        caches.append(cache)
    
    # Lớp đầu ra (kích hoạt sigmoid)
    A_prev = A
    W = parameters[f'W{L}']
    b = parameters[f'b{L}']
    
    Z = np.dot(W, A_prev) + b
    AL = sigmoid(Z)
    
    cache = (A_prev, W, b, Z)
    caches.append(cache)
    
    return AL, caches

# Ví dụ sử dụng
X = np.array([[1.0, 0.5],
              [2.0, 1.5]])

parameters = {
    'W1': np.array([[0.5, -0.3],
                    [0.2, 0.8],
                    [-0.4, 0.6]]),
    'b1': np.array([[0.1], [-0.2], [0.3]]),
    'W2': np.array([[1.0, -0.5, 0.7]]),
    'b2': np.array([[0.5]])
}

predictions, caches = forward_propagation(X, parameters)
print("Predictions:", predictions)
# Output: Predictions: [[0.615 0.657]]
```

### Triển khai Hướng Đối tượng

```python
class NeuralNetwork:
    def __init__(self, layer_dims):
        """
        Arguments:
        layer_dims -- danh sách chứa chiều của mỗi lớp
                     Ví dụ: [2, 3, 1] nghĩa là 2 đầu vào, 3 ẩn, 1 đầu ra
        """
        self.parameters = self.initialize_parameters(layer_dims)
        self.L = len(layer_dims) - 1
    
    def initialize_parameters(self, layer_dims):
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)
        
        for l in range(1, L):
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
        return parameters
    
    def forward(self, X):
        """Lan truyền xuôi"""
        A = X
        caches = []
        
        # Lớp ẩn với ReLU
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(W, A_prev) + b
            A = np.maximum(0, Z)  # ReLU
            
            caches.append((A_prev, W, b, Z, A))
        
        # Lớp đầu ra với sigmoid
        A_prev = A
        W = self.parameters[f'W{self.L}']
        b = self.parameters[f'b{self.L}']
        
        Z = np.dot(W, A_prev) + b
        A = 1 / (1 + np.exp(-Z))  # Sigmoid
        
        caches.append((A_prev, W, b, Z, A))
        
        return A, caches
    
    def predict(self, X):
        """Đưa ra dự đoán (0 hoặc 1)"""
        A, _ = self.forward(X)
        return (A > 0.5).astype(int)

# Sử dụng
nn = NeuralNetwork([2, 3, 1])
X = np.array([[1.0, 0.5], [2.0, 1.5]])
predictions, caches = nn.forward(X)
print("Predictions:", predictions)
```

## Các Vấn đề Thường gặp và Gỡ lỗi

### 1. Không khớp Chiều

**Vấn đề**: Nhân ma trận thất bại do chiều không tương thích.

**Giải pháp**: 
- Kiểm tra $$\mathbf{W}^{[l]}$$ có dạng $$(n^{[l]}, n^{[l-1]})$$
- Kiểm tra $$\mathbf{A}^{[l-1]}$$ có dạng $$(n^{[l-1]}, m)$$
- Dùng print hoặc debugger để xác minh dạng

```python
print(f"Layer {l}:")
print(f"  W shape: {W.shape}")
print(f"  A_prev shape: {A_prev.shape}")
print(f"  b shape: {b.shape}")
print(f"  Z shape: {Z.shape}")
```

### 2. Bất ổn Số học

**Vấn đề**: Tràn trên hoặc tràn dưới với hàm mũ (đặc biệt sigmoid/softmax).

**Giải pháp**: Dùng các mẹo ổn định số:

```python
# Không ổn định
def sigmoid_unstable(z):
    return 1 / (1 + np.exp(-z))

# Phiên bản ổn định
def sigmoid_stable(z):
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
```

### 3. Broadcasting Sai

**Vấn đề**: Độ lệch không được phát sóng đúng.

**Giải pháp**: Đảm bảo độ lệch có dạng $$(n^{[l]}, 1)$$ chứ không phải $$(n^{[l]},)$$

```python
# Đúng
b = np.zeros((n_l, 1))  # Dạng (n_l, 1)

# Sai (có thể gây vấn đề)
b = np.zeros(n_l)  # Dạng (n_l,)
```

## Độ phức tạp của Lan truyền Xuôi

### Độ phức tạp Thời gian

Với mạng có $$L$$ lớp và $$n$$ neuron mỗi lớp:

$$O(L \cdot n^2 \cdot m)$$

trong đó $$m$$ là kích thước batch.

**Phân tích:**
- Mỗi lớp: $$O(n^2 \cdot m)$$ cho nhân ma trận $$\mathbf{W}^{[l]} \mathbf{A}^{[l-1]}$$
- Tổng cộng $$L$$ lớp

### Độ phức tạp Không gian

$$O(L \cdot n \cdot m)$$

Cần lưu kích hoạt của mỗi lớp (cần cho lan truyền ngược).

## Tóm tắt

- **Lan truyền xuôi** tính dự đoán bằng cách truyền đầu vào qua mạng
- Mỗi lớp thực hiện: **biến đổi tuyến tính** → **hàm kích hoạt**
- **Vectorization** cho phép xử lý batch nhiều mẫu hiệu quả
- Quá trình là: $$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}, \quad \mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$
- **Lưu cache** các giá trị trung gian là thiết yếu cho lan truyền ngược hiệu quả
- Xử lý đúng **chiều** và **ổn định số** là then chốt
- Lan truyền xuôi **hiệu quả tính toán** ($$O(L \cdot n^2 \cdot m)$$)

Khi đã hiểu cách mạng đưa ra dự đoán, ta cần học cách huấn luyện chúng. Ở chương tiếp theo, ta sẽ trình bày **lan truyền ngược** và **gradient descent**, các thuật toán cho phép mạng neuron học từ dữ liệu.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Trực giác mạng nơ-ron (Perceptron → Deep Learning)](https://www.facebook.com/reel/793200140509765)
- [Mạng nơ-ron là bộ xấp xỉ hàm](https://www.facebook.com/reel/720970114372332)
