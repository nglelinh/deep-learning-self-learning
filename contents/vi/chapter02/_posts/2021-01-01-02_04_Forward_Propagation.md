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

Nếu lan truyền ngược là quá trình "học từ sai lầm", thì **lan truyền xuôi (*forward propagation*)** chính là quá trình mạng nơ-ron "suy nghĩ và đưa ra câu trả lời". Viết vòng lặp `for` lồng nhau cho từng nơ-ron và từng mẫu dữ liệu bằng Python thuần có thể mất hàng giờ đồng hồ, nhưng chỉ cần viết lại đúng phép tính đó dưới dạng **nhân ma trận theo batch (vectorization)**, GPU có thể xử lý xong hàng chục nghìn ảnh trong vài mili-giây. Bài học này dẫn bạn đi từng bước toán học của lan truyền xuôi, từ một mẫu đơn lẻ đến mini-batch đa chiều và trực giác hình học đằng sau mỗi tầng biến đổi.

## Kiến thức cần nhớ: Bốn trụ cột của Lan truyền Xuôi

1. **Chuỗi Biến đổi Đệ quy qua từng Tầng**:
   Bắt đầu từ $$\mathbf{A}^{(0)} = \mathbf{X}$$, mỗi tầng $$\ell = 1, \dots, L$$ thực hiện đúng hai bước: biến đổi tuyến tính $$\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)}$$ (có broadcasting cho $$\mathbf{b}^{(\ell)}$$) và kích hoạt phi tuyến $$\mathbf{A}^{(\ell)} = \sigma_\ell(\mathbf{Z}^{(\ell)})$$.

2. **Kiểm soát Kích thước Ma trận (Shape Tracking)**:
   Với mini-batch $$B$$ mẫu, nếu $$\mathbf{A}^{(\ell-1)} \in \mathbb{R}^{B \times n_{\ell-1}}$$ và $$\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_{\ell-1} \times n_\ell}$$, thì $$\mathbf{Z}^{(\ell)}, \mathbf{A}^{(\ell)} \in \mathbb{R}^{B \times n_\ell}$$. Theo dõi kích thước ở từng dòng là kỹ năng phòng tránh lỗi `RuntimeError: mat1 and mat2 shapes cannot be multiplied` tốt nhất.

3. **Lưu giữ Bộ nhớ Đệm (Activation Caching)**:
   Trong chế độ huấn luyện, lan truyền xuôi không thể vứt bỏ các giá trị trung gian $$\mathbf{Z}^{(\ell)}, \mathbf{A}^{(\ell)}$$—nó phải giữ lại trong VRAM để phục vụ tính đạo hàm lúc lan truyền ngược (trừ khi bật `torch.no_grad()` lúc suy luận).

4. **Trực giác Hình học của Chuỗi Tầng**:
   Mỗi tầng không di chuyển điểm dữ liệu một cách hỗn loạn mà liên tục xoay, kéo giãn và gấp không gian để đến tầng cuối cùng, các lớp khác nhau nằm gọn trên hai phía đối diện của một siêu phẳng.

### Giải thích: Tính tay Lan truyền Xuôi cho một Mini-Batch 2 mẫu qua mạng 2–2–1

Xét mini-batch gồm 2 mẫu $$\mathbf{X} = \begin{bmatrix} 1 & 0 \\ -1 & 2 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$ đi qua mạng có:
$$\mathbf{W}^{(1)} = \begin{bmatrix} 1 & -1 \\ 2 & 1 \end{bmatrix}, \;\mathbf{b}^{(1)} = \begin{bmatrix} 0 & 1 \end{bmatrix}, \;\mathbf{W}^{(2)} = \begin{bmatrix} 1 \\ -2 \end{bmatrix}, \; b^{(2)} = 0.5.$$

1. **Tầng ẩn 1 — Phép nhân ma trận + Broadcasting bias**:
   $$\mathbf{Z}^{(1)} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)} = \begin{bmatrix} 1(1)+0(2) & 1(-1)+0(1) \\ -1(1)+2(2) & -1(-1)+2(1) \end{bmatrix} + \begin{bmatrix} 0 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ 3 & 3 \end{bmatrix} + \begin{bmatrix} 0 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 3 & 4 \end{bmatrix}.$$
2. **Tầng ẩn 1 — Áp dụng ReLU từng phần tử**:
   Vì mọi phần tử của $$\mathbf{Z}^{(1)}$$ đều $$\ge 0$$, ta có $$\mathbf{A}^{(1)} = \begin{bmatrix} 1 & 0 \\ 3 & 4 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$.
3. **Tầng đầu ra — Biến đổi tuyến tính + Sigmoid**:
   $$\mathbf{Z}^{(2)} = \mathbf{A}^{(1)}\mathbf{W}^{(2)} + b^{(2)} = \begin{bmatrix} 1(1) + 0(-2) + 0.5 \\ 3(1) + 4(-2) + 0.5 \end{bmatrix} = \begin{bmatrix} 1.5 \\ -4.5 \end{bmatrix}.$$
   Qua Sigmoid: $$\hat{\mathbf{y}} = \begin{bmatrix} \sigma(1.5) \\ \sigma(-4.5) \end{bmatrix} \approx \begin{bmatrix} 0.8176 \\ 0.0110 \end{bmatrix}.$$
Chỉ với hai phép nhân ma trận song song, cả hai mẫu đã được dự đoán đồng thời mà không cần bất kỳ vòng lặp `for` nào!

Bài này trình bày chi tiết về lan truyền xuôi (*forward propagation*) — quá trình mạng neuron tính toán và đưa ra dự đoán từ dữ liệu đầu vào.

---

## Lan truyền Xuôi là gì?

**Lan truyền xuôi** là quá trình tính toán đầu ra của mạng neuron ứng với một đầu vào cho trước. Dữ liệu đi theo chiều thuận từ lớp đầu vào, lần lượt qua tất cả các lớp ẩn cho đến lớp đầu ra.

Đây chính là giai đoạn **suy luận** (*inference*) hoặc **dự đoán** của mạng neuron.

### Trực giác hình học của một bước lan truyền xuôi

Với một lớp ẩn dùng ReLU và lớp đầu ra dùng hàm kích hoạt $$\sigma$$, quá trình lan truyền xuôi có thể viết gọn:

$$\mathbf{h} = (W_h \mathbf{x} + b_h)^+,\qquad \hat{\mathbf{y}} = \sigma(W_y \mathbf{h} + b_y)$$

![Forward: hidden ReLU rồi output σ](/deep-learning-self-learning/img/chapter_img/chapter02/mlp_hidden_relu_forward.jpg)
*Hình: Luồng tính toán $$x \to h \to \hat{y}$$ với ReLU ở lớp ẩn và $$\sigma$$ ở lớp đầu ra. (Minh họa từ video nhập môn mạng neuron)*

Mỗi phép biến đổi **affine** (nhân $$W$$ rồi cộng $$b$$) thực hiện xoay, kéo giãn hoặc tịnh tiến không gian; sau đó ReLU "gấp" nửa không gian âm về 0. Chính thao tác phi tuyến này giúp mạng nhiều lớp biểu diễn được các ranh giới phức tạp hơn một phép biến đổi tuyến tính.

![ReLU gấp không gian đặc trưng](/deep-learning-self-learning/img/chapter_img/chapter02/feature_space_folding_relu.jpg)
*Hình: Sau phép biến đổi $$h=(W_h x+b_h)^+$$, tập điểm dữ liệu được gấp và biến dạng phi tuyến trong không gian đặc trưng. (Minh họa từ video nhập môn mạng neuron)*

![Biểu diễn sau biến đổi lớp](/deep-learning-self-learning/img/chapter_img/chapter02/feature_space_after_transform.jpg)
*Hình: Dữ liệu sau khi đi qua các lớp — lan truyền xuôi có thể được hình dung như chuỗi các phép biến đổi hình học liên tiếp. (Minh họa từ video nhập môn mạng neuron)*

**Lan truyền xuôi như quá trình tái tạo hàm số:** Trong bài toán hồi quy 1D, mỗi lần đưa một giá trị $$x$$ qua mạng ta thu được một điểm $$(x, \hat{y})$$; lặp lại trên nhiều giá trị $$x$$ sẽ vẽ ra đường cong mà mạng đã học. Các neuron ẩn sẽ kích hoạt mạnh hoặc yếu khác nhau tùy thuộc vào vùng giá trị của đầu vào.

![Dự đoán nhiều điểm trên đường cong](/deep-learning-self-learning/img/chapter_img/chapter02/nn_predictions_complex_curve.jpg)
*Hình: Thực hiện lan truyền xuôi trên nhiều điểm $$x$$ giúp tái tạo đường cong xấp xỉ. (Minh họa từ video về bản chất mạng neuron)*

## Quy trình Lan truyền Xuôi Từng Bước

Xét một mạng 3 lớp đơn giản:
- **Lớp đầu vào:** $$n^{[0]} = 3$$ đặc trưng
- **Lớp ẩn 1:** $$n^{[1]} = 4$$ neuron với ReLU
- **Lớp ẩn 2:** $$n^{[2]} = 4$$ neuron với ReLU  
- **Lớp đầu ra:** $$n^{[3]} = 1$$ neuron với sigmoid (phân loại nhị phân)

### Lớp 0: Đầu vào

$$\mathbf{a}^{[0]} = \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$$

### Lớp 1: Lớp Ẩn Thứ nhất

**Biến đổi tuyến tính:**

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{a}^{[0]} + \mathbf{b}^{[1]}$$

Trong đó:
- $$\mathbf{W}^{[1]} \in \mathbb{R}^{4 \times 3}$$ (4 neuron, mỗi neuron nhận 3 đầu vào)
- $$\mathbf{b}^{[1]} \in \mathbb{R}^{4}$$ (4 hệ số độ lệch)
- $$\mathbf{z}^{[1]} \in \mathbb{R}^{4}$$ (các giá trị tiền kích hoạt)

**Kích hoạt:**

$$\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \max(0, \mathbf{z}^{[1]})$$

$$\mathbf{a}^{[1]} \in \mathbb{R}^{4}$$ (vector đầu ra của lớp 1)

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

Kết quả thu được là xác suất dự đoán mẫu thuộc lớp dương.

## Lan truyền Xuôi Dạng Ma trận (Vector hóa)

Để tận dụng phần cứng và tăng tốc tính toán, ta xử lý **nhiều mẫu đồng thời** thông qua kỹ thuật **vector hóa** (*vectorization*).

### Xử lý theo Batch

Thay vì đưa từng mẫu một vào mạng, ta xếp $$m$$ mẫu thành các cột của một ma trận:

$$\mathbf{X} = \begin{bmatrix} | & | & & | \\ \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)} \\ | & | & & | \end{bmatrix} \in \mathbb{R}^{n^{[0]} \times m}$$

Mỗi cột tương ứng với một mẫu dữ liệu.

### Phép tính theo Batch

Tại lớp $$l$$:

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

Trong đó:
- $$\mathbf{A}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$$ (mỗi cột là kích hoạt của một mẫu)
- $$\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$$
- $$\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$$
- $$\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$$ (được mở rộng chiều tự động qua tất cả $$m$$ mẫu)

### Cơ chế Broadcasting

Python/NumPy tự động thực hiện *broadcasting* cho $$\mathbf{b}^{[l]}$$ trên toàn bộ các cột của batch:

$$\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]} \times 1} \rightarrow \mathbb{R}^{n^{[l]} \times m}$$

Nhờ đó, mọi cột của ma trận tích đều được cộng cùng một vector độ lệch.

## Ví dụ Tính toán Cụ thể

Hãy theo dõi một ví dụ nhỏ với các con số cụ thể.

### Cấu hình Mạng

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

**Kết quả nhân ma trận:**

Cột thứ nhất: $$\begin{bmatrix} 0.5(1.0) + (-0.3)(2.0) \\ 0.2(1.0) + 0.8(2.0) \\ -0.4(1.0) + 0.6(2.0) \end{bmatrix} = \begin{bmatrix} -0.1 \\ 1.8 \\ 0.8 \end{bmatrix}$$

Cột thứ hai: $$\begin{bmatrix} 0.5(0.5) + (-0.3)(1.5) \\ 0.2(0.5) + 0.8(1.5) \\ -0.4(0.5) + 0.6(1.5) \end{bmatrix} = \begin{bmatrix} -0.2 \\ 1.3 \\ 0.7 \end{bmatrix}$$

Sau khi cộng vector độ lệch:

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

## Các Lỗi Thường gặp và Cách Gỡ lỗi

### 1. Không khớp Kích thước Ma trận

**Vấn đề:** Phép nhân ma trận báo lỗi do kích thước không tương thích.

**Cách xử lý:**
- Kiểm tra $$\mathbf{W}^{[l]}$$ có kích thước $$(n^{[l]}, n^{[l-1]})$$
- Kiểm tra $$\mathbf{A}^{[l-1]}$$ có kích thước $$(n^{[l-1]}, m)$$
- In kích thước từng tensor hoặc dùng debugger để đối chiếu

```python
print(f"Layer {l}:")
print(f"  W shape: {W.shape}")
print(f"  A_prev shape: {A_prev.shape}")
print(f"  b shape: {b.shape}")
print(f"  Z shape: {Z.shape}")
```

### 2. Bất ổn Số học

**Vấn đề:** Hiện tượng tràn số trên (*overflow*) hoặc tràn số dưới (*underflow*) khi tính hàm mũ, đặc biệt trong sigmoid và softmax.

**Cách xử lý:** Dùng dạng công thức ổn định số:

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

### 3. Lỗi Broadcasting ở Độ lệch

**Vấn đề:** Vector độ lệch bị cộng sai chiều do thiếu trục cột.

**Cách xử lý:** Luôn giữ kích thước của độ lệch là $$(n^{[l]}, 1)$$ thay vì mảng 1 chiều $$(n^{[l]},)$$.

```python
# Đúng
b = np.zeros((n_l, 1))  # Dạng (n_l, 1)

# Sai (có thể gây vấn đề)
b = np.zeros(n_l)  # Dạng (n_l,)
```

## Độ phức tạp của Lan truyền Xuôi

### Độ phức tạp Thời gian

Với mạng gồm $$L$$ lớp và mỗi lớp có $$n$$ neuron:

$$O(L \cdot n^2 \cdot m)$$

trong đó $$m$$ là kích thước batch.

**Giải thích:**
- Mỗi lớp tốn $$O(n^2 \cdot m)$$ cho phép nhân ma trận $$\mathbf{W}^{[l]} \mathbf{A}^{[l-1]}$$
- Nhân với tổng số $$L$$ lớp

### Độ phức tạp Không gian

$$O(L \cdot n \cdot m)$$

Bộ nhớ này dùng để lưu lại các giá trị kích hoạt trung gian ở từng lớp phục vụ cho quá trình lan truyền ngược.

## Tóm tắt

- **Lan truyền xuôi** tính đầu ra dự đoán bằng cách đưa dữ liệu đi tuần tự qua các lớp của mạng.
- Mỗi lớp thực hiện hai bước: **biến đổi tuyến tính** rồi **áp dụng hàm kích hoạt**.
- **Vector hóa** giúp tính toán song song trên cả batch dữ liệu với hiệu năng cao.
- Biểu thức tổng quát: $$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}, \quad \mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$.
- **Lưu đệm (*cache*)** các giá trị trung gian là cần thiết để lan truyền ngược tính gradient nhanh chóng.
- Kiểm soát đúng **kích thước ma trận** và **ổn định số học** giúp tránh nhiều lỗi cài đặt phổ biến.
- Độ phức tạp thời gian của lan truyền xuôi là $$O(L \cdot n^2 \cdot m)$$.

Sau khi nắm cách mạng tính toán dự đoán, chương tiếp theo sẽ trình bày **lan truyền ngược** và **gradient descent** — các thuật toán giúp mạng học tham số từ dữ liệu.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video của Machine Learning Thực Chiến:

- [Trực giác mạng nơ-ron (Perceptron → Deep Learning)](https://www.facebook.com/reel/793200140509765)
- [Mạng nơ-ron là bộ xấp xỉ hàm](https://www.facebook.com/reel/720970114372332)

## Video tham khảo

- **Xem trước:** [Forward Propagation in a Deep Network (C1W4L02)](https://www.youtube.com/watch?v=a8i2eJin0lY) — DeepLearningAI, tiếng Anh. Trình bày từng phương trình lan truyền xuôi qua từng lớp, khớp trực tiếp với nội dung bài học.
- [Forward and Backward Propagation (C1W4L06)](https://www.youtube.com/watch?v=qzPQ8cEsVK8) — DeepLearningAI, tiếng Anh. Đặt lượt xuôi bên cạnh lượt ngược để thấy cách sử dụng các giá trị lưu đệm vừa tính.
- [Xây dựng mạng nơ-ron nhân tạo từ con số 0 | Writing a Neural Network from scratch](https://www.youtube.com/watch?v=VBviWbusHd8) — Học Lập Trình cùng Phát, tiếng Việt. Hướng dẫn cài đặt lan truyền xuôi từ đầu bằng tiếng Việt.
