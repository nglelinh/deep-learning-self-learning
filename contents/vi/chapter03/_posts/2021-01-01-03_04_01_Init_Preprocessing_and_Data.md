---
layout: post
title: 03-04-01 Khởi tạo, Tiền xử lý, Batch và Chia tập
chapter: '03'
order: 12
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

Bài này tổng hợp các kỹ thuật nền tảng trong giai đoạn chuẩn bị huấn luyện mạng neuron: khởi tạo trọng số, tiền xử lý dữ liệu, chia mini-batch và phân tách tập dữ liệu.

---

## Khởi tạo Trọng số

Khởi tạo trọng số hợp lý ảnh hưởng trực tiếp tới khả năng hội tụ của mạng. Khởi tạo sai có thể dẫn đến hiện tượng gradient biến mất, bùng nổ hoặc khiến quá trình học diễn ra rất chậm.

### Các Cách Khởi tạo Nên Tránh

#### 1. Khởi tạo toàn 0

```python
W = np.zeros((n_l, n_l_prev))
```

**Hạn chế:** Tất cả neuron trong cùng một lớp tính ra cùng giá trị và nhận gradient giống hệt nhau, nên mạng không thể học được các đặc trưng khác nhau.

#### 2. Khởi tạo cùng một hằng số

```python
W = np.ones((n_l, n_l_prev)) * 0.5
```

**Hạn chế:** Tương tự khởi tạo toàn 0 — không phá vỡ được tính đối xứng giữa các neuron.

### Các Phương pháp Khởi tạo Tốt

#### 1. Giá trị Ngẫu nhiên Nhỏ

```python
W = np.random.randn(n_l, n_l_prev) * 0.01
```

**Ưu điểm:** Phá vỡ đối xứng.  
**Nhược điểm:** Có thể làm tín hiệu và gradient co quá nhỏ trong các mạng nhiều lớp.

#### 2. Khởi tạo Xavier/Glorot

Phù hợp với các hàm kích hoạt **sigmoid** hoặc **tanh**:

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{1}{n^{[l-1]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(1 / n_l_prev)
```

hoặc phiên bản cân bằng cả chiều vào và chiều ra:

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]} + n^{[l]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(2 / (n_l_prev + n_l))
```

**Mục đích:** Giữ cho phương sai của các giá trị kích hoạt ổn định khi đi qua nhiều lớp.

#### 3. Khởi tạo He

Phù hợp với hàm kích hoạt **ReLU** (được sử dụng rộng rãi nhất):

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(2 / n_l_prev)
```

**Vì sao có hệ số 2?** Vì ReLU đưa trung bình một nửa số đầu vào âm về 0, nên phương sai cần tăng gấp đôi để duy trì thang đo kích hoạt.

### Khởi tạo Độ lệch

Khác với trọng số, độ lệch thường có thể khởi tạo bằng 0:

```python
b = np.zeros((n_l, 1))
```

### Hàm Khởi tạo Đầy đủ

```python
def initialize_parameters(layer_dims, initialization_method='he'):
    """
    Khởi tạo tham số mạng
    
    layer_dims: danh sách kích thước lớp [n_x, n_h1, ..., n_y]
    initialization_method: 'zeros', 'random', 'xavier', 'he'
    """
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        if initialization_method == 'zeros':
            parameters[f'W{l}'] = np.zeros((layer_dims[l], layer_dims[l-1]))
        
        elif initialization_method == 'random':
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        
        elif initialization_method == 'xavier':
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1])
        
        elif initialization_method == 'he':
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    return parameters
```

## Tiền xử lý Dữ liệu

### 1. Chuẩn hóa Thang đo Đặc trưng

Đưa các đặc trưng đầu vào về cùng thang đo giúp bề mặt hàm mất mát cân đối hơn và gradient descent hội tụ nhanh hơn.

#### Chuẩn hóa Z-score (*Standardization*)

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

```python
def standardize(X):
    """Chuẩn hóa đặc trưng về mean=0, std=1"""
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    X_norm = (X - mean) / (std + 1e-8)  # Thêm epsilon tránh chia cho không
    return X_norm, mean, std
```

#### Chuẩn hóa Min-Max

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

```python
def min_max_normalize(X):
    """Scale đặc trưng về [0, 1]"""
    x_min = np.min(X, axis=0, keepdims=True)
    x_max = np.max(X, axis=0, keepdims=True)
    X_norm = (X - x_min) / (x_max - x_min + 1e-8)
    return X_norm, x_min, x_max
```

**Cách chọn phương pháp:**
- **Standardization:** Thích hợp khi đặc trưng có phân phối gần chuẩn hoặc xuất hiện điểm ngoại lai.
- **Min-Max:** Thích hợp khi cần ép toàn bộ giá trị vào một khoảng cố định (ví dụ $$[0, 1]$$).

### 2. Xáo trộn Dữ liệu

Cần xáo trộn tập dữ liệu huấn luyện trước mỗi epoch để tránh việc mô hình học theo thứ tự sắp xếp của mẫu dữ liệu.

```python
def shuffle_data(X, Y):
    """Xáo trộn dữ liệu huấn luyện"""
    m = X.shape[1]
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    return X_shuffled, Y_shuffled
```

## Xử lý theo Batch

### Tạo Các Mini-Batch

```python
def create_mini_batches(X, Y, batch_size):
    """
    Tạo danh sách các mini-batch
    
    X: (n_x, m)
    Y: (n_y, m)
    batch_size: kích thước mỗi mini-batch
    
    Returns: danh sách các tuple (X_batch, Y_batch)
    """
    m = X.shape[1]
    mini_batches = []
    
    # Xáo trộn dữ liệu
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    
    # Phân hoạch
    num_complete_batches = m // batch_size
    
    for k in range(num_complete_batches):
        X_batch = X_shuffled[:, k * batch_size:(k + 1) * batch_size]
        Y_batch = Y_shuffled[:, k * batch_size:(k + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))
    
    # Xử lý các mẫu còn lại (nếu m không chia hết cho batch_size)
    if m % batch_size != 0:
        X_batch = X_shuffled[:, num_complete_batches * batch_size:]
        Y_batch = Y_shuffled[:, num_complete_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches
```

## Chia Tập Train / Validation / Test

### Vai trò của Ba Tập dữ liệu

- **Tập huấn luyện (*train*):** Dùng để học tham số mô hình.
- **Tập kiểm định (*validation*):** Dùng để chọn siêu tham số và theo dõi dấu hiệu quá khớp.
- **Tập kiểm tra (*test*):** Dùng để đánh giá khách quan chất lượng cuối cùng (chỉ nên dùng sau khi đã chốt mô hình).

### Tỷ lệ Chia Thường dùng

**Tập dữ liệu nhỏ (dưới 10.000 mẫu):**
- Train: 60%, Validation: 20%, Test: 20%

**Tập dữ liệu trung bình (10.000 – 1.000.000 mẫu):**
- Train: 80%, Validation: 10%, Test: 10%

**Tập dữ liệu lớn (trên 1.000.000 mẫu):**
- Train: 98%, Validation: 1%, Test: 1%

### Cài đặt

```python
def train_val_test_split(X, Y, train_ratio=0.8, val_ratio=0.1):
    """
    Chia dữ liệu thành tập train, validation và test
    """
    m = X.shape[1]
    
    # Xáo trộn trước
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    
    # Tính chỉ số chia
    train_end = int(train_ratio * m)
    val_end = train_end + int(val_ratio * m)
    
    # Chia
    X_train = X_shuffled[:, :train_end]
    Y_train = Y_shuffled[:, :train_end]
    
    X_val = X_shuffled[:, train_end:val_end]
    Y_val = Y_shuffled[:, train_end:val_end]
    
    X_test = X_shuffled[:, val_end:]
    Y_test = Y_shuffled[:, val_end:]
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
```

## Video tham khảo

- **Xem trước:** [L11.6 Xavier Glorot and Kaiming He Initialization](https://www.youtube.com/watch?v=ScWTYHQra5E) — Sebastian Raschka, tiếng Anh. Giải thích cơ sở toán học của quy tắc khởi tạo Xavier và He được trình bày trong bài.
- [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering, tiếng Anh. Bối cảnh tổng thể về khởi tạo trọng số, tiền xử lý dữ liệu và xử lý theo batch.
- [Train, Test, & Validation Sets explained](https://www.youtube.com/watch?v=Zi-0rlM4RDs) — deeplizard, tiếng Anh. Giải thích rõ mục đích của việc chia dữ liệu thành ba tập trước khi huấn luyện.
