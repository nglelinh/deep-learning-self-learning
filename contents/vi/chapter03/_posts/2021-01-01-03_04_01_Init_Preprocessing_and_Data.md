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

Bài này trình bày các kỹ thuật thực tế và thực hành tốt nhất để huấn luyện mạng neuron hiệu quả.

---

## Khởi tạo Trọng số

Khởi tạo đúng là then chốt cho huấn luyện thành công. Khởi tạo kém có thể dẫn đến gradient biến mất/bùng nổ hoặc hội tụ chậm.

### Phương pháp Khởi tạo Xấu

#### 1. Toàn không

```python
W = np.zeros((n_l, n_l_prev))
```

**Vấn đề**: Mọi neuron tính cùng đầu ra và nhận cùng gradient → Không học!

#### 2. Toàn cùng giá trị

```python
W = np.ones((n_l, n_l_prev)) * 0.5
```

**Vấn đề**: Giống toàn không — phá vỡ đối xứng thất bại.

### Phương pháp Khởi tạo Tốt

#### 1. Giá trị Ngẫu nhiên Nhỏ

```python
W = np.random.randn(n_l, n_l_prev) * 0.01
```

**Ưu**: Phá đối xứng  
**Nhược**: Có thể quá nhỏ với mạng sâu

#### 2. Khởi tạo Xavier/Glorot

Cho kích hoạt **sigmoid** hoặc **tanh**:

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{1}{n^{[l-1]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(1 / n_l_prev)
```

hoặc

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]} + n^{[l]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(2 / (n_l_prev + n_l))
```

**Lý do**: Duy trì phương sai kích hoạt qua các lớp.

#### 3. Khởi tạo He

Cho kích hoạt **ReLU** (phổ biến nhất):

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(2 / n_l_prev)
```

**Vì sao hệ số 2?** ReLU triệt tiêu trung bình một nửa số neuron.

### Khởi tạo Độ lệch

Độ lệch thường có thể khởi tạo bằng không:

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

### 1. Scale Đặc trưng

Chuẩn hóa các đặc trưng đầu vào về các khoảng tương tự.

#### Chuẩn hóa Standardization (Z-score)

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

**Khi nào dùng gì:**
- **Standardization**: Khi đặc trưng phân phối chuẩn hoặc có ngoại lai
- **Min-Max**: Khi cần đặc trưng trong khoảng cụ thể (ví dụ: [0, 1])

### 2. Xáo trộn Dữ liệu

Xáo trộn dữ liệu huấn luyện trước mỗi epoch để tránh học các mẫu phụ thuộc thứ tự.

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

### Tạo Mini-Batch

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

## Chia Train/Validation/Test

### Vì sao Ba Tập?

- **Tập huấn luyện**: Học tham số
- **Tập kiểm định**: Tinh chỉnh siêu tham số, giám sát quá khớp
- **Tập kiểm tra**: Đánh giá cuối cùng (chỉ dùng một lần!)

### Tỷ lệ Chia Điển hình

**Tập dữ liệu nhỏ (< 10.000 mẫu):**
- Train: 60%, Val: 20%, Test: 20%

**Tập dữ liệu vừa (10.000 – 1.000.000):**
- Train: 80%, Val: 10%, Test: 10%

**Tập dữ liệu lớn (> 1.000.000):**
- Train: 98%, Val: 1%, Test: 1%

### Triển khai

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
