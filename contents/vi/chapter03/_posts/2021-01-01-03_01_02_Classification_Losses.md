---
layout: post
title: 03-01-02 Hàm Mất mát Phân loại và Lựa chọn
chapter: '03'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

## Hàm Mất mát cho Phân loại Nhị phân

### 1. Entropy Chéo Nhị phân (Log Loss)

**Công thức:**

$$\mathcal{L}_{\text{BCE}}(\hat{y}, y) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

trong đó:
- $$y \in \{0, 1\}$$ là nhãn thật
- $$\hat{y} \in (0, 1)$$ là xác suất dự đoán

**Hàm chi phí:**

$$J_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

**Trực giác:**
- Nếu $$y = 1$$: Mất mát là $$-\log(\hat{y})$$, cực tiểu khi $$\hat{y} \to 1$$
- Nếu $$y = 0$$: Mất mát là $$-\log(1-\hat{y})$$, cực tiểu khi $$\hat{y} \to 0$$

**Tính chất:**
- Dựa trên ước lượng hợp lý cực đại
- Trơn và khả vi
- Phù hợp tốt với tối ưu dựa trên gradient
- Phạt nặng các dự đoán sai nhưng tự tin

**Đạo hàm (với đầu ra sigmoid):**

Với lớp đầu ra dùng sigmoid: $$\hat{y} = \sigma(z)$$

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \hat{y} - y$$

Đạo hàm đơn giản đáng kể này làm huấn luyện hiệu quả!

**Trường hợp dùng:**
- **Phân loại nhị phân**: Ảnh này là mèo hay chó?
- Lớp đầu ra với kích hoạt sigmoid

**Ưu điểm:**
- Diễn giải xác suất đúng đắn
- Gradient mạnh với dự đoán sai
- Phù hợp tốt với đầu ra sigmoid

**Nhược điểm:**
- Có thể sinh mất mát rất lớn với dự đoán sai tự tin
- Yêu cầu xác suất dự đoán (không phải logits trực tiếp)

### 2. Mất mát Hinge (Mất mát SVM)

**Công thức:**

$$\mathcal{L}_{\text{Hinge}}(\hat{y}, y) = \max(0, 1 - y \cdot \hat{y})$$

trong đó $$y \in \{-1, +1\}$$ và $$\hat{y}$$ là điểm số thô (không phải xác suất).

**Tính chất:**
- Dùng trong Support Vector Machines
- Khuyến khích cực đại hóa lề
- Không khả vi tại $$y \cdot \hat{y} = 1$$

**Trường hợp dùng:**
- Phân loại nhị phân với mục tiêu kiểu SVM
- Khi muốn bộ phân loại lề cực đại

## Hàm Mất mát cho Phân loại Đa lớp

### 1. Entropy Chéo Categorical

**Công thức:**

$$\mathcal{L}_{\text{CCE}}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

trong đó:
- $$\mathbf{y}$$ là nhãn thật ở dạng **mã hóa one-hot**: $$y_c \in \{0, 1\}$$, $$\sum_c y_c = 1$$
- $$\hat{\mathbf{y}}$$ là phân phối xác suất dự đoán: $$\hat{y}_c \in (0, 1)$$, $$\sum_c \hat{y}_c = 1$$
- $$C$$ là số lớp

**Hàm chi phí:**

$$J_{\text{CCE}} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_c^{(i)} \log(\hat{y}_c^{(i)})$$

**Dạng đơn giản** (vì chỉ một $$y_c = 1$$):

$$\mathcal{L}_{\text{CCE}} = -\log(\hat{y}_{c^*})$$

trong đó $$c^*$$ là lớp thật.

**Đạo hàm (với đầu ra softmax):**

Với đầu ra softmax: $$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$$

$$\frac{\partial \mathcal{L}_{\text{CCE}}}{\partial z_j} = \hat{y}_j - y_j$$

Một lần nữa, đơn giản đáng kể!

**Trường hợp dùng:**
- **Phân loại đa lớp**: Nhận dạng chữ số (MNIST), phân loại ảnh (ImageNet)
- Lớp đầu ra với kích hoạt softmax

**Ưu điểm:**
- Khung xác suất đúng đắn
- Chuẩn cho bài toán đa lớp
- Gradient đơn giản với softmax

**Nhược điểm:**
- Yêu cầu nhãn mã hóa one-hot
- Có thể bất ổn số (dùng mẹo log-softmax)

### 2. Entropy Chéo Categorical Thưa

**Công thức:**

Giống entropy chéo categorical, nhưng nhận nhãn số nguyên thay vì mã hóa one-hot.

**Định dạng đầu vào:**
- $$y \in \{0, 1, \ldots, C-1\}$$ (chỉ số lớp nguyên)
- $$\hat{\mathbf{y}} \in \mathbb{R}^C$$ (phân phối xác suất)

**Công thức:**

$$\mathcal{L}_{\text{SCCE}}(\hat{\mathbf{y}}, y) = -\log(\hat{y}_y)$$

**Trường hợp dùng:**
- Phân loại đa lớp với nhãn nguyên
- Tiết kiệm bộ nhớ (không cần tạo vectơ one-hot)

### 3. Phân kỳ Kullback-Leibler (KL)

**Công thức:**

$$\mathcal{L}_{\text{KL}}(\mathbf{p}, \mathbf{q}) = \sum_{c=1}^{C} p_c \log\left(\frac{p_c}{q_c}\right) = \sum_{c=1}^{C} [p_c \log(p_c) - p_c \log(q_c)]$$

trong đó $$\mathbf{p}$$ là phân phối thật và $$\mathbf{q}$$ là phân phối dự đoán.

**Tính chất:**
- Đo "khoảng cách" giữa hai phân phối xác suất
- Luôn không âm
- Không đối xứng: $$\text{KL}(p \| q) \neq \text{KL}(q \| p)$$

**Quan hệ với Entropy Chéo:**

$$\text{KL}(p \| q) = H(p, q) - H(p)$$

trong đó $$H(p, q)$$ là entropy chéo và $$H(p)$$ là entropy của $$p$$.

Vì $$H(p)$$ là hằng (phân phối thật không đổi), cực tiểu hóa phân kỳ KL tương đương cực tiểu hóa entropy chéo.

**Trường hợp dùng:**
- Autoencoder biến phân (VAE)
- Khớp phân phối
- Chưng cất tri thức (*knowledge distillation*)

## Các Cân nhắc Thực tế

### Ổn định Số

#### Vấn đề: Log của Số Nhỏ

Tính $$\log(\hat{y})$$ khi $$\hat{y}$$ rất gần 0 có thể gây vấn đề số học.

#### Giải pháp: Cắt giá trị

```python
epsilon = 1e-7
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
loss = -np.mean(y_true * np.log(y_pred_clipped))
```

#### Giải pháp Tốt hơn: Mẹo LogSumExp

Với entropy chéo kèm softmax, tính mất mát trực tiếp từ logits:

```python
def softmax_cross_entropy(logits, labels):
    """Softmax + entropy chéo ổn định số"""
    # Mẹo log-sum-exp
    logits_max = np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = logits_max + np.log(np.sum(np.exp(logits - logits_max), axis=-1, keepdims=True))
    log_softmax = logits - log_sum_exp
    
    # Entropy chéo
    return -np.mean(np.sum(labels * log_softmax, axis=-1))
```

### Ví dụ Triển khai

```python
import numpy as np

class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred, epsilon=1e-7):
        """Binary Cross-Entropy"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred, epsilon=1e-7):
        """Categorical Cross-Entropy"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
    
    @staticmethod
    def sparse_categorical_crossentropy(y_true, y_pred, epsilon=1e-7):
        """Sparse Categorical Cross-Entropy
        
        y_true: nhãn nguyên (m,)
        y_pred: xác suất (m, C)
        """
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        log_likelihood = -np.log(y_pred[range(m), y_true])
        return np.mean(log_likelihood)
    
    @staticmethod
    def huber(y_true, y_pred, delta=1.0):
        """Huber Loss"""
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * np.abs(error) - 0.5 * delta ** 2
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# Ví dụ sử dụng
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0.1, 0.9, 0.8, 0.3])

print("Binary Cross-Entropy:", LossFunctions.binary_crossentropy(y_true, y_pred))

# Ví dụ đa lớp
y_true_categorical = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_categorical = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

print("Categorical Cross-Entropy:", LossFunctions.categorical_crossentropy(y_true_categorical, y_pred_categorical))
```

## Chọn Hàm Mất mát Phù hợp

### Cây Quyết định

```
Loại tác vụ?
├─ Hồi quy
│  ├─ Có ngoại lai? → MAE hoặc Huber Loss
│  └─ Không ngoại lai? → MSE
│
├─ Phân loại nhị phân
│  ├─ Đầu ra xác suất? → Binary Cross-Entropy
│  └─ Dựa trên lề? → Hinge Loss
│
└─ Phân loại đa lớp
   ├─ Nhãn one-hot? → Categorical Cross-Entropy
   └─ Nhãn nguyên? → Sparse Categorical Cross-Entropy
```

### Tham chiếu Nhanh

| Tác vụ | Hàm mất mát | Kích hoạt đầu ra |
|------|--------------|-------------------|
| Hồi quy | MSE, MAE, Huber | Tuyến tính |
| Phân loại nhị phân | Binary Cross-Entropy | Sigmoid |
| Phân loại đa lớp | Categorical Cross-Entropy | Softmax |
| Phân loại đa nhãn | Binary Cross-Entropy (mỗi nhãn) | Sigmoid (mỗi nhãn) |

## Tóm tắt

- **Hàm mất mát** định lượng sự khác biệt giữa dự đoán và giá trị thật
- **Mất mát hồi quy**: MSE (nhạy ngoại lai), MAE (bền vững), Huber (cân bằng)
- **Mất mát phân loại**: Entropy chéo (xác suất, lựa chọn chuẩn)
- **Phân loại nhị phân**: Entropy chéo nhị phân với sigmoid
- **Phân loại đa lớp**: Entropy chéo categorical với softmax
- **Ổn định số** then chốt khi triển khai hàm mất mát
- **Lựa chọn mất mát** nên khớp với tác vụ và đặc điểm dữ liệu

Ở bài tiếp theo, ta sẽ học về gradient descent, thuật toán dùng các hàm mất mát này để cập nhật tham số mạng.
