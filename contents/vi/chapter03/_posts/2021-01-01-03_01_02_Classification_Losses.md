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

Nếu mô hình phân loại ảnh của bạn đoán xác suất bức ảnh là con chó bằng $$0.99$$ khi nhãn thật là con mèo, sai lầm đó không chỉ "sai 1 điểm" mà là một sự tự tin mù quáng cần bị phạt cực nặng. Đó là lý do các bài toán phân loại trong Học sâu hầu như luôn xoay quanh **Cross-Entropy (Log Loss)**—hàm mất mát có nguồn gốc từ Lý thuyết Thông tin và MLE—cùng những biến thể chuyên trị dữ liệu mất cân bằng như **Focal Loss**, **Label Smoothing** và **Hinge Loss**. Bài học này giải phẫu trọn vẹn bộ hàm mất mát phân loại và quy tắc lựa chọn trong từng tình huống.

## Bản đồ kiến thức nền: Bốn trụ cột của Hàm Mất mát Phân loại

1. **Binary Cross-Entropy (BCE) cho Phân loại Nhị phân & Đa nhãn**:
   $$L = -[y\ln p + (1-y)\ln(1-p)]$$ với $$p = \sigma(z)$$. Trong PyTorch, luôn dùng `BCEWithLogitsLoss` (nhận trực tiếp logit $$z$$) thay vì tự chạy `Sigmoid` rồi `BCELoss` để đảm bảo ổn định số học.

2. **Categorical Cross-Entropy cho Phân loại Đa lớp (Softmax + NLL)**:
   Với nhãn one-hot tại lớp đúng $$c$$, mất mát rút gọn cực đẹp thành $$L = -\ln p_c = -z_c + \ln\!\left(\sum_j e^{z_j}\right)$$, và gradient theo vector logit đơn giản đáng kinh ngạc: $$\nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}$$.

3. **Hinge Loss (Biên lề SVM) và Margin-based Losses**:
   $$L = \max(0,\; 1 - y\cdot z)$$ với $$y \in \{-1, +1\}$$: chỉ phạt các mẫu bị đoán sai hoặc nằm quá sát biên quyết định ($$y z < 1$$), bỏ qua hoàn toàn các mẫu đã phân tách đủ xa.

4. **Kỹ thuật nâng cao: Label Smoothing & Focal Loss**:
   Label Smoothing làm mềm nhãn từ `[0, 1, 0]` thành `[0.05, 0.90, 0.05]` chống quá tự tin (*overconfidence*); còn Focal Loss nhân thêm $$(1 - p_t)^\gamma$$ để hạ trọng số mẫu dễ và tập trung sức học vào mẫu khó/lớp hiếm.

### Khung giải thích (nên đọc chậm): Tính tay Cross-Entropy, Gradient $$\mathbf{p} - \mathbf{y}$$ và Focal Loss

Giả sử bài toán 3 lớp có nhãn thật là lớp 0 ($$\mathbf{y} = [1, 0, 0]$$), và mạng xuất ra vector logit $$\mathbf{z} = [1.0,\; 2.0,\; 0.0]$$:

1. **Tính xác suất Softmax**:
   $$e^1 \approx 2.718,\; e^2 \approx 7.389,\; e^0 = 1.000 \implies \text{Tổng} = 11.107.$$
   $$\mathbf{p} \approx \left[\frac{2.718}{11.107},\; \frac{7.389}{11.107},\; \frac{1.000}{11.107}\right] = [0.2447,\; 0.6652,\; 0.0900].$$
2. **Tính Categorical Cross-Entropy**:
   $$L_{\text{CE}} = -\ln(p_0) = -\ln(0.2447) \approx 1.4077.$$
3. **Tính Gradient theo từng logit ($$\nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}$$)**:
   $$\frac{\partial L}{\partial \mathbf{z}} = [0.2447 - 1,\; 0.6652 - 0,\; 0.0900 - 0] = [-0.7553,\; +0.6652,\; +0.0900].$$
   Khi cập nhật ngược dấu gradient ($$\mathbf{z} \leftarrow \mathbf{z} - \eta \nabla_{\mathbf{z}} L$$), logit lớp đúng $$z_0$$ được **kéo tăng lên** (vì gradient âm), còn logit hai lớp sai $$z_1, z_2$$ bị **đẩy xuống** (vì gradient dương), trong đó lớp sai nổi bật nhất $$z_1$$ bị đẩy mạnh nhất!
4. **So với Focal Loss ($$\gamma = 2$$)**:
   Nhân thêm $$(1 - p_0)^2 = (1 - 0.2447)^2 \approx 0.5705$$: nếu một mẫu dễ có $$p_0 = 0.95$$, thừa số $$(1 - 0.95)^2 = 0.0025$$ làm giảm mức phạt xuống **400 lần** để mô hình dồn lực vào các mẫu khó!

## Hàm Mất mát cho Phân loại Nhị phân

### 1. Entropy Chéo Nhị phân (Log Loss)

**Công thức:**

$$\mathcal{L}_{\text{BCE}}(\hat{y}, y) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

trong đó:
- $$y \in \{0, 1\}$$ là nhãn thực tế
- $$\hat{y} \in (0, 1)$$ là xác suất dự đoán

**Hàm chi phí:**

$$J_{\text{BCE}} = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

**Trực giác:**
- Khi $$y = 1$$: Mất mát bằng $$-\log(\hat{y})$$, đạt cực tiểu khi $$\hat{y} \to 1$$
- Khi $$y = 0$$: Mất mát bằng $$-\log(1-\hat{y})$$, đạt cực tiểu khi $$\hat{y} \to 0$$

**Tính chất:**
- Xuất phát từ nguyên lý ước lượng hợp lý cực đại (*maximum likelihood estimation*)
- Trơn và khả vi
- Tương thích tốt với tối ưu dựa trên gradient
- Phạt rất nặng các dự đoán sai nhưng có độ tự tin cao

**Đạo hàm (với đầu ra sigmoid):**

Khi lớp đầu ra sử dụng sigmoid $$\hat{y} = \sigma(z)$$:

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \hat{y} - y$$

Dạng đạo hàm gọn gàng này giúp việc tính gradient và huấn luyện diễn ra thuận lợi.

**Trường hợp sử dụng:**
- **Phân loại nhị phân:** Ví dụ phân biệt ảnh chó và mèo
- Kết hợp với lớp đầu ra dùng hàm kích hoạt sigmoid

**Ưu điểm:**
- Có diễn giải xác suất chặt chẽ
- Tạo gradient đủ mạnh khi mô hình dự đoán sai
- Kết hợp tự nhiên với đầu ra sigmoid

**Nhược điểm:**
- Mất mát có thể tăng rất lớn nếu mô hình dự đoán sai với xác suất gần 0 hoặc 1
- Cần đầu ra dạng xác suất (hoặc dùng phiên bản nhận trực tiếp logits để ổn định số)

### 2. Mất mát Hinge (Mất mát SVM)

**Công thức:**

$$\mathcal{L}_{\text{Hinge}}(\hat{y}, y) = \max(0, 1 - y \cdot \hat{y})$$

trong đó $$y \in \{-1, +1\}$$ và $$\hat{y}$$ là điểm số thô (không phải xác suất).

**Tính chất:**
- Dùng trong máy vector hỗ trợ (*Support Vector Machines*)
- Khuyến khích mở rộng khoảng lề phân tách (*margin*)
- Không khả vi tại $$y \cdot \hat{y} = 1$$

**Trường hợp sử dụng:**
- Bài toán phân loại nhị phân theo mục tiêu kiểu SVM
- Khi cần tối ưu hóa biên lề cực đại

## Hàm Mất mát cho Phân loại Đa lớp

### 1. Entropy Chéo Đa lớp (Categorical Cross-Entropy)

**Công thức:**

$$\mathcal{L}_{\text{CCE}}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

trong đó:
- $$\mathbf{y}$$ là nhãn thực tế ở dạng **mã hóa one-hot**: $$y_c \in \{0, 1\}$$, $$\sum_c y_c = 1$$
- $$\hat{\mathbf{y}}$$ là phân phối xác suất dự đoán: $$\hat{y}_c \in (0, 1)$$, $$\sum_c \hat{y}_c = 1$$
- $$C$$ là tổng số lớp

**Hàm chi phí:**

$$J_{\text{CCE}} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_c^{(i)} \log(\hat{y}_c^{(i)})$$

**Dạng rút gọn** (vì chỉ có đúng một thành phần $$y_{c^*} = 1$$):

$$\mathcal{L}_{\text{CCE}} = -\log(\hat{y}_{c^*})$$

trong đó $$c^*$$ là chỉ số của lớp đúng.

**Đạo hàm (với đầu ra softmax):**

Khi lớp đầu ra sử dụng softmax $$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$$:

$$\frac{\partial \mathcal{L}_{\text{CCE}}}{\partial z_j} = \hat{y}_j - y_j$$

Tương tự trường hợp nhị phân, biểu thức đạo hàm có dạng rất đơn giản.

**Trường hợp sử dụng:**
- **Phân loại đa lớp:** Ví dụ nhận dạng chữ số (MNIST), phân loại ảnh (ImageNet)
- Kết hợp với lớp đầu ra dùng softmax

**Ưu điểm:**
- Nền tảng xác suất rõ ràng
- Lựa chọn tiêu chuẩn cho bài toán đa lớp
- Biểu thức gradient gọn khi ghép với softmax

**Nhược điểm:**
- Đòi hỏi nhãn ở dạng mã hóa one-hot
- Dễ gặp vấn đề ổn định số nếu tính rời rạc (nên dùng kết hợp log-softmax)

### 2. Entropy Chéo Đa lớp Thưa (Sparse Categorical Cross-Entropy)

Về bản chất toán học, hàm này giống hệt Categorical Cross-Entropy nhưng nhận trực tiếp nhãn số nguyên thay vì vector one-hot.

**Định dạng đầu vào:**
- $$y \in \{0, 1, \ldots, C-1\}$$ (chỉ số nguyên của lớp đúng)
- $$\hat{\mathbf{y}} \in \mathbb{R}^C$$ (phân phối xác suất dự đoán)

**Công thức:**

$$\mathcal{L}_{\text{SCCE}}(\hat{\mathbf{y}}, y) = -\log(\hat{y}_y)$$

**Trường hợp sử dụng:**
- Phân loại đa lớp khi nhãn lưu dưới dạng số nguyên
- Tiết kiệm bộ nhớ vì không cần tạo ma trận one-hot (đặc biệt hữu ích khi $$C$$ lớn)

### 3. Phân kỳ Kullback-Leibler (KL Divergence)

**Công thức:**

$$\mathcal{L}_{\text{KL}}(\mathbf{p}, \mathbf{q}) = \sum_{c=1}^{C} p_c \log\left(\frac{p_c}{q_c}\right) = \sum_{c=1}^{C} [p_c \log(p_c) - p_c \log(q_c)]$$

trong đó $$\mathbf{p}$$ là phân phối mục tiêu và $$\mathbf{q}$$ là phân phối dự đoán.

**Tính chất:**
- Đo độ sai khác giữa hai phân phối xác suất
- Luôn không âm (bằng 0 khi và chỉ khi $$\mathbf{p} = \mathbf{q}$$)
- Không đối xứng: $$\text{KL}(p \| q) \neq \text{KL}(q \| p)$$

**Quan hệ với Entropy Chéo:**

$$\text{KL}(p \| q) = H(p, q) - H(p)$$

trong đó $$H(p, q)$$ là entropy chéo và $$H(p)$$ là entropy của phân phối $$p$$.

Do $$H(p)$$ là hằng số đối với tham số mô hình, việc cực tiểu hóa phân kỳ KL tương đương với cực tiểu hóa entropy chéo.

**Trường hợp sử dụng:**
- Autoencoder biến phân (VAE)
- Các bài toán khớp phân phối xác suất
- Chưng cất tri thức (*knowledge distillation*)

## Các Lưu ý Thực tế

### Ổn định Số học

#### Vấn đề: Logarit của Số Rất Nhỏ

Tính $$\log(\hat{y})$$ khi $$\hat{y}$$ tiến sát 0 có thể sinh ra giá trị $$-\infty$$ hoặc `NaN`.

#### Cách xử lý cơ bản: Giới hạn miền giá trị (*clipping*)

```python
epsilon = 1e-7
y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
loss = -np.mean(y_true * np.log(y_pred_clipped))
```

#### Cách xử lý tốt hơn: Kỹ thuật LogSumExp

Khi kết hợp softmax và entropy chéo, nên tính trực tiếp từ logits thay vì qua bước xác suất trung gian:

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

### Bảng Tra cứu Nhanh

| Tác vụ | Hàm mất mát | Kích hoạt đầu ra |
|------|--------------|-------------------|
| Hồi quy | MSE, MAE, Huber | Tuyến tính |
| Phân loại nhị phân | Binary Cross-Entropy | Sigmoid |
| Phân loại đa lớp | Categorical Cross-Entropy | Softmax |
| Phân loại đa nhãn | Binary Cross-Entropy (mỗi nhãn) | Sigmoid (mỗi nhãn) |

## Tóm tắt

- **Hàm mất mát** định lượng mức độ sai lệch giữa dự đoán của mô hình và giá trị thực tế.
- **Nhóm bài toán hồi quy:** MSE (nhạy ngoại lai), MAE (bền vững trước ngoại lai), Huber (cân bằng giữa hai đặc tính).
- **Nhóm bài toán phân loại:** Entropy chéo là lựa chọn tiêu chuẩn nhờ nền tảng xác suất và gradient ổn định.
- **Phân loại nhị phân:** Ghép Binary Cross-Entropy với sigmoid.
- **Phân loại đa lớp:** Ghép Categorical Cross-Entropy với softmax.
- **Ổn định số học** là yếu tố quan trọng khi cài đặt hàm mất mát, đặc biệt với các phép tính hàm mũ và logarit.
- **Lựa chọn hàm mất mát** cần phù hợp với loại bài toán và phân phối dữ liệu.

Ở bài tiếp theo, ta sẽ tìm hiểu gradient descent — thuật toán sử dụng hàm mất mát để cập nhật tham số mạng.

## Video tham khảo

- **Xem trước:** [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) — StatQuest with Josh Starmer, tiếng Anh. Giới thiệu hàm mất mát logistic cho bài toán phân loại nhị phân.
- [But what is cross-entropy? | Compression is Intelligence Part 2](https://www.youtube.com/watch?v=GlYgs6v2YfU) — 3Blue1Brown, tiếng Anh. Giải thích bản chất của cross-entropy từ góc độ độ dài mã hóa thay vì chỉ đưa công thức.
- [Softmax Regression (C2W3L08)](https://www.youtube.com/watch?v=LLux1SW--oM) — DeepLearningAI, tiếng Anh. Trình bày hồi quy softmax cho bài toán phân loại nhiều lớp.
