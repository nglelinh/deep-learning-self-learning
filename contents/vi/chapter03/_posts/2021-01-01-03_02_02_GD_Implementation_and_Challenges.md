---
layout: post
title: 03-02-02 Triển khai Gradient Descent và Thách thức
chapter: '03'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

## Triển khai

### Gradient Descent Cơ bản

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Gradient descent cơ bản cho hồi quy tuyến tính
    
    X: ma trận đầu vào (m, n)
    y: vectơ mục tiêu (m, 1)
    """
    m, n = X.shape
    theta = np.zeros((n, 1))  # Khởi tạo tham số
    cost_history = []
    
    for i in range(num_iterations):
        # Lan truyền xuôi: dự đoán
        y_pred = X @ theta
        
        # Tính chi phí
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)
        
        # Tính gradient
        gradient = (1 / m) * X.T @ (y_pred - y)
        
        # Cập nhật tham số
        theta = theta - learning_rate * gradient
        
        # In tiến độ
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return theta, cost_history

# Ví dụ sử dụng
X = np.random.randn(100, 3)  # 100 mẫu, 3 đặc trưng
y = np.random.randn(100, 1)  # 100 mục tiêu

theta_optimal, costs = gradient_descent(X, y, learning_rate=0.1, num_iterations=1000)
```

### Mini-Batch Gradient Descent

```python
def mini_batch_gradient_descent(X, y, learning_rate=0.01, batch_size=32, num_epochs=10):
    """
    Mini-batch gradient descent
    
    X: ma trận đầu vào (m, n)
    y: vectơ mục tiêu (m, 1)
    batch_size: kích thước mỗi mini-batch
    num_epochs: số lần quét đầy đủ tập dữ liệu
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    
    for epoch in range(num_epochs):
        # Xáo trộn dữ liệu
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Xử lý các mini-batch
        for i in range(0, m, batch_size):
            # Lấy batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Lan truyền xuôi
            y_pred = X_batch @ theta
            
            # Tính gradient trên batch
            batch_size_actual = X_batch.shape[0]
            gradient = (1 / batch_size_actual) * X_batch.T @ (y_pred - y_batch)
            
            # Cập nhật tham số
            theta = theta - learning_rate * gradient
        
        # Tính chi phí trên toàn tập (để giám sát)
        y_pred_full = X @ theta
        cost = (1 / (2 * m)) * np.sum((y_pred_full - y) ** 2)
        cost_history.append(cost)
        
        print(f"Epoch {epoch + 1}/{num_epochs}: Cost = {cost:.4f}")
    
    return theta, cost_history

# Ví dụ sử dụng
theta_optimal, costs = mini_batch_gradient_descent(
    X, y, 
    learning_rate=0.1, 
    batch_size=32, 
    num_epochs=50
)
```

### Kết hợp Lịch Điều chỉnh Tốc độ Học

```python
class LearningRateSchedule:
    def __init__(self, initial_lr, schedule_type='step', **kwargs):
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.kwargs = kwargs
    
    def get_lr(self, iteration):
        if self.schedule_type == 'step':
            decay_rate = self.kwargs.get('decay_rate', 0.5)
            decay_steps = self.kwargs.get('decay_steps', 1000)
            return self.initial_lr * (decay_rate ** (iteration // decay_steps))
        
        elif self.schedule_type == 'exponential':
            decay_rate = self.kwargs.get('decay_rate', 0.95)
            return self.initial_lr * np.exp(-decay_rate * iteration)
        
        elif self.schedule_type == 'inverse':
            decay_rate = self.kwargs.get('decay_rate', 0.01)
            return self.initial_lr / (1 + decay_rate * iteration)
        
        else:
            return self.initial_lr

def gradient_descent_with_schedule(X, y, initial_lr=0.01, num_iterations=1000, schedule_type='step'):
    """Gradient descent với lịch learning rate"""
    m, n = X.shape
    theta = np.zeros((n, 1))
    
    lr_schedule = LearningRateSchedule(initial_lr, schedule_type, decay_rate=0.5, decay_steps=200)
    
    for i in range(num_iterations):
        # Lấy learning rate hiện tại
        lr = lr_schedule.get_lr(i)
        
        # Tính xuôi và gradient
        y_pred = X @ theta
        gradient = (1 / m) * X.T @ (y_pred - y)
        
        # Cập nhật với learning rate hiện tại
        theta = theta - lr * gradient
        
        if i % 100 == 0:
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            print(f"Iteration {i}: LR = {lr:.6f}, Cost = {cost:.4f}")
    
    return theta

# Ví dụ sử dụng
theta = gradient_descent_with_schedule(X, y, initial_lr=0.1, num_iterations=1000)
```

## Tiêu chí Hội tụ

Trong thực tế, ta xác định thời điểm dừng huấn luyện bằng một hoặc nhiều tiêu chí sau:

### 1. Giới hạn Số Vòng lặp

Dừng sau một số vòng lặp hoặc epoch định sẵn.

```python
if iteration >= max_iterations:
    break
```

### 2. Ngưỡng Chi phí

Dừng khi giá trị hàm chi phí giảm xuống dưới một ngưỡng đặt trước.

```python
if cost < threshold:
    break
```

### 3. Độ lớn Gradient

Dừng khi chuẩn của vector gradient đủ nhỏ (gần điểm dừng).

```python
if np.linalg.norm(gradient) < epsilon:
    break
```

### 4. Mức thay đổi Chi phí

Dừng khi hàm chi phí không còn giảm đáng kể giữa các bước liên tiếp.

```python
if abs(cost - previous_cost) < epsilon:
    break
```

### 5. Mất mát trên Tập Kiểm định (Phổ biến nhất trong Học sâu)

Dừng khi mất mát trên tập kiểm định (*validation loss*) ngừng cải thiện sau một số epoch liên tiếp (kỹ thuật dừng sớm — *early stopping*).

```python
if validation_loss > best_validation_loss:
    patience_counter += 1
    if patience_counter >= patience:
        break
else:
    best_validation_loss = validation_loss
    patience_counter = 0
```

## Các Thách thức với Gradient Descent

### 1. Cực tiểu Cục bộ

Các hàm mất mát không lồi (như ở mạng neuron sâu) thường có nhiều cực tiểu cục bộ.

**Hướng xử lý:**
- Khởi tạo ngẫu nhiên để thử nhiều điểm xuất phát khác nhau
- Sử dụng momentum (sẽ học trong bài về các bộ tối ưu nâng cao)
- Áp dụng *simulated annealing*

### 2. Điểm Yên ngựa

Là những điểm có gradient bằng 0 nhưng không phải cực tiểu mà dốc lên theo một số phương và dốc xuống theo các phương khác.

**Hướng xử lý:**
- Sử dụng momentum và các bộ tối ưu có tốc độ học thích nghi để thoát khỏi vùng yên ngựa
- Áp dụng phương pháp tối ưu bậc hai (như phương pháp Newton)

### 3. Vùng Bằng phẳng (*Plateau*)

Những vùng có độ dốc rất nhỏ khiến tham số cập nhật cực chậm.

**Hướng xử lý:**
- Huấn luyện thêm số bước đủ lớn
- Kết hợp lịch trình điều chỉnh tốc độ học
- Chuyển sang các bộ tối ưu thích nghi như Adam hoặc RMSprop

### 4. Gradient Biến mất và Bùng nổ

Trong mạng sâu, gradient có thể suy giảm nhanh về 0 hoặc tăng vọt mất kiểm soát.

**Hướng xử lý:**
- Khởi tạo trọng số hợp lý (Xavier, He)
- Áp dụng batch normalization
- Sử dụng kết nối tắt (*residual connections*)
- Cắt ngưỡng gradient (*gradient clipping*) để xử lý tình trạng bùng nổ

## Tóm tắt

- **Gradient descent** là thuật toán tối ưu nền tảng cho huấn luyện mạng neuron.
- **Quy tắc cập nhật:** $$\theta := \theta - \eta \nabla_{\theta} J(\theta)$$.
- **Mini-batch gradient descent** là biến thể được sử dụng rộng rãi nhất nhờ cân bằng tốt giữa tốc độ và độ ổn định.
- **Tốc độ học** $$\eta$$ cần chọn cẩn thận: quá nhỏ sẽ hội tụ chậm, quá lớn sẽ gây mất ổn định.
- **Lịch trình tốc độ học** giúp cải thiện tốc độ hội tụ và chất lượng nghiệm cuối cùng.
- **Tiêu chí hội tụ** (đặc biệt là dừng sớm theo tập kiểm định) giúp xác định thời điểm kết thúc huấn luyện hợp lý.
- **Các thách thức phổ biến** gồm cực tiểu cục bộ, điểm yên ngựa, vùng bằng phẳng và vấn đề gradient biến mất/bùng nổ.

Ở bài tiếp theo, ta sẽ tìm hiểu **lan truyền ngược** — thuật toán tính gradient hiệu quả cho mạng neuron.

## Video tham khảo

- **Xem trước:** [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trực tiếp tiêu chí dừng khi bước cập nhật đủ nhỏ.
- [Stochastic Gradient Descent, Clearly Explained!!!](https://www.youtube.com/watch?v=vMh0zPT0tLI) — StatQuest with Josh Starmer, tiếng Anh. Giải thích nhiễu trong SGD — thách thức thường gặp khi cài đặt thực tế.
- [PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) — Patrick Loeber, tiếng Anh. Trình bày vòng lặp huấn luyện thực tế với mô hình, hàm mất mát và bộ tối ưu trong PyTorch.
