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

### Với Lịch Learning Rate

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

Làm sao biết khi nào dừng huấn luyện?

### 1. Số Vòng lặp Cực đại

Dừng sau số vòng lặp/epoch cố định.

```python
if iteration >= max_iterations:
    break
```

### 2. Ngưỡng Chi phí

Dừng khi chi phí xuống dưới một ngưỡng.

```python
if cost < threshold:
    break
```

### 3. Độ lớn Gradient

Dừng khi gradient rất nhỏ (gần điểm dừng).

```python
if np.linalg.norm(gradient) < epsilon:
    break
```

### 4. Thay đổi Chi phí

Dừng khi chi phí không còn giảm đáng kể.

```python
if abs(cost - previous_cost) < epsilon:
    break
```

### 5. Mất mát Kiểm định (Phổ biến nhất trong Học sâu)

Dừng khi mất mát kiểm định không còn cải thiện (dừng sớm — *early stopping*).

```python
if validation_loss > best_validation_loss:
    patience_counter += 1
    if patience_counter >= patience:
        break
else:
    best_validation_loss = validation_loss
    patience_counter = 0
```

## Thách thức với Gradient Descent

### 1. Cực tiểu Địa phương

Hàm không lồi (như mạng neuron) có nhiều cực tiểu địa phương.

**Giải pháp:**
- Khởi tạo ngẫu nhiên (thử các điểm xuất phát khác nhau)
- Momentum (trình bày trong bộ tối ưu nâng cao)
- Simulated annealing

### 2. Điểm Yên ngựa

Các điểm nơi gradient bằng không nhưng không phải cực tiểu.

**Giải pháp:**
- Momentum và learning rate thích nghi giúp thoát
- Phương pháp bậc hai (phương pháp Newton)

### 3. Plateau

Vùng phẳng nơi gradient rất nhỏ.

**Giải pháp:**
- Kiên nhẫn (chờ lâu hơn)
- Lịch learning rate
- Bộ tối ưu thích nghi (Adam, RMSprop)

### 4. Gradient Biến mất / Bùng nổ

Gradient trở nên quá nhỏ hoặc quá lớn trong mạng sâu.

**Giải pháp:**
- Khởi tạo đúng (Xavier, He)
- Batch normalization
- Kết nối dư (*residual connections*)
- Gradient clipping (cho gradient bùng nổ)

## Tóm tắt

- **Gradient descent** là thuật toán tối ưu nền tảng cho mạng neuron
- **Quy tắc cập nhật**: $$\theta := \theta - \eta \nabla_{\theta} J(\theta)$$
- **Mini-batch gradient descent** là biến thể được dùng phổ biến nhất
- **Learning rate** $$\eta$$ then chốt: quá nhỏ → chậm, quá lớn → bất ổn
- **Lịch learning rate** có thể cải thiện hội tụ
- **Tiêu chí hội tụ** giúp xác định khi nào dừng huấn luyện
- **Thách thức** gồm cực tiểu địa phương, điểm yên ngựa, và vấn đề gradient

Ở bài tiếp theo, ta sẽ xét **lan truyền ngược**, thuật toán tính gradient hiệu quả trong mạng neuron.
