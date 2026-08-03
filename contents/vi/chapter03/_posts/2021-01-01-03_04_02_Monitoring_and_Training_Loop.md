---
layout: post
title: 03-04-02 Giám sát, Dừng sớm và Vòng lặp Huấn luyện
chapter: '03'
order: 13
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

## Giám sát Huấn luyện

### Các Chỉ số Cần Theo dõi

1. **Mất mát huấn luyện**: Nên giảm ổn định
2. **Mất mát kiểm định**: Nên giảm; nếu tăng, đang quá khớp!
3. **Độ chính xác huấn luyện**: Nên tăng
4. **Độ chính xác kiểm định**: Nên tăng; khoảng cách với độ chính xác huấn luyện chỉ ra quá khớp

### Trực quan hóa

```python
import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Vẽ lịch sử huấn luyện"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Biểu đồ mất mát
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Biểu đồ độ chính xác
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Dừng sớm (Early Stopping)

Dừng huấn luyện khi mất mát kiểm định không còn cải thiện.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        patience: số epoch chờ trước khi dừng
        min_delta: thay đổi tối thiểu để coi là cải thiện
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_parameters = None
    
    def __call__(self, val_loss, parameters):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_parameters = parameters.copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_parameters = parameters.copy()
            self.counter = 0
        
        return self.early_stop

# Sử dụng trong vòng lặp huấn luyện
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    # Huấn luyện...
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    
    if early_stopping(val_loss, parameters):
        print("Early stopping triggered!")
        parameters = early_stopping.best_parameters
        break
```

## Gradient Clipping

Ngăn gradient bùng nổ bằng cách cắt độ lớn gradient.

### Cắt theo Giá trị

```python
def clip_gradients_by_value(gradients, max_value=5.0):
    """Cắt gradient về [-max_value, max_value]"""
    clipped_gradients = {}
    for key in gradients.keys():
        clipped_gradients[key] = np.clip(gradients[key], -max_value, max_value)
    return clipped_gradients
```

### Cắt theo Chuẩn

```python
def clip_gradients_by_norm(gradients, max_norm=5.0):
    """Cắt gradient theo chuẩn toàn cục"""
    # Tính chuẩn toàn cục
    total_norm = 0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Cắt nếu cần
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        clipped_gradients = {}
        for key, grad in gradients.items():
            clipped_gradients[key] = grad * clip_coef
        return clipped_gradients
    else:
        return gradients
```

## Vòng lặp Huấn luyện Đầy đủ

```python
def train_model(X_train, Y_train, X_val, Y_val, layer_dims, 
                learning_rate=0.01, batch_size=32, num_epochs=100,
                initialization='he', early_stopping_patience=10):
    """
    Hàm huấn luyện đầy đủ với mọi thực hành tốt nhất
    """
    # Khởi tạo
    parameters = initialize_parameters(layer_dims, initialization)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Lịch sử
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        # Tạo mini-batch
        mini_batches = create_mini_batches(X_train, Y_train, batch_size)
        epoch_loss = 0
        
        # Xử lý mỗi mini-batch
        for X_batch, Y_batch in mini_batches:
            # Lan truyền xuôi
            AL, caches = forward_propagation(X_batch, parameters)
            
            # Tính chi phí
            batch_cost = compute_cost(AL, Y_batch)
            epoch_loss += batch_cost
            
            # Lan truyền ngược
            gradients = backward_propagation(AL, Y_batch, caches)
            
            # Gradient clipping (tùy chọn)
            gradients = clip_gradients_by_norm(gradients, max_norm=5.0)
            
            # Cập nhật tham số
            parameters = update_parameters(parameters, gradients, learning_rate)
        
        # Mất mát trung bình trên mọi batch
        epoch_loss /= len(mini_batches)
        train_losses.append(epoch_loss)
        
        # Tính độ chính xác huấn luyện
        train_predictions = predict(X_train, parameters)
        train_acc = np.mean(train_predictions == Y_train)
        train_accs.append(train_acc)
        
        # Kiểm định
        val_predictions, val_caches = forward_propagation(X_val, parameters)
        val_loss = compute_cost(val_predictions, Y_val)
        val_losses.append(val_loss)
        
        val_pred_labels = (val_predictions > 0.5).astype(int)
        val_acc = np.mean(val_pred_labels == Y_val)
        val_accs.append(val_acc)
        
        # In tiến độ
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Dừng sớm
        if early_stopping(val_loss, parameters):
            print(f"Early stopping at epoch {epoch}")
            parameters = early_stopping.best_parameters
            break
    
    # Vẽ lịch sử
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    return parameters, (train_losses, val_losses, train_accs, val_accs)
```

## Tinh chỉnh Siêu tham số

### Các Siêu tham số Then chốt

1. **Learning rate** (quan trọng nhất)
2. **Kích thước batch**
3. **Số lớp**
4. **Số neuron mỗi lớp**
5. **Hàm kích hoạt**
6. **Phương pháp khởi tạo**

### Chiến lược Tinh chỉnh

#### 1. Tìm kiếm Thủ công

Thử các giá trị khác nhau dựa trên trực giác và kinh nghiệm.

#### 2. Grid Search

Thử mọi tổ hợp của một tập giá trị định trước.

```python
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]

best_val_acc = 0
best_params = None

for lr in learning_rates:
    for bs in batch_sizes:
        model, history = train_model(X_train, Y_train, X_val, Y_val, 
                                     layer_dims, learning_rate=lr, batch_size=bs)
        val_acc = history[3][-1]  # Độ chính xác kiểm định cuối
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (lr, bs)

print(f"Best: LR={best_params[0]}, BS={best_params[1]}, Val Acc={best_val_acc:.4f}")
```

#### 3. Random Search

Thường hiệu quả hơn grid search.

```python
import random

num_trials = 20
best_val_acc = 0

for trial in range(num_trials):
    lr = 10 ** random.uniform(-4, -1)  # Log-uniform giữa 0.0001 và 0.1
    bs = random.choice([16, 32, 64, 128])
    
    model, history = train_model(X_train, Y_train, X_val, Y_val,
                                 layer_dims, learning_rate=lr, batch_size=bs)
    val_acc = history[3][-1]
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"New best: LR={lr:.6f}, BS={bs}, Val Acc={val_acc:.4f}")
```

## Tóm tắt

- **Khởi tạo đúng** (He cho ReLU) ngăn vấn đề huấn luyện
- **Tiền xử lý dữ liệu** (standardization/normalization) cải thiện hội tụ
- **Xử lý mini-batch** cân bằng tốc độ và ổn định
- **Chia Train/Val/Test** cho phép đánh giá đúng
- **Giám sát** các chỉ số train/val phát hiện quá khớp
- **Dừng sớm** ngăn quá khớp và tiết kiệm thời gian
- **Gradient clipping** ngăn gradient bùng nổ
- **Tinh chỉnh siêu tham số** thiết yếu cho hiệu năng tối ưu

Với các kỹ thuật này, ta đã có đủ công cụ để huấn luyện mạng neuron hiệu quả. Chương tiếp theo trình bày Mạng Neuron Tích chập cho các tác vụ thị giác máy tính!
