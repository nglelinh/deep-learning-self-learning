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

Huấn luyện một mạng nơ-ron mà không theo dõi biểu đồ `train_loss`, `val_loss` và chuẩn gradient giống như lái máy bay xuyên mây mà tắt toàn bộ bảng đồng hồ buồng lái: bạn không biết mình đang tăng độ cao êm đẹp hay sắp lao xuống đất. Nhiều mô hình đạt đỉnh hiệu năng ở epoch 18, nhưng vì người chạy cứ để mặc tới epoch 100 nên mô hình học vẹt hoàn toàn tập huấn luyện và tụt dốc không phanh trên tập thật. Bài học này hướng dẫn bạn xây dựng một **vòng lặp huấn luyện chuyên nghiệp** với hệ thống giám sát chỉ số, lưu checkpoint tốt nhất và cơ chế **Dừng sớm (*Early Stopping*)**.

## Kiến thức cần nhớ: Bốn thành phần của Vòng lặp Huấn luyện Chuyên nghiệp

1. **Tách biệt Chế độ Huấn luyện (`model.train()`) và Đánh giá (`model.eval()` + `torch.no_grad()`)**:
   Lúc đánh giá bắt buộc chuyển sang `eval()` để tắt Dropout và khóa thống kê chạy (*running stats*) của BatchNorm, đồng thời bọc `torch.no_grad()` để không tốn VRAM lưu đồ thị đạo hàm.

2. **Bộ chỉ số Giám sát Đa chiều (Metrics & Diagnostics)**:
   Không chỉ nhìn Loss: theo dõi song song Accuracy/F1/MAE, learning rate hiện hành, và chuẩn $$L_2$$ của gradient (`grad_norm`) để phát hiện sớm dấu hiệu bùng nổ hay bão hòa.

3. **Dừng sớm (Early Stopping) với `patience` và `min_delta`**:
   Theo dõi `best_val_loss`; mỗi khi `val_loss` cải thiện ít nhất `min_delta`, lưu lại trọng số (*model checkpoint*) và reset bộ đếm `counter = 0`. Nếu `counter` chạm ngưỡng `patience`, dừng sớm và nạp lại bản trọng số tốt nhất.

4. **Tính Tái lập Thực nghiệm (Reproducibility & Logging)**:
   Cố định hạt giống ngẫu nhiên (`seed`), ghi nhật ký đầy đủ siêu tham số và biểu đồ học (TensorBoard / Weights & Biases) để mọi kết quả tốt đều có thể lặp lại chính xác.

### Giải thích: Mô phỏng tay cơ chế Early Stopping với `patience = 3`, `min_delta = 0.01`

Theo dõi chuỗi `val_loss` qua 9 epoch liên tiếp để thấy chính xác thời điểm lưu checkpoint và thời điểm dừng:

| Epoch | `val_loss` | So với `best_loss - 0.01` | Hành động | `best_loss` | `counter` |
| :---: | :---: | :--- | :--- | :---: | :---: |
| 1 | `0.85` | Cải thiện lần đầu | **Lưu checkpoint (Epoch 1)** | `0.85` | `0` |
| 2 | `0.62` | `0.62 < 0.84` (Tốt hơn) | **Lưu checkpoint (Epoch 2)** | `0.62` | `0` |
| 3 | `0.48` | `0.48 < 0.61` (Tốt hơn) | **Lưu checkpoint (Epoch 3)** | `0.48` | `0` |
| 4 | `0.475` | `0.475 >= 0.47` (Cải thiện quá nhỏ `< 0.01`) | Không lưu, tăng bộ đếm | `0.48` | `1` |
| 5 | `0.45` | `0.45 < 0.47` (Tốt hơn!) | **Lưu checkpoint (Epoch 5)**, reset | `0.45` | `0` |
| 6 | `0.46` | Tăng nhẹ | Tăng bộ đếm | `0.45` | `1` |
| 7 | `0.49` | Tiếp tục tăng (overfitting) | Tăng bộ đếm | `0.45` | `2` |
| 8 | `0.52` | Tăng liên tiếp 3 lần | `counter == 3` $$\to$$ **DỪNG SỚM!** | `0.45` | `3` |

Sau khi dừng ở Epoch 8, ta **không dùng** trọng số Epoch 8 (đang có `val_loss = 0.52`), mà nạp lại bản checkpoint tốt nhất ở **Epoch 5** (`val_loss = 0.45`) để đánh giá trên tập Test!

## Giám sát Quá trình Huấn luyện

### Các Chỉ số Cần Theo dõi

1. **Mất mát huấn luyện (*training loss*):** Cần giảm ổn định qua các epoch.
2. **Mất mát kiểm định (*validation loss*):** Cần giảm song hành; nếu quay đầu tăng trong khi mất mát huấn luyện vẫn giảm, mô hình đang bị quá khớp (*overfitting*).
3. **Độ chính xác huấn luyện (*training accuracy*):** Cần tăng dần.
4. **Độ chính xác kiểm định (*validation accuracy*):** Cần tăng theo; khoảng cách lớn giữa độ chính xác huấn luyện và kiểm định là dấu hiệu rõ rệt của quá khớp.

### Trực quan hóa Lịch sử Huấn luyện

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

Kỹ thuật này dừng huấn luyện khi mất mát trên tập kiểm định ngừng cải thiện sau một số epoch chờ (*patience*) và khôi phục lại bộ tham số tốt nhất đã lưu.

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

## Cắt ngưỡng Gradient (Gradient Clipping)

Giúp hạn chế hiện tượng gradient bùng nổ bằng cách giới hạn biên độ hoặc chuẩn của vector gradient.

### Cắt theo Giá trị

```python
def clip_gradients_by_value(gradients, max_value=5.0):
    """Cắt gradient về [-max_value, max_value]"""
    clipped_gradients = {}
    for key in gradients.keys():
        clipped_gradients[key] = np.clip(gradients[key], -max_value, max_value)
    return clipped_gradients
```

### Cắt theo Chuẩn Toàn cục

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

## Vòng lặp Huấn luyện Hoàn chỉnh

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

### Các Siêu tham số Quan trọng

1. **Tốc độ học (*learning rate*)** — thường có ảnh hưởng lớn nhất
2. **Kích thước batch (*batch size*)**
3. **Số lượng lớp**
4. **Số neuron mỗi lớp**
5. **Hàm kích hoạt**
6. **Phương pháp khởi tạo trọng số**

### Các Chiến lược Tìm kiếm

#### 1. Thử nghiệm Thủ công

Điều chỉnh dần các giá trị dựa trên kinh nghiệm và quan sát đường cong huấn luyện.

#### 2. Tìm kiếm Lưới (*Grid Search*)

Thử toàn bộ các tổ hợp có thể từ một tập giá trị định trước.

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

#### 3. Tìm kiếm Ngẫu nhiên (*Random Search*)

Lấy mẫu ngẫu nhiên trong không gian siêu tham số — thường hiệu quả hơn grid search khi có nhiều siêu tham số với mức độ nhạy khác nhau.

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

- **Khởi tạo phù hợp** (như khởi tạo He cho ReLU) giúp tránh nhiều lỗi hội tụ ngay từ đầu.
- **Tiền xử lý dữ liệu** (standardization hoặc min-max) cải thiện tốc độ và độ ổn định khi tối ưu.
- **Mini-batch** cân bằng tốt giữa tốc độ cập nhật và độ mượt của gradient.
- **Chia tập Train/Val/Test** đảm bảo đánh giá khách quan và hạn chế quá khớp.
- **Giám sát đồng thời** chỉ số trên tập huấn luyện và tập kiểm định giúp phát hiện sớm dấu hiệu quá khớp.
- **Dừng sớm** giúp tránh huấn luyện thừa và giữ lại phiên bản tham số tốt nhất.
- **Cắt ngưỡng gradient** ngăn chặn hiện tượng gradient bùng nổ.
- **Tinh chỉnh siêu tham số** đóng vai trò quan trọng để đạt hiệu năng tốt trên bài toán thực tế.

Chương tiếp theo sẽ giới thiệu **Mạng Neuron Tích chập (CNN)** chuyên dụng cho các bài toán thị giác máy tính.

## Video tham khảo

- **Xem trước:** [Other Regularization Methods (C2W1L08)](https://www.youtube.com/watch?v=BOCLq2gpcGU) — DeepLearningAI, tiếng Anh. Trình bày dừng sớm (*early stopping*) cùng các phương pháp chính quy hóa liên quan.
- [PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) — Patrick Loeber, tiếng Anh. Minh họa vòng lặp huấn luyện chuẩn ghi nhận loss và cập nhật tham số.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering, tiếng Anh. Phân tích cách đọc đường cong huấn luyện và lựa chọn siêu tham số.
