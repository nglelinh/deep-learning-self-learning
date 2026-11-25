---
layout: post
title: 09-01-02 Cài đặt Dropout và Cạm bẫy
chapter: '09'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter09
---

Một trong những lỗi "âm thầm nhưng chí mạng" hay gặp nhất trong các dự án PyTorch thực tế là **quên gọi `model.eval()` trước khi chạy vòng đánh giá validation/test**. Kết quả là Dropout vẫn tiếp tục tắt ngẫu nhiên 50% neuron trong lúc chấm điểm, khiến độ chính xác tập kiểm định vừa tụt thảm hại vừa nhảy loạn xạ sau mỗi lần chạy! Bài thực hành này giúp bạn làm chủ cả cài đặt chuẩn PyTorch, bản viết tay NumPy (cả forward lẫn backward) và các quy tắc kết hợp với BatchNorm.

## Kiến thức cần nhớ: Bốn điểm kỹ thuật cốt tử khi cài đặt Dropout

1. **Lan truyền ngược qua lớp Dropout**:
   Trong lượt backward, gradient chỉ được truyền ngược qua **đúng những neuron đã sống sót** trong lượt forward (nhân lại với chính mặt nạ `mask / (1 - p)` đã lưu ở cache); các neuron bị tắt nhận gradient bằng `0`.
2. **Quản lý chặt chẽ hai chế độ `model.train()` và `model.eval()`**:
   Luôn bọc vòng đánh giá trong `model.eval()` kết hợp `with torch.no_grad():`, và nhớ gọi lại `model.train()` trước khi sang epoch huấn luyện kế tiếp.
3. **Phân biệt `nn.Dropout` (phần tử) và `nn.Dropout2d` (toàn kênh)**:
   Trên bản đồ đặc trưng CNN, các điểm ảnh kề nhau tương quan rất mạnh nên tắt lẻ từng pixel hầu như vô tác dụng; `nn.Dropout2d` tắt nguyên cả kênh đặc trưng để ép CNN thực sự học đặc trưng độc lập.
4. **Tránh xung đột giữa Dropout và BatchNorm**:
   Đặt Dropout ngay *trước* BatchNorm làm thay đổi phương sai giữa train và eval (hiện tượng *variance shift*, Li et al. 2019). Quy tắc an toàn: đặt Dropout sau BatchNorm hoặc chỉ dùng ở tầng phân loại cuối.

### Giải thích: Tính tay cả Forward và Backward của một lớp Inverted Dropout bằng NumPy

Giả sử đầu vào $$\mathbf{x} = [1{,}0, \; -2{,}0, \; 4{,}0, \; 3{,}0]$$, tỷ lệ tắt $$p = 0{,}2$$ (tỷ lệ giữ $$1 - p = 0{,}8$$), mặt nạ ngẫu nhiên rút được là $$\mathbf{m} = [1, \; 0, \; 1, \; 1]$$, và gradient từ tầng trên truyền xuống là $$\mathbf{g}_{\text{out}} = [0{,}4, \; 0{,}8, \; -0{,}4, \; 1{,}6]$$:

1. **Bước Forward (hệ số khuếch đại $$\frac{1}{1 - p} = \frac{1}{0{,}8} = 1{,}25$$)**:
   $$\mathbf{y} = \mathbf{x} \odot \mathbf{m} \times 1{,}25 = [1{,}0\cdot 1,\; -2{,}0\cdot 0,\; 4{,}0\cdot 1,\; 3{,}0\cdot 1] \times 1{,}25 = \mathbf{[1{,}25, \; 0{,}00, \; 5{,}00, \; 3{,}75]}$$
2. **Bước Backward (dùng lại CHÍNH xác mặt nạ $$\mathbf{m}$$ và hệ số $$1{,}25$$)**:
   $$\mathbf{g}_{\text{in}} = \mathbf{g}_{\text{out}} \odot \mathbf{m} \times 1{,}25 = [0{,}4\cdot 1,\; 0{,}8\cdot 0,\; -0{,}4\cdot 1,\; 1{,}6\cdot 1] \times 1{,}25 = \mathbf{[0{,}50, \; 0{,}00, \; -0{,}50, \; 2{,}00]}$$

Chú ý phần tử thứ 2: dù gradient từ trên truyền xuống là `0.8`, vì neuron này đã bị tắt ở lượt forward nên `g_in[1] = 0.00` — nó hoàn toàn không bị cập nhật trọng số ở bước này!

## 4. Đoạn Mã

### Cài đặt bằng PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPWithDropout(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10,
                 dropout_rate=0.5):
        super(MLPWithDropout, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Tầng 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)  # Dropout sau kích hoạt
        
        # Tầng 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Tầng đầu ra (không dropout)
        x = self.fc3(x)
        return x

# Huấn luyện
model = MLPWithDropout(dropout_rate=0.5)
model.train()  # Bật dropout

x_train = torch.randn(32, 784)  # Batch 32
output_train = model(x_train)
print(f"Shape đầu ra huấn luyện: {output_train.shape}")  # (32, 10)

# Kiểm tra
model.eval()  # Tắt dropout

x_test = torch.randn(10, 784)
output_test = model(x_test)
print(f"Shape đầu ra kiểm tra: {output_test.shape}")  # (10, 10)

# Kiểm tra hiệu ứng dropout
model.train()
outputs = []
for _ in range(5):
    out = model(x_train[0:1])  # Cùng đầu vào
    outputs.append(out)

print("Huấn luyện (có dropout) - đầu ra biến thiên:")
print(torch.stack(outputs).std(dim=0).mean().item())

model.eval()
outputs_test = []
for _ in range(5):
    out = model(x_train[0:1])  # Cùng đầu vào
    outputs_test.append(out)

print("Kiểm tra (không dropout) - đầu ra giống hệt:")
print(torch.stack(outputs_test).std(dim=0).mean().item())
```

### Cài đặt Dropout thủ công bằng NumPy

```python
class DropoutManual:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, X, training=True):
        """
        X: (batch, features)
        """
        if training:
            # Sinh mặt nạ: 1 với xác suất (1-p), 0 với xác suất p
            keep_prob = 1 - self.dropout_rate
            self.mask = np.random.binomial(1, keep_prob, size=X.shape)
            
            # Áp dụng mặt nạ và scale
            return X * self.mask / keep_prob
        else:
            return X
    
    def backward(self, dout):
        """Gradient chỉ chảy qua các neuron được giữ"""
        keep_prob = 1 - self.dropout_rate
        return dout * self.mask / keep_prob

# Ví dụ
dropout = DropoutManual(dropout_rate=0.5)
X = np.random.randn(4, 100)

# Huấn luyện
X_train = dropout.forward(X, training=True)
print(f"Đơn vị bị loại: {np.sum(X_train == 0)}/{X_train.size}")

# Kiểm tra  
X_test = dropout.forward(X, training=False)
print(f"Đơn vị bị loại lúc test: {np.sum(X_test == 0)}/{X_test.size}")
```

### Spatial Dropout cho CNN

```python
class SpatialDropout2D(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        x: (batch, channels, height, width)
        Loại bỏ toàn bộ bản đồ đặc trưng
        """
        if not self.training:
            return x
        
        # Shape mặt nạ: (batch, channels, 1, 1)
        # Cùng mặt nạ cho mọi vị trí không gian trong một kênh
        batch, channels = x.shape[:2]
        mask = torch.bernoulli(torch.ones(batch, channels, 1, 1) * 
                              (1 - self.dropout_rate))
        
        return x * mask / (1 - self.dropout_rate)

# Ví dụ
spatial_dropout = SpatialDropout2D(dropout_rate=0.3)
x = torch.randn(2, 64, 32, 32)  # Batch=2, 64 kênh, 32×32
out = spatial_dropout(x)
print(f"Toàn bộ kênh bị loại: {(out.sum(dim=(2,3)) == 0).sum().item()}")
```

## 5. Khái niệm liên quan

### Chuẩn hóa Batch (*Batch Normalization*)
- Cũng mang lại hiệu ứng chính quy hóa nhẹ nhờ nhiễu thống kê mini-batch.
- Trong nhiều kiến trúc CNN hiện đại, BatchNorm giúp giảm nhu cầu dùng dropout.
- Nếu kết hợp cả hai, thường đặt dropout sau BatchNorm và hàm kích hoạt.

### Tăng cường dữ liệu (*Data Augmentation*)
- Chính quy hóa từ phía dữ liệu bằng cách tạo thêm biến thể hợp lệ cho tập huấn luyện.
- Bổ trợ tốt cho dropout.

### Dừng sớm (*Early Stopping*)
- Dừng huấn luyện khi hàm mất mát trên tập kiểm định bắt đầu tăng liên tục, có thể kết hợp song song với dropout.

### Phương pháp tổ hợp (*Ensemble*)
- Huấn luyện nhiều mô hình độc lập rồi lấy trung bình dự đoán; dropout xấp xỉ ý tưởng này trong một mạng duy nhất.

### DropConnect
- Vô hiệu hóa ngẫu nhiên các kết nối (trọng số) thay vì toàn bộ neuron; ý tưởng tương đồng nhưng ít phổ biến hơn dropout.

## 6. Các bài báo nền tảng

**["Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)](https://jmlr.org/papers/v15/srivastava14a.html)**  
*Tác giả*: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov  
Bài báo tiêu chuẩn về dropout, phân tích cơ chế ngăn đồng thích nghi giữa các neuron và chứng minh cải thiện rõ rệt trên nhiều tập chuẩn như MNIST, CIFAR và ImageNet.

**["Improving neural networks by preventing co-adaptation of feature detectors" (2012)](https://arxiv.org/abs/1207.0580)**  
*Tác giả*: Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, và cộng sự  
Công trình sớm giới thiệu ý tưởng dropout và diễn giải kỹ thuật này như một tổ hợp gồm số lượng lớn theo hàm mũ các mạng con chia sẻ tham số.

**["Regularization of Neural Networks using DropConnect" (2013)](http://proceedings.mlr.press/v28/wan13.html)**  
*Tác giả*: Li Wan, Matthew Zeiler, Sixin Zhang, Yann LeCun, Rob Fergus  
Đề xuất DropConnect — tổng quát hóa cơ chế loại bỏ ngẫu nhiên từ cấp độ neuron sang cấp độ từng kết nối trọng số.

**["Spatial Dropout" (2015) — một phần của "Efficient Object Localization Using CNNs"](https://arxiv.org/abs/1411.4280)**  
*Tác giả*: Jonathan Tompson, Ross Goroshin, Arjun Jain, Yann LeCun, Christoph Bregler  
Giới thiệu *spatial dropout* cho tầng tích chập bằng cách loại bỏ nguyên bản đồ đặc trưng thay vì từng điểm ảnh rời rạc, phù hợp với tính tương quan không gian của CNN.

**["A Theoretically Grounded Application of Dropout in RNNs" (2016)](https://arxiv.org/abs/1512.05287)**  
*Tác giả*: Yarin Gal, Zoubin Ghahramani  
Phân tích cách dùng dropout trong RNN (đề xuất *variational dropout* giữ cố định mặt nạ qua các bước thời gian) và thiết lập liên hệ lý thuyết giữa dropout với suy luận Bayes xấp xỉ để ước lượng độ bất định.

## Cạm bẫy thường gặp và mẹo thực hành

### ⚠️ Cạm bẫy 1: Quên tắt dropout khi kiểm tra
**Vấn đề:** Dropout vẫn hoạt động lúc đánh giá khiến kết quả dự đoán biến động ngẫu nhiên.  
**Giải pháp:** Luôn chuyển sang chế độ đánh giá bằng `model.eval()`.

```python
# Sai
output = model(test_data)  # Dropout vẫn bật nếu model ở chế độ train!

# Đúng
model.eval()
with torch.no_grad():
    output = model(test_data)
```

### ⚠️ Cạm bẫy 2: Tỷ lệ dropout quá cao
**Vấn đề:** Đặt tỷ lệ quá lớn làm mất quá nhiều dung lượng biểu diễn của mạng.  
**Giải pháp:** Chọn tỷ lệ phù hợp với từng loại tầng (khoảng 0.5 cho tầng kết nối đầy đủ, thấp hơn cho tầng tích chập hoặc đầu vào).

```python
# Quá mạnh - có thể hại hiệu năng
dropout = nn.Dropout(0.9)  # Loại 90%!

# Điểm xuất phát tốt hơn
dropout_fc = nn.Dropout(0.5)      # Kết nối đầy đủ
dropout_conv = nn.Dropout(0.2)    # Tích chập
dropout_input = nn.Dropout(0.1)   # Tầng đầu vào
```

### ⚠️ Cạm bẫy 3: Kết hợp không phù hợp với Batch Normalization
**Vấn đề:** Dropout làm thay đổi phương sai kích hoạt giữa hai chế độ train và eval, dễ tương tác kém với thống kê chạy của BatchNorm.  
**Giải pháp:** Thường ưu tiên dùng BatchNorm riêng lẻ; nếu cần kết hợp, đặt dropout sau lớp BatchNorm và hàm kích hoạt.

```python
# Thực hành hiện đại: Dùng BN, bỏ dropout
x = conv(x)
x = batch_norm(x)
x = relu(x)
# Không cần dropout!

# Nếu dùng cả hai: dropout sau BN
x = conv(x)
x = batch_norm(x)
x = relu(x)
x = dropout(x)  # Sau kích hoạt
```

### ✅ Mẹo 1: Dùng tỷ lệ khác nhau cho từng tầng
```python
class SmartDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_input = nn.Dropout(0.2)  # Bảo thủ
        self.dropout_hidden = nn.Dropout(0.5)  # Chuẩn
        self.dropout_deep = nn.Dropout(0.3)    # Thấp hơn cho tầng sâu
```

### ✅ Mẹo 2: Monte Carlo Dropout (ước lượng độ bất định)
```python
def mc_dropout_predict(model, x, num_samples=10):
    """
    Nhiều lượt xuôi ngẫu nhiên để ước lượng bất định
    """
    model.train()  # Giữ dropout BẬT
    predictions = []
    
    for _ in range(num_samples):
        with torch.no_grad():
            pred = model(x)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)
    
    return mean_pred, uncertainty

# Lấy dự đoán kèm bất định
mean, std = mc_dropout_predict(model, x_test)
print(f"Dự đoán: {mean}, Bất định: {std}")
```

### ✅ Mẹo 3: Dropout theo lịch trình (*Scheduled Dropout*)
```python
class ScheduledDropout:
    """Tăng dần tỷ lệ dropout trong quá trình huấn luyện"""
    def __init__(self, initial_rate=0.1, final_rate=0.5, total_epochs=100):
        self.initial = initial_rate
        self.final = final_rate
        self.total = total_epochs
    
    def get_rate(self, epoch):
        progress = min(epoch / self.total, 1.0)
        return self.initial + (self.final - self.initial) * progress
```

## Điểm chính

- **Dropout** vô hiệu hóa ngẫu nhiên một phần neuron trong lúc huấn luyện để ngăn đồng thích nghi và tạo hiệu ứng tổ hợp.
- **Tỷ lệ tham khảo**: khoảng 0.5 cho tầng kết nối đầy đủ, 0.1–0.3 cho tầng tích chập và khoảng 0.1–0.2 cho tầng đầu vào.
- **Chia tỉ lệ (inverted dropout)**: nhân các đơn vị giữ lại với $$\frac{1}{1-p}$$ khi huấn luyện để giữ nguyên kỳ vọng khi suy luận.
- **Chế độ vận hành**: luôn nhớ chuyển `model.train()` khi huấn luyện và `model.eval()` khi đánh giá.
- **Bối cảnh hiện đại**: ít dùng nặng ở các tầng tích chập đã có BatchNorm, nhưng vẫn là kỹ thuật nền tảng cần nắm vững.

Tiếp theo là bài về **Chuẩn hóa Batch (Batch Normalization)** — kỹ thuật chuẩn hóa kích hoạt giúp ổn định và tăng tốc quá trình huấn luyện mạng sâu.

## Video tham khảo

- **Xem trước:** [Dropout Regularization (C2W1L06)](https://www.youtube.com/watch?v=D8PJAL-MZv8) — DeepLearningAI (tiếng Anh). Giải thích cách tạo mặt nạ và hệ số inverted dropout trước khi cài đặt.
- [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering (tiếng Anh). Minh họa cách tích hợp dropout đúng chế độ train/eval trong vòng huấn luyện.
- [Regularization (C2W1L04)](https://www.youtube.com/watch?v=6g0t3Phly2M) — DeepLearningAI (tiếng Anh). Nêu rõ lưu ý khi đánh giá mô hình có sử dụng dropout.
