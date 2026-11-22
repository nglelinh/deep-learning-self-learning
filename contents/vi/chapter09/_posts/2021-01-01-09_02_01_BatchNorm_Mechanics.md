---
layout: post
title: 09-02-01 Cơ chế Chuẩn hóa Batch
chapter: '09'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter09
---

## Chuẩn hóa Batch là gì?

**Chuẩn hóa Batch** (*Batch Normalization* — BatchNorm hay BN), do Ioffe và Szegedy giới thiệu năm 2015, chuẩn hóa đầu vào của từng tầng theo mỗi mini-batch về trung bình 0 và phương sai 1, sau đó áp dụng phép biến đổi tỉ lệ và tịnh tiến có tham số học được. Đây là một trong những kỹ thuật ổn định huấn luyện được sử dụng rộng rãi nhất trong học sâu.

### Vấn đề đặt ra: Dịch chuyển hiệp biến nội (*Internal Covariate Shift*)

Trong quá trình huấn luyện mạng sâu, khi trọng số ở các tầng trước liên tục cập nhật:
- Phân phối đầu vào của các tầng phía sau thay đổi theo từng bước.
- Mỗi tầng phải liên tục thích nghi với phân phối mới.
- Quá trình hội tụ chậm hơn và mạng nhạy cảm với cách khởi tạo trọng số.

**Cách tiếp cận của BatchNorm:** chuẩn hóa trực tiếp đầu vào tầng để giữ thang đo và phân phối ổn định hơn xuyên suốt quá trình huấn luyện.

## Cách BatchNorm hoạt động

### Lượt truyền xuôi khi huấn luyện

Với một mini-batch kích hoạt $$\mathbf{x} = \{x_1, x_2, \ldots, x_m\}$$:

**Bước 1: Tính trung bình và phương sai của mini-batch**

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma^2_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**Bước 2: Chuẩn hóa**

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$$

trong đó $$\epsilon$$ (ví dụ $$10^{-5}$$) là hằng số nhỏ tránh chia cho 0.

**Bước 3: Co giãn và tịnh tiến (tham số học được)**

$$y_i = \gamma \hat{x}_i + \beta$$

trong đó $$\gamma$$ (hệ số co giãn) và $$\beta$$ (độ dịch) được học qua lan truyền ngược để mạng vẫn giữ khả năng biểu diễn ban đầu khi cần.

### Khi suy luận (kiểm tra)

Thay vì tính trên batch kiểm tra, mô hình dùng thống kê tích lũy (trung bình trượt theo hàm mũ thu được trong lúc huấn luyện):

$$\hat{x} = \frac{x - \mu_{\text{pop}}}{\sqrt{\sigma^2_{\text{pop}} + \epsilon}}$$

$$y = \gamma \hat{x} + \beta$$

## Cài đặt bằng NumPy

```python
import numpy as np

class BatchNorm1D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        num_features: số đặc trưng/kênh
        eps: hằng số nhỏ cho ổn định số
        momentum: cho cập nhật trung bình/phương sai chạy
        """
        self.eps = eps
        self.momentum = momentum
        
        # Tham số học được
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Thống kê chạy (cho suy luận)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache cho backprop
        self.cache = None
    
    def forward(self, x, training=True):
        """
        x: đầu vào dạng (batch_size, num_features)
        training: có đang ở chế độ huấn luyện không
        """
        if training:
            # Tính thống kê batch
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Chuẩn hóa
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Scale và dịch
            out = self.gamma * x_normalized + self.beta
            
            # Cập nhật thống kê chạy
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
            
            # Cache cho lượt ngược
            self.cache = (x, x_normalized, batch_mean, batch_var)
            
        else:
            # Dùng thống kê chạy
            x_normalized = (x - self.running_mean) / \
                          np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, dout):
        """
        Lan truyền ngược qua chuẩn hóa batch
        dout: gradient từ tầng sau
        """
        x, x_normalized, mean, var = self.cache
        N, D = x.shape
        
        # Gradient của tham số
        self.dgamma = np.sum(dout * x_normalized, axis=0)
        self.dbeta = np.sum(dout, axis=0)
        
        # Gradient của x đã chuẩn hóa
        dx_normalized = dout * self.gamma
        
        # Gradient của phương sai
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * \
                     (var + self.eps)**(-1.5), axis=0)
        
        # Gradient của trung bình
        dmean = np.sum(dx_normalized * -1 / np.sqrt(var + self.eps), axis=0) + \
                dvar * np.mean(-2 * (x - mean), axis=0)
        
        # Gradient của x
        dx = dx_normalized / np.sqrt(var + self.eps) + \
             dvar * 2 * (x - mean) / N + \
             dmean / N
        
        return dx

# Ví dụ sử dụng
batch_norm = BatchNorm1D(num_features=128)

# Huấn luyện
x_train = np.random.randn(32, 128)
out_train = batch_norm.forward(x_train, training=True)
print(f"Trung bình đầu ra train: {np.mean(out_train, axis=0)[:5]}")
print(f"Độ lệch chuẩn đầu ra train: {np.std(out_train, axis=0)[:5]}")

# Kiểm tra
x_test = np.random.randn(10, 128)
out_test = batch_norm.forward(x_test, training=False)
print(f"Test dùng thống kê chạy")
```

## Vì sao Chuẩn hóa Batch hiệu quả

### 1. Ổn định phân phối kích hoạt giữa các tầng
Giữ cho thang đo đầu vào của từng tầng nhất quán hơn qua các bước cập nhật, giảm bớt tác động của sự thay đổi phân phối ở các tầng trước.

### 2. Cho phép dùng tốc độ học lớn hơn
Luồng gradient ổn định hơn giúp tăng tốc độ học và rút ngắn thời gian hội tụ đáng kể, đồng thời giảm độ nhạy cảm với cách khởi tạo trọng số.

### 3. Tạo hiệu ứng chính quy hóa nhẹ
Thống kê trung bình và phương sai ước lượng trên từng mini-batch mang một lượng nhiễu nhỏ, góp phần hạn chế quá khớp và giảm nhu cầu dùng dropout ở một số kiến trúc.

### 4. Làm trơn bề mặt hàm mất mát
Các phân tích lý thuyết và thực nghiệm cho thấy BatchNorm làm bề mặt tối ưu mượt hơn, giúp hướng gradient ổn định và dễ dự đoán hơn trong quá trình tối ưu hóa.

## Video tham khảo

- **Xem trước:** [Why Does Batch Norm Work? (C2W3L06)](https://www.youtube.com/watch?v=nUUqwaxLnWs) — DeepLearningAI (tiếng Anh). Trình bày công thức chuẩn hóa theo trung bình–phương sai và cơ chế co giãn lại bằng $$\gamma, \beta$$.
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) — Andrej Karpathy (tiếng Anh). Minh họa trực tiếp cách tính thống kê batch theo từng kênh trong mã nguồn.
- [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering (tiếng Anh). Phân tích vai trò ổn định huấn luyện của BatchNorm trong bài giảng mạng sâu.
