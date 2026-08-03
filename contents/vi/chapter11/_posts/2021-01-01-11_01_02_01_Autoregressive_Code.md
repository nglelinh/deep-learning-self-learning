---
layout: post
title: 11-01-02-01 Cài đặt Mô hình Tự hồi quy
chapter: '11'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter11
---

## 4. Đoạn Code

Hãy triển khai các cách tiếp cận mô hình hóa sinh khác nhau trên tập dữ liệu toy để hiểu cơ chế của chúng:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Sinh dữ liệu toy: hỗn hợp 8 Gaussian trên một đường tròn
def generate_mixture_data(n_samples=10000):
    """
    Sinh dữ liệu 2D từ hỗn hợp 8 Gaussian sắp xếp theo vòng tròn.
    
    Tập dữ liệu toy này cho phép trực quan hóa các phân phối đã học và so sánh
    các cách tiếp cận mô hình hóa sinh khác nhau. Mỗi Gaussian biểu diễn một
    "mode" — mô hình sinh nên học cách sinh từ mọi mode.
    """
    n_modes = 8
    radius = 2.0
    std = 0.02
    
    # Góc cho các mode cách đều quanh vòng tròn
    thetas = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
    
    # Tâm của các Gaussian
    centers = np.array([[radius * np.cos(t), radius * np.sin(t)] for t in thetas])
    
    # Lấy mẫu từ hỗn hợp
    data = []
    for _ in range(n_samples):
        # Chọn mode đều
        mode_idx = np.random.randint(n_modes)
        # Lấy mẫu từ Gaussian đã chọn
        sample = centers[mode_idx] + std * np.random.randn(2)
        data.append(sample)
    
    return np.array(data), centers

# Sinh dữ liệu huấn luyện
print("="*70)
print("Mô hình Sinh trên Tập Dữ liệu Toy 2D")
print("="*70)

data_train, true_centers = generate_mixture_data(n_samples=10000)
data_tensor = torch.FloatTensor(data_train)

print(f"Đã sinh {len(data_train)} mẫu từ 8 mode Gaussian")
print(f"Hình dạng dữ liệu: {data_train.shape}")  # (10000, 2)
print(f"Tâm các mode:\n{true_centers.round(3)}")

# 1. Mô hình tự hồi quy đơn giản
class AutoregressiveModel(nn.Module):
    """
    Mô hình tự hồi quy 2D đơn giản: p(x) = p(x2|x1) * p(x1)
    
    Mô hình hóa p(x1) như hỗn hợp logistic, p(x2|x1) như hỗn hợp điều kiện.
    Minh họa mô hình mật độ tường minh — ta có thể tính p(x) chính xác.
    """
    
    def __init__(self, n_components=10):
        super().__init__()
        
        # p(x1): hỗn hợp logistic
        self.x1_logits = nn.Parameter(torch.randn(n_components))
        self.x1_means = nn.Parameter(torch.randn(n_components))
        self.x1_scales = nn.Parameter(torch.ones(n_components) * 0.1)
        
        # p(x2|x1): mạng neuron xuất tham số hỗn hợp
        self.x2_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_components * 3)  # logits, means, scales cho hỗn hợp
        )
        
        self.n_components = n_components
    
    def log_prob(self, x):
        """
        Tính log p(x) = log p(x1) + log p(x2|x1)
        
        Đây là điều khiến đây là mô hình mật độ tường minh — ta có thể đánh giá
        xác suất của bất kỳ điểm nào, cho phép huấn luyện hợp lý cực đại.
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        # log p(x1): log hỗn hợp logistic
        logits_1 = self.x1_logits.unsqueeze(0)  # (1, n_components)
        means_1 = self.x1_means.unsqueeze(0)
        scales_1 = torch.abs(self.x1_scales).unsqueeze(0) + 0.01
        
        # Log-prob logistic cho mỗi thành phần
        z = (x1 - means_1) / scales_1
        log_probs_1 = -z - 2 * torch.nn.functional.softplus(-z) - torch.log(scales_1)
        
        # Log-prob hỗn hợp dùng log-sum-exp
        log_p_x1 = torch.logsumexp(logits_1 + log_probs_1, dim=1) - \
                   torch.logsumexp(logits_1, dim=1)
        
        # log p(x2|x1): hỗn hợp điều kiện
        params_2 = self.x2_net(x1)
        params_2 = params_2.view(-1, self.n_components, 3)
        
        logits_2 = params_2[:, :, 0]
        means_2 = params_2[:, :, 1]
        scales_2 = torch.abs(params_2[:, :, 2]) + 0.01
        
        z_2 = (x2 - means_2) / scales_2
        log_probs_2 = -z_2 - 2 * torch.nn.functional.softplus(-z_2) - torch.log(scales_2)
        
        log_p_x2_given_x1 = torch.logsumexp(logits_2 + log_probs_2, dim=1) - \
                            torch.logsumexp(logits_2, dim=1)
        
        # Tổng log-prob
        return log_p_x1 + log_p_x2_given_x1
    
    def sample(self, n_samples):
        """
        Sinh mẫu: trước hết lấy mẫu x1, rồi x2|x1
        
        Minh họa sinh tuần tự — đặc trưng của tự hồi quy.
        Lấy mẫu chính xác từ phân phối đã học.
        """
        # Lấy mẫu x1
        probs_1 = torch.softmax(self.x1_logits, dim=0)
        components = torch.multinomial(probs_1, n_samples, replacement=True)
        
        means = self.x1_means[components]
        scales = torch.abs(self.x1_scales[components])
        
        # Mẫu logistic (xấp xỉ dùng Gaussian)
        x1 = means + scales * torch.randn(n_samples)
        
        # Lấy mẫu x2|x1
        params_2 = self.x2_net(x1.unsqueeze(1))
        params_2 = params_2.view(n_samples, self.n_components, 3)
        
        # Lấy mẫu thành phần cho mỗi x1
        logits_2 = params_2[:, :, 0]
        probs_2 = torch.softmax(logits_2, dim=1)
        components_2 = torch.multinomial(probs_2, 1).squeeze()
        
        # Lấy tham số cho các thành phần đã chọn
        means_2 = params_2[range(n_samples), components_2, 1]
        scales_2 = torch.abs(params_2[range(n_samples), components_2, 2])
        
        x2 = means_2 + scales_2 * torch.randn(n_samples)
        
        return torch.stack([x1, x2], dim=1)

# Huấn luyện mô hình tự hồi quy
print("\n1. Huấn luyện Mô hình Tự hồi quy (Mật độ Tường minh)")
print("-" * 70)

ar_model = AutoregressiveModel(n_components=10)
ar_optimizer = optim.Adam(ar_model.parameters(), lr=0.001)

ar_model.train()
for epoch in range(200):
    # Xáo dữ liệu
    indices = torch.randperm(len(data_tensor))
    
    # Huấn luyện mini-batch
    batch_size = 128
    epoch_loss = 0
    
    for i in range(0, len(data_tensor), batch_size):
        batch = data_tensor[indices[i:i+batch_size]]
        
        # Tính âm log-hợp lý
        log_probs = ar_model.log_prob(batch)
        loss = -log_probs.mean()  # Negative log-likelihood
        
        ar_optimizer.zero_grad()
        loss.backward()
        ar_optimizer.step()
        
        epoch_loss += loss.item()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: NLL = {epoch_loss/(len(data_tensor)/batch_size):.4f}")

# Sinh mẫu
ar_model.eval()
with torch.no_grad():
    samples_ar = ar_model.sample(1000)
    print(f"\nĐã sinh {len(samples_ar)} mẫu")
    print(f"Mean mẫu: {samples_ar.mean(dim=0).numpy()}")
    print(f"Mean dữ liệu:   {data_tensor.mean(dim=0).numpy()}")

```
