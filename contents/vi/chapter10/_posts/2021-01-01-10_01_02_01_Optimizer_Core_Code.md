---
layout: post
title: 10-01-02-01 Cài đặt Cốt lõi Bộ tối ưu
chapter: '10'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter10
---

## 4. Đoạn Mã

Hãy cài đặt các bộ tối ưu từ đầu để hiểu cơ chế của chúng:

```python
import numpy as np
import matplotlib.pyplot as plt

class SGDMomentum:
    """
    Stochastic Gradient Descent với Momentum.
    
    Duy trì trung bình có trọng số mũ của gradient (vận tốc)
    và dùng nó cho cập nhật thay vì gradient thô. Tăng tốc
    theo hướng nhất quán, dập dao động.
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        """
        params: danh sách mảng tham số cần tối ưu
        lr: learning rate
        momentum: hệ số cho vận tốc (β trong phương trình)
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        
        # Khởi tạo vận tốc bằng zero
        # Mỗi tham số có vận tốc riêng cùng shape
        self.velocities = [np.zeros_like(p) for p in params]
    
    def step(self, grads):
        """
        Cập nhật tham số bằng momentum.
        
        grads: danh sách gradient (cùng cấu trúc với params)
        
        Cập nhật vận tốc v_t = β*v_{t-1} + g_t tạo trọng số mũ:
        gradient gần đây đóng góp đầy đủ, gradient cũ hơn
        đóng góp với trọng số β^k. β=0.9 điển hình nghĩa là ta
        hiệu quả trung bình ~10 gradient gần nhất.
        """
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Cập nhật vận tốc: trung bình trượt mũ của gradient
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            
            # Cập nhật tham số bằng vận tốc
            # Lưu ý: một số công thức dùng (1-β)*g thay vì g
            # Ta theo quy ước PyTorch
            param -= self.lr * self.velocities[i]

class RMSprop:
    """
    RMSprop: Root Mean Square Propagation.
    
    Thích nghi learning rate theo từng tham số dựa trên trung bình
    trượt mũ của bình phương gradient. Tham số với gradient
    liên tục lớn nhận learning rate hiệu dụng nhỏ hơn.
    """
    
    def __init__(self, params, lr=0.001, beta=0.9, epsilon=1e-8):
        """
        beta: tỷ lệ suy giảm cho trung bình bình phương gradient
        epsilon: hằng số nhỏ cho ổn định số
        """
        self.params = params
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        
        # Khởi tạo trung bình bình phương gradient
        self.sq_grads = [np.zeros_like(p) for p in params]
    
    def step(self, grads):
        """
        Cập nhật bằng learning rate thích nghi.
        
        Phép chia bởi √E[g²] nghĩa là tham số với gradient điển hình lớn
        nhận cập nhật nhỏ hơn (để ngăn bất ổn), trong khi tham số với
        gradient điển hình nhỏ nhận cập nhật lớn hơn (để tiến).
        """
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Cập nhật trung bình trượt bình phương gradient
            # E[g²]_t = β*E[g²]_{t-1} + (1-β)*g²_t
            self.sq_grads[i] = (self.beta * self.sq_grads[i] + 
                               (1 - self.beta) * grad**2)
            
            # Learning rate thích nghi: lr / √E[g²]
            # Thêm epsilon ngăn chia cho zero
            adapted_lr = self.lr / (np.sqrt(self.sq_grads[i]) + self.epsilon)
            
            # Cập nhật tham số
            param -= adapted_lr * grad

class Adam:
    """
    Adam: Adaptive Moment Estimation.
    
    Kết hợp momentum (moment bậc nhất) và RMSprop (moment bậc hai).
    Bao gồm hiệu chỉnh độ lệch cho hành vi đúng đầu huấn luyện.
    Bộ tối ưu chuẩn thực tế cho nhiều tác vụ học sâu.
    """
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        beta1: tỷ lệ suy giảm moment bậc nhất (momentum)
        beta2: tỷ lệ suy giảm moment bậc hai (RMSprop)
        
        Giá trị mặc định hoạt động tốt trên nhiều tác vụ — sức mạnh
        của Adam là độ vững với lựa chọn siêu tham số.
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Khởi tạo moment
        self.m = [np.zeros_like(p) for p in params]  # Moment bậc nhất
        self.v = [np.zeros_like(p) for p in params]  # Moment bậc hai
        
        self.t = 0  # Bước thời gian (cho hiệu chỉnh độ lệch)
    
    def step(self, grads):
        """
        Cập nhật Adam với hiệu chỉnh độ lệch.
        
        Hiệu chỉnh độ lệch then chốt đầu huấn luyện khi m_t và v_t
        lệch về zero. Không hiệu chỉnh, cập nhật sớm quá nhỏ,
        làm chậm huấn luyện ban đầu. Hệ số hiệu chỉnh 1/(1-β^t)
        tăng khi t tăng, rồi tiến về 1.
        """
        self.t += 1  # Tăng bước thời gian
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Cập nhật ước lượng moment bậc nhất lệch (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Cập nhật ước lượng moment bậc hai lệch (RMSprop)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Tính moment đã hiệu chỉnh độ lệch
            # Các hiệu chỉnh này lớn nhất sớm (khi t nhỏ)
            # và tiến về 1 khi t → ∞
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Cập nhật tham số
            # Kết hợp hướng momentum (m_hat) với scale thích nghi (√v_hat)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdamW:
    """
    AdamW: Adam với weight decay tách rời.
    
    Tách chính quy hóa L2 khỏi tối ưu dựa trên gradient.
    Tổng quát hóa tốt hơn Adam, đặc biệt với Transformer.
    """
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, grads):
        """
        Cập nhật Adam với weight decay tách rời.
        
        Khác biệt then chốt so với Adam: weight decay được áp dụng trực tiếp
        lên tham số (θ ← θ - λθ) thay vì được cộng vào gradient.
        Điều này đảm bảo cường độ chính quy hóa độc lập với scale
        learning rate thích nghi.
        """
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Cập nhật moment (giống Adam)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Hiệu chỉnh độ lệch
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Cập nhật với weight decay tách rời
            # Weight decay xảy ra ngoài scale thích nghi
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + 
                               self.weight_decay * param)
