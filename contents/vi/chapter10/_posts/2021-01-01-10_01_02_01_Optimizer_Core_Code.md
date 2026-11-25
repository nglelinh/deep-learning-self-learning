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

Viết lại trọn bộ SGD, Momentum, Nesterov, AdaGrad, RMSprop và Adam bằng NumPy thuần chỉ tốn chưa đầy 100 dòng code, nhưng giá trị nhận lại cực kỳ lớn: bạn sẽ hiểu chính xác biến đếm bước `self.t` tăng ở đâu trong Adam, tại sao phép toán `in-place` tiết kiệm bộ nhớ GPU, và cách viết dạng triển khai tương đương của Nesterov mà không phải gọi thêm một lượt forward/backward thứ hai!

## Kiến thức cần nhớ: Bốn chi tiết kỹ thuật khi viết mã lõi Bộ tối ưu

1. **Quản lý trạng thái per-parameter theo danh sách/từ điển**:
   Mỗi tensor trọng số (ma trận $$W_1$$, vector $$b_1$$, $$W_2$$, ...) cần các bộ đệm `m` và `v` có cùng chính xác shape, khởi tạo bằng `np.zeros_like(param)`.
2. **Dạng cài đặt 1-pass của Nesterov Momentum**:
   Thay vì tính gradient ở điểm dự đoán $$\theta - \eta \beta v$$ (rất bất tiện với API tự động đạo hàm), các framework như PyTorch dùng phép đổi biến tương đương để chỉ cần gọi gradient tại điểm hiện tại nhưng vẫn tạo ra hiệu ứng "nhìn trước" giống hệt.
3. **Thứ tự tăng bộ đếm `t += 1` trong Adam**:
   Phải tăng `self.t += 1` **trước** khi tính `1 - beta1 ** self.t` và `1 - beta2 ** self.t`; nếu để `t = 0` ở bước đầu sẽ lập tức xảy ra lỗi chia cho 0 (`ZeroDivisionError` / `NaN`)!
4. **Hằng số $$\epsilon$$ đặt ngoài dấu căn bậc hai**:
   Trong cài đặt Adam chuẩn (`m_hat / (np.sqrt(v_hat) + eps)`), $$\epsilon$$ nằm ngoài dấu căn (mặc định `1e-8`, nhưng khi huấn luyện FP16/BF16 nên tăng lên `1e-6` để tránh underflow làm tròn về 0).

### Giải thích: Tính tay so sánh tốc độ giảm bước của AdaGrad và RMSprop sau 3 bước gradient lớn

Giả sử một tham số liên tục nhận gradient $$g_1 = g_2 = g_3 = 10{,}0$$, tốc độ học cơ sở $$\eta = 1{,}0$$, và hệ số suy giảm của RMSprop là $$\rho = 0{,}9$$ (bỏ qua $$\epsilon$$ nhỏ):

| Bước $$t$$ | Tích lũy AdaGrad $$G_t = \sum g_i^2$$ | Bước nhảy AdaGrad $$\frac{\eta g_t}{\sqrt{G_t}}$$ | EMA RMSprop $$s_t = 0{,}9 s_{t-1} + 0{,}1 g_t^2$$ | Bước nhảy RMSprop $$\frac{\eta g_t}{\sqrt{s_t}}$$ |
| :---: | :---: | :---: | :---: | :---: |
| $$t = 1$$ | $$100$$ | $$\frac{10}{\sqrt{100}} = \mathbf{1{,}000}$$ | $$10{,}0$$ | $$\frac{10}{\sqrt{10}} \approx \mathbf{3{,}162}$$ |
| $$t = 2$$ | $$200$$ | $$\frac{10}{\sqrt{200}} \approx \mathbf{0{,}707}$$ | $$19{,}0$$ | $$\frac{10}{\sqrt{19}} \approx \mathbf{2{,}294}$$ |
| $$t = 3$$ | $$300$$ | $$\frac{10}{\sqrt{300}} \approx \mathbf{0{,}577}$$ | $$27{,}1$$ | $$\frac{10}{\sqrt{27{,}1}} \approx \mathbf{1{,}921}$$ |
| $$t \to \infty$$ | $$\to \infty$$ | **$$\to 0$$ (tắt hẳn!)** | Hội tụ về $$100{,}0$$ | **Hội tụ về $$1{,}000$$ (ổn định!)** |

Khi $$t$$ lớn, mẫu số của AdaGrad phình vô hạn kéo tốc độ học về 0, còn mẫu số của RMSprop tự cân bằng đúng về giá trị bình phương gradient hiện hành ($$10^2 = 100$$) giúp mô hình tiếp tục học bền bỉ!

## 4. Đoạn mã

Dưới đây là phần cài đặt các bộ tối ưu từ đầu bằng NumPy để minh họa trực tiếp cơ chế cập nhật của từng thuật toán:

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
        lớn ở các bước đầu và giảm dần về 1 khi t tăng.
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
        lên tham số (θ ← θ - ηλθ) thay vì được cộng vào gradient.
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
```

## Video tham khảo

- **Xem trước:** [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, tiếng Anh. Nắm rõ hai bước cập nhật moment trước khi dùng các lớp cài đặt sẵn trong framework.
- [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, tiếng Anh. Giúp phân biệt rõ hai bộ đệm riêng biệt của momentum và RMSprop khi tự cài đặt.
- [PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) — Patrick Loeber, tiếng Anh. Cách gọi `step()` trong PyTorch thay cho đoạn cập nhật viết tay sau khi đã hiểu cơ chế bên trong.
