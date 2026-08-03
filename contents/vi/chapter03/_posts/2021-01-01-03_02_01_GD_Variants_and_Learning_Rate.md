---
layout: post
title: 03-02-01 Các Biến thể Gradient Descent và Learning Rate
chapter: '03'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

Bài này giới thiệu gradient descent, thuật toán tối ưu nền tảng dùng để huấn luyện mạng neuron.

---

## Bài toán Tối ưu trong Học sâu

Huấn luyện mạng neuron là một **bài toán tối ưu**: Tìm tham số $$\theta = \{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \ldots, \mathbf{W}^{[L]}, \mathbf{b}^{[L]}\}$$ cực tiểu hóa hàm chi phí:

$$\theta^* = \arg\min_{\theta} J(\theta)$$

trong đó $$J(\theta)$$ là mất mát trung bình trên mọi mẫu huấn luyện.

## Gradient Descent: Ý tưởng Cốt lõi

![Gradient Descent Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/600px-Gradient_descent.svg.png)
*Hình ảnh: Minh họa Gradient Descent trên hàm mất mát. Nguồn: Wikimedia Commons*

**Gradient descent** là thuật toán tối ưu lặp, dịch tham số theo hướng giảm hàm chi phí nhanh nhất.

### Gradient

**Gradient** $$\nabla_{\theta} J(\theta)$$ là vectơ các đạo hàm riêng:

$$\nabla_{\theta} J = \begin{bmatrix} \frac{\partial J}{\partial \theta_1} \\ \frac{\partial J}{\partial \theta_2} \\ \vdots \\ \frac{\partial J}{\partial \theta_n} \end{bmatrix}$$

**Tính chất then chốt**: Gradient chỉ theo hướng **tăng dốc nhất**. Do đó, gradient âm chỉ theo hướng **giảm dốc nhất**.

### Quy tắc Cập nhật

Quy tắc cập nhật gradient descent là:

$$\theta := \theta - \eta \nabla_{\theta} J(\theta)$$

trong đó:
- $$\eta$$ là **tốc độ học** (*learning rate*) (siêu tham số vô hướng dương)
- $$:=$$ ký hiệu gán/cập nhật
- $$\nabla_{\theta} J(\theta)$$ là gradient của hàm chi phí

**Với mỗi tham số trong mạng neuron:**

$$\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \eta \frac{\partial J}{\partial \mathbf{W}^{[l]}}$$

$$\mathbf{b}^{[l]} := \mathbf{b}^{[l]} - \eta \frac{\partial J}{\partial \mathbf{b}^{[l]}}$$

### Trực giác Hình học

Hình dung ta đứng trên sườn núi và muốn xuống thung lũng (cực tiểu):
1. Kiểm tra độ dốc xung quanh (tính gradient)
2. Bước xuống dốc (theo hướng gradient âm)
3. Lặp lại cho đến khi chạm đáy (hội tụ)

Tốc độ học $$\eta$$ quyết định **kích thước bước**.

## Các Biến thể của Gradient Descent

### 1. Batch Gradient Descent (GD Gốc)

Dùng **toàn bộ mẫu huấn luyện** để tính gradient ở mỗi bước.

**Thuật toán:**
```
Lặp đến khi hội tụ:
    1. Tính gradient dùng tất cả m mẫu:
       ∇J(θ) = (1/m) Σᵢ₌₁ᵐ ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
    
    2. Cập nhật tham số:
       θ := θ - η ∇J(θ)
```

**Ưu điểm:**
- Đảm bảo hội tụ tới cực tiểu toàn cục (với hàm lồi)
- Hội tụ ổn định
- Có thể dùng các đảm bảo hội tụ lý thuyết

**Nhược điểm:**
- **Rất chậm** với tập dữ liệu lớn (phải xử lý toàn bộ dữ liệu trước một lần cập nhật)
- Yêu cầu toàn bộ tập dữ liệu trong bộ nhớ
- Có thể kẹt ở cực tiểu địa phương (với hàm không lồi)

### 2. Stochastic Gradient Descent (SGD)

Dùng **một mẫu huấn luyện ngẫu nhiên** mỗi lần để tính gradient.

**Thuật toán:**
```
Lặp đến khi hội tụ:
    1. Xáo trộn ngẫu nhiên dữ liệu huấn luyện
    
    2. Với mỗi mẫu i:
        a. Tính gradient chỉ dùng mẫu i:
           ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
        
        b. Cập nhật tham số:
           θ := θ - η ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
```

**Ưu điểm:**
- Cập nhật **nhanh hơn nhiều** (có thể bắt đầu học ngay)
- Có thể thoát cực tiểu địa phương nhờ cập nhật nhiễu
- Học trực tuyến khả thi (xử lý luồng dữ liệu)
- Tiết kiệm bộ nhớ

**Nhược điểm:**
- **Ước lượng gradient nhiễu** → quỹ đạo hội tụ thất thường
- Không thực sự "hội tụ" (dao động quanh cực tiểu)
- Khó song song hóa hơn

### 3. Mini-Batch Gradient Descent (Phổ biến nhất)

Dùng một **batch nhỏ các mẫu** (thường 32, 64, 128, hoặc 256) để tính gradient.

**Thuật toán:**
```
Lặp đến khi hội tụ:
    1. Xáo trộn ngẫu nhiên dữ liệu huấn luyện
    
    2. Chia dữ liệu thành các mini-batch kích thước B
    
    3. Với mỗi mini-batch:
        a. Tính gradient dùng batch:
           ∇J_batch(θ) = (1/B) Σᵢ∈batch ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
        
        b. Cập nhật tham số:
           θ := θ - η ∇J_batch(θ)
```

**Ưu điểm:**
- **Tốt nhất của cả hai thế giới**: Cập nhật nhanh + hội tụ ổn định
- **Song song hóa cao**: Tận dụng GPU/TPU hiệu quả
- Phương sai ước lượng gradient giảm
- Tiết kiệm bộ nhớ (xử lý batch, không phải toàn bộ tập)

**Nhược điểm:**
- Thêm kích thước batch như một siêu tham số
- Vẫn còn một số nhiễu (ít hơn SGD)

### Bảng So sánh

| Biến thể | Mẫu mỗi lần cập nhật | Tốc độ | Ổn định | Bộ nhớ | Song song hóa |
|---------|-------------------|-------|-----------|---------|-----------------|
| Batch GD | Tất cả (m) | Chậm | Cao | Cao | Khó |
| SGD | 1 | Nhanh | Thấp | Thấp | Khó |
| Mini-batch GD | Kích thước batch (B) | **Nhanh** | **Trung bình** | **Thấp** | **Dễ** |

**Khuyến nghị**: Dùng **mini-batch gradient descent** với kích thước batch 32–256.

## Tốc độ Học (Learning Rate)

Tốc độ học $$\eta$$ là một trong những siêu tham số quan trọng nhất.

### Ảnh hưởng của Learning Rate

#### Quá nhỏ ($$\eta$$ quá thấp)
- Hội tụ rất chậm
- Có thể mất quá nhiều thời gian huấn luyện
- Có thể kẹt ở plateau

#### Quá lớn ($$\eta$$ quá cao)
- Huấn luyện bất ổn
- Có thể vượt quá cực tiểu
- Mất mát có thể phân kỳ (tăng)

#### Vừa phải
- Mất mát giảm mượt, ổn định
- Thời gian huấn luyện hợp lý
- Hội tụ tới nghiệm tốt

### Giá trị Điển hình

- **Điểm bắt đầu tốt**: 0.001, 0.01, 0.1
- **Mạng sâu**: Thường 0.001 – 0.01
- **Mạng nông**: Có thể dùng tốc độ cao hơn (0.01 – 0.1)

### Lịch Learning Rate

Thay vì learning rate cố định, dùng **lịch** thay đổi $$\eta$$ trong huấn luyện:

#### 1. Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

trong đó:
- $$\eta_0$$ là learning rate ban đầu
- $$\gamma \in (0, 1)$$ là hệ số suy giảm (ví dụ: 0.5)
- $$k$$ là khoảng bước (ví dụ: mỗi 10 epoch)

**Ví dụ**: Bắt đầu ở 0.1, nhân 0.5 mỗi 10 epoch

#### 2. Exponential Decay

$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

trong đó $$\lambda$$ là hằng số suy giảm.

#### 3. Suy giảm 1/t

$$\eta_t = \frac{\eta_0}{1 + \lambda t}$$

#### 4. Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

trong đó $$T$$ là tổng số vòng lặp.

**Warm restarts**: Định kỳ đặt lại learning rate về giá trị ban đầu.

#### 5. Learning Rate Warm-up

Bắt đầu với learning rate rất nhỏ và tăng dần đến giá trị mục tiêu:

$$\eta_t = \eta_0 \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)$$

**Trường hợp dùng**: Huấn luyện batch lớn, transformer
