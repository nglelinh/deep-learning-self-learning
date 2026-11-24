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

Nếu bạn có 10 triệu ảnh huấn luyện, việc tính gradient trên toàn bộ 10 triệu ảnh rồi mới nhích trọng số đi đúng **1 bước** (Batch GD) sẽ chậm tới mức không thể chấp nhận; ngược lại, cập nhật ngay sau từng ảnh đơn lẻ (Pure SGD) lại quá nhiễu và lãng phí 99% sức mạnh song song của GPU. Câu trả lời thực chiến nằm ở **Mini-Batch Gradient Descent** kết hợp **Lịch điều chỉnh Tốc độ Học (*Learning Rate Schedule*)**. Bài học này mổ xẻ cơ chế toán học của cả ba biến thể và nghệ thuật chọn, làm ấm (*warmup*) và giảm dần (*decay*) tốc độ học.

## Bản đồ kiến thức nền: Bốn trụ cột về Biến thể GD và Learning Rate

1. **Batch vs. Stochastic (SGD) vs. Mini-Batch GD**:
   So sánh trên 3 trục: phương sai ước lượng gradient (tỷ lệ $$\propto 1/B$$), tần suất cập nhật mỗi epoch ($$\lceil N/B \rceil$$ bước), và khả năng khai thác phép nhân ma trận song song (GEMM) trên GPU.

2. **Tác dụng Bất ngờ của Nhiễu Mini-Batch (Implicit Regularization)**:
   Batch nhỏ (32–256) tạo nhiễu giúp mô hình nhảy ra khỏi các cực tiểu sắc nhọn (*sharp minima*) và dừng ở cực tiểu phẳng rộng (*flat minima*), nhờ đó khái quát hóa tốt hơn trên tập test!

3. **Lịch trình Learning Rate (Step Decay, Cosine Annealing, Warmup)**:
   Giai đoạn đầu cần $$\eta$$ lớn để vượt nhanh qua vùng phẳng và khám phá toàn cục (sau vài bước *linear warmup* tránh sốc khởi tạo); giai đoạn cuối hạ nhỏ $$\eta$$ theo đường cong Cosine để hạ cánh êm xuống đáy.

4. **Quy tắc Mở rộng Tuyến tính (Linear Scaling Rule)**:
   Khi tăng kích thước batch lên $$k$$ lần để huấn luyện phân tán trên nhiều GPU, phương sai gradient giảm $$k$$ lần nên ta thường tăng learning rate tương ứng (cỡ $$k$$ hoặc $$\sqrt{k}$$ lần kèm warmup).

### Khung giải thích (nên đọc chậm): Tính tay tốc độ cập nhật và Cosine Decay Learning Rate

**Phần A — So sánh số lần cập nhật trọng số trong 1 Epoch** với tập dữ liệu $$N = 50{,}000$$ mẫu:
- **Batch GD** ($$B = 50{,}000$$): đúng **1 lần cập nhật / epoch** (rất mượt nhưng siêu chậm).
- **Mini-Batch GD** ($$B = 64$$): $$\lceil 50{,}000 / 64 \rceil = \mathbf{782}$$ lần cập nhật / epoch (cân bằng tối ưu).
- **Pure SGD** ($$B = 1$$): **50{,}000 lần cập nhật / epoch** (rất nhiễu, không tận dụng được GPU).

**Phần B — Tính tay Learning Rate theo lịch Cosine Annealing** từ $$\eta_{\max} = 0.01$$ xuống $$\eta_{\min} = 0$$ qua $$T = 100$$ epoch:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right).$$
- Tại $$t = 0$$: $$\eta_0 = 0.005(1 + \cos 0) = \mathbf{0.0100}$$.
- Tại $$t = 50$$ (nửa chặng đường): $$\eta_{50} = 0.005(1 + \cos \frac{\pi}{2}) = \mathbf{0.0050}$$.
- Tại $$t = 75$$: $$\eta_{75} = 0.005(1 + \cos \frac{3\pi}{4}) \approx 0.005(1 - 0.7071) = \mathbf{0.00146}$$.
- Tại $$t = 100$$: $$\eta_{100} = 0.005(1 + \cos\pi) = \mathbf{0.0000}$$.
Đường cong Cosine giảm chậm lúc đầu (giữ sức khám phá), giảm nhanh ở giữa và giảm cực êm ở cuối để tinh chỉnh trọng số sát đáy nhất!

Bài này giới thiệu gradient descent — thuật toán tối ưu nền tảng dùng để huấn luyện mạng neuron, cùng các biến thể chính và cách điều chỉnh tốc độ học (*learning rate*).

---

## Bài toán Tối ưu trong Học sâu

Huấn luyện mạng neuron về bản chất là một **bài toán tối ưu**: tìm bộ tham số $$\theta = \{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \ldots, \mathbf{W}^{[L]}, \mathbf{b}^{[L]}\}$$ sao cho hàm chi phí đạt cực tiểu:

$$\theta^* = \arg\min_{\theta} J(\theta)$$

trong đó $$J(\theta)$$ là giá trị mất mát trung bình trên toàn bộ tập dữ liệu huấn luyện.

## Gradient Descent: Ý tưởng Cốt lõi

![Gradient Descent Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/600px-Gradient_descent.svg.png)
*Hình ảnh: Minh họa thuật toán Gradient Descent trên bề mặt hàm mất mát. Nguồn: Wikimedia Commons*

**Gradient descent** là thuật toán tối ưu lặp, cập nhật tham số theo hướng làm giảm hàm chi phí nhanh nhất tại điểm hiện tại.

### Gradient

**Gradient** $$\nabla_{\theta} J(\theta)$$ là vector tập hợp các đạo hàm riêng của hàm chi phí theo từng tham số:

$$\nabla_{\theta} J = \begin{bmatrix} \frac{\partial J}{\partial \theta_1} \\ \frac{\partial J}{\partial \theta_2} \\ \vdots \\ \frac{\partial J}{\partial \theta_n} \end{bmatrix}$$

**Tính chất quan trọng:** Vector gradient chỉ theo hướng **tăng dốc nhất** của hàm số. Do đó, hướng ngược lại ($$-\nabla_{\theta} J$$) là hướng **giảm dốc nhất**.

### Quy tắc Cập nhật

Công thức cập nhật tham số của gradient descent là:

$$\theta := \theta - \eta \nabla_{\theta} J(\theta)$$

trong đó:
- $$\eta$$ là **tốc độ học** (*learning rate*), một siêu tham số dương
- $$:=$$ ký hiệu phép gán cập nhật
- $$\nabla_{\theta} J(\theta)$$ là gradient của hàm chi phí tại $$\theta$$

**Áp dụng cho từng tham số trong mạng neuron:**

$$\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \eta \frac{\partial J}{\partial \mathbf{W}^{[l]}}$$

$$\mathbf{b}^{[l]} := \mathbf{b}^{[l]} - \eta \frac{\partial J}{\partial \mathbf{b}^{[l]}}$$

### Trực giác Hình học

Có thể hình dung thuật toán giống như việc đứng trên sườn núi và tìm đường đi xuống thung lũng (cực tiểu):
1. Quan sát độ dốc xung quanh vị trí hiện tại (tính gradient).
2. Bước đi xuống theo hướng dốc nhất (hướng ngược chiều gradient).
3. Lặp lại quá trình cho tới khi chạm đáy (hội tụ).

Trong phép ví von này, tốc độ học $$\eta$$ quyết định **độ dài của mỗi bước chân**.

## Các Biến thể của Gradient Descent

### 1. Batch Gradient Descent

Sử dụng **toàn bộ mẫu huấn luyện** để tính gradient trước mỗi lần cập nhật tham số.

**Thuật toán:**
```
Lặp đến khi hội tụ:
    1. Tính gradient dùng tất cả m mẫu:
       ∇J(θ) = (1/m) Σᵢ₌₁ᵐ ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
    
    2. Cập nhật tham số:
       θ := θ - η ∇J(θ)
```

**Ưu điểm:**
- Đảm bảo hội tụ tới cực tiểu toàn cục với hàm lồi
- Hướng cập nhật chính xác và ổn định
- Dễ phân tích lý thuyết về tốc độ hội tụ

**Nhược điểm:**
- **Rất chậm** trên tập dữ liệu lớn vì phải duyệt toàn bộ dữ liệu cho mỗi bước cập nhật
- Đòi hỏi bộ nhớ lớn để xử lý toàn bộ tập dữ liệu
- Có thể bị kẹt tại cực tiểu cục bộ hoặc điểm yên ngựa với hàm không lồi

### 2. Stochastic Gradient Descent (SGD)

Tính gradient và cập nhật tham số dựa trên **từng mẫu huấn luyện ngẫu nhiên**.

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
- Tốc độ cập nhật **nhanh hơn nhiều**, mô hình bắt đầu cải thiện ngay trong epoch đầu tiên
- Nhiễu trong ước lượng gradient có thể giúp thoát khỏi cực tiểu cục bộ nông
- Hỗ trợ học trực tuyến (*online learning*) theo luồng dữ liệu
- Tiết kiệm bộ nhớ

**Nhược điểm:**
- **Ước lượng gradient nhiều nhiễu**, làm quỹ đạo hội tụ dao động mạnh
- Thường dao động quanh cực tiểu thay vì dừng hẳn tại điểm cực tiểu
- Khó tận dụng khả năng tính toán song song của phần cứng hiện đại

### 3. Mini-Batch Gradient Descent (Phổ biến nhất)

Dùng một **nhóm nhỏ mẫu dữ liệu** (*mini-batch*, thường gồm 32, 64, 128 hoặc 256 mẫu) cho mỗi bước tính gradient.

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
- **Cân bằng ưu điểm của cả hai phương pháp:** Cập nhật nhanh đồng thời giữ độ ổn định tốt
- **Dễ song song hóa:** Tận dụng hiệu quả năng lực tính toán ma trận của GPU/TPU
- Giảm phương sai của ước lượng gradient so với SGD đơn lẻ
- Tiết kiệm bộ nhớ vì chỉ nạp từng batch một

**Nhược điểm:**
- Thêm kích thước batch (*batch size*) làm một siêu tham số cần chọn
- Vẫn tồn tại một mức nhiễu nhất định (nhưng thấp hơn nhiều so với SGD)

### Bảng So sánh

| Biến thể | Mẫu mỗi lần cập nhật | Tốc độ | Độ ổn định | Bộ nhớ | Khả năng song song hóa |
|---------|-------------------|-------|-----------|---------|-----------------|
| Batch GD | Tất cả (m) | Chậm | Cao | Cao | Khó |
| SGD | 1 | Nhanh | Thấp | Thấp | Khó |
| Mini-batch GD | Kích thước batch (B) | **Nhanh** | **Trung bình** | **Thấp** | **Dễ** |

**Khuyến nghị:** Sử dụng **mini-batch gradient descent** với kích thước batch trong khoảng 32–256 làm điểm khởi đầu.

## Tốc độ Học (Learning Rate)

Tốc độ học $$\eta$$ là một trong những siêu tham số có ảnh hưởng lớn nhất tới quá trình huấn luyện.

### Ảnh hưởng của Tốc độ Học

#### Quá nhỏ ($$\eta$$ quá thấp)
- Hội tụ rất chậm
- Tốn nhiều thời gian và tài nguyên huấn luyện
- Dễ bị chững lại ở các vùng bằng phẳng (*plateau*)

#### Quá lớn ($$\eta$$ quá cao)
- Quá trình huấn luyện mất ổn định
- Bước cập nhật vượt quá điểm cực tiểu
- Giá trị mất mát có thể tăng vọt (phân kỳ)

#### Phù hợp
- Hàm mất mát giảm đều và ổn định
- Thời gian hội tụ hợp lý
- Đạt được nghiệm có chất lượng tốt

### Giá trị Khởi đầu Điển hình

- **Các mốc thường thử:** 0.001, 0.01, 0.1
- **Mạng sâu:** Thường bắt đầu trong khoảng 0.001 – 0.01
- **Mạng nông:** Có thể dùng mức cao hơn, khoảng 0.01 – 0.1

### Lịch Điều chỉnh Tốc độ Học (Learning Rate Schedules)

Thay vì giữ cố định $$\eta$$ trong suốt quá trình huấn luyện, ta thường dùng **lịch trình** thay đổi $$\eta$$ theo bước hoặc epoch:

#### 1. Giảm theo Bậc (Step Decay)

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

trong đó:
- $$\eta_0$$ là tốc độ học ban đầu
- $$\gamma \in (0, 1)$$ là hệ số suy giảm (ví dụ: 0.5)
- $$k$$ là chu kỳ giảm (ví dụ: mỗi 10 epoch)

**Ví dụ:** Bắt đầu từ 0.1 và nhân với 0.5 sau mỗi 10 epoch.

#### 2. Giảm theo Hàm mũ (Exponential Decay)

$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

trong đó $$\lambda$$ là hằng số suy giảm.

#### 3. Suy giảm $$1/t$$

$$\eta_t = \frac{\eta_0}{1 + \lambda t}$$

#### 4. Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

trong đó $$T$$ là tổng số bước lặp.

**Warm restarts:** Định kỳ đưa tốc độ học trở lại giá trị cao ban đầu để giúp mô hình thoát khỏi các vùng cực tiểu hẹp.

#### 5. Khởi động Tốc độ Học (Learning Rate Warm-up)

Bắt đầu từ tốc độ học rất nhỏ rồi tăng dần tới giá trị mục tiêu trong giai đoạn đầu:

$$\eta_t = \eta_0 \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)$$

**Trường hợp sử dụng:** Huấn luyện với kích thước batch lớn và các kiến trúc Transformer.

## Video tham khảo

- **Xem trước:** [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown, tiếng Anh. Minh họa trực quan vai trò của tốc độ học như độ dài bước đi trên bề mặt hàm chi phí.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, tiếng Anh. Tính chi tiết từng bước cập nhật bằng tay giúp phân biệt rõ batch, stochastic và mini-batch.
- [Gradient Descent là gì? | Tự học AI từ con số 0](https://www.youtube.com/watch?v=f3h5CU1808g) — Gà AI, tiếng Việt. Giải thích gradient descent qua ví dụ trực quan bằng tiếng Việt.
