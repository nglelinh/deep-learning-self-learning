---
layout: post
title: 02-03-01 Hàm Kích hoạt Cổ điển
chapter: '02'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

Bài này trình bày sâu về hàm kích hoạt (*activation functions*), các tính chất của chúng, và cách chọn hàm phù hợp cho mạng neuron.

---

## Vì sao Hàm Kích hoạt Quan trọng

**Không có hàm kích hoạt**, mạng neuron chỉ học được các biến đổi tuyến tính. Dù xếp bao nhiêu lớp, hợp thành của các hàm tuyến tính vẫn là tuyến tính:

$$f(g(h(\mathbf{x}))) = \mathbf{A}_1(\mathbf{A}_2(\mathbf{A}_3 \mathbf{x})) = (\mathbf{A}_1 \mathbf{A}_2 \mathbf{A}_3)\mathbf{x} = \mathbf{A}\mathbf{x}$$

**Hàm kích hoạt đưa vào phi tuyến tính**, cho phép mạng học các mẫu phức tạp và xấp xỉ các hàm tùy ý.

## Các Tính chất Mong muốn của Hàm Kích hoạt

Một hàm kích hoạt lý tưởng nên có:

1. **Phi tuyến tính**: Cho phép học các mẫu phức tạp
2. **Khả vi**: Cho phép học dựa trên gradient
3. **Đơn điệu**: Bảo toàn thứ tự (hữu ích cho tối ưu)
4. **Hiệu quả tính toán**: Nhanh khi tính xuôi và ngược
5. **Bị chặn hoặc không bị chặn phù hợp**: Tùy theo tác vụ
6. **Tâm tại không**: Hỗ trợ luồng gradient (cho lớp ẩn)
7. **Tránh bão hòa**: Ngăn gradient biến mất

Không có hàm kích hoạt nào thỏa mãn hoàn hảo mọi tính chất, nên lựa chọn phụ thuộc trường hợp sử dụng cụ thể.

## Các Hàm Kích hoạt Phổ biến

### 1. Sigmoid (Hàm Logistic)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Đạo hàm:**

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Tính chất:**
- Miền giá trị: $$(0, 1)$$
- Trơn, khả vi mọi nơi
- Đơn điệu tăng
- Bão hòa ở cả hai đầu

**Ưu điểm:**
- Diễn giải xác suất rõ ràng
- Gradient trơn
- Từng phổ biến lịch sử

**Nhược điểm:**
- **Bài toán gradient biến mất**: Gradient gần 0 khi $$|z| > 4$$
- **Không tâm tại không**: Đầu ra luôn dương
- **Tốn tính toán**: Cần tính hàm mũ

**Trường hợp dùng:**
- Lớp đầu ra cho phân loại nhị phân
- Kích hoạt cổng trong LSTM
- Thường tránh dùng ở lớp ẩn của mạng sâu

### 2. Tang Hyperbolic (tanh)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1} = 2\sigma(2z) - 1$$

**Đạo hàm:**

$$\tanh'(z) = 1 - \tanh^2(z)$$

**Tính chất:**
- Miền giá trị: $$(-1, 1)$$
- Tâm tại không (cải thiện so với sigmoid)
- Bão hòa ở cả hai đầu

**Ưu điểm:**
- Tâm tại không (luồng gradient tốt hơn)
- Gradient mạnh hơn sigmoid (miền đạo hàm: $$(0, 1]$$)

**Nhược điểm:**
- Vẫn chịu gradient biến mất
- Tốn tính toán

**Trường hợp dùng:**
- Lớp ẩn (tốt hơn sigmoid nhưng kém hơn ReLU)
- Ô RNN/LSTM
- Khi đầu ra tâm tại không mang lại lợi ích

### 3. Rectified Linear Unit (ReLU)

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

![Đồ thị hàm ReLU](/deep-learning-self-learning/img/chapter_img/chapter02/relu_function.jpg)
*Hình: $$\mathrm{ReLU}(x)=\max(0,x)=(x)^+$$ — đoạn $$y=0$$ với $$x<0$$ và $$y=x$$ với $$x\geq 0$$. (Minh họa từ video nhập môn neural network)*

**Đạo hàm:**

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \\ \text{undefined} & \text{if } z = 0 \end{cases}$$

(Trong thực tế, ta định nghĩa $$\text{ReLU}'(0) = 0$$ hoặc $$0.5$$)

**Tính chất:**
- Miền giá trị: $$[0, \infty)$$
- Không bão hòa với giá trị dương
- Kích hoạt thưa (nhiều neuron cho ra 0)

**Ưu điểm:**
- **Hiệu quả tính toán**: Chỉ cần ngưỡng hóa tại không
- **Giảm gradient biến mất**: Gradient bằng 1 với đầu vào dương
- **Biểu diễn thưa**: Độ thưa tự nhiên
- **Thành công thực nghiệm**: Hoạt động rất tốt trong thực tế

**Nhược điểm:**
- **Không tâm tại không**: Mọi đầu ra đều không âm
- **Bài toán ReLU chết** (*dying ReLU*): Neuron có thể trở nên không hoạt động vĩnh viễn
  - Nếu $$z < 0$$ luôn, gradient luôn bằng 0, không có học
  - Có thể xảy ra với learning rate cao hoặc khởi tạo kém

**Trường hợp dùng:**
- **Lựa chọn mặc định** cho lớp ẩn trong mạng sâu
- CNN, ResNet, hầu hết kiến trúc hiện đại

### 4. Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

trong đó $$\alpha$$ là hằng số nhỏ (thường 0.01)

**Đạo hàm:**

$$\text{LeakyReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

**Ưu điểm:**
- **Ngăn ReLU chết**: Vẫn có gradient nhỏ với đầu vào âm
- Hiệu quả tính toán
- Giữ mọi lợi ích của ReLU

**Nhược điểm:**
- Thêm siêu tham số $$\alpha$$
- Không phải lúc nào cũng tốt hơn ReLU trong thực tế

**Biến thể:**
- **Parametric ReLU (PReLU)**: $$\alpha$$ được học trong huấn luyện
- **Randomized Leaky ReLU (RReLU)**: $$\alpha$$ được lấy mẫu ngẫu nhiên khi huấn luyện

### 5. Exponential Linear Unit (ELU)

$$\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

trong đó $$\alpha > 0$$ (thường $$\alpha = 1$$)

**Đạo hàm:**

$$\text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha e^z = \text{ELU}(z) + \alpha & \text{if } z \leq 0 \end{cases}$$

**Tính chất:**
- Trơn mọi nơi
- Giá trị âm đẩy trung bình kích hoạt gần không hơn

**Ưu điểm:**
- **Gần tâm tại không hơn**: Cho phép đầu ra âm
- **Không có ReLU chết**: Gradient tồn tại với mọi đầu vào
- **Trơn**: Phong cảnh tối ưu tốt hơn
- Thường dẫn đến học nhanh hơn và hiệu năng tốt hơn

**Nhược điểm:**
- Tốn tính toán hơn (hàm mũ)
- Thêm siêu tham số $$\alpha$$

**Trường hợp dùng:**
- Thay thế ReLU khi chấp nhận thêm chi phí tính toán
- Tác vụ mà kích hoạt tâm tại không mang lại lợi ích

### 6. Scaled Exponential Linear Unit (SELU)

$$\text{SELU}(z) = \lambda \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

trong đó $$\lambda \approx 1.0507$$ và $$\alpha \approx 1.6733$$

**Tính chất:**
- **Tự chuẩn hóa** (*self-normalizing*): Dưới một số điều kiện, kích hoạt tự hội tụ về trung bình không và phương sai đơn vị
- Yêu cầu khởi tạo cụ thể (LeCun normal)
- Yêu cầu kiến trúc cụ thể (lớp liên kết đầy đủ)

**Ưu điểm:**
- Có thể cho phép mạng rất sâu không cần batch normalization
- Có đảm bảo lý thuyết về hội tụ

**Nhược điểm:**
- Yêu cầu nghiêm ngặt về kiến trúc mạng
- Chưa được áp dụng rộng rãi
- Không hoạt động tốt với dropout hoặc lớp tích chập

### 7. Swish (SiLU - Sigmoid Linear Unit)

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

**Tính chất:**
- Trơn, không đơn điệu
- Không bị chặn phía trên, bị chặn phía dưới
- Tự cổng (*self-gated*): đầu vào được điều chế bởi sigmoid của chính nó

**Ưu điểm:**
- **Hiệu năng tốt hơn**: Thực nghiệm cho thấy vượt ReLU trên một số tác vụ
- Gradient trơn
- Tính không đơn điệu có thể có lợi

**Nhược điểm:**
- Tốn tính toán hơn ReLU
- Cần tinh chỉnh cẩn thận

**Trường hợp dùng:**
- Kiến trúc hiện đại (EfficientNet dùng Swish)
- Khi chi phí tính toán không phải ràng buộc then chốt

### 8. GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z)$$

trong đó $$\Phi(z)$$ là hàm phân phối tích lũy của phân phối chuẩn chuẩn.

**Xấp xỉ:**

$$\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right)\right)$$

**Tính chất:**
- Trơn, không đơn điệu
- Diễn giải như bộ chính quy hóa ngẫu nhiên

**Ưu điểm:**
- **Hiệu năng state-of-the-art**: Dùng trong BERT, GPT
- Trơn mọi nơi
- Nắm bắt một số khía cạnh của dropout và zoneout

**Nhược điểm:**
- Tốn tính toán
- Khó diễn giải hơn

**Trường hợp dùng:**
- **Mô hình Transformer**: BERT, GPT-2, GPT-3
- Tác vụ NLP
- Mô hình quy mô lớn hiện đại

### 9. Softmax (Lớp Đầu ra)

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

**Tính chất:**
- Chuyển logits thành phân phối xác suất
- Miền đầu ra: $$(0, 1)$$ với $$\sum_i p_i = 1$$

**Đạo hàm (của lớp $$i$$ theo $$z_j$$):**

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \begin{cases} \text{softmax}(\mathbf{z})_i(1 - \text{softmax}(\mathbf{z})_i) & \text{if } i = j \\ -\text{softmax}(\mathbf{z})_i \cdot \text{softmax}(\mathbf{z})_j & \text{if } i \neq j \end{cases}$$

**Trường hợp dùng:**
- Lớp đầu ra **phân loại đa lớp**
- Cơ chế attention
- Mọi tình huống cần phân phối xác suất trên các lớp

### 10. Softplus

$$\text{softplus}(z) = \ln(1 + e^z)$$

**Tính chất:**
- Xấp xỉ trơn của ReLU
- Luôn dương
- Tiệm cận ReLU khi $$z$$ lớn

**Đạo hàm:**

$$\text{softplus}'(z) = \frac{e^z}{1 + e^z} = \sigma(z)$$

**Trường hợp dùng:**
- Đôi khi dùng ở lớp ẩn
- Mô hình sinh (đảm bảo đầu ra dương)

## Bảng So sánh Tóm tắt

| Kích hoạt | Miền | Tâm không | Gradient biến mất | Đơn vị chết | Chi phí tính toán | Dùng phổ biến |
|------------|-------|---------------|-------------------|-------------|-------------------|------------|
| Sigmoid | (0,1) | Không | Có | Không | Cao | Đầu ra (nhị phân) |
| tanh | (-1,1) | Có | Có | Không | Cao | Ẩn (cũ), RNN |
| ReLU | [0,∞) | Không | Không (với z>0) | Có | **Thấp** | **Ẩn (mặc định)** |
| Leaky ReLU | (-∞,∞) | Không | Không | Không | **Thấp** | Ẩn |
| ELU | (-α,∞) | ~Có | Không | Không | Trung bình | Ẩn |
| SELU | (-λα,∞) | Có (tự chuẩn hóa) | Không | Không | Trung bình | Ẩn (cụ thể) |
| Swish | (-∞,∞) | Không | Không | Không | Trung bình | Ẩn (hiện đại) |
| GELU | (-∞,∞) | Không | Không | Không | Cao | **Transformer** |
| Softmax | (0,1) | N/A | Có | Không | Cao | **Đầu ra (đa lớp)** |

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Trực giác mạng nơ-ron (Perceptron → Deep Learning)](https://www.facebook.com/reel/793200140509765)
