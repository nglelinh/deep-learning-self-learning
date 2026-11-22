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

Bài này phân tích chi tiết các hàm kích hoạt (*activation functions*), đặc tính toán học của chúng và cách lựa chọn hàm phù hợp cho mạng neuron.

---

## Vì sao Hàm Kích hoạt Quan trọng

**Nếu không có hàm kích hoạt**, mạng neuron chỉ có thể biểu diễn các phép biến đổi tuyến tính. Dù xếp chồng bao nhiêu lớp, hợp thành của nhiều hàm tuyến tính vẫn chỉ là một hàm tuyến tính duy nhất:

$$f(g(h(\mathbf{x}))) = \mathbf{A}_1(\mathbf{A}_2(\mathbf{A}_3 \mathbf{x})) = (\mathbf{A}_1 \mathbf{A}_2 \mathbf{A}_3)\mathbf{x} = \mathbf{A}\mathbf{x}$$

**Hàm kích hoạt bổ sung tính phi tuyến**, giúp mạng học được các mẫu phức tạp và xấp xỉ nhiều lớp hàm khác nhau.

## Các Tính chất Mong muốn của Hàm Kích hoạt

Một hàm kích hoạt tốt thường hướng tới các tính chất sau:

1. **Phi tuyến:** Cho phép mạng học các mối quan hệ phức tạp
2. **Khả vi:** Hỗ trợ tối ưu hóa dựa trên gradient
3. **Đơn điệu:** Bảo toàn thứ tự đầu vào (thuận lợi cho quá trình tối ưu)
4. **Tính toán hiệu quả:** Thực thi nhanh ở cả lượt xuôi và lượt ngược
5. **Miền giá trị phù hợp:** Bị chặn hoặc không bị chặn tùy theo yêu cầu bài toán
6. **Đối xứng quanh 0 (*zero-centered*):** Giúp luồng gradient ổn định hơn ở các lớp ẩn
7. **Hạn chế bão hòa:** Tránh hiện tượng gradient biến mất

Không có hàm kích hoạt nào thỏa mãn đồng thời mọi tiêu chí ở mức tối ưu, nên việc lựa chọn cần căn cứ vào từng bài toán và kiến trúc cụ thể.

## Các Hàm Kích hoạt Phổ biến

### 1. Sigmoid (Hàm Logistic)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Đạo hàm:**

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Tính chất:**
- Miền giá trị: $$(0, 1)$$
- Trơn và khả vi tại mọi điểm
- Đơn điệu tăng
- Bão hòa ở cả hai phía

**Ưu điểm:**
- Dễ diễn giải dưới dạng xác suất
- Gradient biến thiên mượt
- Được sử dụng rộng rãi trong giai đoạn đầu của mạng neuron

**Nhược điểm:**
- **Gradient biến mất:** Đạo hàm rất gần 0 khi $$|z| > 4$$
- **Không đối xứng quanh 0:** Đầu ra luôn dương
- **Chi phí tính toán cao:** Cần tính hàm mũ

**Trường hợp sử dụng:**
- Lớp đầu ra cho bài toán phân loại nhị phân
- Các cổng điều khiển trong LSTM
- Thường tránh dùng ở lớp ẩn của mạng sâu

### 2. Tang Hyperbolic (tanh)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1} = 2\sigma(2z) - 1$$

**Đạo hàm:**

$$\tanh'(z) = 1 - \tanh^2(z)$$

**Tính chất:**
- Miền giá trị: $$(-1, 1)$$
- Đối xứng quanh 0 (cải thiện so với sigmoid)
- Bão hòa ở cả hai phía

**Ưu điểm:**
- Đối xứng quanh 0, hỗ trợ luồng gradient tốt hơn
- Đạo hàm lớn hơn sigmoid (nhận giá trị trong $$(0, 1]$$)

**Nhược điểm:**
- Vẫn gặp hiện tượng gradient biến mất ở hai đầu bão hòa
- Tốn chi phí tính toán hàm mũ

**Trường hợp sử dụng:**
- Lớp ẩn (tốt hơn sigmoid nhưng thường kém hiệu quả hơn ReLU)
- Các ô nhớ trong RNN/LSTM
- Những bài toán cần đầu ra đối xứng quanh 0

### 3. Rectified Linear Unit (ReLU)

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

![Đồ thị hàm ReLU](/deep-learning-self-learning/img/chapter_img/chapter02/relu_function.jpg)
*Hình: $$\mathrm{ReLU}(x)=\max(0,x)=(x)^+$$ — bằng $$0$$ với $$x<0$$ và bằng $$x$$ với $$x\geq 0$$. (Minh họa từ video nhập môn mạng neuron)*

**Đạo hàm:**

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \\ \text{undefined} & \text{if } z = 0 \end{cases}$$

(Trong thực tế cài đặt, ta thường quy ước $$\text{ReLU}'(0) = 0$$ hoặc $$0.5$$.)

**Tính chất:**
- Miền giá trị: $$[0, \infty)$$
- Không bị bão hòa với đầu vào dương
- Tạo kích hoạt thưa (nhiều neuron xuất giá trị 0)

**Ưu điểm:**
- **Tính toán rất nhanh:** Chỉ cần so sánh ngưỡng tại 0
- **Giảm gradient biến mất:** Đạo hàm bằng 1 trên toàn miền dương
- **Biểu diễn thưa:** Tự nhiên đưa nhiều kích hoạt về 0
- **Hiệu quả thực nghiệm cao:** Hoạt động ổn định trong phần lớn mô hình

**Nhược điểm:**
- **Không đối xứng quanh 0:** Đầu ra luôn không âm
- **Hiện tượng ReLU chết** (*dying ReLU*): Neuron có thể ngừng hoạt động vĩnh viễn
  - Nếu $$z < 0$$ trên mọi mẫu, đạo hàm luôn bằng 0 nên trọng số không cập nhật
  - Thường xảy ra khi tốc độ học quá lớn hoặc khởi tạo chưa phù hợp

**Trường hợp sử dụng:**
- **Lựa chọn mặc định** cho lớp ẩn trong mạng sâu
- CNN, ResNet và phần lớn các kiến trúc tiêu chuẩn

### 4. Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

trong đó $$\alpha$$ là một hằng số dương nhỏ (thường chọn 0.01).

**Đạo hàm:**

$$\text{LeakyReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

**Ưu điểm:**
- **Khắc phục ReLU chết:** Vẫn duy trì đạo hàm nhỏ ở miền âm
- Tính toán đơn giản và nhanh
- Giữ được các ưu điểm chính của ReLU

**Nhược điểm:**
- Xuất hiện thêm siêu tham số $$\alpha$$
- Không phải lúc nào cũng cải thiện rõ rệt so với ReLU trong thực tế

**Các biến thể:**
- **Parametric ReLU (PReLU):** Hệ số $$\alpha$$ được học trực tiếp trong quá trình huấn luyện
- **Randomized Leaky ReLU (RReLU):** Hệ số $$\alpha$$ được lấy mẫu ngẫu nhiên trong khi huấn luyện

### 5. Exponential Linear Unit (ELU)

$$\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

trong đó $$\alpha > 0$$ (thường chọn $$\alpha = 1$$).

**Đạo hàm:**

$$\text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha e^z = \text{ELU}(z) + \alpha & \text{if } z \leq 0 \end{cases}$$

**Tính chất:**
- Trơn tại mọi điểm (khi $$\alpha = 1$$)
- Cho phép giá trị âm giúp kéo trung bình kích hoạt về gần 0 hơn

**Ưu điểm:**
- **Gần đối xứng quanh 0 hơn:** Có đầu ra âm ở miền $$z < 0$$
- **Tránh ReLU chết:** Gradient khác 0 với mọi đầu vào
- **Trơn:** Giúp bề mặt tối ưu mượt hơn
- Thường giúp hội tụ nhanh và ổn định hơn

**Nhược điểm:**
- Chi phí tính toán cao hơn ReLU do có hàm mũ
- Thêm siêu tham số $$\alpha$$

**Trường hợp sử dụng:**
- Thay thế ReLU khi chấp nhận chi phí tính toán cao hơn một chút
- Các bài toán hưởng lợi từ kích hoạt có trung bình gần 0

### 6. Scaled Exponential Linear Unit (SELU)

$$\text{SELU}(z) = \lambda \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

trong đó $$\lambda \approx 1.0507$$ và $$\alpha \approx 1.6733$$.

**Tính chất:**
- **Tự chuẩn hóa** (*self-normalizing*): Dưới các điều kiện nhất định, kích hoạt tự hội tụ về trung bình 0 và phương sai 1
- Đòi hỏi phương pháp khởi tạo tương thích (LeCun normal)
- Thiết kế cho kiến trúc nhất định (chủ yếu các lớp liên kết đầy đủ)

**Ưu điểm:**
- Có thể huấn luyện mạng rất sâu mà không cần batch normalization
- Có cơ sở lý thuyết rõ ràng về tính hội tụ phân phối

**Nhược điểm:**
- Điều kiện áp dụng khá chặt chẽ về kiến trúc mạng
- Ít được sử dụng rộng rãi bằng ReLU hay GELU
- Không tương thích tốt với dropout tiêu chuẩn hoặc lớp tích chập

### 7. Swish (SiLU — Sigmoid Linear Unit)

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

**Tính chất:**
- Trơn, không đơn điệu
- Không bị chặn trên, bị chặn dưới
- Tự đóng cổng (*self-gated*): đầu vào được điều chỉnh trọng số bởi chính giá trị sigmoid của nó

**Ưu điểm:**
- **Hiệu năng tốt:** Thực nghiệm cho thấy thường vượt ReLU trên nhiều bài toán sâu
- Gradient trơn
- Tính không đơn điệu đôi khi giúp tối ưu tốt hơn

**Nhược điểm:**
- Tốn tính toán hơn ReLU
- Cần điều chỉnh siêu tham số cẩn thận

**Trường hợp sử dụng:**
- Các kiến trúc hiện đại (ví dụ EfficientNet sử dụng Swish)
- Khi độ chính xác quan trọng hơn chi phí tính toán tăng thêm

### 8. GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z)$$

trong đó $$\Phi(z)$$ là hàm phân phối tích lũy của phân phối chuẩn tắc.

**Công thức xấp xỉ:**

$$\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right)\right)$$

**Tính chất:**
- Trơn, không đơn điệu
- Có thể diễn giải như một cơ chế chính quy hóa ngẫu nhiên theo độ lớn đầu vào

**Ưu điểm:**
- **Hiệu năng cao trong thực tế:** Được sử dụng phổ biến trong BERT, GPT
- Trơn tại mọi điểm
- Mang ý tưởng gần với dropout và zoneout ở dạng tất định

**Nhược điểm:**
- Chi phí tính toán cao
- Kém trực quan hơn ReLU

**Trường hợp sử dụng:**
- **Các mô hình Transformer:** BERT, GPT-2, GPT-3
- Các tác vụ xử lý ngôn ngữ tự nhiên (NLP)
- Các mô hình quy mô lớn hiện đại

### 9. Softmax (Lớp Đầu ra)

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

**Tính chất:**
- Chuyển vector logits thành một phân phối xác suất
- Miền đầu ra: $$(0, 1)$$ với $$\sum_i p_i = 1$$

**Đạo hàm (của đầu ra lớp $$i$$ theo $$z_j$$):**

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \begin{cases} \text{softmax}(\mathbf{z})_i(1 - \text{softmax}(\mathbf{z})_i) & \text{if } i = j \\ -\text{softmax}(\mathbf{z})_i \cdot \text{softmax}(\mathbf{z})_j & \text{if } i \neq j \end{cases}$$

**Trường hợp sử dụng:**
- Lớp đầu ra cho **phân loại đa lớp**
- Cơ chế attention
- Mọi tình huống cần tạo phân phối xác suất trên các lớp

### 10. Softplus

$$\text{softplus}(z) = \ln(1 + e^z)$$

**Tính chất:**
- Là bản xấp xỉ trơn của ReLU
- Đầu ra luôn dương
- Tiệm cận ReLU khi $$z$$ lớn

**Đạo hàm:**

$$\text{softplus}'(z) = \frac{e^z}{1 + e^z} = \sigma(z)$$

**Trường hợp sử dụng:**
- Đôi khi dùng ở lớp ẩn
- Các mô hình sinh khi cần đảm bảo đầu ra luôn dương

## Bảng So sánh Tóm tắt

| Kích hoạt | Miền | Đối xứng quanh 0 | Gradient biến mất | Đơn vị chết | Chi phí tính toán | Cách dùng phổ biến |
|------------|-------|---------------|-------------------|-------------|-------------------|------------|
| Sigmoid | (0,1) | Không | Có | Không | Cao | Đầu ra (nhị phân) |
| tanh | (-1,1) | Có | Có | Không | Cao | Ẩn (cũ), RNN |
| ReLU | [0,∞) | Không | Không (với z>0) | Có | **Thấp** | **Ẩn (mặc định)** |
| Leaky ReLU | (-∞,∞) | Không | Không | Không | **Thấp** | Ẩn |
| ELU | (-α,∞) | Gần có | Không | Không | Trung bình | Ẩn |
| SELU | (-λα,∞) | Có (tự chuẩn hóa) | Không | Không | Trung bình | Ẩn (kiến trúc cụ thể) |
| Swish | (-∞,∞) | Không | Không | Không | Trung bình | Ẩn (hiện đại) |
| GELU | (-∞,∞) | Không | Không | Không | Cao | **Transformer** |
| Softmax | (0,1) | N/A | Có | Không | Cao | **Đầu ra (đa lớp)** |

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ video của Machine Learning Thực Chiến:

- [Trực giác mạng nơ-ron (Perceptron → Deep Learning)](https://www.facebook.com/reel/793200140509765)

## Video tham khảo

- **Xem trước:** [Why Non-linear Activation Functions (C1W3L07)](https://www.youtube.com/watch?v=NkOv_k7r6no) — DeepLearningAI, tiếng Anh. Giải thích các tính chất cần có của một hàm kích hoạt phi tuyến.
- [Neural Networks Pt. 3: ReLU In Action!!!](https://www.youtube.com/watch?v=68BZ5f7P94E) — StatQuest with Josh Starmer, tiếng Anh. Minh họa ReLU qua một lượt lan truyền xuôi bằng số cụ thể.
- [Neural Networks Part 5: ArgMax and SoftMax](https://www.youtube.com/watch?v=KpKog-L9veg) — StatQuest with Josh Starmer, tiếng Anh. Giải thích Softmax — hàm kích hoạt đầu ra dùng cho bài toán phân loại đa lớp.
