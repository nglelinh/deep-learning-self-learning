---
layout: post
title: 02-02-02 Độ sâu, Độ rộng và Mẫu Thiết kế
chapter: '02'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

Nếu Định lý Xấp xỉ Phổ quát khẳng định chỉ **một tầng ẩn duy nhất** là đủ để xấp xỉ mọi hàm liên tục, vậy tại sao các kiến trúc hiện đại như ResNet hay Llama lại chồng tới 50, 100 thậm chí hơn 100 tầng sâu? Khoảng cách giữa *"tồn tại trên lý thuyết"* và *"học được thực tế với chi phí hữu hạn"* chính là bí mật của nghệ thuật thiết kế kiến trúc. Bài học này phân tích sâu sự đánh đổi giữa **độ sâu** (*depth*) và **độ rộng** (*width*), giới thiệu các mẫu thiết kế kinh điển như thắt cổ chai (*bottleneck*), hình phễu (*pyramid*) và kết nối tắt (*skip connections*).

## Bản đồ kiến thức nền: Bốn nguyên lý Thiết kế Độ sâu, Độ rộng và Mẫu Kiến trúc

1. **Định lý Xấp xỉ Phổ quát vs. Hiệu quả Độ sâu (Expressivity vs. Efficiency)**:
   Mạng 1 tầng ẩn có thể cần số nơ-ron cỡ $$O(2^d)$$ để biểu diễn một hàm có cấu trúc phân cấp, trong khi mạng sâu chỉ cần $$O(d)$$ nơ-ron nhờ tính hợp hàm lặp lại.

2. **Bệnh lý của Mạng quá sâu (Degradation Problem)**:
   Trước khi có Residual Connection (2015), một mạng 56 tầng thậm chí có *train error* cao hơn mạng 20 tầng—không phải vì overfitting mà vì tối ưu hóa một chuỗi dài các ánh xạ đồng nhất cực kỳ khó khăn.

3. **Các Mẫu Thiết kế Kiến trúc Kinh điển (Design Patterns)**:
   - **Hình phễu (Funnel/Pyramid)**: Giảm dần chiều (ví dụ `512 -> 256 -> 128 -> 10`) để nén thông tin thành quyết định cô đọng.
   - **Thắt cổ chai (Bottleneck / Autoencoder)**: Ép qua chiều nhỏ rồi mở rộng lại (như `256 -> 64 -> 256` trong ResNet) để giảm FLOPs mà giữ dung lượng.
   - **Kết nối tắt (Residual $$\mathbf{y} = \mathbf{x} + F(\mathbf{x})$$)**: Tạo "đường cao tốc" cho cả tín hiệu lẫn gradient đi thẳng qua mạng sâu.

4. **Quy tắc Thực chiến khi Chọn Kích thước Mạng**:
   Bắt đầu từ kiến trúc gọn, kiểm chứng khả năng overfit dữ liệu huấn luyện trước, sau đó mới mở rộng độ sâu/độ rộng kết hợp Regularization và Normalization.

### Khung giải thích (nên đọc chậm): Tính tay mức tiết kiệm FLOPs của khối Bottleneck so với khối Phẳng

Giả sử bạn cần xử lý một vector đặc trưng 256 chiều đi qua hai lớp liên tiếp và trả về 256 chiều:

1. **Khối Phẳng (Standard 2-layer block: `256 -> 256 -> 256`)**:
   - Số trọng số lớp 1: $$256 \times 256 = 65{,}536$$.
   - Số trọng số lớp 2: $$256 \times 256 = 65{,}536$$.
   - **Tổng trọng số**: $$131{,}072$$ tham số.
2. **Khối Thắt cổ chai (Bottleneck block: `256 -> 64 -> 256`)**:
   - Nén chiều (`256 -> 64`): $$256 \times 64 = 16{,}384$$.
   - Mở rộng lại (`64 -> 256`): $$64 \times 256 = 16{,}384$$.
   - **Tổng trọng số**: $$32{,}768$$ tham số.
3. **So sánh**: Khối Bottleneck cắt giảm đúng **4 lần** (75%) số tham số và chi phí nhân ma trận! Khi xếp 50 khối như vậy trong một mạng sâu, đây chính là bí quyết giúp các kiến trúc hiện đại vừa cực sâu vừa chạy kịp thời gian thực.

## Độ sâu và Độ rộng của Mạng

### Độ rộng

**Độ rộng** (*width*) của một lớp là số lượng neuron nằm trong lớp đó.

- **Mạng rộng hơn** (nhiều neuron hơn ở mỗi lớp):
  - Tăng khả năng học các mẫu phức tạp ngay trong một lớp
  - Tăng số lượng tham số, từ đó dễ dẫn đến quá khớp (*overfitting*) nếu dữ liệu ít
  - Tăng chi phí tính toán trên mỗi lớp

### Độ sâu

**Độ sâu** (*depth*) của mạng là số lượng lớp trong mạng.

- **Mạng sâu hơn** (nhiều lớp hơn):
  - Có khả năng học các biểu diễn phân cấp (*hierarchical representations*)
  - Nâng cao sức biểu đạt để mô hình hóa các hàm phức tạp hơn
  - Có thể khó huấn luyện hơn do hiện tượng gradient biến mất hoặc bùng nổ (*vanishing/exploding gradients*)
  - Thuật ngữ "học sâu" (*deep learning*) cũng xuất phát từ việc sử dụng mạng nhiều lớp

### Định lý Xấp xỉ Phổ quát

<div class="textbook-theorem" markdown="1">
**Định lý**: Một mạng neuron truyền thẳng (*feedforward*) với:
- Một lớp ẩn duy nhất
- Số lượng neuron hữu hạn
- Hàm kích hoạt thích hợp (ví dụ: sigmoid, ReLU)

có thể xấp xỉ bất kỳ hàm liên tục nào trên một tập compact của $$\mathbb{R}^n$$ với độ chính xác tùy ý.
</div>

**Lưu ý quan trọng:**
- Đây là **định lý tồn tại**, không phải công thức hướng dẫn thiết kế thực tế
- Định lý không chỉ rõ cần bao nhiêu neuron (số lượng có thể tăng theo cấp số nhân)
- Mạng sâu hơn thường xấp xỉ được cùng một hàm với ít tham số hơn nhiều
- Mạng sâu hơn có xu hướng tự hình thành đặc trưng phân cấp trong quá trình học

**Hình dung độ sâu và độ rộng qua bài toán xấp xỉ 1D:** Trên cùng tập dữ liệu gợn sóng, mạng dung lượng thấp cho đường khớp kém, còn khi tăng kích thước lớp ẩn và huấn luyện đủ lâu, đường dự đoán bám sát dữ liệu hơn nhiều (xem chi tiết ở bài Lý thuyết Kiến trúc).

![Khớp kém khi dung lượng thấp](/deep-learning-self-learning/img/chapter_img/chapter02/nn_shallow_poor_fit.jpg)
*Hình: Mạng quá mỏng hoặc quá nông không đủ dung lượng để xấp xỉ hàm. (Minh họa từ video về bản chất mạng neuron)*

![Khớp tốt khi đủ dung lượng](/deep-learning-self-learning/img/chapter_img/chapter02/nn_deeper_good_fit.jpg)
*Hình: Khi đủ độ sâu/độ rộng và được huấn luyện đầy đủ, mô hình xấp xỉ hàm rất sát. (Minh họa từ video về bản chất mạng neuron)*

## Các Mẫu Thiết kế Phổ biến

### Độ rộng Giảm dần

Một cấu trúc quen thuộc là giảm dần số neuron qua các lớp:

```
Input (784) → 512 → 256 → 128 → 64 → Output (10)
```

**Lý do:** Nén dần thông tin đầu vào thành các biểu diễn trừu tượng ở mức cao hơn.

### Kiến trúc Đồng hồ cát / Cổ chai (*Hourglass/Bottleneck*)

Thu hẹp chiều xuống một lớp trung gian nhỏ rồi mở rộng trở lại:

```
Input (784) → 256 → 64 → 256 → Output (784)
```

**Ứng dụng phổ biến:** Autoencoder dùng cho bài toán giảm chiều và tái tạo dữ liệu.

### Độ rộng Đồng nhất

Giữ nguyên số neuron ở tất cả các lớp ẩn:

```
Input (784) → 256 → 256 → 256 → Output (10)
```

**Lý do:** Cấu trúc đơn giản, giảm số siêu tham số cần tinh chỉnh.

## Hàm Kích hoạt theo Lớp

Các lớp khác nhau trong cùng một mạng có thể sử dụng hàm kích hoạt khác nhau.

**Cấu hình thường gặp:**
- **Lớp ẩn:** ReLU (hoặc các biến thể như Leaky ReLU, ELU)
  - Tính toán nhanh
  - Giảm nhẹ hiện tượng gradient biến mất
- **Lớp đầu ra:** Chọn tùy theo tác vụ
  - Phân loại nhị phân: Sigmoid
  - Phân loại đa lớp: Softmax
  - Hồi quy: Tuyến tính (hàm đồng nhất)

## Lớp Liên kết Đầy đủ so với Các Kiến trúc Khác

### Lớp Liên kết Đầy đủ (*Fully Connected / Dense*)

Mỗi neuron ở lớp $$l$$ đều kết nối tới toàn bộ neuron của lớp $$l-1$$.

**Ưu điểm:**
- Linh hoạt, không ràng buộc cứng về cấu trúc kết nối
- Có thể học nhiều dạng mẫu khác nhau nếu có đủ neuron

**Nhược điểm:**
- Số tham số lớn ($$n^{[l]} \times n^{[l-1]}$$)
- Không tận dụng được cấu trúc sẵn có của dữ liệu đầu vào
- Kém hiệu quả với dữ liệu có cấu trúc không gian hoặc tuần tự (ảnh, chuỗi)

### Các Kiến trúc Chuyên biệt

Với từng loại dữ liệu cụ thể, các kiến trúc chuyên biệt cho hiệu quả cao hơn nhiều:

- **Lớp tích chập** (*convolutional layers*): Phù hợp cho ảnh nhờ khai thác cấu trúc không gian
- **Lớp hồi quy** (*recurrent layers*): Phù hợp cho dữ liệu chuỗi nhờ khai thác cấu trúc thời gian
- **Cơ chế attention**: Xử lý hiệu quả các phụ thuộc tầm xa

Các kiến trúc này sẽ được trình bày chi tiết ở những chương sau.

## Biểu diễn Mạng

### Biểu diễn Đồ thị

Mạng neuron truyền thẳng thường được biểu diễn dưới dạng đồ thị có hướng không chu trình (DAG):

```
      Input Layer    Hidden Layer 1   Hidden Layer 2   Output Layer
         (3)             (4)              (4)              (2)
    
    x₁  ○────────────────●──────────────────●──────────────────●  ŷ₁
                        ╱│╲              ╱│╲              ╱│
    x₂  ○──────────────●─●─●────────────●─●─●────────────●─●
                        ╲│╱              ╲│╱              ╲│
    x₃  ○────────────────●──────────────────●──────────────────●  ŷ₂
```

### Biểu diễn Ma trận

Để tối ưu hiệu năng tính toán, các phép toán trên cả batch dữ liệu được viết dưới dạng nhân ma trận:

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

trong đó:
- $$\mathbf{A}^{[l-1]}$$: ma trận kích hoạt (mỗi cột tương ứng với một mẫu)
- $$\mathbf{W}^{[l]}$$: ma trận trọng số
- $$\mathbf{b}^{[l]}$$: vector độ lệch (được mở rộng chiều tự động — *broadcast* — trên toàn bộ các mẫu)

## Triển khai Thực tế

### Ví dụ: Mạng Neuron Đơn giản bằng Python

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: danh sách kích thước các lớp, gồm đầu vào và đầu ra
        Ví dụ: [784, 128, 64, 10] cho MNIST
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        
        # Khởi tạo trọng số và độ lệch
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # Khởi tạo He cho mạng dùng ReLU
            w = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2.0 / layer_sizes[i-1])
            b = np.zeros((layer_sizes[i], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward(self, x):
        """
        x: đầu vào có dạng (input_size, num_examples)
        Trả về: đầu ra có dạng (output_size, num_examples)
        """
        a = x
        activations = [x]
        zs = []
        
        # Lan truyền xuôi qua các lớp ẩn
        for i in range(self.num_layers - 2):
            z = self.weights[i] @ a + self.biases[i]
            a = self.relu(z)
            zs.append(z)
            activations.append(a)
        
        # Lớp đầu ra (softmax)
        z = self.weights[-1] @ a + self.biases[-1]
        a = self.softmax(z)
        zs.append(z)
        activations.append(a)
        
        return a, activations, zs
    
    def predict(self, x):
        """Trả về dự đoán lớp"""
        output, _, _ = self.forward(x)
        return np.argmax(output, axis=0)

# Ví dụ sử dụng
network = NeuralNetwork([784, 128, 64, 10])
x = np.random.randn(784, 5)  # 5 mẫu
output, _, _ = network.forward(x)
print(f"Output shape: {output.shape}")  # (10, 5)
print(f"Predictions: {network.predict(x)}")
```

## Các Cân nhắc Thiết kế

### Số Lớp

- **1–2 lớp ẩn:** Phù hợp cho bài toán đơn giản và tập dữ liệu nhỏ
- **3–5 lớp ẩn:** Phù hợp cho bài toán có độ phức tạp trung bình
- **Trên 5 lớp ẩn:** Phù hợp cho bài toán phức tạp, tập dữ liệu lớn (học sâu)

### Số Neuron trên mỗi Lớp

Một số quy tắc thực nghiệm thường dùng:
- Khởi đầu với kích thước lớp ẩn nằm giữa kích thước đầu vào và đầu ra
- Các mức kích thước phổ biến: 32, 64, 128, 256, 512
- Tăng số neuron giúp tăng dung lượng nhưng đồng thời tăng nguy cơ quá khớp
- Dựa vào kết quả trên tập kiểm định (*validation set*) để điều chỉnh

### Tìm kiếm Kiến trúc

Quá trình chọn kiến trúc phù hợp thường được thực hiện qua:
- **Thử nghiệm thủ công:** Thay đổi và so sánh các cấu hình dựa trên kinh nghiệm
- **Grid search:** Thử lần lượt mọi tổ hợp trong lưới siêu tham số định trước
- **Random search:** Lấy mẫu ngẫu nhiên các cấu hình, thường hiệu quả hơn grid search
- **Neural Architecture Search (NAS):** Phương pháp tìm kiếm kiến trúc tự động (chủ đề nâng cao)

## Tóm tắt

- **Mạng neuron** bao gồm lớp đầu vào, các lớp ẩn và lớp đầu ra.
- **Mạng feedforward (MLP)** là kiến trúc cơ bản nhất, trong đó dữ liệu truyền theo một chiều từ đầu vào tới đầu ra.
- **Lan truyền xuôi** tính toán dự đoán bằng cách đưa đầu vào đi tuần tự qua các lớp.
- **Độ sâu** (số lớp) và **độ rộng** (số neuron mỗi lớp) cùng quyết định dung lượng biểu diễn của mạng.
- **Định lý Xấp xỉ Phổ quát** khẳng định khả năng xấp xỉ về mặt tồn tại, nhưng không đảm bảo hiệu quả tham số hay quá trình tối ưu.
- **Lớp liên kết đầy đủ** nối mọi neuron của hai lớp kề nhau, linh hoạt nhưng tốn nhiều tham số.
- **Thiết kế kiến trúc hợp lý** cần cân đối giữa bản chất bài toán, lượng dữ liệu và tài nguyên tính toán.

Ở bài tiếp theo, ta sẽ tìm hiểu chi tiết hơn về các hàm kích hoạt và vai trò của chúng trong quá trình huấn luyện.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ video của Machine Learning Thực Chiến:

- [Mạng nơ-ron là bộ xấp xỉ hàm](https://www.facebook.com/reel/720970114372332)

## Video tham khảo

- **Xem trước:** [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Giải thích vai trò của việc tăng độ sâu trước khi đi vào các mẫu thiết kế.
- [Vanishing/Exploding Gradients (C2W1L10)](https://www.youtube.com/watch?v=qhXZsFVxGKo) — DeepLearningAI, tiếng Anh. Phân tích hiện tượng gradient biến mất — nguyên nhân khiến việc tăng độ sâu thiếu kiểm soát dễ thất bại.
- [The GELU, SiLU and SwiGLU activation functions, clearly explained!!!](https://www.youtube.com/watch?v=2FaI2Fen1mQ) — StatQuest with Josh Starmer, tiếng Anh. Giới thiệu các hàm kích hoạt hiện đại như GELU, SiLU và SwiGLU dùng trong các thiết kế mạng ngày nay.
