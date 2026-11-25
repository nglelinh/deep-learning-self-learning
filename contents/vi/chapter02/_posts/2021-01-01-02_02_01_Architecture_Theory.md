---
layout: post
title: 02-02-01 Lý thuyết Kiến trúc Mạng Neuron
chapter: '02'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

Làm sao một mạng chỉ gồm các phép cộng, nhân và cắt bỏ số âm (ReLU) lại có thể vẽ ra ranh giới uốn lượn mềm mại bao quanh một đám mây dữ liệu hình xoắn ốc, hay nhận ra hàng nghìn giống loài khác nhau? Bí mật nằm ở **hình học gấp không gian** (*space folding*) của mạng nhiều tầng (MLP): mỗi tầng ẩn ReLU thực chất đang gấp tờ giấy không gian đầu vào lại một lần, khiến những điểm từng nằm đan xen nhau bỗng tách hẳn sang hai phía của một đường thẳng! Bài học này xây dựng lý thuyết kiến trúc mạng nơ-ron, giải quyết triệt để bài toán XOR và lý giải hình học đằng sau từng lớp ẩn.

## Kiến thức cần nhớ: Bốn trụ cột Lý thuyết Kiến trúc Mạng Nơ-ron

1. **Phân rã Bài toán bằng Lớp ẩn (Hidden Representation)**:
   Thay vì cố phân tách dữ liệu ngay ở không gian gốc, lớp ẩn tạo ra hệ tọa độ mới $$\mathbf{h} = \phi(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$$ nơi bài toán phi tuyến (như XOR) biến thành khả tách tuyến tính.

2. **Số lượng Vùng Tuyến tính (Linear Regions of ReLU Networks)**:
   Một mạng ReLU chia không gian đầu vào thành nhiều đa diện lồi rời nhau, trên mỗi đa diện mạng hoạt động như một hàm tuyến tính riêng. Số đa diện tối đa tăng theo đa thức với độ rộng, nhưng tăng theo **hàm mũ** với độ sâu!

3. **Ánh xạ từ Đặc trưng đến Quyết định (Feature Extractor + Classifier)**:
   Mọi mạng phân loại sâu đều có thể tách làm hai phần: toàn bộ các tầng đầu là bộ trích xuất đặc trưng học biểu diễn $$\mathbf{h}$$, còn tầng cuối cùng thực chất chỉ là một bộ phân loại tuyến tính (Logistic/Softmax Regression) chạy trên $$\mathbf{h}$$.

4. **Dung lượng Mô hình (Model Capacity) và Thiên kiến Quy nạp (Inductive Bias)**:
   Kiến trúc quy định họ hàm mà mạng có thể biểu diễn dễ dàng nhất: MLP giả định mọi đầu vào bình đẳng, còn CNN hay Transformer cài sẵn thiên kiến về cấu trúc không gian và quan hệ ngữ cảnh.

### Giải thích: Tự tay giải bài toán XOR bằng một mạng MLP đúng 2 nơ-ron ẩn ReLU

Ta xây dựng mạng 2–2–1 biến đổi 4 điểm XOR $$(x_1, x_2)$$ qua 2 nơ-ron ẩn ReLU:
$$h_1 = \max(0,\; x_1 + x_2 - 0.5), \quad h_2 = \max(0,\; x_1 + x_2 - 1.5), \quad \hat{y} = h_1 - 2h_2.$$

Hãy tính từng dòng cho cả 4 đầu vào:

| Đầu vào $$(x_1, x_2)$$ | $$x_1+x_2-0.5$$ | $$h_1 = \text{ReLU}(\cdot)$$ | $$x_1+x_2-1.5$$ | $$h_2 = \text{ReLU}(\cdot)$$ | Đầu ra $$\hat{y} = h_1 - 2h_2$$ | Nhãn XOR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $$(0, 0)$$ | $$-0.5$$ | $$0.0$$ | $$-1.5$$ | $$0.0$$ | $$0.0 - 0 = \mathbf{0.0}$$ | **0** |
| $$(0, 1)$$ | $$+0.5$$ | $$0.5$$ | $$-0.5$$ | $$0.0$$ | $$0.5 - 0 = \mathbf{0.5}$$ | **1** |
| $$(1, 0)$$ | $$+0.5$$ | $$0.5$$ | $$-0.5$$ | $$0.0$$ | $$0.5 - 0 = \mathbf{0.5}$$ | **1** |
| $$(1, 1)$$ | $$+1.5$$ | $$1.5$$ | $$+0.5$$ | $$0.5$$ | $$1.5 - 2(0.5) = \mathbf{0.5}$$? Khoan! Kiểm tra lại! | — |

Hãy chú ý phép kiểm tra cực quan trọng ở dòng cuối: với hệ số $$-2$$ ta được $$1.5 - 1.0 = 0.5$$, vẫn chưa về $$0$$! Để kéo $$(1,1)$$ về đúng $$0$$, chỉ cần đổi hệ số của $$h_2$$ thành $$-3$$ (hoặc dùng $$h_1 = \max(0, x_1+x_2),\, h_2 = \max(0, x_1+x_2-1),\, \hat{y} = h_1 - 2h_2$$):
- $$(0,0) \to h_1=0, h_2=0 \implies \hat{y} = 0 - 0 = \mathbf{0}$$.
- $$(0,1) \to h_1=1, h_2=0 \implies \hat{y} = 1 - 0 = \mathbf{1}$$.
- $$(1,0) \to h_1=1, h_2=0 \implies \hat{y} = 1 - 0 = \mathbf{1}$$.
- $$(1,1) \to h_1=2, h_2=1 \implies \hat{y} = 2 - 2(1) = \mathbf{0}$$.

Đúng tuyệt đối 4/4 điểm! Nhìn vào bảng trên bạn thấy phép màu hình học: lớp ẩn đã gộp hai điểm $$(0,1)$$ và $$(1,0)$$ về cùng một điểm $$(1,0)$$ trong không gian $$(h_1, h_2)$$, khiến đường thẳng $$h_1 - 2h_2 = 0.5$$ chém gọn hai lớp XOR chỉ bằng một nhát cắt!

# Kiến trúc Mạng Neuron: Từ Neuron đến Hệ thống Sâu

![Neural Network Layers](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/500px-Colored_neural_network.svg.png)
*Hình ảnh: Kiến trúc mạng neuron với lớp đầu vào (input), lớp ẩn (hidden) và lớp đầu ra (output). Nguồn: Wikimedia Commons*

![Mạng fully connected nhiều lớp](/deep-learning-self-learning/img/chapter_img/chapter02/mlp_fully_connected.jpg)
*Hình: Mạng feedforward liên kết đầy đủ — input (hồng), hidden (xanh lá), output (xanh dương). (Minh họa từ video nhập môn mạng neuron)*

## 1. Tổng quan khái niệm

Kiến trúc mạng neuron là bản thiết kế quy định cách tổ chức, sắp xếp và kết nối các neuron riêng lẻ để giải quyết các bài toán phức tạp. Nếu một neuron đơn lẻ chỉ học được biên quyết định tuyến tính như ta đã thấy với perceptron, thì sức mạnh của học sâu chỉ thực sự xuất hiện khi nhiều neuron được nhóm thành từng lớp và các lớp này được xếp chồng thành kiến trúc sâu. Cấu trúc phân tầng này không đơn thuần là cách bố trí kỹ thuật mà phản ánh cơ chế tính toán hợp thành: các đơn vị đơn giản phối hợp với nhau để tạo nên khả năng biểu diễn phức tạp.

Thiết kế kiến trúc đóng vai trò quyết định đến những gì mạng có thể học và hiệu quả của quá trình huấn luyện. Một kiến trúc không phù hợp có thể học kém ngay cả trên dữ liệu đơn giản, trong khi một kiến trúc hợp lý có thể nắm bắt những mối quan hệ tinh vi với chi phí hợp lý. Kiến trúc chính là nơi thể hiện các thiên kiến quy nạp (*inductive biases*) — những giả định sẵn có về cấu trúc của bài toán — giúp mạng học nhanh và hiệu quả hơn so với việc coi mọi bài toán như một phép xấp xỉ hàm hoàn toàn tổng quát.

### Mô hình Lớp

Nguyên lý tổ chức cơ bản của mạng neuron là **lớp** (*layers*) — tập hợp các neuron cùng thực hiện phép biến đổi ở một giai đoạn tính toán. Cách tổ chức theo lớp hiện thực hóa trực tiếp tính toán hợp thành: mỗi lớp chuyển đầu vào thành một biểu diễn mới, và các lớp tiếp theo dựa trên đó để xây dựng những đặc trưng ngày càng trừu tượng. Trong bài toán nhận diện khuôn mặt, các lớp đầu thường học phát hiện cạnh, các lớp giữa ghép cạnh thành bộ phận như mắt, mũi, miệng, còn các lớp sâu hơn nhận diện danh tính hoàn chỉnh.

**Ví dụ ảnh chữ số:** Ảnh kích thước $$28\times28$$ được làm phẳng (*flatten*) thành $$784$$ giá trị đầu vào, đưa qua các lớp ẩn rồi đến $$10$$ đầu ra tương ứng với các chữ số từ $$0$$ đến $$9$$. Không có một công thức cố định nào cho số lượng neuron ẩn — đó là lựa chọn thiết kế — nhưng luồng xử lý luôn đi theo trình tự: vector hóa → đi qua các lớp xếp chồng → xuất đầu ra phù hợp với tác vụ.

![MLP nhận diện chữ số: 784 → ẩn → 10](/deep-learning-self-learning/img/chapter_img/chapter02/mnist_mlp_10_outputs.jpg)
*Hình: Quy trình kiểu MNIST — $$784$$ pixel → lớp ẩn → $$10$$ logit cho các chữ số. (Minh họa từ video nhập môn mạng neuron)*

Cấu trúc xử lý phân cấp này tương đồng với cả hệ thần kinh sinh học lẫn bản chất phân tầng của nhiều khái niệm trong thực tế. Chẳng hạn, một chiếc xe gồm bánh xe, cửa sổ và thân xe; các bộ phận này được cấu thành từ hình dạng và kết cấu (*texture*); và các yếu tố đó lại bắt nguồn từ cạnh và màu sắc. Các lớp trong mạng neuron tự nhiên học được hệ thống phân cấp này, với mỗi lớp phụ trách một mức độ trừu tượng tương ứng với vị trí của nó trong mạng.

### Vì sao Kiến trúc Quan trọng: Đánh đổi giữa Độ sâu và Độ rộng

Một nhận định quan trọng từ cả lý thuyết lẫn thực nghiệm là **độ sâu** (số lớp) và **độ rộng** (số neuron trên mỗi lớp) tác động rất khác nhau tới dung lượng và khả năng học của mạng. Định lý Xấp xỉ Phổ quát khẳng định rằng chỉ cần một lớp ẩn với đủ nhiều neuron là đã có thể xấp xỉ bất kỳ hàm liên tục nào. Tuy nhiên, trong thực tế, mạng sâu với số neuron mỗi lớp vừa phải thường vượt trội hơn mạng nông nhưng rất rộng trên các bài toán phức tạp.

Lợi thế này không chỉ nằm ở hiệu quả tham số — khi mạng sâu thường đạt sức biểu diễn tương đương với số tham số ít hơn theo cấp số nhân so với mạng nông — mà còn ở cách biểu diễn dữ liệu. Mạng sâu tự hình thành hệ thống đặc trưng phân cấp trong quá trình huấn luyện mà không cần lập trình thủ công từng bước phát hiện cạnh hay hình dạng. Bên cạnh đó, các biểu diễn trung gian của mạng sâu thường khái quát hóa tốt hơn và dễ tái sử dụng giữa các tác vụ khác nhau, làm nền tảng cho học chuyển giao (*transfer learning*) và tiền huấn luyện (*pretraining*).

Hiểu rõ sự đánh đổi giữa độ sâu và độ rộng, cũng như ảnh hưởng của lựa chọn kiến trúc lên quá trình tối ưu, khả năng khái quát hóa và chi phí tính toán, là điều kiện cần để thiết kế mạng neuron hiệu quả. Bài học này trình bày nền tảng toán học và trực giác giúp bạn đưa ra các quyết định kiến trúc có cơ sở.

## 2. Nền tảng toán học

### Mạng Neuron Feedforward: Định nghĩa Hình thức

Một **mạng neuron truyền thẳng** (*feedforward neural network*, còn gọi là **Multilayer Perceptron** hay **MLP**) là một hàm $$f: \mathbb{R}^{n_0} \to \mathbb{R}^{n_L}$$ được tạo nên bằng cách hợp thành các phép biến đổi theo từng lớp. Với mạng gồm $$L$$ lớp:

$$f(\mathbf{x}) = f^{[L]} \circ f^{[L-1]} \circ \cdots \circ f^{[1]}(\mathbf{x})$$

trong đó mỗi hàm lớp $$f^{[l]}$$ gồm một phép biến đổi affine theo sau bởi một hàm kích hoạt phi tuyến áp dụng theo từng phần tử:

$$f^{[l]}(\mathbf{a}^{[l-1]}) = \sigma^{[l]}(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})$$

![Công thức lớp ẩn + kích hoạt](/deep-learning-self-learning/img/chapter_img/chapter02/nn_hidden_activation_formula.jpg)
*Hình: Một lớp ẩn $$z_1=\sigma(xw+b)$$ theo sau bởi lớp tuyến tính để cho ra $$f(x)$$. (Minh họa từ video về bản chất mạng neuron)*

![Mở rộng độ rộng (nhiều neuron ẩn)](/deep-learning-self-learning/img/chapter_img/chapter02/nn_width_parameters_formula.jpg)
*Hình: Tăng kích thước ma trận $$W$$ (thêm neuron ẩn) làm tăng số tham số và khả năng biểu diễn. (Minh họa từ video về bản chất mạng neuron)*

![Sơ đồ lớp ẩn](/deep-learning-self-learning/img/chapter_img/chapter02/nn_hidden_layer_diagram.jpg)
*Hình: Kết nối input → hidden → output tương ứng với ma trận trọng số $$W$$. (Minh họa từ video về bản chất mạng neuron)*

Dưới đây ta sẽ phân tích chi tiết từng thành phần trong công thức này.

### Tính toán Từng Lớp

Xét mạng có $$L$$ lớp (không tính lớp đầu vào). Tại mỗi lớp $$l \in \{1, 2, \ldots, L\}$$, phép tính gồm hai bước:

**Tiền kích hoạt (biến đổi tuyến tính):**
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

**Kích hoạt (biến đổi phi tuyến):**
$$\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})$$

Kích thước của các đại lượng:
- $$\mathbf{a}^{[l-1]} \in \mathbb{R}^{n_{l-1}}$$: đầu vào của lớp $$l$$ (đầu ra từ lớp trước)
- $$\mathbf{W}^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$$: ma trận trọng số
- $$\mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$$: vector độ lệch
- $$\mathbf{z}^{[l]} \in \mathbb{R}^{n_l}$$: vector tiền kích hoạt
- $$\mathbf{a}^{[l]} \in \mathbb{R}^{n_l}$$: vector hậu kích hoạt (đầu ra của lớp)
- $$n_l$$: số lượng neuron ở lớp $$l$$

### Vai trò của Từng Thành phần

**Ma trận trọng số $$\mathbf{W}^{[l]}$$:** Mỗi hàng $$\mathbf{w}_i^{[l]}$$ quy định tổ hợp tuyến tính đầu vào cho một neuron. Phép nhân ma trận $$\mathbf{W}^{[l]} \mathbf{a}^{[l-1]}$$ tính đồng thời giá trị tiền kích hoạt của toàn bộ neuron trong lớp. Trọng số là các tham số học được, liên tục điều chỉnh trong quá trình huấn luyện để nắm bắt cấu trúc dữ liệu.

**Vector độ lệch $$\mathbf{b}^{[l]}$$:** Dịch chuyển hàm kích hoạt sang trái hoặc phải, cho phép neuron kích hoạt ngay cả khi các đầu vào đều gần bằng 0. Nếu thiếu độ lệch, một neuron ReLU khi nhận đầu vào toàn 0 sẽ luôn trả về 0, làm giảm đáng kể khả năng biểu đạt. Độ lệch vì vậy rất cần thiết để học các ngưỡng kích hoạt phù hợp.

**Hàm kích hoạt $$\sigma^{[l]}$$:** Bổ sung tính phi tuyến để mạng học được các biên quyết định phức tạp. Nếu không có hàm kích hoạt phi tuyến, việc xếp chồng nhiều lớp trở nên vô ích vì hợp của nhiều phép biến đổi tuyến tính vẫn chỉ là một phép biến đổi tuyến tính duy nhất. Các lựa chọn phổ biến gồm:
- ReLU: $$\sigma(z) = \max(0, z)$$
- Sigmoid: $$\sigma(z) = \frac{1}{1+e^{-z}}$$
- Tanh: $$\sigma(z) = \tanh(z)$$

### Thiết kế Lớp Đầu ra

Cấu trúc lớp đầu ra phụ thuộc trực tiếp vào bản chất bài toán để đảm bảo đầu ra tương thích với hàm mất mát.

**Phân loại nhị phân** ($$y \in \{0, 1\}$$):
- Một neuron đầu ra với hàm kích hoạt sigmoid
- Diễn giải: $$\hat{y} = P(y=1|\mathbf{x})$$
- Đầu ra: $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0,1)$$
- Hàm mất mát: Entropy chéo nhị phân $$\mathcal{L} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

**Phân loại đa lớp** ($$y \in \{1,2,\ldots,K\}$$):
- $$K$$ neuron đầu ra với hàm kích hoạt softmax
- Diễn giải: $$\hat{y}_k = P(y=k|\mathbf{x})$$
- Đầu ra: $$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$ với $$\sum_{k=1}^K \hat{y}_k = 1$$
- Hàm mất mát: Entropy chéo đa lớp (*categorical cross-entropy*) $$\mathcal{L} = -\sum_{k=1}^K y_k \log \hat{y}_k$$

Hàm softmax có các tính chất rất hữu ích: khả vi, đầu ra tạo thành một phân phối xác suất hợp lệ, và đóng vai trò phiên bản trơn ("mềm") của phép `argmax`, nhờ đó huấn luyện được bằng gradient descent.

**Hồi quy** ($$y \in \mathbb{R}$$):
- Một hoặc nhiều neuron đầu ra với kích hoạt tuyến tính (hàm đồng nhất)
- Đầu ra: $$\hat{y} = z$$ (không áp dụng hàm phi tuyến)
- Hàm mất mát: Sai số bình phương trung bình $$\mathcal{L} = \frac{1}{2}(y - \hat{y})^2$$

### Lan truyền Xuôi: Toàn bộ Quy trình

Với đầu vào $$\mathbf{x} \in \mathbb{R}^{n_0}$$, quá trình lan truyền xuôi diễn ra như sau:

$$
\begin{align}
\mathbf{a}^{[0]} &= \mathbf{x} \quad \text{(khởi tạo bằng đầu vào)} \\
\\
\text{For } l &= 1 \text{ to } L: \\
\mathbf{z}^{[l]} &= \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \quad \text{(biến đổi affine)} \\
\mathbf{a}^{[l]} &= \sigma^{[l]}(\mathbf{z}^{[l]}) \quad \text{(kích hoạt phi tuyến)} \\
\\
\hat{\mathbf{y}} &= \mathbf{a}^{[L]} \quad \text{(đầu ra cuối)}
\end{align}
$$

Chuỗi tính toán tuần tự này xây dựng các biểu diễn dữ liệu ngày càng trừu tượng, giúp mạng xấp xỉ được nhiều loại hàm phức tạp.

### Số Tham số và Độ phức tạp

Tổng số tham số học được của mạng là:

$$\text{Parameters} = \sum_{l=1}^{L} (n_l \times n_{l-1} + n_l) = \sum_{l=1}^{L} n_l(n_{l-1} + 1)$$

Ví dụ cụ thể với kiến trúc [784, 128, 64, 10]:
- Lớp 1: $$128 \times 784 + 128 = 100,480$$ tham số
- Lớp 2: $$64 \times 128 + 64 = 8,256$$ tham số  
- Lớp 3: $$10 \times 64 + 10 = 650$$ tham số
- **Tổng**: $$109,386$$ tham số

Số lượng tham số tăng theo bậc hai theo độ rộng lớp nhưng chỉ tăng tuyến tính theo độ sâu. Đây là một trong những lý do khiến mạng sâu và hẹp thường sử dụng tham số hiệu quả hơn mạng nông nhưng rộng khi cần cùng một năng lực biểu diễn.

### Định lý Xấp xỉ Phổ quát

<div class="textbook-theorem" markdown="1">
**Định lý** (Cybenko 1989, Hornik et al. 1989): Cho $$\sigma$$ là hàm liên tục đơn điệu tăng, bị chặn, không hằng (ví dụ: sigmoid). Khi đó với mọi hàm liên tục $$g$$ trên tập compact $$K \subset \mathbb{R}^n$$, mọi $$\epsilon > 0$$, và mọi độ đo xác suất $$\mu$$ trên $$K$$, tồn tại mạng neuron một lớp ẩn $$f$$ sao cho:

$$\int_K |f(\mathbf{x}) - g(\mathbf{x})| d\mu(\mathbf{x}) < \epsilon$$
</div>

**Ý nghĩa:** Mạng neuron có thể xấp xỉ bất kỳ hàm liên tục nào với độ chính xác tùy ý. Về nguyên lý, điều này khẳng định mạng neuron là bộ xấp xỉ hàm phổ quát (*universal function approximator*).

**Trực giác thực nghiệm (1D):** Trên cùng một tập điểm gợn sóng, mạng nông hoặc dung lượng thấp cho đường khớp rất kém; khi tăng độ rộng hoặc độ sâu và huấn luyện đầy đủ, đường dự đoán bám sát dữ liệu hơn nhiều.

![Mạng nông: khớp kém](/deep-learning-self-learning/img/chapter_img/chapter02/nn_shallow_poor_fit.jpg)
*Hình: Dung lượng thấp — đường dự đoán (hồng) lệch xa dữ liệu. (Minh họa từ video về bản chất mạng neuron)*

![Mạng đủ sâu/rộng: khớp tốt](/deep-learning-self-learning/img/chapter_img/chapter02/nn_deeper_good_fit.jpg)
*Hình: Sau khi mở rộng kiến trúc và huấn luyện — mô hình khớp sát dữ liệu. (Minh họa từ video về bản chất mạng neuron)*

![Dự đoán điểm trên đường cong phức tạp](/deep-learning-self-learning/img/chapter_img/chapter02/nn_predictions_complex_curve.jpg)
*Hình: Lan truyền xuôi tại nhiều giá trị $$x$$ tái tạo lại toàn bộ đường cong hàm số. (Minh họa từ video về bản chất mạng neuron)*

![Neuron sáng theo vùng input](/deep-learning-self-learning/img/chapter_img/chapter02/nn_prediction_neuron_activation.jpg)
*Hình: Với mỗi vùng $$x$$, các nhóm neuron ẩn khác nhau kích hoạt — minh họa cách mạng chia nhỏ không gian đầu vào. (Minh họa từ video về bản chất mạng neuron)*

**Các lưu ý quan trọng:**
1. **Tồn tại không đồng nghĩa với dễ học:** Định lý khẳng định nghiệm tồn tại nhưng không đảm bảo gradient descent sẽ tìm được nghiệm đó.
2. **Yêu cầu về độ rộng:** Mạng một lớp ẩn có thể cần số lượng neuron tăng theo cấp số nhân (theo số chiều đầu vào hoặc độ chính xác $$1/\epsilon$$).
3. **Hiệu quả của độ sâu:** Mạng sâu hơn thường đạt mức xấp xỉ tương đương với số tham số ít hơn nhiều.
4. **Không chỉ dẫn thiết kế cụ thể:** Định lý không cho biết nên chọn hàm kích hoạt, phương pháp khởi tạo hay tốc độ học nào.

Nói cách khác, định lý giải thích *vì sao* mạng neuron có khả năng biểu diễn về mặt nguyên lý, còn hiệu quả thực tế của học sâu đến từ các tiến bộ về độ sâu, kiến trúc, thuật toán tối ưu và chính quy hóa.

## 3. Ví dụ / Trực giác

Để nắm chắc cách hoạt động của kiến trúc mạng neuron, hãy theo dõi luồng thông tin qua từng bước trong một ví dụ cụ thể.

### Ví dụ: Mạng 3 Lớp cho MNIST

Xét một mạng phân loại ảnh chữ số viết tay (ảnh xám kích thước 28×28 thuộc 10 lớp):

**Kiến trúc:** [784 → 128 → 64 → 10]
- **Đầu vào**: 784 pixel (ảnh 28×28 được làm phẳng)
- **Lớp ẩn 1**: 128 neuron với ReLU
- **Lớp ẩn 2**: 64 neuron với ReLU  
- **Lớp đầu ra**: 10 neuron với softmax

### Luồng Thông tin Chi tiết

**Bước 1: Đầu vào (Lớp 0)**

Nhận một ảnh 28×28 của chữ số "3" và làm phẳng thành vector:
$$\mathbf{a}^{[0]} = [0.2, 0.1, 0.0, 0.8, 0.9, \ldots] \in \mathbb{R}^{784}$$

Mỗi phần tử biểu diễn cường độ sáng của một pixel (0 là đen, 1 là trắng). Mạng xem bức ảnh như một điểm trong không gian 784 chiều.

**Bước 2: Lớp Ẩn Thứ nhất (Lớp 1)**

Lớp này gồm 128 neuron, mỗi neuron học cách phát hiện một mẫu cục bộ khác nhau:

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{a}^{[0]} + \mathbf{b}^{[1]}$$

trong đó $$\mathbf{W}^{[1]} \in \mathbb{R}^{128 \times 784}$$ và $$\mathbf{b}^{[1]} \in \mathbb{R}^{128}$$.

Chẳng hạn, trong 128 neuron này:
- Neuron 1 có thể kích hoạt mạnh với cạnh dọc ở góc trên bên trái
- Neuron 2 có thể kích hoạt với đường cong tròn  
- Neuron 3 có thể kích hoạt với nét chéo
- Và tương tự cho các neuron còn lại

Sau khi qua hàm kích hoạt ReLU: $$\mathbf{a}^{[1]} = \max(0, \mathbf{z}^{[1]}) \in \mathbb{R}^{128}$$

Một số neuron cho giá trị dương lớn, số khác bị ReLU đưa về 0. Mạng đã chuyển dữ liệu pixel thô sang một biểu diễn đặc trưng, tương ứng với thông tin như: "ảnh này có cạnh dọc rõ, đường cong ở mức trung bình, nét ngang yếu."

**Bước 3: Lớp Ẩn Thứ hai (Lớp 2)**

Lớp thứ hai kết hợp các đặc trưng mức thấp từ Lớp 1 thành các cấu trúc trừu tượng hơn:

$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$

trong đó $$\mathbf{W}^{[2]} \in \mathbb{R}^{64 \times 128}$$.

Các neuron ở đây có thể phản ứng với các bộ phận lớn hơn:
- Neuron 1: "vòng cong phía trên" (kết hợp các đường cong và cạnh ở nửa trên)
- Neuron 2: "vòng cong phía dưới" (tổ hợp đường cong ở nửa dưới)
- Neuron 3: "nét dọc" (kết hợp các cạnh dọc)

Sau ReLU: $$\mathbf{a}^{[2]} = \max(0, \mathbf{z}^{[2]}) \in \mathbb{R}^{64}$$

Biểu diễn ở giai đoạn này đã giàu ngữ nghĩa hơn: "ảnh có cả vòng cong trên và dưới, giống đặc trưng của các chữ số 3, 8 hoặc 0."

**Bước 4: Lớp Đầu ra (Lớp 3)**

Lớp cuối cùng tính điểm phân loại cho từng chữ số:

$$\mathbf{z}^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]} \in \mathbb{R}^{10}$$

trong đó $$\mathbf{W}^{[3]} \in \mathbb{R}^{10 \times 64}$$.

Các giá trị thô (logits) này sau đó được chuyển sang phân phối xác suất bằng softmax:

$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z}^{[3]})$$

Kết quả có dạng:
$$\hat{\mathbf{y}} = [0.01, 0.02, 0.05, 0.82, 0.03, 0.01, 0.02, 0.02, 0.01, 0.01]$$

Mạng dự đoán với xác suất 82% rằng ảnh là chữ số "3" (chỉ số 3), đồng thời gán xác suất nhỏ cho các chữ số khác có vài đặc trưng tương đồng.

### Vì sao Cấu trúc Lớp này Hoạt động Hiệu quả

**Học đặc trưng phân cấp:** Lớp 1 học cạnh và đường cong, Lớp 2 ghép chúng thành bộ phận chữ số, còn Lớp 3 tổng hợp các bộ phận để đưa ra kết luận cuối cùng. Hệ thống phân cấp này tự hình thành qua huấn luyện mà không cần thiết kế thủ công.

**Biểu diễn phân tán:** Khái niệm chữ số "3" không nằm gói gọn ở một neuron duy nhất mà được mã hóa qua tổ hợp kích hoạt của cả 64 neuron ở Lớp 2. Cách biểu diễn này vừa có độ bền cao (mất một vài neuron không làm hỏng hoàn toàn dự đoán) vừa tiết kiệm (cùng các đặc trưng có thể tái sử dụng cho nhiều chữ số).

**Giảm chiều dần:** Dữ liệu đi từ 784 chiều nén dần xuống 128 → 64 → 10. Quá trình thu hẹp chiều buộc mạng giữ lại các thông tin phân biệt quan trọng và loại bỏ nhiễu không cần thiết.

### Trực giác: Vì sao Độ sâu Hiệu quả hơn Độ rộng

So sánh hai kiến trúc thay thế cho cùng một bài toán:

**Nông – Rộng:** [784 → 4096 → 10]  
- Một lớp ẩn rất rộng với 4096 neuron
- Tổng số tham số: khoảng 3.26M
- Mỗi neuron phải học trực tiếp mẫu hoàn chỉnh từ pixel thô
- Không hình thành hệ thống đặc trưng phân cấp rõ rệt

**Sâu – Hẹp:** [784 → 128 → 64 → 32 → 10]
- Ba lớp ẩn nhỏ hơn
- Tổng số tham số: khoảng 111K (ít hơn gần 29 lần)
- Tự hình thành phân cấp đặc trưng qua các lớp
- Thường khái quát hóa tốt hơn và dễ huấn luyện hơn

Mạng sâu chiếm ưu thế vì hầu hết dữ liệu trong thực tế đều có cấu trúc hợp thành: khuôn mặt tạo nên từ mắt, mũi, miệng; câu văn tạo nên từ cụm từ, từ và chữ cái. Kiến trúc nhiều lớp phản ánh trực tiếp cấu trúc phân tầng đó.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video của Machine Learning Thực Chiến:

- [Trực giác mạng nơ-ron (Perceptron → Deep Learning)](https://www.facebook.com/reel/793200140509765)
- [Mạng nơ-ron là bộ xấp xỉ hàm](https://www.facebook.com/reel/720970114372332)

## Video tham khảo

- **Xem trước:** [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Diễn giải mỗi lớp như một ánh xạ affine kết hợp phi tuyến, khớp với công thức trong bài.
- [The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) — StatQuest with Josh Starmer, tiếng Anh. Minh họa chi tiết một lớp ẩn với kích thước ma trận hiển thị rõ ràng.
- [Lecture 4 | Introduction to Neural Networks](https://www.youtube.com/watch?v=d14TUNcbn1k) — Stanford University School of Engineering, tiếng Anh. Bài giảng trình bày phương trình các lớp và lý giải vai trò của phép hợp hàm trong mạng sâu.
