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

# Kiến trúc Mạng Neuron: Từ Neuron đến Hệ thống Sâu

![Neural Network Layers](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/500px-Colored_neural_network.svg.png)
*Hình ảnh: Kiến trúc neural network với input, hidden và output layers. Nguồn: Wikimedia Commons*

![Mạng fully connected nhiều lớp](/deep-learning-self-learning/img/chapter_img/chapter02/mlp_fully_connected.jpg)
*Hình: Mạng feedforward fully connected — input (hồng), hidden (xanh lá), output (xanh dương). (Minh họa từ video nhập môn neural network)*

## 1. Tổng quan khái niệm
Kiến trúc mạng neuron là bản thiết kế định nghĩa cách các neuron riêng lẻ được tổ chức, kết nối và cấu trúc để giải quyết các bài toán phức tạp. Trong khi một neuron đơn chỉ học được biên quyết định tuyến tính (như ta đã thấy với perceptron), sức mạnh thực sự của học sâu xuất hiện khi ta hợp thành nhiều neuron thành các lớp và xếp các lớp này thành kiến trúc sâu. Cấu trúc hợp thành này không chỉ là tiện lợi kỹ thuật — nó phản ánh một hiểu biết sâu sắc về cách trí tuệ phức tạp có thể nảy sinh từ các đơn vị tính toán đơn giản hoạt động phối hợp.

Hiểu kiến trúc là then chốt vì cách ta tổ chức neuron quyết định về cơ bản những gì mạng có thể học và học hiệu quả như thế nào. Một kiến trúc thiết kế kém có thể thất bại ngay cả với các mẫu đơn giản, trong khi kiến trúc thiết kế tốt có thể khám phá các quan hệ tinh vi trong dữ liệu với hiệu quả đáng kể. Kiến trúc thể hiện các thiên kiến quy nạp (*inductive biases*) của ta — các giả định về cấu trúc bài toán — cho phép mạng học hiệu quả hơn so với việc coi mọi bài toán như xấp xỉ hàm hoàn toàn tổng quát.

### Mô hình Lớp

Nguyên lý tổ chức cơ bản của mạng neuron là **lớp** (*layers*) — các nhóm neuron thực hiện biến đổi ở cùng một giai đoạn tính toán. Cấu trúc lớp này tự nhiên hiện thực hóa tính toán hợp thành: mỗi lớp biến đổi đầu vào thành một biểu diễn mới, và các lớp tiếp theo xây trên các biểu diễn này để tạo đặc trưng ngày càng trừu tượng. Khi nhận diện khuôn mặt, các lớp sớm có thể phát hiện cạnh, các lớp giữa kết hợp cạnh thành đặc trưng khuôn mặt (mắt, mũi, miệng), và các lớp sâu nhận diện danh tính hoàn chỉnh.

**Ví dụ ảnh chữ số.** Ảnh $$28\times28$$ được flatten thành $$784$$ đầu vào, qua các lớp ẩn, rồi tới $$10$$ đầu ra (lớp $$0$$–$$9$$). Không có “công thức cố định” cho số neuron ẩn — đó là lựa chọn kiến trúc — nhưng pipeline luôn là: vector hóa → chồng lớp → đầu ra theo tác vụ.

![MLP nhận diện chữ số: 784 → ẩn → 10](/deep-learning-self-learning/img/chapter_img/chapter02/mnist_mlp_10_outputs.jpg)
*Hình: Pipeline MNIST-style — $$784$$ pixel → lớp ẩn → $$10$$ logit cho các chữ số. (Minh họa từ video nhập môn neural network)*

Xử lý phân cấp này phản ánh cả hệ thống neuron sinh học lẫn bản chất hợp thành của nhiều khái niệm thực tế. Một "chiếc xe" gồm bánh xe, cửa sổ và cửa; các thành phần này gồm hình dạng và texture; các thứ đó gồm cạnh và màu. Các lớp mạng neuron tự nhiên nắm bắt hệ thống phân cấp này qua các biến đổi được học, với mỗi lớp học mức trừu tượng phù hợp với vị trí của nó trong pipeline xử lý.

### Vì sao Kiến trúc Quan trọng: Đánh đổi Độ sâu so với Độ rộng

Một hiểu biết then chốt từ cả lý thuyết lẫn thực hành là **độ sâu** (số lớp) và **độ rộng** (số neuron mỗi lớp) có ảnh hưởng về cơ bản khác nhau lên dung lượng mạng và việc học. Định lý Xấp xỉ Phổ quát cho biết một lớp ẩn đơn với đủ nhiều neuron có thể xấp xỉ bất kỳ hàm liên tục nào. Song trong thực tế, mạng sâu với tương đối ít neuron mỗi lớp vượt trội rõ rệt so với mạng nông rộng trên các tác vụ phức tạp.

Điều này không chỉ về hiệu quả tham số, dù mạng sâu thường đạt cùng sức mạnh biểu diễn với số tham số ít hơn theo cấp số nhân so với mạng nông. Mạng sâu học đặc trưng phân cấp một cách tự nhiên — ta không cần bảo chúng phát hiện cạnh trước, rồi hình dạng, rồi đối tượng; điều này tự xuất hiện từ quá trình huấn luyện. Chúng cũng khái quát hóa tốt hơn: các biểu diễn trung gian do mạng sâu học chuyển giao được giữa các tác vụ, cho phép các kỹ thuật mạnh như học chuyển giao (*transfer learning*) và tiền huấn luyện mà mạng nông không hỗ trợ tốt bằng.

Hiểu đánh đổi giữa độ sâu và độ rộng, và cách lựa chọn kiến trúc ảnh hưởng động lực huấn luyện, khái quát hóa và hiệu quả tính toán, là thiết yếu để thiết kế mạng neuron hiệu quả. Bài này cung cấp hiểu biết nền tảng về cách mạng được cấu trúc, vì sao các cấu trúc này hoạt động, và cách đưa ra quyết định kiến trúc có cơ sở cho các ứng dụng của mình.

## 2. Nền tảng toán học
### Mạng Neuron Feedforward: Định nghĩa Hình thức

Một **mạng neuron feedforward** (còn gọi là **Multilayer Perceptron** hay **MLP**) là một hàm $$f: \mathbb{R}^{n_0} \to \mathbb{R}^{n_L}$$ định nghĩa bởi hợp thành các biến đổi lớp. Với mạng có $$L$$ lớp, hàm là:

$$f(\mathbf{x}) = f^{[L]} \circ f^{[L-1]} \circ \cdots \circ f^{[1]}(\mathbf{x})$$

trong đó mỗi hàm lớp $$f^{[l]}$$ là một biến đổi affine theo sau bởi phi tuyến tính theo từng phần tử:

$$f^{[l]}(\mathbf{a}^{[l-1]}) = \sigma^{[l]}(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})$$

![Công thức lớp ẩn + kích hoạt](/deep-learning-self-learning/img/chapter_img/chapter02/nn_hidden_activation_formula.jpg)
*Hình: Một lớp ẩn $$z_1=\sigma(xw+b)$$ rồi lớp tuyến tính ra $$f(x)$$. (Minh họa từ video về bản chất mạng nơ-ron)*

![Mở rộng độ rộng (nhiều neuron ẩn)](/deep-learning-self-learning/img/chapter_img/chapter02/nn_width_parameters_formula.jpg)
*Hình: Tăng chiều $$W$$ (nhiều neuron ẩn) tăng số tham số và sức biểu diễn. (Minh họa từ video về bản chất mạng nơ-ron)*

![Sơ đồ lớp ẩn](/deep-learning-self-learning/img/chapter_img/chapter02/nn_hidden_layer_diagram.jpg)
*Hình: Kết nối input → hidden → output tương ứng ma trận $$W$$. (Minh họa từ video về bản chất mạng nơ-ron)*

Ta sẽ phân tích cẩn thận từng thành phần và hiểu vì sao công thức tưởng đơn giản này lại mạnh mẽ đến vậy.

### Tính toán Từng Lớp

Với mạng có $$L$$ lớp (không đếm đầu vào), lớp $$l \in \{1, 2, \ldots, L\}$$ tính:

**Tiền kích hoạt (biến đổi tuyến tính)**:
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

**Kích hoạt (biến đổi phi tuyến)**:
$$\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})$$

Các chiều là:
- $$\mathbf{a}^{[l-1]} \in \mathbb{R}^{n_{l-1}}$$: đầu vào của lớp $$l$$ (đầu ra từ lớp trước)
- $$\mathbf{W}^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$$: ma trận trọng số
- $$\mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$$: vectơ độ lệch
- $$\mathbf{z}^{[l]} \in \mathbb{R}^{n_l}$$: giá trị tiền kích hoạt
- $$\mathbf{a}^{[l]} \in \mathbb{R}^{n_l}$$: giá trị hậu kích hoạt (đầu ra lớp)
- $$n_l$$: số neuron ở lớp $$l$$

### Vai trò của Từng Thành phần

**Ma trận trọng số $$\mathbf{W}^{[l]}$$**: Mỗi hàng $$\mathbf{w}_i^{[l]}$$ định nghĩa tổ hợp tuyến tính đầu vào của một neuron. Phép nhân ma trận $$\mathbf{W}^{[l]} \mathbf{a}^{[l-1]}$$ tính tiền kích hoạt của mọi neuron song song. Các trọng số là tham số có thể học được, thích nghi trong huấn luyện để nắm bắt mẫu trong dữ liệu.

**Vectơ độ lệch $$\mathbf{b}^{[l]}$$**: Dịch hàm kích hoạt sang trái hoặc phải, cho phép neuron kích hoạt ngay cả khi đầu vào gần không. Không có độ lệch, một neuron ReLU với mọi đầu vào bằng không sẽ luôn cho ra không, hạn chế khả năng biểu đạt. Độ lệch then chốt để học các ngưỡng phù hợp.

**Hàm kích hoạt $$\sigma^{[l]}$$**: Đưa vào phi tuyến tính, cho phép mạng học biên quyết định phi tuyến. Không có hàm kích hoạt, xếp lớp sẽ vô nghĩa — nhiều biến đổi tuyến tính hợp thành thành một biến đổi tuyến tính đơn. Các lựa chọn phổ biến gồm:
- ReLU: $$\sigma(z) = \max(0, z)$$
- Sigmoid: $$\sigma(z) = \frac{1}{1+e^{-z}}$$
- Tanh: $$\sigma(z) = \tanh(z)$$

### Thiết kế Lớp Đầu ra

Cấu trúc lớp đầu ra phụ thuộc cơ bản vào tác vụ, vì nó phải sinh đầu ra ở định dạng phù hợp với hàm mất mát.

**Phân loại nhị phân** ($$y \in \{0, 1\}$$):
- Một neuron đầu ra với kích hoạt sigmoid
- Diễn giải: $$\hat{y} = P(y=1|\mathbf{x})$$
- Đầu ra: $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0,1)$$
- Mất mát: Entropy chéo nhị phân $$\mathcal{L} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

**Phân loại đa lớp** ($$y \in \{1,2,\ldots,K\}$$):
- $$K$$ neuron đầu ra với kích hoạt softmax
- Diễn giải: $$\hat{y}_k = P(y=k|\mathbf{x})$$
- Đầu ra: $$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$ với $$\sum_{k=1}^K \hat{y}_k = 1$$
- Mất mát: Entropy chéo categorical $$\mathcal{L} = -\sum_{k=1}^K y_k \log \hat{y}_k$$

Hàm softmax có các tính chất thanh lịch: khả vi, đầu ra tạo thành phân phối xác suất, và "làm mềm" phép argmax (do đó có tên), cho phép học dựa trên gradient.

**Hồi quy** ($$y \in \mathbb{R}$$):
- Một hoặc nhiều neuron đầu ra với kích hoạt tuyến tính (đồng nhất)
- Đầu ra: $$\hat{y} = z$$ (không có hàm kích hoạt)
- Mất mát: Sai số bình phương trung bình $$\mathcal{L} = \frac{1}{2}(y - \hat{y})^2$$

### Lan truyền Xuôi: Bức tranh Đầy đủ

Cho đầu vào $$\mathbf{x} \in \mathbb{R}^{n_0}$$, lan truyền xuôi tính:

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

Tính toán tuần tự này xây các biểu diễn ngày càng phức tạp. Mỗi lớp học đặc trưng ở mức trừu tượng khác nhau, với hợp thành các lớp cho phép mạng biểu diễn các hàm cực kỳ phức tạp.

### Số Tham số và Độ phức tạp

Tổng số tham số có thể học là:

$$\text{Parameters} = \sum_{l=1}^{L} (n_l \times n_{l-1} + n_l) = \sum_{l=1}^{L} n_l(n_{l-1} + 1)$$

Ví dụ cụ thể với kiến trúc [784, 128, 64, 10]:
- Lớp 1: $$128 \times 784 + 128 = 100,480$$ tham số
- Lớp 2: $$64 \times 128 + 64 = 8,256$$ tham số  
- Lớp 3: $$10 \times 64 + 10 = 650$$ tham số
- **Tổng**: $$109,386$$ tham số

Số tham số tăng theo bậc hai với độ rộng lớp nhưng chỉ tuyến tính với độ sâu, giải thích vì sao mạng sâu hẹp thường hiệu quả tham số hơn mạng nông rộng với dung lượng biểu diễn tương tự.

### Định lý Xấp xỉ Phổ quát

**Định lý** (Cybenko 1989, Hornik et al. 1989): Cho $$\sigma$$ là hàm liên tục đơn điệu tăng, bị chặn, không hằng (ví dụ: sigmoid). Khi đó với mọi hàm liên tục $$g$$ trên tập compact $$K \subset \mathbb{R}^n$$, mọi $$\epsilon > 0$$, và mọi độ đo xác suất $$\mu$$ trên $$K$$, tồn tại mạng neuron một lớp ẩn $$f$$ sao cho:

$$\int_K |f(\mathbf{x}) - g(\mathbf{x})| d\mu(\mathbf{x}) < \epsilon$$

**Ý nghĩa**: Mạng neuron có thể xấp xỉ bất kỳ hàm liên tục nào với độ chính xác tùy ý. Điều này đáng chú ý — nghĩa là mạng neuron là bộ xấp xỉ hàm phổ quát, có khả năng biểu diễn bất kỳ quan hệ nào ta muốn học.

**Trực giác thực nghiệm (1D).** Cùng một tập điểm gợn sóng: mạng nông / dung lượng thấp cho đường khớp kém; tăng độ rộng/độ sâu + huấn luyện đủ thì đường dự đoán bám dữ liệu gần như hoàn hảo.

![Mạng nông: khớp kém](/deep-learning-self-learning/img/chapter_img/chapter02/nn_shallow_poor_fit.jpg)
*Hình: Dung lượng thấp — đường dự đoán (hồng) lệch xa dữ liệu. (Minh họa từ video về bản chất mạng nơ-ron)*

![Mạng đủ sâu/rộng: khớp tốt](/deep-learning-self-learning/img/chapter_img/chapter02/nn_deeper_good_fit.jpg)
*Hình: Sau khi mở rộng kiến trúc và huấn luyện — mô hình khớp gần như hoàn hảo. (Minh họa từ video về bản chất mạng nơ-ron)*

![Dự đoán điểm trên đường cong phức tạp](/deep-learning-self-learning/img/chapter_img/chapter02/nn_predictions_complex_curve.jpg)
*Hình: Lan truyền xuôi tại nhiều $$x$$ “vẽ” lại toàn bộ hàm. (Minh họa từ video về bản chất mạng nơ-ron)*

![Neuron sáng theo vùng input](/deep-learning-self-learning/img/chapter_img/chapter02/nn_prediction_neuron_activation.jpg)
*Hình: Với mỗi $$x$$, tập neuron ẩn khác nhau “sáng” — trực giác chia không gian input. (Minh họa từ video về bản chất mạng nơ-ron)*

**Các lưu ý quan trọng**:
1. **Tồn tại ≠ Khả học**: Định lý đảm bảo nghiệm tồn tại nhưng không cho biết cách tìm qua gradient descent
2. **Yêu cầu về độ rộng**: Có thể cần số neuron theo cấp số nhân (theo chiều đầu vào hoặc độ chính xác $$1/\epsilon$$)
3. **Hiệu quả độ sâu**: Mạng sâu hơn thường đạt cùng xấp xỉ với ít tham số hơn theo cấp số nhân
4. **Không hướng dẫn kiến trúc**: Không cho biết dùng hàm kích hoạt, khởi tạo, hay learning rate nào

Định lý giải thích *vì sao* mạng neuron hoạt động về nguyên tắc, nhưng thành công thực tế của học sâu đến từ các hiểu biết bổ sung về độ sâu, kiến trúc, tối ưu và chính quy hóa mà định lý không đề cập.

## 3. Ví dụ / Trực giác

Để củng cố hiểu biết về kiến trúc mạng neuron, ta lần theo một ví dụ cụ thể từng bước, quan sát thông tin biến đổi khi chảy qua các lớp.

### Ví dụ: Mạng 3 Lớp cho MNIST

Xét mạng được thiết kế để phân loại chữ số viết tay (ảnh xám 28×28 thành 10 lớp):

**Kiến trúc**: [784 → 128 → 64 → 10]
- **Đầu vào**: 784 pixel (28×28 làm phẳng)
- **Lớp ẩn 1**: 128 neuron với ReLU
- **Lớp ẩn 2**: 64 neuron với ReLU  
- **Lớp đầu ra**: 10 neuron với softmax

### Luồng Thông tin: Đi qua Chi tiết

**Bước 1: Đầu vào (Lớp 0)**

Ta nhận ảnh 28×28 của chữ số "3". Làm phẳng thành vectơ:
$$\mathbf{a}^{[0]} = [0.2, 0.1, 0.0, 0.8, 0.9, \ldots] \in \mathbb{R}^{784}$$

Mỗi giá trị biểu diễn cường độ pixel (0=đen, 1=trắng). Mạng nhìn đây như một điểm trong không gian 784 chiều.

**Bước 2: Lớp Ẩn Thứ nhất (Lớp 1)**

Lớp này có 128 neuron, mỗi neuron tìm các mẫu khác nhau:

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{a}^{[0]} + \mathbf{b}^{[1]}$$

trong đó $$\mathbf{W}^{[1]} \in \mathbb{R}^{128 \times 784}$$ và $$\mathbf{b}^{[1]} \in \mathbb{R}^{128}$$.

Mỗi trong 128 neuron tính:
- Neuron 1 có thể kích hoạt với cạnh dọc ở góc trên-trái
- Neuron 2 có thể kích hoạt với đường cong tròn  
- Neuron 3 có thể kích hoạt với nét chéo
- ... và cứ thế

Sau kích hoạt ReLU: $$\mathbf{a}^{[1]} = \max(0, \mathbf{z}^{[1]}) \in \mathbb{R}^{128}$$

Một số neuron phát mạnh (giá trị gần cực đại), số khác không phát (bị ReLU triệt tiêu). Mạng đã biến đổi biểu diễn pixel thô thành biểu diễn đặc trưng: "ảnh này có cạnh dọc mạnh, đường cong vừa, nét ngang yếu."

**Bước 3: Lớp Ẩn Thứ hai (Lớp 2)**

Lớp này kết hợp các đặc trưng mức thấp từ Lớp 1 thành khái niệm mức cao hơn:

$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$

trong đó $$\mathbf{W}^{[2]} \in \mathbb{R}^{64 \times 128}$$.

64 neuron này có thể nhận diện:
- Neuron 1: "vòng trên" (kết hợp đường cong và cạnh ở vị trí trên)
- Neuron 2: "vòng dưới" (tổ hợp đường cong khác)
- Neuron 3: "nét dọc" (kết hợp cạnh dọc)

Sau ReLU: $$\mathbf{a}^{[2]} = \max(0, \mathbf{z}^{[2]}) \in \mathbb{R}^{64}$$

Biểu diễn nay còn trừu tượng hơn: "ảnh này có vòng trên và vòng dưới, đặc trưng của các chữ số như 3, 8, hoặc có thể 0."

**Bước 4: Lớp Đầu ra (Lớp 3)**

Lớp cuối đưa ra quyết định phân loại:

$$\mathbf{z}^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]} \in \mathbb{R}^{10}$$

trong đó $$\mathbf{W}^{[3]} \in \mathbb{R}^{10 \times 64}$$.

Điều này cho điểm số thô (logits) cho mỗi chữ số. Để chuyển thành xác suất:

$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z}^{[3]})$$

Kết quả dạng:
$$\hat{\mathbf{y}} = [0.01, 0.02, 0.05, 0.82, 0.03, 0.01, 0.02, 0.02, 0.01, 0.01]$$

Mạng tin 82% đây là "3" (chỉ số 3), với một phần xác suất trên các chữ số khác chia sẻ đặc trưng tương tự.

### Vì sao Cấu trúc Lớp này Hoạt động

**Học đặc trưng phân cấp**: Lớp 1 học cạnh và đường cong. Lớp 2 kết hợp chúng thành bộ phận chữ số. Lớp 3 kết hợp bộ phận thành dự đoán chữ số hoàn chỉnh. Hệ thống phân cấp này tự xuất hiện từ huấn luyện — ta không bao giờ bảo tường minh mạng phát hiện cạnh trước!

**Biểu diễn phân tán**: Chữ số "3" không được biểu diễn bởi một neuron đơn mà bởi mẫu kích hoạt trên cả 64 neuron ở Lớp 2. Điều này làm biểu diễn bền vững (mất vài neuron không hủy khái niệm) và hiệu quả (cùng đặc trưng giúp nhận diện nhiều chữ số).

**Giảm chiều**: Ta bắt đầu với 784 chiều và nén qua 128 → 64 → 10. Mỗi lần giảm buộc mạng trích xuất thông tin thiết yếu hơn, loại bỏ nhiễu và chi tiết không liên quan trong khi bảo toàn đặc trưng phân biệt.

### Trực giác: Vì sao Độ sâu Thắng Độ rộng

Xét hai mạng thay thế cho cùng tác vụ:

**Nông-Rộng**: [784 → 4096 → 10]  
- Một lớp ẩn khổng lồ với 4096 neuron
- Tổng tham số: ~3.2M
- Mỗi neuron phải học mẫu hoàn chỉnh từ pixel thô
- Không có hệ thống phân cấp đặc trưng tường minh

**Sâu-Hẹp**: [784 → 128 → 64 → 32 → 10]
- Bốn lớp ẩn nhỏ hơn
- Tổng tham số: ~110K (ít hơn 29×!)
- Hệ thống phân cấp đặc trưng tự xuất hiện
- Khái quát hóa tốt hơn, dễ huấn luyện hơn

Mạng sâu thắng vì hầu hết mẫu thực tế mang tính hợp thành. Khuôn mặt gồm mắt, mũi, miệng (không phải mẫu pixel ngẫu nhiên). Câu gồm cụm từ, gồm từ, gồm chữ cái. Mạng sâu tự nhiên nắm bắt cấu trúc hợp thành này qua kiến trúc lớp của chúng.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Trực giác mạng nơ-ron (Perceptron → Deep Learning)](https://www.facebook.com/reel/793200140509765)
- [Mạng nơ-ron là bộ xấp xỉ hàm](https://www.facebook.com/reel/720970114372332)
