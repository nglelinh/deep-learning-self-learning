---
layout: post
title: 02-01 Perceptron và Neuron Nhân tạo
chapter: '02'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

# Perceptron và Neuron Nhân tạo

![Perceptron Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Perceptron_moj.png/400px-Perceptron_moj.png)
*Hình ảnh: Sơ đồ cấu trúc của một Perceptron. Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm

Hành trình tìm hiểu học sâu bắt đầu từ đơn vị cơ bản nhất: neuron nhân tạo (*artificial neuron*). Dù học sâu hiện đại đã tiến rất xa so với perceptron đơn giản do Frank Rosenblatt giới thiệu năm 1958, việc nắm rõ điểm xuất phát lịch sử này giúp ta hiểu vì sao các mạng neuron ngày nay lại được thiết kế như vậy. Perceptron là một trong những nỗ lực đầu tiên nhằm xây dựng hệ thống có thể tự học từ ví dụ, lấy cảm hứng đơn giản hóa từ cách neuron sinh học xử lý thông tin.

Về bản chất, perceptron là một bộ phân loại nhị phân. Nó nhận nhiều đầu vào dạng số, kết hợp chúng thông qua các trọng số (*weights*) học được, rồi đưa ra một đầu ra nhị phân cho biết mẫu dữ liệu thuộc lớp nào trong hai lớp. Điểm quan trọng ở cơ chế này là khả năng tự động học trọng số từ dữ liệu: thuật toán lặp lại việc điều chỉnh trọng số cho đến khi phân loại đúng các mẫu huấn luyện. Dù đơn giản so với chuẩn mực hiện đại, ý tưởng này đã tạo ra bước ngoặt lớn vào thời điểm ra đời và đặt nền móng cho toàn bộ lĩnh vực mạng neuron sau này.

Hiểu rõ perceptron rất cần thiết vì nó giới thiệu nhiều khái niệm cốt lõi vẫn được dùng xuyên suốt học sâu. Cơ chế tổng có trọng số phản ánh mức độ đóng góp khác nhau của từng đặc trưng vào quyết định cuối cùng. Hạng tử độ lệch (*bias*) cho phép dịch chuyển biên quyết định ra khỏi gốc tọa độ. Quy tắc học minh họa cách cập nhật tham số dựa trên sai số dự đoán. Quan trọng hơn cả, hạn chế căn bản của perceptron — không thể giải các bài toán không tách tuyến tính như XOR — đã trực tiếp thúc đẩy sự ra đời của mạng nhiều lớp và các kiến trúc sâu trong học sâu hiện đại.

Bước chuyển từ perceptron sang neuron nhân tạo hiện đại nằm ở việc thay thế hàm bước cứng bằng các hàm kích hoạt trơn và khả vi. Thay đổi tưởng chừng nhỏ này lại có tác động rất lớn: hàm kích hoạt trơn cho phép tối ưu dựa trên gradient thông qua lan truyền ngược (*backpropagation*), từ đó huấn luyện được mạng nhiều lớp. Đồng thời, chúng bổ sung tính phi tuyến cần thiết để mạng neuron xấp xỉ các hàm phức tạp. Việc lựa chọn hàm kích hoạt ảnh hưởng trực tiếp đến tốc độ hội tụ cũng như khả năng biểu diễn dữ liệu, khiến đây trở thành một quyết định kiến trúc quan trọng trong học sâu.

## 2. Nền tảng toán học

![Biological vs Artificial Neuron](https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Blausen_0657_MultipolarNeuron.png/500px-Blausen_0657_MultipolarNeuron.png)
*Hình ảnh: Neuron sinh học (trái) đã truyền cảm hứng cho neuron nhân tạo. Nguồn: Wikimedia Commons*

Perceptron thực hiện một phép tính khá đơn giản, nhưng công thức toán học của nó làm rõ bản chất của bộ phân loại tuyến tính và biên quyết định. Cho vector đầu vào $$\mathbf{x} = (x_1, x_2, \ldots, x_n)$$ với mỗi $$x_i$$ là một đặc trưng, và vector trọng số tương ứng $$\mathbf{w} = (w_1, w_2, \ldots, w_n)$$, perceptron trước hết tính tổng có trọng số:

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

Đại lượng $$z$$, thường gọi là giá trị tiền kích hoạt (*pre-activation*) hoặc logit, biểu diễn tổ hợp tuyến tính của các đầu vào. Mỗi trọng số $$w_i$$ quyết định mức độ ảnh hưởng của đặc trưng $$x_i$$ tới kết quả cuối cùng. Hạng tử độ lệch $$b$$ đóng vai trò như một ngưỡng dịch chuyển, giúp đặt biên quyết định ở vị trí phù hợp trong không gian đầu vào thay vì buộc phải đi qua gốc tọa độ.

Tiếp theo, perceptron áp dụng hàm bước Heaviside lên tổ hợp tuyến tính này để tạo ra đầu ra nhị phân:

$$y = H(z) = \begin{cases} 
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}$$

![Đồ thị hàm bước Heaviside](/deep-learning-self-learning/img/chapter_img/chapter02/heaviside_step_function.jpg)
*Hình: Hàm bước Heaviside $$H(x)$$ — đầu ra chỉ nhận giá trị $$0$$ hoặc $$1$$. (Minh họa từ video nhập môn mạng neuron)*

Hàm bước tạo ra một biên quyết định sắc nét: mọi điểm ở một phía được gán nhãn dương (1), phía còn lại được gán nhãn âm (0). Về mặt hình học, các trọng số xác định một siêu phẳng (*hyperplane*) trong không gian đầu vào theo phương trình $$\mathbf{w}^T \mathbf{x} + b = 0$$, và các điểm dữ liệu được phân loại tùy theo phía của siêu phẳng mà chúng nằm trên đó.

Vector trọng số $$\mathbf{w}$$ vuông góc (trực giao) với siêu phẳng quyết định. Có thể thấy ngay điều này: nếu hai điểm $$\mathbf{x}_1$$ và $$\mathbf{x}_2$$ cùng nằm trên siêu phẳng thì $$\mathbf{w}^T(\mathbf{x}_1 - \mathbf{x}_2) = 0$$, chứng tỏ $$\mathbf{w}$$ trực giao với mọi vector nằm trong siêu phẳng. Độ lớn của $$\mathbf{w}$$ quyết định tốc độ biến thiên của $$z$$ khi di chuyển theo phương vuông góc với siêu phẳng, còn độ lệch $$b$$ điều khiển khoảng cách từ siêu phẳng tới gốc tọa độ dọc theo hướng $$\mathbf{w}$$.

Thuật toán học perceptron đưa ra cách đơn giản để tìm bộ trọng số phù hợp khi dữ liệu tách tuyến tính được. Bắt đầu từ các trọng số khởi tạo (thường bằng 0 hoặc giá trị ngẫu nhiên nhỏ), thuật toán duyệt qua từng mẫu huấn luyện $$(\mathbf{x}_i, y_i)$$. Nếu dự đoán đúng, trọng số được giữ nguyên; nếu phân loại sai, thuật toán cập nhật theo công thức:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$$

$$b \leftarrow b + \eta (y_i - \hat{y}_i)$$

Trong đó, $$\eta$$ là tốc độ học (*learning rate*) điều chỉnh kích thước bước cập nhật, còn $$(y_i - \hat{y}_i)$$ là sai số. Khi nhãn thật là 1 nhưng mô hình dự đoán 0, sai số bằng +1 nên trọng số được dịch chuyển theo hướng của $$\mathbf{x}_i$$, làm tăng khả năng mẫu này được phân vào lớp 1 ở các lần sau. Ngược lại, khi dự đoán 1 nhưng nhãn thật là 0, trọng số được dịch ra xa $$\mathbf{x}_i$$. Trực giác hình học này — xoay và dịch biên quyết định để phân tách đúng các điểm dữ liệu — là nền tảng của nhiều thuật toán học máy.

Định lý Hội tụ Perceptron đảm bảo rằng nếu tập dữ liệu huấn luyện tách tuyến tính được, thuật toán sẽ tìm ra siêu phẳng phân tách sau hữu hạn bước. Tuy nhiên, định lý này cũng bộc lộ giới hạn căn bản của perceptron: nó không thể giải quyết các bài toán mà hai lớp không tách tuyến tính được. Ví dụ kinh điển là hàm XOR, nơi các điểm lớp dương và lớp âm nằm xen kẽ nhau khiến không tồn tại đường thẳng (hay siêu phẳng ở không gian nhiều chiều) nào phân tách được chúng. Hạn chế này là một trong những nguyên nhân dẫn đến "mùa đông AI" đầu tiên vào thập niên 1970, khi người ta nhận ra perceptron một lớp không đủ khả năng giải nhiều bài toán thực tế.

Neuron nhân tạo hiện đại khắc phục hạn chế đó trong khi vẫn giữ lại phép tính cốt lõi là tổng có trọng số. Thay cho hàm bước, ta sử dụng một hàm kích hoạt trơn $$\sigma$$:

$$a = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

![Activation Functions](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Gjl-t%28x%29.svg/500px-Gjl-t%28x%29.svg.png)
*Hình ảnh: Các hàm kích hoạt phổ biến (Sigmoid, Tanh, ReLU). Nguồn: Wikimedia Commons*

Cách chọn hàm $$\sigma$$ tác động lớn đến hành vi của neuron. Hàm sigmoid $$\sigma(z) = \frac{1}{1+e^{-z}}$$ chuyển tiếp mượt giữa 0 và 1, cho phép diễn giải như xác suất và quan trọng nhất là khả vi tại mọi điểm. Hàm tang hyperbolic $$\tanh(z)$$ nhận giá trị trong khoảng $$(-1, 1)$$ và đối xứng quanh 0, thường giúp luồng gradient ổn định hơn. Đơn vị tuyến tính chỉnh lưu (ReLU), định nghĩa là $$\max(0, z)$$, trở thành lựa chọn phổ biến nhất trong học sâu hiện đại vì tính toán nhanh, không bị bão hòa ở miền dương (giảm hiện tượng gradient biến mất) và tạo ra biểu diễn thưa khi các đầu vào âm đều cho đầu ra bằng 0.

## 3. Ví dụ / Trực giác

### 3.1. Trực giác hình học: từ 1D đến 3D

Trước khi quay lại bài toán AND và XOR, hãy hình dung perceptron như một **ranh giới** trong không gian đầu vào.

**Một chiều (ví dụ nhiệt độ):** Neuron nhận một giá trị $$x$$ (độ C). Trọng số và độ lệch xác định một ngưỡng — chẳng hạn tại $$20$$: một phía là trạng thái *Inactive*, phía kia là *Active*. Perceptron chỉ thực hiện việc bật hoặc tắt tùy thuộc đầu vào nằm bên nào của ngưỡng.

![Ranh giới 1D: Inactive vs Active](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_1d_threshold.jpg)
*Hình: Perceptron 1D đặt ranh giới (ví dụ tại vị trí 20) trên trục Input Space. (Minh họa từ video nhập môn mạng neuron)*

Khi đổi **dấu của trọng số**, vùng kích hoạt sẽ đảo ngược. Ví dụ, $$\hat{y} = H(-x + 20)$$ kích hoạt khi $$x$$ nhỏ (vùng *Active* nằm bên trái ngưỡng 20).

![Đảo ranh giới bằng dấu weight](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_1d_weight_sign.jpg)
*Hình: $$\hat{y}=H(-x+20)$$ — trọng số âm đảo ngược vùng Active và Inactive. (Minh họa từ video nhập môn mạng neuron)*

**Hai chiều (ví dụ Temp và Humidity):** Với hai đặc trưng, biên quyết định trở thành một **đường thẳng** trên mặt phẳng. Các điểm thuộc hai lớp nằm về hai phía của đường thẳng (đường màu vàng trong hình dưới).

![Biên quyết định 2D](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_2d_decision_boundary.jpg)
*Hình: Perceptron hai đầu vào (Temp, Humidity) và đường phân tách tuyến tính trên mặt phẳng. (Minh họa từ video nhập môn mạng neuron)*

Ta có thể viết gọn phép tính dưới dạng vector/ma trận:

![Dạng vector của perceptron](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_vector_form.jpg)
*Hình: $$\hat{y}=H(w_1x_1+w_2x_2+b)=H(\mathbf{W}\mathbf{x}+b)$$. (Minh họa từ video nhập môn mạng neuron)*

**Ba chiều trở lên:** Khi thêm đặc trưng thứ ba (ví dụ Wind Speed), biên quyết định mở rộng thành một **siêu phẳng** trong không gian đầu vào:

![Siêu phẳng quyết định 3D](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_3d_hyperplane.jpg)
*Hình: Ba đầu vào (Temp, Humidity, Wind Speed) và siêu phẳng phân loại trong không gian 3D. (Minh họa từ video nhập môn mạng neuron)*

![Dữ liệu tách tuyến tính được](/deep-learning-self-learning/img/chapter_img/chapter02/linearly_separable.jpg)
*Hình: Hai lớp tách tuyến tính được (*linearly separable*) — chỉ cần một đường thẳng để phân tách. Khi dữ liệu không tách tuyến tính được, cần bổ sung tính phi tuyến và nhiều lớp. (Minh họa từ video nhập môn mạng neuron)*

### 3.2. Ví dụ logic: AND và XOR

Để hiểu rõ hơn cách perceptron vận hành, hãy xét một ví dụ cụ thể cho thấy cả ưu điểm lẫn giới hạn của nó: hàm logic AND, chỉ trả về 1 khi cả hai đầu vào đều bằng 1. Dù bài toán này đơn giản, việc một mô hình có thể tự học quan hệ logic chỉ từ các ví dụ đã là một bước tiến quan trọng vào thời điểm đó.

Giả sử ta muốn perceptron học hàm AND từ bốn mẫu: $$(0,0) \to 0$$, $$(0,1) \to 0$$, $$(1,0) \to 0$$, $$(1,1) \to 1$$. Khởi tạo $$w_1 = 0, w_2 = 0, b = 0$$ và chọn tốc độ học $$\eta = 1$$. Với mẫu đầu tiên $$(0,0)$$ có nhãn 0, mô hình dự đoán $$H(0 \cdot 0 + 0 \cdot 0 + 0) = H(0) = 1$$, bị sai. Sai số là $$0 - 1 = -1$$, nên ta cập nhật: $$w_1 \leftarrow 0 + 1 \cdot (-1) \cdot 0 = 0$$, $$w_2 \leftarrow 0$$, $$b \leftarrow 0 + 1 \cdot (-1) = -1$$. Lúc này độ lệch đã trở thành -1, tạo ra một ngưỡng kích hoạt.

Tiếp tục lặp qua các mẫu huấn luyện, perceptron sẽ hội tụ về một bộ tham số chẳng hạn như $$w_1 = 1, w_2 = 1, b = -1.5$$. Kiểm tra lại: với đầu vào $$(1,1)$$, ta có $$z = 1 \cdot 1 + 1 \cdot 1 - 1.5 = 0.5$$, suy ra $$H(0.5) = 1$$ ✓. Với $$(1,0)$$, ta có $$z = 1 \cdot 1 + 1 \cdot 0 - 1.5 = -0.5$$, suy ra $$H(-0.5) = 0$$ ✓. Như vậy, perceptron đã học được quy tắc rằng cả hai đầu vào phải cùng bằng 1 để tổng vượt ngưỡng 1.5.

Bây giờ hãy xét vì sao một perceptron đơn lẻ không thể giải bài toán XOR. Hàm XOR trả về 1 khi hai đầu vào khác nhau: $$(0,0) \to 0$$, $$(0,1) \to 1$$, $$(1,0) \to 1$$, $$(1,1) \to 0$$. Khi biểu diễn bốn điểm này trên mặt phẳng 2D, hai mẫu dương $$(0,1)$$ và $$(1,0)$$ nằm trên một đường chéo, còn hai mẫu âm $$(0,0)$$ và $$(1,1)$$ nằm trên đường chéo kia. Không có đường thẳng nào có thể tách riêng hai nhóm này — ta cần một biên quyết định phi tuyến, chẳng hạn một đường cong hoặc tổ hợp nhiều đoạn thẳng. Giới hạn hình học này cho thấy một mô hình tuyến tính một lớp không thể biểu diễn mọi quan hệ logic.

Chính hạn chế này đã thúc đẩy sự phát triển của mạng nhiều lớp. Nếu xếp hai perceptron ở lớp đầu rồi nối đầu ra của chúng vào một perceptron thứ ba ở lớp sau, ta có thể giải được XOR: lớp đầu học cách phát hiện $$x_1 \text{ OR } x_2$$ và $$x_1 \text{ AND } x_2$$, còn lớp thứ hai học quy tắc "OR nhưng không AND", tức là đúng bằng XOR. Ví dụ này minh họa một nguyên lý cốt lõi: tăng độ sâu (nhiều lớp) cho phép mạng học các biên quyết định phức tạp hơn. Mỗi lớp mới kết hợp các đặc trưng từ lớp trước theo những cách mới, mở rộng mạnh mẽ tập các hàm mà mạng có thể biểu diễn.

Sự liên hệ với sinh học, dù chỉ mang tính gần đúng, vẫn đem lại trực giác hữu ích. Neuron sinh học trong não nhận tín hiệu qua sợi nhánh (*dendrite*), tổng hợp tín hiệu tại thân tế bào (*soma*), và phát xung điện thế hoạt động dọc theo sợi trục (*axon*) khi tín hiệu vượt ngưỡng. Độ mạnh của khớp nối synapse tương ứng với trọng số — synapse càng mạnh thì ảnh hưởng tới việc neuron có phát xung hay không càng lớn. Tuy nhiên, không nên diễn giải phép ẩn dụ này quá sát nghĩa. Neuron sinh học phức tạp hơn rất nhiều, với các cơ chế hóa sinh, quá trình xử lý trên sợi nhánh và động lực học theo thời gian mà mô hình tổng có trọng số đơn giản không thể tái hiện. Neuron nhân tạo là một mô hình toán học và kỹ thuật lấy cảm hứng từ sinh học, chứ không phải bản mô phỏng chính xác hoạt động của não bộ.

## 4. Đoạn Mã

Dưới đây là cài đặt của cả perceptron cổ điển và neuron hiện đại để so sánh điểm giống và khác nhau giữa chúng. Trước tiên là phiên bản NumPy cho perceptron cổ điển nhằm thể hiện rõ quá trình cập nhật trọng số:

```python
import numpy as np

class Perceptron:
    """Perceptron cổ điển với kích hoạt hàm bước"""
    
    def __init__(self, n_features, learning_rate=0.01, n_iterations=1000):
        """
        Khởi tạo perceptron với trọng số ngẫu nhiên nhỏ.
        
        Vì sao trọng số ngẫu nhiên nhỏ? Cần phá đối xứng - nếu mọi trọng số
        khởi đầu bằng nhau, mọi neuron học cùng đặc trưng. Giá trị nhỏ đảm bảo
        ta bắt đầu trong vùng gradient (với kích hoạt trơn) có ý nghĩa.
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        # Bắt đầu bằng không cho perceptron cổ điển (đơn giản và hiệu quả với dữ liệu tách tuyến tính)
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors_ = []  # Theo dõi sai số mỗi epoch để chẩn đoán
    
    def activation(self, z):
        """
        Hàm bước Heaviside: cho ra 1 nếu z >= 0, ngược lại 0
        
        Tạo biên quyết định sắc nét. Mọi thứ trên ngưỡng được phân lớp 1,
        dưới ngưỡng là 0. Bản chất tất-cả-hoặc-không-gì vừa là sức mạnh
        (quyết định rõ) vừa là điểm yếu của perceptron
        (không khả vi, không dùng được gradient descent).
        """
        return np.where(z >= 0, 1, 0)
    
    def predict(self, X):
        """
        Đưa ra dự đoán cho đầu vào X.
        
        Phép tính X @ weights là tích ma trận-vectơ theo batch.
        Với mỗi mẫu (hàng của X), ta tính tích vô hướng với trọng số,
        cộng độ lệch, và áp dụng kích hoạt. Cách vectorized này nhanh hơn
        lặp qua các mẫu nhiều bậc độ lớn.
        """
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
    
    def fit(self, X, y):
        """
        Huấn luyện perceptron bằng quy tắc học perceptron.
        
        Vì sao điều này hoạt động? Quy tắc học perceptron có diễn giải
        hình học đẹp: khi phân loại sai một điểm, ta điều chỉnh biên quyết định
        tiến về điểm đó (nếu phải dương) hoặc ra xa (nếu phải âm). Với dữ liệu
        tách tuyến tính, quá trình này được đảm bảo hội tụ.
        """
        for iteration in range(self.n_iterations):
            errors = 0
            for i, x_i in enumerate(X):
                # Tính dự đoán
                z = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(z)
                
                # Chỉ cập nhật nếu dự đoán sai
                error = y[i] - y_pred
                if error != 0:
                    # Quy tắc cập nhật: w ← w + η(y - ŷ)x
                    # Khi y=1, ŷ=0: error=+1, tiến về x (tăng tích vô hướng)
                    # Khi y=0, ŷ=1: error=-1, ra xa x (giảm tích vô hướng)
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error
                    errors += abs(error)
            
            self.errors_.append(errors)
            
            # Dừng sớm nếu hội tụ
            if errors == 0:
                print(f"Converged at iteration {iteration}")
                break
        
        return self

# Minh họa học hàm AND
print("="*60)
print("Training Perceptron on AND gate")
print("="*60)

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron = Perceptron(n_features=2, learning_rate=0.1, n_iterations=100)
perceptron.fit(X_and, y_and)

print(f"\nLearned weights: {perceptron.weights}")
print(f"Learned bias: {perceptron.bias:.2f}")
print(f"Predictions: {perceptron.predict(X_and)}")
print(f"True labels:  {y_and}")
print(f"\nDecision boundary equation: {perceptron.weights[0]:.2f}*x1 + {perceptron.weights[1]:.2f}*x2 + {perceptron.bias:.2f} = 0")

# Minh họa bất khả XOR
print("\n" + "="*60)
print("Attempting to learn XOR (will fail!)")
print("="*60)

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(n_features=2, learning_rate=0.1, n_iterations=1000)
perceptron_xor.fit(X_xor, y_xor)

predictions_xor = perceptron_xor.predict(X_xor)
print(f"\nPredictions: {predictions_xor}")
print(f"True labels:  {y_xor}")
print(f"Errors remaining: {perceptron_xor.errors_[-1]}")
print("\nNote: Perceptron cannot solve XOR because it's not linearly separable!")
```

Tiếp theo là cài đặt neuron hiện đại bằng PyTorch với các hàm kích hoạt trơn, cho phép huấn luyện dựa trên gradient:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ModernNeuron(nn.Module):
    """
    Neuron nhân tạo hiện đại với hàm kích hoạt trơn.
    
    Khác biệt then chốt so với perceptron là hàm kích hoạt.
    Thay vì hàm bước chỉ 0 hoặc 1, ta dùng hàm trơn có thể
    cho ra mọi giá trị trong một khoảng và, then chốt, khả vi.
    Tính khả vi này cho phép lan truyền ngược và gradient descent.
    """
    
    def __init__(self, input_size, activation='relu'):
        super(ModernNeuron, self).__init__()
        # Lớp tuyến tính: y = Wx + b
        # PyTorch mặc định khởi tạo bằng Kaiming uniform,
        # được thiết kế cho kích hoạt ReLU
        self.linear = nn.Linear(input_size, 1)
        
        # Chọn hàm kích hoạt
        # Mỗi hàm có tính chất và trường hợp dùng khác nhau
        if activation == 'relu':
            # ReLU: max(0, z) - phổ biến nhất, ngăn gradient biến mất
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            # Sigmoid: 1/(1+e^(-z)) - đầu ra [0,1], tốt cho xác suất
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            # Tanh: (e^z - e^(-z))/(e^z + e^(-z)) - đầu ra [-1,1], tâm không
            self.activation = nn.Tanh()
        else:
            # Tuyến tính: f(z) = z - cho hồi quy
            self.activation = nn.Identity()
    
    def forward(self, x):
        """
        Lan truyền xuôi qua neuron.
        
        Phép tính giống perceptron (tổng có trọng số + độ lệch)
        nhưng ta áp dụng hàm kích hoạt trơn. Tính trơn này then chốt:
        thay đổi nhỏ của trọng số gây thay đổi nhỏ của đầu ra,
        cho phép gradient descent hoạt động hiệu quả.
        """
        z = self.linear(x)  # Tổ hợp tuyến tính: z = w^T x + b
        return self.activation(z)  # Áp dụng kích hoạt phi tuyến

# Minh họa neuron hiện đại có thể học XOR với nhiều lớp
class TwoLayerNetwork(nn.Module):
    """
    Mạng hai lớp CÓ THỂ giải XOR.
    
    Minh họa vì sao độ sâu quan trọng: lớp đầu tạo không gian biểu diễn
    mới trong đó XOR trở nên tách tuyến tính, và lớp thứ hai sau đó
    có thể tách bằng biên tuyến tính.
    """
    
    def __init__(self, input_size=2, hidden_size=4):
        super(TwoLayerNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Lớp đầu tạo đặc trưng phi tuyến
        h = torch.relu(self.hidden(x))
        # Lớp thứ hai kết hợp các đặc trưng này
        return torch.sigmoid(self.output(h))

# Huấn luyện trên XOR
print("\n" + "="*60)
print("Training 2-Layer Network on XOR")
print("="*60)

X_xor_torch = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor_torch = torch.tensor([[0.], [1.], [1.], [0.]])

model = TwoLayerNetwork(input_size=2, hidden_size=4)
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Vòng lặp huấn luyện
for epoch in range(5000):
    # Lan truyền xuôi
    predictions = model(X_xor_torch)
    loss = criterion(predictions, y_xor_torch)
    
    # Lan truyền ngược
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch:4d}, Loss: {loss.item():.4f}')

# Kiểm tra mô hình đã học
model.eval()
with torch.no_grad():
    final_predictions = model(X_xor_torch)
    binary_predictions = (final_predictions > 0.5).float()
    
print(f"\nFinal Predictions (probabilities):")
for i, (inp, pred, true) in enumerate(zip(X_xor_torch, final_predictions, y_xor_torch)):
    print(f"  Input {inp.numpy()} → Pred: {pred.item():.4f}, True: {int(true.item())}, " +
          f"Classified as: {int(binary_predictions[i].item())}")

print(f"\nSuccess! The network learned XOR, something a single perceptron cannot do.")
print("This demonstrates why depth (multiple layers) is fundamental to neural networks.")
```

Ta cũng có thể so sánh trực tiếp cách các hàm kích hoạt khác nhau tác động lên cùng một tập giá trị đầu vào:

```python
# So sánh các hàm kích hoạt
z = torch.linspace(-3, 3, 100)

relu_out = torch.relu(z)
sigmoid_out = torch.sigmoid(z)
tanh_out = torch.tanh(z)

print("\n" + "="*60)
print("Activation Function Characteristics")
print("="*60)

print(f"\nFor z = -2.0:")
print(f"  ReLU:    {torch.relu(torch.tensor(-2.0)).item():.4f}  (zero for negative)")
print(f"  Sigmoid: {torch.sigmoid(torch.tensor(-2.0)).item():.4f}  (near 0, but not exactly)")
print(f"  Tanh:    {torch.tanh(torch.tensor(-2.0)).item():.4f}  (negative output)")

print(f"\nFor z = 2.0:")
print(f"  ReLU:    {torch.relu(torch.tensor(2.0)).item():.4f}  (linear for positive)")
print(f"  Sigmoid: {torch.sigmoid(torch.tensor(2.0)).item():.4f}  (approaching 1)")
print(f"  Tanh:    {torch.tanh(torch.tensor(2.0)).item():.4f}  (approaching 1)")

print(f"\nKey observations:")
print("  - ReLU: Outputs exactly zero for negative inputs, creates sparsity")
print("  - Sigmoid: Always positive, good for probabilities but can saturate")
print("  - Tanh: Zero-centered (outputs can be negative), better gradient flow than sigmoid")
```

Lý do ReLU trở nên phổ biến rộng rãi nằm ở đặc tính của đạo hàm. Với sigmoid hoặc tanh, khi đầu vào có độ lớn cao (dương hoặc âm), độ dốc trở nên rất nhỏ. Hiện tượng bão hòa này làm gradient suy giảm nhanh khi truyền ngược qua nhiều lớp, gây khó khăn cho việc huấn luyện mạng sâu. Ngược lại, ReLU không bị bão hòa ở miền dương — đạo hàm luôn bằng 1 — giúp gradient truyền qua nhiều lớp mà không bị suy giảm. Tính chất này đóng vai trò quan trọng giúp huấn luyện được các mạng sâu hơn nhiều trong thập niên 2010.

Tuy vậy, ReLU cũng có nhược điểm riêng là hiện tượng "ReLU chết" (*dying ReLU*). Nếu đầu vào của một neuron luôn mang giá trị âm trong quá trình huấn luyện, đầu ra và đạo hàm của nó đều bằng 0, khiến trọng số không còn được cập nhật nữa. Tình trạng này thường xảy ra khi khởi tạo kém hoặc tốc độ học quá lớn. Các biến thể như Leaky ReLU ($$\max(\alpha z, z)$$ với $$\alpha \approx 0.01$$) khắc phục điểm yếu này bằng cách giữ lại một độ dốc nhỏ ở miền âm, đảm bảo gradient không triệt tiêu hoàn toàn.

## 5. Các Khái niệm Liên quan

Đặt perceptron và neuron nhân tạo trong bức tranh chung của học máy sẽ giúp thấy rõ mối liên hệ giữa các phương pháp. Nếu thay hàm bước trong perceptron bằng hàm kích hoạt sigmoid, ta thu được đúng mô hình hồi quy logistic (*logistic regression*), trong đó xác suất thuộc lớp dương được mô hình hóa là $$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$. Hồi quy logistic thường được huấn luyện theo ước lượng hợp lý cực đại (*maximum likelihood estimation*), tương đương với việc cực tiểu hóa hàm mất mát entropy chéo nhị phân (*binary cross-entropy*). Đây cũng chính là hàm mất mát tiêu chuẩn cho lớp đầu ra của mạng neuron trong bài toán phân loại nhị phân.

So sánh với máy vector hỗ trợ (*Support Vector Machine* — SVM) cũng giúp làm sáng tỏ thêm bản chất của perceptron. Cả hai đều tìm siêu phẳng phân tách cho dữ liệu tách tuyến tính. Tuy nhiên, SVM tối ưu siêu phẳng có lề cực đại (*maximum-margin hyperplane*) — tức siêu phẳng nằm xa các điểm dữ liệu gần nhất của cả hai lớp nhất có thể, qua đó mang lại đảm bảo lý thuyết tốt hơn về khả năng khái quát hóa. Ngược lại, perceptron dừng lại ở bất kỳ siêu phẳng phân tách nào tìm được mà không tối ưu lề. Dù SVM có lợi thế về mặt lý thuyết ở trường hợp tuyến tính, mạng neuron sâu xây từ các đơn vị kiểu perceptron lại linh hoạt hơn với dữ liệu phức tạp, nhiều chiều vì có thể tự học các đặc trưng phi tuyến qua nhiều lớp.

Bước tiến từ perceptron đơn lẻ sang mạng perceptron nhiều lớp (*Multi-Layer Perceptron* — MLP) là một cột mốc quan trọng của học máy. MLP xếp chồng nhiều lớp neuron nối tiếp nhau, đầu ra của lớp trước làm đầu vào cho lớp sau, giúp mạng học biểu diễn phân cấp (*hierarchical representation*). Chẳng hạn, lớp đầu phát hiện các mẫu đơn giản (như cạnh trong ảnh hoặc cụm từ ngắn trong văn bản), lớp giữa kết hợp chúng thành các đặc trưng mức trung bình (như hình dạng hoặc ngữ nghĩa cụm từ), còn các lớp sâu hơn tổng hợp thành khái niệm ở mức trừu tượng cao hơn. Khả năng học phân cấp này là đặc trưng cốt lõi giúp mạng neuron sâu vượt qua giới hạn của perceptron một lớp.

Sự tương đồng với hệ thần kinh sinh học cũng gợi ý vì sao biểu diễn phân tán (*distributed representation*) lại hiệu quả. Trong não, thông tin và khái niệm không được lưu ở một neuron đơn lẻ mà phân bố trên mẫu kích hoạt của nhiều neuron. Tương tự, trong mạng neuron nhân tạo, biểu diễn được trải đều trên nhiều đơn vị. Cơ chế này mang lại tính bền vững — nếu một vài neuron bị tắt (chẳng hạn khi dùng dropout), mạng vẫn duy trì được hoạt động. Đồng thời, nó cho phép biểu diễn số lượng khái niệm rất lớn chỉ với số lượng neuron tăng tuyến tính.

Cuối cùng, nhu cầu sử dụng hàm kích hoạt trơn gắn chặt với bài toán tối ưu hóa. Thuật toán huấn luyện chủ đạo của mạng neuron là gradient descent, vốn đòi hỏi đạo hàm có nghĩa. Đạo hàm của hàm bước bằng 0 hầu như khắp nơi và không xác định tại ngưỡng, khiến tối ưu theo gradient không thể thực hiện. Các hàm kích hoạt trơn như sigmoid, tanh hay ReLU cung cấp tín hiệu đạo hàm dẫn đường cho quá trình cập nhật tham số. Lựa chọn hàm kích hoạt ảnh hưởng trực tiếp đến độ lớn gradient và tốc độ học giữa các lớp, đặc biệt quan trọng khi mạng có nhiều lớp sâu.

## 6. Các Bài báo Nền tảng

**["The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958)](https://psycnet.apa.org/record/1959-09865-001)**  
*Tác giả*: Frank Rosenblatt  
Bài báo nền tảng này giới thiệu mô hình perceptron và chứng minh rằng một neuron nhân tạo đơn giản có thể tự học từ ví dụ. Rosenblatt vừa trình bày tính chất hội tụ về mặt lý thuyết, vừa xây dựng phần cứng thực tế có khả năng nhận dạng mẫu. Công trình này tạo nên làn sóng kỳ vọng lớn cho lĩnh vực trí tuệ nhân tạo, trước khi những hạn chế của nó được phân tích kỹ hơn về sau. Giá trị lịch sử của bài báo nằm không chỉ ở bản thân thuật toán mà còn ở việc đặt nền móng cho tư duy học từ dữ liệu trong học máy hiện đại.

**["Perceptrons: An Introduction to Computational Geometry" (1969)](https://mitpress.mit.edu/books/perceptrons)**  
*Tác giả*: Marvin Minsky và Seymour Papert  
Cuốn sách kinh điển này (không có trên arXiv) phân tích chặt chẽ các giới hạn toán học của perceptron, chứng minh rằng perceptron một lớp không thể giải các bài toán như XOR. Những phân tích nghiêm ngặt và kết luận thận trọng của tác phẩm đã góp phần làm giảm mạnh nguồn tài trợ nghiên cứu mạng neuron trong hơn một thập niên, mở đầu mùa đông AI lần thứ nhất. Thực tế, Minsky và Papert có đề cập rằng mạng nhiều lớp có thể vượt qua hạn chế này, nhưng khi đó thuật toán huấn luyện phù hợp (lan truyền ngược) chưa được phổ biến. Tác phẩm đến nay vẫn có giá trị trong việc tìm hiểu nền tảng toán học của bộ phân loại tuyến tính cũng như lịch sử phát triển của mạng neuron.

**["Learning representations by back-propagating errors" (1986)](https://www.nature.com/articles/323533a0)**  
*Tác giả*: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams  
Bài báo này khơi dậy trở lại nghiên cứu mạng neuron bằng cách trình bày cách huấn luyện mạng nhiều lớp thông qua thuật toán lan truyền ngược. Ý tưởng cốt lõi là áp dụng quy tắc dây chuyền để tính gradient theo từng lớp, nhờ đó phân bổ sai số cho mọi trọng số trong mạng chứ không chỉ ở lớp đầu ra. Nhờ vậy, mạng nhiều lớp có thể học được XOR và nhiều bài toán phi tuyến mà perceptron đơn lẻ không giải được, mở đường cho sự phát triển của học sâu hiện đại.

**["Deep Sparse Rectifier Neural Networks" (2011)](http://proceedings.mlr.press/v15/glorot11a.html)**  
*Tác giả*: Xavier Glorot, Antoine Bordes, Yoshua Bengio  
Bài báo giới thiệu Rectified Linear Unit (ReLU) như một hàm kích hoạt hiệu quả hơn cho mạng sâu và chứng minh bằng thực nghiệm ưu thế của nó so với sigmoid và tanh. Các tác giả chỉ ra rằng ReLU giúp giảm hiện tượng gradient biến mất, tính toán đơn giản (chỉ gồm phép lấy cực đại) và tạo ra biểu diễn thưa khi nhiều neuron có đầu ra bằng 0. Việc áp dụng rộng rãi ReLU là một trong những yếu tố then chốt giúp huấn luyện thành công các mạng có hàng chục đến hàng trăm lớp.

**["Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" (2015)](https://arxiv.org/abs/1502.01852)**  
*Tác giả*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Bài báo giới thiệu PReLU (Parametric ReLU, trong đó độ dốc ở miền âm là tham số học được) cùng phương pháp khởi tạo He thiết kế riêng cho mạng dùng ReLU. Nghiên cứu cho thấy với cách khởi tạo phù hợp, mạng rất sâu (22 lớp vào thời điểm công bố) không chỉ huấn luyện ổn định mà còn vượt qua mức hiệu năng của con người trên bài toán phân loại ImageNet. Khởi tạo He sử dụng độ lệch chuẩn $$\sqrt{2/n_{in}}$$ (tương ứng phương sai $$2/n_{in}$$) thay vì $$\sqrt{1/n_{in}}$$ của khởi tạo Xavier, nhằm bù trừ việc ReLU triệt tiêu trung bình một nửa số neuron và giữ cho thang đo kích hoạt cũng như gradient ổn định qua các lớp sâu.

## Cạm bẫy và Mẹo Thường gặp

Một sai lầm phổ biến khi cài đặt mạng neuron là khởi tạo tất cả trọng số bằng cùng một giá trị, đặc biệt là bằng 0. Ý tưởng bắt đầu từ trạng thái "trung tính" nghe có vẻ hợp lý nhưng lại gây ra hiện tượng đối xứng (*symmetry problem*): nếu mọi neuron trong cùng một lớp có trọng số giống hệt nhau, chúng sẽ nhận gradient như nhau khi lan truyền ngược và cập nhật giống hệt nhau trong suốt quá trình huấn luyện. Khi đó, một lớp gồm 100 neuron cũng chỉ học được đúng một đặc trưng như một neuron đơn lẻ. Khởi tạo ngẫu nhiên giúp phá vỡ tính đối xứng này để từng neuron phát triển theo hướng riêng.

Thang đo khởi tạo trọng số cũng quan trọng không kém. Nếu trọng số khởi tạo quá lớn, kích hoạt dễ rơi vào vùng bão hòa (với sigmoid/tanh) hoặc tăng bùng nổ qua các lớp, kéo theo gradient bùng nổ và làm quá trình huấn luyện mất ổn định. Ngược lại, nếu trọng số quá nhỏ, kích hoạt và gradient sẽ co dần về 0 qua các lớp, khiến mạng sâu học rất chậm. Giải pháp là điều chỉnh phương sai khởi tạo theo số chiều đầu vào của lớp: khởi tạo Xavier ($$\mathcal{N}(0, 1/n_{in})$$) phù hợp với sigmoid và tanh, còn khởi tạo He ($$\mathcal{N}(0, 2/n_{in})$$) được thiết kế cho ReLU để bù lại việc triệt tiêu đầu vào âm.

Hiện tượng ReLU chết là một lỗi thường gặp trong thực tế. Khi giá trị tiền kích hoạt của một neuron ReLU luôn âm, cả đầu ra lẫn gradient của nó đều bằng 0 nên trọng số không thể cập nhật thêm. Nguyên nhân có thể do khởi tạo chưa phù hợp, tốc độ học quá cao gây ra bước nhảy lớn đẩy neuron sang miền âm, hoặc do đặc điểm lệch trong dữ liệu. Một khi đã rơi vào trạng thái này, neuron thường không tự phục hồi trong lần huấn luyện đó. Để theo dõi, ta có thể kiểm tra tỷ lệ neuron luôn xuất ra 0; nếu tỷ lệ này vượt 20–30%, cần xem lại cấu hình huấn luyện. Các biện pháp khắc phục gồm chuyển sang Leaky ReLU, giảm tốc độ học, chọn cách khởi tạo phù hợp hoặc áp dụng batch normalization (sẽ học ở các chương sau) để ổn định phân phối kích hoạt.

Một kỹ thuật đơn giản nhưng hữu ích là khởi tạo độ lệch của neuron ReLU bằng một giá trị dương nhỏ (chẳng hạn 0.01) trong khi vẫn khởi tạo ngẫu nhiên các trọng số. Cách làm này giúp đa số neuron ở trạng thái hoạt động ngay từ đầu thay vì lập tức rơi vào vùng bằng 0, tạo điều kiện cho chúng bắt đầu học trước khi có nguy cơ bị chết.

Cuối cùng, hiểu ý nghĩa hình học của trọng số sẽ hỗ trợ đáng kể cho việc gỡ lỗi và diễn giải mô hình. Vector trọng số xác định hướng trong không gian đầu vào mà neuron đặc biệt nhạy cảm; độ lớn của nó thể hiện mức độ phản ứng khi đầu vào thay đổi theo hướng đó. Trong thị giác máy tính, khi trực quan hóa mẫu đầu vào kích hoạt cực đại từng neuron, ta thường thấy các lớp đầu học phát hiện cạnh theo hướng nhất định, còn các lớp sâu hơn dần nhận diện kết cấu (*texture*), bộ phận và toàn bộ đối tượng.

## Điểm Then chốt

Dù có cấu trúc đơn giản, perceptron đã đặt ra những ý niệm nền tảng vẫn còn nguyên giá trị trong học sâu: học từ ví dụ bằng cách cập nhật trọng số theo sai số, biểu diễn phép tính qua tổng có trọng số, và nhận thức rằng mô hình tuyến tính một lớp có giới hạn nên cần đến tính phi tuyến và độ sâu. Neuron nhân tạo hiện đại mở rộng perceptron bằng các hàm kích hoạt trơn, khả vi, cho phép huấn luyện mạng sâu tùy ý bằng gradient descent. Trong đó, ReLU trở thành lựa chọn mặc định cho các lớp ẩn nhờ chi phí tính toán thấp và khả năng hạn chế gradient biến mất. Bên cạnh đó, khởi tạo trọng số đúng cách — điển hình là khởi tạo He cho mạng ReLU — giúp phá vỡ đối xứng và giữ thang đo kích hoạt lẫn gradient ổn định. Nắm chắc không chỉ công thức mà cả lý do chúng hoạt động và các tình huống dễ gặp lỗi là cơ sở quan trọng để làm chủ học sâu.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ video của Machine Learning Thực Chiến:

- [Trực giác mạng nơ-ron (Perceptron → Deep Learning)](https://www.facebook.com/reel/793200140509765)

## Video tham khảo

- **Xem trước:** [The Perceptron Explained](https://www.youtube.com/watch?v=i1G7PXZMnSc) — Alice Heiman, tiếng Anh. Giải thích perceptron như tổng có trọng số cộng ngưỡng, khớp với mô hình mở đầu của bài.
- [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Minh họa điều gì thay đổi khi xếp chồng nhiều perceptron và thay ngưỡng cứng bằng hàm kích hoạt trơn.
- [Tất tần tật về Neural Network trong 12 phút](https://www.youtube.com/watch?v=sWPNm_GhhCA) — Việt Nguyễn AI, tiếng Việt. Giới thiệu mạch phát triển từ một neuron đến cả mạng bằng tiếng Việt.
