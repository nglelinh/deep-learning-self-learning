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
Hành trình vào học sâu bắt đầu bằng việc hiểu khối xây dựng cơ bản nhất: neuron nhân tạo (*artificial neuron*). Dù học sâu hiện đại đã phát triển xa vượt khỏi perceptron đơn giản do Frank Rosenblatt giới thiệu năm 1958, việc nắm điểm xuất phát lịch sử này là thiết yếu để hiểu vì sao các mạng neuron đương đại được thiết kế như hiện nay. Perceptron đại diện cho nỗ lực đầu tiên của con người nhằm tạo ra một máy có thể học từ ví dụ, mô phỏng theo cách cực kỳ đơn giản hóa cách neuron sinh học xử lý thông tin.

Về cốt lõi, perceptron là một bộ phân loại nhị phân. Nó nhận nhiều đầu vào số, kết hợp chúng bằng các trọng số đã học, và sinh một đầu ra nhị phân chỉ ra đầu vào thuộc lớp nào trong hai lớp. Điều làm cơ chế tưởng chừng đơn giản này sâu sắc là nó có thể học các trọng số đó tự động từ ví dụ, điều chỉnh lặp lại cho đến khi phân loại đúng dữ liệu huấn luyện. Khả năng học này, dù thô sơ so với chuẩn mực hiện đại, đã mang tính cách mạng vào thời điểm đó và đặt nền tảng khái niệm cho mọi phát triển sau này của mạng neuron.

Hiểu perceptron là then chốt vì nó giới thiệu nhiều khái niệm vẫn tồn tại xuyên suốt học sâu. Ý niệm đầu vào có trọng số nắm bắt việc các đặc trưng khác nhau đóng góp khác nhau vào quyết định. Hạng tử độ lệch (*bias*) cho phép biên quyết định dịch xa gốc tọa độ. Quy tắc học minh họa cách điều chỉnh tham số dựa trên sai số. Có lẽ quan trọng nhất, hạn chế cơ bản của perceptron — không giải được các bài toán không tách tuyến tính như XOR — trực tiếp thúc đẩy nhu cầu nhiều lớp và các kiến trúc sâu định hình học sâu hiện đại.

Sự chuyển từ perceptron sang neuron nhân tạo hiện đại nằm ở việc thay hàm bước cứng bằng các hàm kích hoạt trơn, khả vi. Thay đổi tưởng nhỏ này mang hệ quả sâu sắc. Hàm kích hoạt trơn cho phép học dựa trên gradient qua lan truyền ngược, cho phép huấn luyện mạng nhiều lớp. Chúng đưa vào phi tuyến tính cần thiết để mạng neuron xấp xỉ các hàm phức tạp. Lựa chọn hàm kích hoạt ảnh hưởng mọi thứ từ tốc độ huấn luyện đến khả năng biểu diễn các kiểu mẫu nhất định, khiến đây trở thành một trong những quyết định kiến trúc quan trọng nhất trong học sâu.

## 2. Nền tảng toán học
![Biological vs Artificial Neuron](https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Blausen_0657_MultipolarNeuron.png/500px-Blausen_0657_MultipolarNeuron.png)
*Hình ảnh: Neuron sinh học (trái) đã truyền cảm hứng cho neuron nhân tạo. Nguồn: Wikimedia Commons*

Perceptron thực hiện một phép tính đáng kể đơn giản, song việc hiểu công thức toán học của nó hé lộ những hiểu biết sâu về bộ phân loại tuyến tính và biên quyết định. Cho vectơ đầu vào $$\mathbf{x} = (x_1, x_2, \ldots, x_n)$$ trong đó mỗi $$x_i$$ là một đặc trưng, và vectơ trọng số tương ứng $$\mathbf{w} = (w_1, w_2, \ldots, w_n)$$, perceptron trước hết tính tổng có trọng số:

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

Đại lượng $$z$$, thường gọi là tiền kích hoạt (*pre-activation*) hoặc logit, biểu diễn tổ hợp tuyến tính của các đầu vào. Mỗi trọng số $$w_i$$ quyết định mức độ mạnh mà đầu vào tương ứng $$x_i$$ ảnh hưởng quyết định cuối. Hạng tử độ lệch $$b$$ cung cấp một ngưỡng cho phép đặt biên quyết định tối ưu trong không gian đầu vào, độc lập với việc mọi đầu vào có bằng không hay không.

Perceptron sau đó áp dụng hàm bước Heaviside lên tổ hợp tuyến tính này để sinh đầu ra nhị phân:

$$y = H(z) = \begin{cases} 
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}$$

![Đồ thị hàm bước Heaviside](/deep-learning-self-learning/img/chapter_img/chapter02/heaviside_step_function.jpg)
*Hình: Hàm bước Heaviside $$H(x)$$ — đầu ra chỉ nhận giá trị $$0$$ hoặc $$1$$. (Minh họa từ video nhập môn neural network)*

Hàm bước tạo ra biên quyết định sắc nét. Mọi điểm ở một phía được phân vào lớp dương (1), phía còn lại vào lớp âm (0). Diễn giải hình học thanh lịch: các trọng số định nghĩa một siêu phẳng trong không gian đầu vào theo phương trình $$\mathbf{w}^T \mathbf{x} + b = 0$$. Các điểm được phân loại dựa trên phía của siêu phẳng mà chúng rơi vào.

Vectơ trọng số $$\mathbf{w}$$ vuông góc (trực giao) với siêu phẳng quyết định. Đây là sự thật hình học cơ bản: nếu hai điểm $$\mathbf{x}_1$$ và $$\mathbf{x}_2$$ nằm trên siêu phẳng thì $$\mathbf{w}^T(\mathbf{x}_1 - \mathbf{x}_2) = 0$$, nghĩa là $$\mathbf{w}$$ trực giao với mọi vectơ trong siêu phẳng. Độ lớn của $$\mathbf{w}$$ quyết định tốc độ thay đổi của kích hoạt $$z$$ khi ta di chuyển vuông góc với siêu phẳng, trong khi độ lệch $$b$$ điều khiển khoảng cách của siêu phẳng tới gốc tọa độ theo hướng của $$\mathbf{w}$$.

Thuật toán học perceptron cung cấp cách đơn giản mà thanh lịch để tìm trọng số phù hợp khi dữ liệu tách tuyến tính được. Bắt đầu từ trọng số ban đầu (thường bằng không hoặc giá trị ngẫu nhiên nhỏ), thuật toán xử lý từng mẫu huấn luyện $$(\mathbf{x}_i, y_i)$$. Khi dự đoán đúng, không làm gì. Khi phân loại sai, nó điều chỉnh trọng số theo:

     $$\mathbf{w} \leftarrow \mathbf{w} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$$
     $$b \leftarrow b + \eta (y_i - \hat{y}_i)$$
   
Ở đây $$\eta$$ là tốc độ học (*learning rate*) điều khiển kích thước bước, và $$(y_i - \hat{y}_i)$$ là sai số. Nếu nhãn thật là 1 nhưng ta dự đoán 0, sai số là +1, và ta dịch trọng số theo hướng của $$\mathbf{x}_i$$, khiến đầu vào này có khả năng cao hơn được phân vào lớp 1 sau này. Nếu ta dự đoán 1 nhưng thật là 0, ta dịch xa $$\mathbf{x}_i$$. Trực giác hình học này — di chuyển biên quyết định về phía các điểm phân loại đúng và ra xa các điểm phân loại sai — nằm dưới nền tảng của nhiều phương pháp học máy.

Định lý Hội tụ Perceptron đảm bảo rằng nếu dữ liệu huấn luyện tách tuyến tính được, thuật toán sẽ tìm được siêu phẳng tách trong hữu hạn bước. Tuy nhiên, định lý cũng hé lộ hạn chế cơ bản của perceptron: nó không giải được các bài toán trong đó các lớp không tách tuyến tính. Ví dụ kinh điển là bài toán XOR, trong đó các lớp dương và âm xen kẽ sao cho không có đường thẳng nào (hoặc siêu phẳng ở chiều cao hơn) tách được chúng. Hạn chế này đã khơi mào "mùa đông AI" đầu tiên vào những năm 1970 khi rõ ràng rằng perceptron một lớp không giải được nhiều bài toán thực tế.

Neuron nhân tạo hiện đại khắc phục các hạn chế này trong khi giữ lại hiểu biết cốt lõi về tổng có trọng số. Thay cho hàm bước, ta áp dụng hàm kích hoạt trơn $$\sigma$$:

$$a = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

![Activation Functions](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Gjl-t%28x%29.svg/500px-Gjl-t%28x%29.svg.png)
*Hình ảnh: Các hàm kích hoạt phổ biến (Sigmoid, Tanh, ReLU). Nguồn: Wikimedia Commons*

Lựa chọn $$\sigma$$ ảnh hưởng mạnh hành vi của neuron. Hàm sigmoid $$\sigma(z) = \frac{1}{1+e^{-z}}$$ chuyển tiếp trơn giữa 0 và 1, cung cấp diễn giải xác suất và, then chốt, khả vi mọi nơi. Tang hyperbolic $$\tanh(z)$$ nằm trong $$-1$$ đến $$1$$ và tâm tại không, thường cải thiện luồng gradient. Đơn vị tuyến tính chỉnh lưu (ReLU), định nghĩa là $$\max(0, z)$$, đã trở thành thống trị trong học sâu hiện đại vì hiệu quả tính toán, không bão hòa với đầu vào dương (tránh gradient biến mất), và tạo độ thưa hữu ích khi các neuron kích hoạt âm cho ra đúng bằng không.

## 3. Ví dụ / Trực giác

### 3.1. Trực giác hình học: từ 1D đến 3D

Trước khi quay lại AND/XOR, hãy “nhìn” perceptron như một **ranh giới** trong không gian đầu vào.

**Một chiều (ví dụ nhiệt độ).** Neuron nhận một số $$x$$ (độ C). Bias và weight đặt một ngưỡng — ví dụ tại $$20$$: phía Inactive, phía kia Active. Perceptron chỉ “bật/tắt” theo phía của ngưỡng đó.

![Ranh giới 1D: Inactive vs Active](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_1d_threshold.jpg)
*Hình: Perceptron 1D đặt ranh giới (ví dụ tại vị trí 20) trên trục Input Space. (Minh họa từ video nhập môn neural network)*

Đổi **dấu weight** đảo chiều vùng kích hoạt. Ví dụ $$\hat{y} = H(-x + 20)$$ kích hoạt khi $$x$$ nhỏ (Active bên trái ngưỡng 20).

![Đảo ranh giới bằng dấu weight](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_1d_weight_sign.jpg)
*Hình: $$\hat{y}=H(-x+20)$$ — weight âm đảo vùng Active/Inactive. (Minh họa từ video nhập môn neural network)*

**Hai chiều (ví dụ Temp + Humidity).** Hai đặc trưng → biên quyết định là một **đường thẳng**. Các điểm hai lớp nằm hai phía đường (vàng trong hình dưới).

![Biên quyết định 2D](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_2d_decision_boundary.jpg)
*Hình: Perceptron hai đầu vào (Temp, Humidity) và đường phân tách tuyến tính trên mặt phẳng. (Minh họa từ video nhập môn neural network)*

Công thức gọn bằng vector/matrix:

![Dạng vector của perceptron](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_vector_form.jpg)
*Hình: $$\hat{y}=H(w_1x_1+w_2x_2+b)=H(\mathbf{W}\mathbf{x}+b)$$. (Minh họa từ video nhập môn neural network)*

**Ba chiều trở lên.** Thêm feature (ví dụ Wind Speed) → biên là **siêu phẳng** trong không gian đầu vào:

![Siêu phẳng quyết định 3D](/deep-learning-self-learning/img/chapter_img/chapter02/perceptron_3d_hyperplane.jpg)
*Hình: Ba đầu vào (Temp, Humidity, Wind Speed) và siêu phẳng phân loại trong không gian 3D. (Minh họa từ video nhập môn neural network)*

![Dữ liệu tách tuyến tính được](/deep-learning-self-learning/img/chapter_img/chapter02/linearly_separable.jpg)
*Hình: Hai lớp *linearly separable* — một đường thẳng đủ để tách. Khi không tách được tuyến tính, cần phi tuyến và nhiều lớp. (Minh họa từ video nhập môn neural network)*

### 3.2. Ví dụ logic: AND và XOR

Để thực sự hiểu cách perceptron hoạt động, ta xét một ví dụ cụ thể hé lộ cả sức mạnh lẫn hạn chế của nó. Xét hàm logic AND đơn giản, cho ra 1 chỉ khi cả hai đầu vào đều là 1. Dù trông tầm thường, việc một máy học được quan hệ này chỉ từ ví dụ từng là bước đột phá.

Giả sử ta muốn perceptron học AND từ các ví dụ: (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1. Khởi tạo $$w_1 = 0, w_2 = 0, b = 0$$ và dùng learning rate $$\eta = 1$$. Với mẫu đầu (0,0) nhãn 0, dự đoán là $$H(0 \cdot 0 + 0 \cdot 0 + 0) = H(0) = 1$$, sai! Sai số là $$0 - 1 = -1$$, nên ta cập nhật: $$w_1 \leftarrow 0 + 1 \cdot (-1) \cdot 0 = 0$$, $$w_2 \leftarrow 0$$, $$b \leftarrow 0 + 1 \cdot (-1) = -1$$. Giờ ta có độ lệch -1, cung cấp một ngưỡng.

Tiếp tục quá trình qua các mẫu huấn luyện, perceptron cuối cùng hội tụ về các trọng số như $$w_1 = 1, w_2 = 1, b = -1.5$$. Kiểm chứng: Với đầu vào (1,1), ta được $$z = 1 \cdot 1 + 1 \cdot 1 - 1.5 = 0.5$$, nên $$H(0.5) = 1$$ ✓. Với (1,0), $$z = 1 \cdot 1 + 1 \cdot 0 - 1.5 = -0.5$$, nên $$H(-0.5) = 0$$ ✓. Perceptron đã học yêu cầu cả hai đầu vào cùng vượt ngưỡng 1.5 khi kết hợp.

Bây giờ xét vì sao bài toán XOR bất khả với một perceptron đơn. XOR cho ra 1 khi các đầu vào khác nhau: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. Nếu vẽ các điểm này trong không gian 2D, các mẫu dương (0,1) và (1,0) nằm chéo đối diện, còn mẫu âm (0,0) và (1,1) ở đường chéo kia. Không có đường thẳng nào tách được các lớp — ta cần biên phi tuyến, như đường cong hoặc nhiều đoạn thẳng. Bất khả hình học này hé lộ hạn chế tính toán cơ bản: mô hình tuyến tính một lớp không nắm bắt được một số quan hệ logic nhất định.

Hạn chế này thúc đẩy phát triển mạng nhiều lớp. Nếu xếp hai perceptron nuôi vào một perceptron thứ ba, ta có thể giải XOR. Lớp đầu có thể học phát hiện $$x_1$$ OR $$x_2$$, và $$x_1$$ AND $$x_2$$, còn lớp thứ hai học "OR nhưng không AND", đúng bằng XOR. Điều này minh họa nguyên lý then chốt: độ sâu (nhiều lớp) cho phép học các biên quyết định ngày càng phức tạp. Mỗi lớp bổ sung có thể kết hợp đặc trưng từ các lớp trước theo cách mới, mở rộng theo cấp số nhân không gian các hàm biểu diễn được.

Cảm hứng sinh học, dù không hoàn hảo, cung cấp trực giác hữu ích. Neuron thật trong não nhận tín hiệu qua dendrite, tích hợp tín hiệu trong thân tế bào (soma), và phát điện thế hoạt động dọc axon nếu tín hiệu tích hợp vượt ngưỡng. Độ mạnh synapse tương ứng trọng số của ta — synapse mạnh hơn đóng góp nhiều hơn vào việc neuron có phát xung hay không. Tuy nhiên, ta phải cẩn thận không đẩy phép ẩn dụ quá xa. Neuron sinh học phức tạp hơn nhiều so với neuron nhân tạo, với hóa sinh tinh vi, tính toán dendrite phức tạp, và động lực thời gian mà mô hình tổng có trọng số đơn giản của ta không nắm bắt. Neuron nhân tạo là công cụ kỹ thuật lấy cảm hứng từ sinh học, không phải mô hình chính xác của chức năng não.

## 4. Đoạn Mã

Ta triển khai cả perceptron cổ điển và neuron hiện đại để hiểu sự giống và khác nhau. Bắt đầu với triển khai NumPy sạch làm lộ rõ quá trình học:

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

Bây giờ ta triển khai neuron hiện đại với hàm kích hoạt trơn cho phép học dựa trên gradient:

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

Ta cũng hiểu cách các hàm kích hoạt khác nhau định hình hành vi neuron bằng cách xét ảnh hưởng của chúng lên cùng một đầu vào:

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

Lý do ReLU trở nên thống trị xứng đáng được giải thích sâu hơn. Khi dùng sigmoid hoặc tanh, gradient trở nên rất nhỏ khi đầu vào có độ lớn lớn (dương hoặc âm). "Bão hòa" này nghĩa là trong lan truyền ngược, gradient suy giảm khi truyền ngược qua các lớp, khiến việc huấn luyện mạng sâu trở nên khó khăn. ReLU không bão hòa với đầu vào dương — gradient đúng bằng 1 — cho phép gradient chảy không đổi qua nhiều lớp. Tính chất này cho phép huấn luyện mạng sâu hơn nhiều và then chốt với cuộc cách mạng học sâu thập niên 2010.

Tuy nhiên, ReLU đưa vào thách thức riêng: bài toán "ReLU chết". Nếu đầu vào của một neuron luôn âm trong huấn luyện, đầu ra luôn bằng không, và gradient cũng luôn bằng không, nghĩa là nó không bao giờ cập nhật và thực chất đã chết. Điều này có thể xảy ra với khởi tạo kém hoặc learning rate quá cao. Các biến thể như Leaky ReLU ($$\max(\alpha z, z)$$ với $$\alpha \approx 0.01$$) khắc phục bằng cách cho phép giá trị âm nhỏ, đảm bảo gradient không bao giờ biến mất hoàn toàn.

## 5. Các Khái niệm Liên quan

Hiểu đúng perceptron và neuron nhân tạo đòi hỏi nhìn thấy cách chúng kết nối với bối cảnh rộng hơn của học máy và học sâu. Perceptron về bản chất là dạng đơn giản hóa của hồi quy logistic khi ta thay hàm bước bằng kích hoạt sigmoid. Trong hồi quy logistic, ta mô hình xác suất thuộc lớp dưới dạng $$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$, đúng bằng một perceptron với kích hoạt sigmoid. Mối liên hệ còn sâu hơn: hồi quy logistic thường được huấn luyện bằng ước lượng hợp lý cực đại, với trường hợp nhị phân dẫn đến cực tiểu hóa mất mát entropy chéo nhị phân. Cùng hàm mất mát này được dùng để huấn luyện lớp đầu ra của mạng neuron cho phân loại nhị phân.

Mối quan hệ với Support Vector Machines (SVM) cũng mang tính soi sáng. Như perceptron, SVM tìm siêu phẳng tách cho dữ liệu tách tuyến tính. Tuy nhiên, SVM tối ưu siêu phẳng lề cực đại — siêu phẳng xa nhất có thể so với các điểm dữ liệu gần nhất của cả hai lớp. Cực đại hóa lề cung cấp đảm bảo khái quát hóa tốt hơn. Perceptron, ngược lại, thỏa mãn với bất kỳ siêu phẳng tách nào và không tối ưu lề. Dù có lợi thế lý thuyết này của SVM, mạng neuron sâu xây từ các đơn vị kiểu perceptron đã chứng tỏ thực tế hơn với các bài toán phức tạp, chiều cao, vì chúng có thể học đặc trưng phi tuyến qua nhiều lớp.

Sự tiến hóa từ perceptron sang Multi-Layer Perceptron (MLP) là một trong những phát triển quan trọng nhất trong học máy. MLP đơn giản là nhiều lớp neuron, trong đó đầu ra của mỗi lớp trở thành đầu vào của lớp tiếp theo. Việc xếp chồng này cho phép mạng học biểu diễn phân cấp. Lớp đầu có thể học phát hiện mẫu đơn giản (cạnh trong ảnh, hoặc tổ hợp từ phổ biến trong văn bản). Lớp thứ hai kết hợp các mẫu đơn giản thành đặc trưng mức trung (hình dạng từ cạnh, hoặc nghĩa cụm từ). Các lớp sâu hơn xây các khái niệm mức cao hơn nữa. Học phân cấp này có lẽ là khía cạnh mạnh mẽ nhất của mạng neuron sâu và chỉ có thể nhờ ta đã vượt qua perceptron một lớp.

Liên hệ với mạng neuron sinh học, dù hạn chế, cung cấp trực giác hữu ích về vì sao biểu diễn phân tán hoạt động. Trong não, ký ức và khái niệm không được lưu trong từng neuron đơn lẻ mà trong các mẫu hoạt động trên nhiều neuron. Tương tự, trong mạng neuron nhân tạo, biểu diễn được phân tán trên nhiều neuron. Sự phân tán này mang lại độ bền — nếu vài neuron hỏng hoặc bị loại (như trong dropout), mạng vẫn có thể hoạt động. Nó cũng cho phép mạng biểu diễn theo cấp số nhân nhiều khái niệm với số neuron tuyến tính, một tính chất gọi là hiệu quả biểu diễn phần nào giải thích thành công của học sâu.

Cuối cùng, hiểu vì sao ta cần hàm kích hoạt trơn kết nối với chủ đề rộng hơn về tối ưu. Gradient descent, thuật toán chính để huấn luyện mạng neuron, đòi hỏi gradient. Gradient của hàm bước bằng không hầu như mọi nơi (và không xác định tại ngưỡng), khiến tối ưu dựa trên gradient bất khả. Các hàm kích hoạt trơn như sigmoid, tanh, và đặc biệt ReLU cung cấp gradient có ý nghĩa dẫn dắt quá trình học. Lựa chọn hàm kích hoạt ảnh hưởng không chỉ việc ta có tính được gradient hay không mà còn độ lớn của chúng, quyết định tốc độ học của các lớp khác nhau — một cân nhắc then chốt trong mạng sâu khi gradient phải truyền qua nhiều lớp.

## 6. Các Bài báo Nền tảng

**["The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958)](https://psycnet.apa.org/record/1959-09865-001)**  
*Tác giả*: Frank Rosenblatt  
Bài báo nền tảng này giới thiệu perceptron và chứng minh rằng một neuron nhân tạo đơn giản có thể học từ ví dụ. Rosenblatt cho thấy cả tính chất hội tụ lý thuyết lẫn triển khai thực tế, xây các máy vật lý có thể nhận dạng mẫu. Thành công của perceptron khơi dậy sự lạc quan to lớn về trí tuệ nhân tạo, dù sau đó bị điều tiết bởi phát hiện các hạn chế của nó. Bài báo có ý nghĩa lịch sử không chỉ vì thuật toán mà còn vì thiết lập mô hình học từ dữ liệu nằm dưới nền tảng của toàn bộ học máy hiện đại.

**["Perceptrons: An Introduction to Computational Geometry" (1969)](https://mitpress.mit.edu/books/perceptrons)**  
*Tác giả*: Marvin Minsky và Seymour Papert  
Dù không có trên arXiv, cuốn sách ảnh hưởng này phân tích chặt chẽ các hạn chế của perceptron, chứng minh rằng perceptron một lớp không giải được các bài toán như XOR. Phân tích kỹ lưỡng đến mức và kết luận ảm đạm đến mức đã góp phần vào mùa đông AI đầu tiên, với tài trợ nghiên cứu mạng neuron cạn kiệt hơn một thập niên. Trớ trêu thay, Minsky và Papert ghi nhận rằng mạng nhiều lớp có thể khắc phục các hạn chế này, nhưng thiếu thuật toán huấn luyện (lan truyền ngược chưa được tái phát hiện) khiến nhận xét này không ngăn được sự suy thoái của lĩnh vực. Cuốn sách vẫn quan trọng để hiểu cả nền tảng toán của bộ phân loại tuyến tính lẫn sự phát triển lịch sử của mạng neuron.

**["Learning representations by back-propagating errors" (1986)](https://www.nature.com/articles/323533a0)**  
*Tác giả*: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams  
Bài báo này phục hưng nghiên cứu mạng neuron bằng cách chỉ ra cách huấn luyện mạng nhiều lớp các đơn vị kiểu perceptron bằng lan truyền ngược. Hiểu biết then chốt là bằng cách tính gradient từng lớp dùng quy tắc dây chuyền, ta có thể gán công (hoặc lỗi) cho mọi trọng số trong mạng, không chỉ lớp đầu ra. Điều này cho phép huấn luyện mạng đủ sâu để giải XOR và nhiều bài toán khác mà perceptron đơn không xử lý được. Bài báo đánh dấu sự trỗi dậy trở lại của chủ nghĩa liên kết và đặt nền cho học sâu hiện đại.

**["Deep Sparse Rectifier Neural Networks" (2011)](http://proceedings.mlr.press/v15/glorot11a.html)**  
*Tác giả*: Xavier Glorot, Antoine Bordes, Yoshua Bengio  
Bài báo này giới thiệu Rectified Linear Unit (ReLU) như hàm kích hoạt vượt trội cho mạng sâu và chứng minh thực nghiệm các ưu điểm so với sigmoid và tanh. Các tác giả cho thấy ReLU cho phép huấn luyện mạng sâu hơn bằng cách tránh bài toán gradient biến mất từng làm khổ các hàm kích hoạt trước đó. Neuron ReLU cũng hiệu quả tính toán (chỉ một phép max) và tạo biểu diễn thưa (nhiều neuron cho ra đúng bằng không), vừa có lợi tính toán vừa có thể diễn giải. Việc chấp nhận ReLU là yếu tố then chốt của cuộc cách mạng học sâu, cho phép huấn luyện mạng với hàng chục thậm chí hàng trăm lớp.

**["Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" (2015)](https://arxiv.org/abs/1502.01852)**  
*Tác giả*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
Bài báo này giới thiệu cả PReLU (Parametric ReLU, trong đó độ dốc với đầu vào âm được học) và khởi tạo He, một lược đồ khởi tạo trọng số được thiết kế riêng cho mạng ReLU. Bài báo chứng minh rằng với khởi tạo đúng, mạng cực sâu (22 lớp lúc đó, tưởng rất sâu) không chỉ huấn luyện thành công mà còn vượt hiệu năng con người trên phân loại ImageNet. Lược đồ khởi tạo He, dùng phương sai $$\sqrt{2/n_{in}}$$ thay vì $$\sqrt{1/n_{in}}$$ (Xavier), tính đến việc ReLU triệt tiêu trung bình một nửa số neuron, duy trì độ lớn kích hoạt và gradient phù hợp qua mạng sâu.

## Cạm bẫy và Mẹo Thường gặp

Một sai lầm phổ biến nhất khi triển khai neuron là khởi tạo mọi trọng số cùng một giá trị, đặc biệt bằng không. Điều này có vẻ hợp lý — bắt đầu từ vị trí "trung tính" và để dữ liệu dẫn dắt học — nhưng mang hệ quả tàn khốc gọi là bài toán đối xứng. Nếu mọi neuron trong một lớp khởi đầu với trọng số giống hệt, chúng nhận gradient giống hệt trong lan truyền ngược và do đó cập nhật giống hệt. Chúng vẫn giống hệt suốt huấn luyện, học đúng cùng đặc trưng. Một lớp 100 neuron với trọng số giống hệt không mạnh hơn một neuron đơn. Khởi tạo ngẫu nhiên phá đối xứng, đảm bảo mỗi neuron đi theo quỹ đạo học khác nhau và học phát hiện các mẫu khác nhau.

Thang đo khởi tạo cũng quan trọng sâu sắc, dù lý do tinh tế. Nếu trọng số quá lớn, kích hoạt có thể bão hòa (với sigmoid/tanh) hoặc bùng nổ (tăng theo cấp số nhân qua các lớp), trong khi gradient cũng có thể bùng nổ, gây bất ổn huấn luyện. Nếu trọng số quá nhỏ, kích hoạt co về không qua các lớp, và gradient biến mất, khiến học chậm không tưởng, đặc biệt trong mạng sâu. Giải pháp là scale trọng số ban đầu dựa trên chiều lớp. Khởi tạo Xavier ($$\mathcal{N}(0, 1/n_{in})$$) hoạt động tốt với sigmoid và tanh, duy trì phương sai kích hoạt qua các lớp. Khởi tạo He ($$\mathcal{N}(0, 2/n_{in})$$) được thiết kế riêng cho ReLU, tính đến tính chất triệt tiêu đầu vào âm.

Bài toán ReLU chết xứng đáng chú ý đặc biệt vì là chế độ thất bại phổ biến trong thực tế. Khi đầu vào của neuron ReLU trở nên âm trong huấn luyện và vẫn âm, neuron cho ra không và có gradient bằng không, nên không bao giờ cập nhật. Điều này có thể do khởi tạo không may, learning rate quá cao gây cập nhật trọng số lớn đẩy neuron vào vùng âm, hoặc thiên lệch hệ thống trong dữ liệu. Một khi neuron chết, nó chết vĩnh viễn trong lần huấn luyện đó. Để chẩn đoán, theo dõi tỷ lệ neuron luôn cho ra không. Nếu hơn 20–30% chết, có lẽ có vấn đề. Giải pháp gồm dùng Leaky ReLU (có gradient nhỏ ngay cả với đầu vào âm), giảm learning rate, cải thiện khởi tạo, hoặc dùng batch normalization (sẽ trình bày sau) để giữ kích hoạt trong khoảng hợp lý.

Một kỹ thuật mạnh thường bị bỏ qua là dùng độ lệch dương nhỏ cho neuron ReLU. Trong khi trọng số nên ngẫu nhiên, khởi tạo độ lệch bằng giá trị dương nhỏ như 0.01 đảm bảo hầu hết neuron ban đầu hoạt động (cho ra giá trị dương) thay vì bắt đầu trong vùng không. Điều này cho chúng cơ hội học trước khi có thể chết. Đây là mẹo đơn giản có thể cải thiện rõ rệt huấn luyện trong mạng rất sâu.

Hiểu diễn giải hình học của trọng số giúp gỡ lỗi và diễn giải mô hình. Vectơ trọng số định nghĩa một hướng trong không gian đầu vào mà neuron đang "nhìn" theo. Độ lớn của nó quyết định độ nhạy — trọng số lớn hơn nghĩa là neuron phản ứng mạnh hơn với thay đổi theo hướng đó. Trong xử lý ảnh, ta có thể trực quan hóa những gì neuron đã học bằng cách tìm mẫu đầu vào kích hoạt cực đại nó, thường hé lộ rằng neuron mức thấp học phát hiện cạnh có hướng, trong khi neuron sâu hơn học phát hiện các mẫu ngày càng phức tạp như texture, bộ phận đối tượng, hoặc cuối cùng đối tượng hoàn chỉnh.

## Điểm Then chốt

Perceptron, dù đơn giản, giới thiệu các khái niệm nền tảng vẫn tồn tại xuyên suốt học sâu: ý tưởng rằng ta có thể học từ ví dụ bằng cách điều chỉnh trọng số dựa trên sai số, rằng tổ hợp có trọng số của đầu vào có thể thực hiện tính toán, và rằng mô hình tuyến tính có hạn chế cố hữu đòi hỏi phi tuyến tính và độ sâu. Neuron hiện đại mở rộng perceptron bằng cách dùng hàm kích hoạt trơn, khả vi, cho phép học dựa trên gradient qua mạng sâu tùy ý. Lựa chọn hàm kích hoạt ảnh hưởng sâu động lực huấn luyện, với ReLU nổi lên như lựa chọn thống trị cho lớp ẩn nhờ hiệu quả tính toán và kháng gradient biến mất. Khởi tạo đúng phá đối xứng đồng thời duy trì thang đo kích hoạt và gradient phù hợp, với khởi tạo He là chuẩn cho mạng ReLU. Hiểu sâu các khái niệm nền tảng này — không chỉ công thức là gì mà vì sao chúng hoạt động và khi nào thất bại — là thiết yếu cho bất kỳ ai muốn làm chủ học sâu thay vì chỉ áp dụng hời hợt.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Trực giác mạng nơ-ron (Perceptron → Deep Learning)](https://www.facebook.com/reel/793200140509765)
