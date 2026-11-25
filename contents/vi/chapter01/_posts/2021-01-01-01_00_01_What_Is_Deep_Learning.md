---
layout: post
title: 01-00-01 Học sâu là gì
chapter: '01'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter01
---

Suốt nửa thế kỷ đầu của Trí tuệ Nhân tạo, các nhà khoa học tin rằng cách tốt nhất để máy tính thông minh là chép toàn bộ tri thức nhân loại vào cơ sở dữ liệu luật logic. Kết quả là máy có thể đánh thắng nhà vô địch cờ vua thế giới nhưng lại bó tay trước một việc đứa trẻ 3 tuổi làm trong chớp mắt: phân biệt con mèo với con chó trong ánh sáng mờ. **Học sâu** (*Deep Learning*) lật ngược hoàn toàn cách tiếp cận đó bằng cách cho phép mạng nơ-ron nhiều tầng tự học cấu trúc phân cấp của thế giới trực tiếp từ dữ liệu thô. Bài học này giải phẫu định nghĩa, bốn đặc trưng cốt lõi và sức mạnh biểu diễn của Học sâu.

## Kiến thức cần nhớ: Bốn đặc tính định hình bản sắc của Học sâu

1. **Phân cấp khái niệm (Hierarchical Feature Learning)**:
   Thay vì nhảy một bước từ điểm ảnh sang nhãn lớp, mạng xếp chồng nhiều phép biến đổi phi tuyến $$f(\mathbf{x}) = f_L(\dots f_2(f_1(\mathbf{x})))$$, mỗi tầng chắt lọc mức trừu tượng cao hơn tầng trước.

2. **Tối ưu hóa Đầu-cuối (End-to-End Learning)**:
   Mọi thành phần từ bộ trích xuất đặc trưng đến bộ phân loại cuối cùng đều được tối ưu đồng thời theo một hàm mất mát duy nhất, tránh lỗi tích lũy giữa các mô-đun rời rạc.

3. **Biểu diễn Phân tán (Distributed Representations)**:
   Nếu biểu diễn cục bộ (one-hot) cần $$N$$ nơ-ron để mã hóa $$N$$ khái niệm, thì biểu diễn phân tán nhị phân trên $$k$$ nơ-ron có thể mã hóa tới $$2^k$$ tổ hợp khái niệm khác nhau—tạo nên năng lực khái quát hóa vượt trội.

4. **Định luật Mở rộng (Scaling Laws)**:
   Hiệu năng Học sâu tiếp tục cải thiện một cách có quy luật khi tăng đồng thời dữ liệu ($$D$$), số tham số ($$N$$) và năng lực tính toán ($$C$$), thay vì sớm chạm trần như nhiều thuật toán cổ điển.

### Giải thích: Tính tay sức mạnh tổ hợp của Biểu diễn Phân tán vs. Biểu diễn Cục bộ

Giả sử bạn muốn mô hình biểu diễn các vật thể theo 4 thuộc tính độc lập: *Màu sắc* (Đỏ/Xanh), *Hình dạng* (Tròn/Vuông), *Kích thước* (Lớn/Nhỏ), *Chất liệu* (Gỗ/Kim loại):

1. **Cách biểu diễn cục bộ (1 nơ-ron cho mỗi tổ hợp riêng biệt)**:
   Số tổ hợp khác nhau là $$2 \times 2 \times 2 \times 2 = 2^4 = 16$$. Bạn cần tới **16 nơ-ron độc lập**, và nếu mô hình chưa từng thấy "Khối vuông màu xanh bằng gỗ cỡ nhỏ", nó không biết bất kỳ thông tin gì về vật thể đó.
2. **Cách biểu diễn phân tán (4 nơ-ron đại diện cho 4 trục thuộc tính)**:
   Bạn chỉ cần **4 nơ-ron**, mỗi nơ-ron phụ trách một đặc trưng (ví dụ vector `[0, 1, 0, 1]` = Xanh, Vuông, Nhỏ, Kim loại).
3. **Mở rộng lên 20 thuộc tính nhị phân**:
   - Biểu diễn cục bộ cần $$2^{20} = 1{,}048{,}576$$ nơ-ron (và hàng triệu mẫu cho từng tổ hợp).
   - Biểu diễn phân tán chỉ cần **20 nơ-ron**—và có thể khái quát ngay sang tổ hợp chưa từng gặp! Đó là lý do toán học sâu xa khiến mạng nơ-ron hiểu được những tình huống mới ngoài tập huấn luyện.

# Giới thiệu về Học sâu

## 1. Tổng quan khái niệm

![Deep Learning Hierarchy](https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/AI-ML-DL.svg/800px-AI-ML-DL.svg.png)
*Hình: Mối quan hệ giữa AI, Học máy và Học sâu. Nguồn: Wikimedia Commons*

Về bản chất, **học sâu** (*deep learning*) là một nhánh của học máy (*machine learning*) sử dụng mạng nơ-ron nhân tạo (*artificial neural networks*) gồm nhiều tầng — do đó gọi là "sâu" — để tự động học các biểu diễn phân cấp của dữ liệu. Điểm tạo nên bước ngoặt của học sâu không chỉ nằm ở hiệu năng thực tế, mà còn ở việc nó thay đổi căn bản cách xây dựng các hệ thống thông minh.

Để đánh giá đúng ý nghĩa của học sâu, cần nhìn lại giai đoạn trước đó. Học máy truyền thống đòi hỏi chuyên gia thiết kế thủ công các đặc trưng (*features*) — những mẫu hoặc thuộc tính mà thuật toán dùng để ra quyết định. Với nhận dạng ảnh, điều này nghĩa là phải xây dựng thủ công các bộ phát hiện cạnh, bộ phân tích kết cấu (*texture*) và bộ mô tả hình dạng. Với nhận dạng tiếng nói, người thiết kế phải xây dựng biểu diễn âm vị (*phoneme*) và mô hình âm học dựa trên lý thuyết ngôn ngữ học. Việc thiết kế đặc trưng đòi hỏi kiến thức chuyên sâu về từng miền bài toán và thường cần nhiều năm tinh chỉnh lặp đi lặp lại.

Học sâu khắc phục nút thắt này thông qua **học biểu diễn** (*representation learning*) — khả năng tự động khám phá các biểu diễn cần thiết cho việc phát hiện hoặc phân loại trực tiếp từ dữ liệu thô. Một mạng nơ-ron sâu học đặc trưng ở nhiều mức trừu tượng khác nhau: trong thị giác máy tính, tầng thứ nhất có thể học phát hiện cạnh; tầng thứ hai kết hợp các cạnh thành hình dạng đơn giản; tầng thứ ba ghép các hình dạng thành bộ phận của đối tượng; và các tầng sâu hơn nhận diện đối tượng hoàn chỉnh. Mạng tự hình thành hệ thống đặc trưng phân cấp này mà không cần hướng dẫn thủ công ngoài dữ liệu huấn luyện và mục tiêu tối ưu.

Việc tự động hóa quá trình học đặc trưng mang lại nhiều hệ quả quan trọng. Học sâu có thể xử lý những bài toán mà con người khó thiết kế đặc trưng thủ công hiệu quả. Cùng một nhóm kiến trúc cơ bản — với các điều chỉnh phù hợp — có thể áp dụng cho nhiều tác vụ khác nhau như nhận diện khuôn mặt, dịch máy, sinh ảnh, chơi trò chơi hay dự đoán cấu trúc protein. Khi bổ sung thêm dữ liệu và năng lực tính toán, hiệu năng của mô hình tiếp tục cải thiện thay vì sớm bão hòa như nhiều hệ thống cổ điển phụ thuộc vào đặc trưng thủ công.

### Các đặc trưng then chốt của học sâu

**1. Học đặc trưng phân cấp:** Mạng sâu học đặc trưng theo nhiều tầng trừu tượng. Các tầng thấp nắm bắt những mẫu đơn giản (cạnh, màu sắc, âm vị cơ bản), còn các tầng cao kết hợp chúng thành khái niệm phức tạp hơn (đối tượng, khuôn mặt, ngữ nghĩa). Cấu trúc phân cấp này tương đồng với cách thị giác và nhận thức sinh học xử lý thông tin — xây dựng hiểu biết từng lớp từ đơn giản đến phức tạp.

**2. Học từ đầu đến cuối (*end-to-end learning*):** Thay vì xây dựng một chuỗi mô-đun rời rạc trong đó từng thành phần được tối ưu riêng lẻ, học sâu cho phép tối ưu đồng thời toàn bộ hệ thống. Trong dịch máy, thay vì tách riêng các bước phân tích cú pháp, căn chỉnh và sinh câu, một mạng nơ-ron duy nhất có thể học ánh xạ trực tiếp từ ngôn ngữ nguồn sang ngôn ngữ đích, trong đó mọi thành phần đều được điều chỉnh cùng hướng tới chất lượng bản dịch cuối cùng.

**3. Khả năng mở rộng theo dữ liệu và tính toán:** Học máy truyền thống thường gặp hiện tượng lợi ích giảm dần — sau một ngưỡng nhất định, việc thêm dữ liệu không còn cải thiện đáng kể hiệu năng. Ngược lại, hiệu năng của học sâu tiếp tục tăng khi mở rộng dữ liệu và năng lực tính toán; đặc tính mở rộng (*scaling*) này là nền tảng cho sự phát triển của các mô hình ngôn ngữ lớn và hệ thống thị giác máy tính hiện đại. Đây vừa là ưu thế (giúp mô hình đạt hiệu năng rất cao trên nhiều tác vụ) vừa là thách thức (đòi hỏi tập dữ liệu lớn và chi phí tính toán cao).

**4. Biểu diễn phân tán (*distributed representations*):** Mạng sâu biểu diễn mỗi khái niệm bằng một mẫu kích hoạt trải rộng trên nhiều nơ-ron, thay vì gán riêng một nơ-ron cho từng khái niệm. Cơ chế này hỗ trợ khả năng khái quát hóa: thông tin học được về "chó" có thể giúp mô hình nhận diện "sói" vì chúng chia sẻ nhiều đặc trưng biểu diễn. Đồng thời, biểu diễn phân tán tăng độ bền vững: nếu một số nơ-ron bị loại bỏ trong quá trình huấn luyện (chẳng hạn khi dùng *dropout*), phần còn lại vẫn duy trì được thông tin biểu diễn.

### Vì sao học sâu quan trọng

Tác động của học sâu đã vượt xa phạm vi nghiên cứu học thuật, tạo ra thay đổi sâu rộng trong nhiều ngành công nghiệp và lĩnh vực đời sống:

- **Thị giác máy tính (*computer vision*):** Phát triển từ các hệ thống nhận dạng chữ số còn hạn chế trong thập niên 1990 đến các mô hình vượt hiệu năng con người trên nhiều tác vụ thị giác, nhận diện hàng nghìn lớp đối tượng, sinh ảnh chân thực và hỗ trợ xe tự hành.
- **Xử lý ngôn ngữ tự nhiên (*natural language processing*):** Chuyển từ các hệ thống dựa trên luật cứng nhắc sang các mô hình ngôn ngữ nơ-ron có khả năng soạn thảo văn bản, trả lời câu hỏi, dịch thuật đa ngôn ngữ với chất lượng gần với con người và duy trì hội thoại mạch lạc.
- **Y tế:** Hỗ trợ phát hiện bệnh từ ảnh y khoa với độ chính xác tiệm cận chuyên gia, dự báo diễn tiến bệnh nhân, rút ngắn quá trình khám phá thuốc và cá nhân hóa phác đồ điều trị.
- **Khám phá khoa học:** Hỗ trợ tìm kiếm vật liệu mới, dự đoán cấu trúc protein (như AlphaFold giải quyết bài toán gập protein tồn tại nửa thế kỷ), đề xuất giả thuyết từ tài liệu khoa học và hỗ trợ thiết kế thí nghiệm.

Bên cạnh đó, học sâu cũng giúp mở rộng khả năng tiếp cận công nghệ AI. Các thư viện mã nguồn mở như PyTorch và TensorFlow, cùng kho mô hình tiền huấn luyện công khai và tài liệu học tập phong phú, cho phép bất kỳ ai có máy tính cá nhân đều có thể thử nghiệm và nghiên cứu. Điều này thúc đẩy sự tham gia của đông đảo nhà nghiên cứu và kỹ sư trên toàn thế giới.

## 2. Nền tảng toán học

Về mặt toán học, học sâu là bài toán **xấp xỉ hàm**. Cho tập huấn luyện $$\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$$, trong đó $$\mathbf{x}_i$$ là đầu vào (ảnh, văn bản, tín hiệu cảm biến) và $$y_i$$ là đầu ra mong muốn (nhãn, bản dịch, hành động), ta cần tìm hàm $$f_\theta$$ tham số hóa bởi $$\theta$$ sao cho ánh xạ từ đầu vào sang đầu ra đạt độ chính xác cao.

![Mạng nơ-ron = bộ xấp xỉ hàm](/deep-learning-self-learning/img/chapter_img/chapter01/nn_function_approx_title.jpg)
*Hình: "Hộp đen" mạng nơ-ron thực chất đang khớp một đường cong $$y = f(x)$$ từ dữ liệu. (Minh họa từ video về bản chất mạng nơ-ron)*

![Tổng quan: mạng vs đường cong mục tiêu](/deep-learning-self-learning/img/chapter_img/chapter01/nn_overview_network_vs_curve.jpg)
*Hình: Cùng một mạng nơ-ron, khi thay đổi đầu vào $$x$$, sẽ tạo ra các điểm nằm trên một đường cong phức tạp. (Minh họa từ video về bản chất mạng nơ-ron)*

**Từ mô hình tuyến tính đến phi tuyến:** Hồi quy tuyến tính $$f(x)=xw+b$$ chỉ biểu diễn được đường thẳng nên không đủ để khớp dữ liệu có dạng đường cong uốn lượn.

![Công thức hồi quy tuyến tính](/deep-learning-self-learning/img/chapter_img/chapter01/nn_linear_regression_formula.jpg)
*Hình: Nơ-ron tuyến tính là dạng mở rộng của hàm $$f(x)=xw+b$$. (Minh họa từ video về bản chất mạng nơ-ron)*

![Tuyến tính thất bại trên dữ liệu phi tuyến](/deep-learning-self-learning/img/chapter_img/chapter01/nn_linear_fails_nonlinear.jpg)
*Hình: Mô hình tuyến tính có giới hạn rõ rệt — không thể khớp được đường cong phức tạp. (Minh họa từ video về bản chất mạng nơ-ron)*

Khi bổ sung hàm kích hoạt phi tuyến (chẳng hạn dạng sigmoid), mạng nơ-ron có thể khớp được các đường cong trơn và phức tạp hơn:

![Khớp đường cong dạng sigmoid](/deep-learning-self-learning/img/chapter_img/chapter01/nn_sigmoid_curve_fit.jpg)
*Hình: Ví dụ về hàm trơn mà mạng nơ-ron có thể xấp xỉ. (Minh họa từ video về bản chất mạng nơ-ron)*

### Định lý xấp xỉ phổ quát

Một kết quả lý thuyết nền tảng khẳng định rằng mạng nơ-ron chỉ cần một tầng ẩn, nếu có đủ số nơ-ron, có thể xấp xỉ bất kỳ hàm liên tục nào trên một miền đóng và bị chặn với độ chính xác tùy ý. Về mặt toán học, với mọi hàm liên tục $$g: \mathbb{R}^n \to \mathbb{R}^m$$ và mọi $$\epsilon > 0$$, tồn tại mạng nơ-ron $$f_\theta$$ sao cho:

$$\|f_\theta(\mathbf{x}) - g(\mathbf{x})\| < \epsilon \quad \text{for all } \mathbf{x}$$

Kết quả này cho thấy mạng nơ-ron là **bộ xấp xỉ hàm phổ quát** (*universal function approximators*).

![Kết luận: universal function approximator](/deep-learning-self-learning/img/chapter_img/chapter01/nn_conclusion_universal_approx.jpg)
*Hình: Kết luận cốt lõi — mạng nơ-ron là bộ xấp xỉ hàm phổ quát. (Minh họa từ video về bản chất mạng nơ-ron)*

Tuy nhiên, định lý này có những giới hạn quan trọng. Thứ nhất, nó chỉ khẳng định sự tồn tại của tham số $$\theta$$ chứ không đảm bảo thuật toán *gradient descent* sẽ tìm được tham số đó. Thứ hai, số lượng nơ-ron cần thiết ở tầng ẩn có thể tăng theo hàm mũ, điều không khả thi trong thực tế. Thứ ba, định lý xét mạng nông nên chưa giải thích được vì sao mạng sâu thường hiệu quả hơn khi áp dụng vào bài toán thực tiễn.

### Vì sao độ sâu quan trọng

Dù mạng nông đủ mạnh về mặt lý thuyết, **mạng sâu** thường tiết kiệm tham số hơn theo cấp số mũ đối với nhiều hàm thực tế. Xét một hàm có cấu trúc hợp thành gồm $$k$$ mức: $$f = f_k \circ f_{k-1} \circ \cdots \circ f_1$$. Một mạng nông có thể cần số nơ-ron tăng theo hàm mũ để xấp xỉ hàm này, trong khi mạng sâu có $$k$$ tầng có thể biểu diễn tự nhiên với số tham số chỉ tăng theo đa thức.

Trực giác toán học đằng sau điều này là nhiều hàm trong tự nhiên vốn có cấu trúc hợp thành. Để nhận diện khuôn mặt, hệ thống trước hết phát hiện các cạnh, sau đó ghép cạnh thành các bộ phận (mắt, mũi, miệng), rồi tổng hợp các bộ phận thành khuôn mặt hoàn chỉnh. Cấu trúc phân cấp đó được mô tả trực tiếp qua chuỗi phép biến đổi giữa các tầng:

$$\mathbf{h}^{(1)} = \sigma(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{h}^{(2)} = \sigma(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
$$\vdots$$
$$\mathbf{y} = \mathbf{W}^{(L)}\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}$$

trong đó $$\sigma$$ là hàm kích hoạt phi tuyến (ReLU, sigmoid, tanh), $$\mathbf{W}^{(l)}$$ là ma trận trọng số và $$\mathbf{b}^{(l)}$$ là vector độ lệch (*bias*) tại tầng $$l$$.

### Mục tiêu học

Huấn luyện mạng nơ-ron là quá trình tìm tập tham số $$\theta = \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \ldots, \mathbf{W}^{(L)}, \mathbf{b}^{(L)}\}$$ nhằm cực tiểu hóa hàm mất mát $$\mathcal{L}(\theta)$$ đo sai số dự đoán:

$$\theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$$

Với bài toán phân loại, ta thường dùng hàm mất mát entropy chéo (*cross-entropy*):
$$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_j y_j \log \hat{y}_j$$

Với bài toán hồi quy, ta thường dùng sai số bình phương trung bình:
$$\mathcal{L}(\hat{y}, y) = \frac{1}{2}(y - \hat{y})^2$$

Quá trình tối ưu được thực hiện bằng **gradient descent**, tức cập nhật tham số lặp đi lặp lại theo hướng làm giảm hàm mất mát:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

trong đó $$\eta$$ là tốc độ học (*learning rate*). Khả năng tính gradient hiệu quả nhờ lan truyền ngược (*backpropagation*) — áp dụng quy tắc dây chuyền qua từng tầng — là yếu tố giúp việc huấn luyện mạng sâu trở nên khả thi.

### Vì sao học sâu hoạt động hiệu quả: cân bằng bias–variance

Thành công của học sâu có thể được nhìn qua lăng kính cân bằng độ chệch–phương sai (*bias–variance tradeoff*) cổ điển. Độ chệch cao (*underfitting*) xảy ra khi mô hình quá đơn giản để nắm bắt cấu trúc dữ liệu, còn phương sai cao (*overfitting*) xảy ra khi mô hình khớp cả nhiễu thay vì quy luật thực. Mạng sâu có dung lượng biểu diễn rất lớn (độ chệch thấp), nhưng khi được chính quy hóa phù hợp lại có khả năng chống quá khớp tốt và giữ phương sai thấp ngay cả khi số tham số lên tới hàng triệu — thường lớn hơn nhiều so với số mẫu huấn luyện.

Hiện tượng này thoạt nhìn mâu thuẫn với lý thuyết học thống kê cổ điển, vốn khuyến nghị mô hình nên đơn giản hơn dữ liệu. Các nghiên cứu lý thuyết gần đây về hiện tượng "giảm kép" (*double descent*) và "chính quy hóa ngầm" (*implicit regularization*) cho thấy các mạng thừa tham số (*overparameterized*) khi huấn luyện bằng *gradient descent* có xu hướng ưu tiên những hàm đơn giản hơn, tạo ra một cơ chế chính quy hóa tự động mà khung lý thuyết cổ điển chưa mô tả đầy đủ.

## 3. Ví dụ và trực giác

Để hình dung cụ thể cách học sâu vận hành, ta xét bài toán kinh điển: huấn luyện mạng nhận dạng chữ số viết tay.

### Bài toán: nhận dạng chữ số MNIST

Giả sử ta có các ảnh xám kích thước 28×28 của chữ số viết tay (từ 0 đến 9) và muốn xây dựng hệ thống nhận diện đúng chữ số trong từng ảnh. Mỗi ảnh có thể biểu diễn dưới dạng vector 784 chiều (28×28 = 784 điểm ảnh, mỗi điểm ảnh có cường độ sáng từ 0 đến 255).

**Cách tiếp cận truyền thống** đòi hỏi thiết kế đặc trưng thủ công, chẳng hạn:
- Đếm số vòng kín (0, 6, 8, 9 có vòng kín; 1 và 7 không có).
- Phát hiện các nét dọc và nét ngang.
- Đo tỷ lệ chiều cao trên chiều rộng.
- Xác định các điểm đầu mút và giao điểm của nét.

Cách làm này đòi hỏi nhiều công sức chuyên môn và khó khái quát hóa sang các bài toán tương tự (ví dụ chữ cái viết hoa hoặc các kiểu chữ khác).

**Cách tiếp cận học sâu** đưa trực tiếp toàn bộ 784 giá trị điểm ảnh vào mạng nơ-ron:

```
Input (784 pixels) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (64 neurons) → Output (10 classes)
```

Trong quá trình huấn luyện, mạng tự động học các tầng đặc trưng:
- **Tầng 1:** hình thành các bộ phát hiện cạnh — những nơ-ron kích hoạt mạnh trước nét dọc, nét ngang hoặc đường cong tại các vị trí khác nhau.
- **Tầng 2:** kết hợp các cạnh thành mẫu nét phức tạp hơn — nét dọc dài (cho 1, 7), vòng tròn (cho 0, 6, 8, 9), hoặc các tổ hợp đường cong đặc trưng.
- **Tầng đầu ra:** tổng hợp các mẫu này để phân loại chữ số hoàn chỉnh.

### Quá trình học diễn ra như thế nào

Ban đầu, các trọng số được khởi tạo ngẫu nhiên. Khi đưa vào một ảnh chữ số "3":
1. **Lan truyền xuôi (*forward pass*):** Mạng đưa ra dự đoán ngẫu nhiên, chẳng hạn gán xác suất 70% cho chữ số "7".
2. **Tính sai số:** Nhãn đúng là "3" trong khi dự đoán là "7", tạo ra giá trị mất mát lớn.
3. **Lan truyền ngược (*backpropagation*):**
   - Tầng đầu ra xác định cần tăng mức kích hoạt của nơ-ron ứng với lớp "3" và giảm mức kích hoạt của nơ-ron ứng với lớp "7".
   - Các tầng ẩn tính toán mức độ đóng góp của từng kích hoạt vào sai số để xác định hướng điều chỉnh trọng số tương ứng.
4. **Cập nhật trọng số:** Điều chỉnh nhẹ toàn bộ trọng số nhằm giảm sai số cho mẫu này.
5. **Lặp lại:** Sau khi xử lý hàng nghìn ảnh chữ số "3" với nhiều nét viết khác nhau, mạng dần rút ra các đặc trưng cốt lõi giúp nhận diện chữ số "3".

Điểm đáng chú ý là quy trình tương đối đơn giản này — lan truyền xuôi, tính sai số, lan truyền ngược và cập nhật trọng số — khi lặp lại đủ nhiều lần trên dữ liệu sẽ tự hình thành các đặc trưng phân cấp cần thiết cho bài toán nhận dạng.

### Vì sao học phân cấp quan trọng

Xét ví dụ bài toán nhận diện khuôn mặt:
- **Đặc trưng mức thấp (Tầng 1):** phát hiện cạnh theo nhiều hướng và các vùng màu cơ bản.
- **Đặc trưng mức trung (Tầng 2–3):** ghép cạnh thành hình dạng đơn giản như đường cong, góc và kết cấu bề mặt.
- **Đặc trưng mức cao (Tầng 4–5):** kết hợp hình dạng thành bộ phận khuôn mặt — mắt (cặp vùng tối kèm điểm sáng), mũi (vùng tam giác có đổ bóng), miệng (vùng tối nằm ngang, có thể kèm răng).
- **Khái niệm hoàn chỉnh (Tầng đầu ra):** tổng hợp các bộ phận để xác định danh tính khuôn mặt.

Mỗi tầng học một mức biểu diễn trừu tượng cao hơn, phản ánh tự nhiên bản chất hợp thành của nhận thức thị giác. Mạng nhận ra rằng mắt, mũi, miệng là những thành phần có thể tái sử dụng trên mọi khuôn mặt, tương tự như nét thẳng và đường cong là các thành phần dùng chung ở mọi chữ số.

Biểu diễn phân cấp và phân tán này cũng giúp cải thiện hiệu quả sử dụng dữ liệu. Khi mạng đã học được các nơ-ron phát hiện cạnh hay vòng tròn từ bài toán chữ số, những nơ-ron đó có thể tiếp tục hữu ích cho việc nhận diện chữ cái, khuôn mặt hoặc vật thể khác — giúp **học chuyển giao** (*transfer learning*) hoạt động hiệu quả nhờ các đặc trưng mức thấp được chia sẻ.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ video của kênh *Machine Learning Thực Chiến*. Nguồn gốc để tra cứu:

- [Mạng nơ-ron là bộ xấp xỉ hàm](https://www.facebook.com/reel/720970114372332)

## Video tham khảo

- **Xem trước:** [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Giới thiệu trực quan mạng nơ-ron nhiều tầng như một bộ xấp xỉ hàm, sát với định nghĩa trong bài.
- [The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) — StatQuest with Josh Starmer, tiếng Anh. Trình bày chậm rãi về nút, trọng số và ý tưởng khớp đường cong, giải thích rõ từng đại lượng.
- [Tất tần tật về Neural Network trong 12 phút](https://www.youtube.com/watch?v=sWPNm_GhhCA) — Việt Nguyễn AI, tiếng Việt. Giải thích cấu trúc mạng nơ-ron nhiều tầng bằng tiếng Việt, phù hợp để ôn nhanh khái niệm.
