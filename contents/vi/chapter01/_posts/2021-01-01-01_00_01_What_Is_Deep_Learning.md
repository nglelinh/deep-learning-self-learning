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

# Giới thiệu về Học sâu

## 1. Tổng quan khái niệm

![Deep Learning Hierarchy](https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/AI-ML-DL.svg/800px-AI-ML-DL.svg.png)
*Hình: Mối quan hệ giữa AI, Học máy và Học sâu. Nguồn: Wikimedia Commons*

Học sâu (*deep learning*) là một trong những tiến bộ công nghệ mang tính chuyển đổi nhất của thế kỷ XXI. Về bản chất, **học sâu** là một nhánh của học máy (*machine learning*) sử dụng mạng nơ-ron nhân tạo (*artificial neural networks*) với nhiều tầng — do đó gọi là "sâu" — để tự động học các biểu diễn phân cấp của dữ liệu. Điều làm nên tính cách mạng của học sâu không chỉ là hiệu năng thực tế, mà còn ở chỗ nó thay đổi căn bản cách tiếp cận khi xây dựng hệ thống thông minh.

Để đánh giá đúng ý nghĩa của học sâu, cần nhìn lại giai đoạn trước đó. Học máy truyền thống đòi hỏi chuyên gia thiết kế thủ công các đặc trưng (*features*) — những mẫu hoặc thuộc tính mà thuật toán dùng để ra quyết định. Với nhận dạng ảnh, điều này nghĩa là phải thiết kế tay các bộ phát hiện cạnh, bộ phân tích texture và bộ mô tả hình dạng. Với nhận dạng tiếng nói, phải xây dựng biểu diễn phoneme và mô hình âm học dựa trên lý thuyết ngôn ngữ học. Việc thiết kế đặc trưng vừa là nghệ thuật vừa là khoa học, đòi hỏi tri thức miền sâu và thường cần nhiều năm tinh chỉnh lặp lại.

Học sâu loại bỏ nút thắt này thông qua **học biểu diễn** (*representation learning*) — khả năng tự động khám phá các biểu diễn cần thiết cho phát hiện hoặc phân loại trực tiếp từ dữ liệu thô. Một mạng nơ-ron sâu học đặc trưng ở nhiều mức trừu tượng: trong thị giác máy tính, tầng thứ nhất có thể học phát hiện cạnh; tầng thứ hai kết hợp các cạnh thành hình dạng đơn giản; tầng thứ ba ghép các hình dạng thành bộ phận đối tượng; các tầng sâu hơn nhận diện đối tượng hoàn chỉnh. Quan trọng hơn, mạng tự khám phá các đặc trưng phân cấp này mà không cần hướng dẫn thủ công ngoài dữ liệu huấn luyện và mục tiêu học.

Việc học đặc trưng tự động có hệ quả sâu rộng. Học sâu có thể xử lý các bài toán mà ta chưa biết cách thiết kế đặc trưng tốt bằng tay. Cùng một kiến trúc cơ bản — với các biến thể phù hợp — có thể xuất sắc ở nhiều nhiệm vụ khác nhau: nhận diện khuôn mặt, dịch ngôn ngữ, sinh ảnh, chơi game hay gập protein. Khi thu thập thêm dữ liệu và tăng khả năng tính toán, hiệu năng tiếp tục được cải thiện, thay vì bão hòa như thường thấy ở các hệ thống cổ điển được tinh chỉnh thủ công.

### Các đặc trưng then chốt của học sâu

**1. Học đặc trưng phân cấp**: Mạng sâu học đặc trưng ở nhiều mức trừu tượng. Các tầng thấp nắm bắt mẫu đơn giản (cạnh, màu sắc, phoneme cơ bản), trong khi các tầng cao kết hợp chúng thành khái niệm phức tạp (đối tượng, khuôn mặt, nghĩa ngữ nghĩa). Cấu trúc phân cấp này phản ánh cách ta hiểu về thị giác và nhận thức sinh học — xây dựng hiểu biết từng lớp từ đơn giản đến phức tạp.

**2. Học đầu-cuối** (*end-to-end learning*): Thay vì xây dựng pipeline mô-đun trong đó mỗi thành phần được tối ưu riêng, học sâu cho phép tối ưu toàn hệ thống một cách đồng thời. Với dịch máy, thay vì các mô-đun riêng cho phân tích cú pháp, căn chỉnh và sinh câu, một mạng nơ-ron duy nhất học ánh xạ trực tiếp từ ngôn ngữ nguồn sang ngôn ngữ đích, với mọi thành phần được tối ưu cùng hướng tới chất lượng dịch cuối cùng.

**3. Khả năng mở rộng theo dữ liệu và tính toán**: Học máy truyền thống thường thể hiện lợi ích giảm dần — thêm dữ liệu sau một ngưỡng nhất định mang lại ít cải thiện. Hiệu năng học sâu tiếp tục tăng khi có thêm dữ liệu và tính toán; tính chất scaling này đã thúc đẩy cuộc cách mạng ở các mô hình ngôn ngữ lớn và hệ thống thị giác máy tính. Khả năng mở rộng vừa là điểm mạnh (cho phép hiệu năng vượt con người trên nhiều tác vụ) vừa là thách thức (đòi hỏi tập dữ liệu khổng lồ và tài nguyên tính toán lớn).

**4. Biểu diễn phân tán** (*distributed representations*): Mạng sâu biểu diễn khái niệm dưới dạng mẫu kích hoạt trải trên nhiều nơ-ron, thay vì dành một nơ-ron riêng cho từng khái niệm. Cách này hỗ trợ khái quát hóa: tri thức về "chó" có thể hỗ trợ hiểu "sói" vì chúng chia sẻ nhiều đặc trưng biểu diễn. Đồng thời, biểu diễn phân tán mang lại độ bền vững: nếu một số nơ-ron hỏng hoặc bị loại trong huấn luyện (dropout), biểu diễn vẫn hoạt động.

### Vì sao học sâu quan trọng

Tác động của học sâu vượt xa sự tò mò học thuật. Nó đã thay đổi căn bản nhiều ngành công nghiệp và nhiều khía cạnh đời sống:

- **Thị giác máy tính** (*computer vision*): Từ nhận dạng chữ số còn hạn chế những năm 1990 đến các hệ thống vượt hiệu năng con người trên nhiều tác vụ thị giác, nhận diện hàng nghìn loại đối tượng, sinh ảnh chân thực và hỗ trợ xe tự hành.

- **Xử lý ngôn ngữ tự nhiên** (*natural language processing*): Từ hệ thống dựa trên luật cứng nhắc đến các mô hình ngôn ngữ nơ-ron có thể viết bài luận, trả lời câu hỏi, dịch giữa các ngôn ngữ với chất lượng gần con người và đối thoại mạch lạc.

- **Y tế**: Từ chẩn đoán thủ công chậm và dễ sai đến các hệ thống AI phát hiện bệnh từ ảnh y khoa với độ chính xác ngang chuyên gia, dự báo kết cục bệnh nhân, tăng tốc khám phá thuốc và cá nhân hóa phác đồ điều trị.

- **Khám phá khoa học**: Từ nghiên cứu thuần túy dựa trên giả thuyết đến các hệ thống AI khám phá vật liệu mới, dự đoán cấu trúc protein (AlphaFold giải quyết thách thức kéo dài nửa thế kỷ), sinh giả thuyết từ tài liệu và thiết kế thí nghiệm.

Quan trọng không kém, học sâu đã dân chủ hóa AI. Các framework mã nguồn mở như PyTorch và TensorFlow, các mô hình tiền huấn luyện miễn phí và tài nguyên học tập phong phú khiến AI mạnh mẽ trở nên tiếp cận được với bất kỳ ai có máy tính xách tay và sự tò mò. Sự dân chủ hóa này thúc đẩy đổi mới khi hàng triệu nhà nghiên cứu và kỹ sư trên thế giới cùng đóng góp cho lĩnh vực.

## 2. Nền tảng toán học

Về mặt toán học, học sâu là bài toán **xấp xỉ hàm**. Cho tập huấn luyện $$\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$$ trong đó $$\mathbf{x}_i$$ là đầu vào (ảnh, văn bản, tín hiệu cảm biến) và $$y_i$$ là đầu ra mong muốn (nhãn, bản dịch, hành động), ta tìm hàm $$f_\theta$$ tham số hóa bởi $$\theta$$ ánh xạ chính xác từ đầu vào sang đầu ra.

![Mạng nơ-ron = bộ xấp xỉ hàm](/deep-learning-self-learning/img/chapter_img/chapter01/nn_function_approx_title.jpg)
*Hình: “Hộp đen” mạng nơ-ron thực chất đang khớp một đường cong $$y = f(x)$$ từ dữ liệu. (Minh họa từ video về bản chất mạng nơ-ron)*

![Tổng quan: mạng vs đường cong mục tiêu](/deep-learning-self-learning/img/chapter_img/chapter01/nn_overview_network_vs_curve.jpg)
*Hình: Cùng một mạng, khi cho $$x$$ khác nhau, sinh ra các điểm trên đường cong phức tạp. (Minh họa từ video về bản chất mạng nơ-ron)*

**Trực giác tuyến tính → phi tuyến.** Hồi quy tuyến tính $$f(x)=xw+b$$ chỉ vẽ được đường thẳng; với dữ liệu gợn sóng thì **không đủ**.

![Công thức hồi quy tuyến tính](/deep-learning-self-learning/img/chapter_img/chapter01/nn_linear_regression_formula.jpg)
*Hình: Neuron tuyến tính là mở rộng của $$f(x)=xw+b$$. (Minh họa từ video về bản chất mạng nơ-ron)*

![Tuyến tính thất bại trên dữ liệu phi tuyến](/deep-learning-self-learning/img/chapter_img/chapter01/nn_linear_fails_nonlinear.jpg)
*Hình: Mô hình tuyến tính bị giới hạn — không khớp được đường cong phức tạp. (Minh họa từ video về bản chất mạng nơ-ron)*

Với kích hoạt phi tuyến (ví dụ sigmoid-like), một mạng có thể khớp được các hình dạng mềm:

![Khớp đường cong dạng sigmoid](/deep-learning-self-learning/img/chapter_img/chapter01/nn_sigmoid_curve_fit.jpg)
*Hình: Ví dụ hàm “mềm” mà mô hình có thể xấp xỉ. (Minh họa từ video về bản chất mạng nơ-ron)*

### Định lý xấp xỉ phổ quát

Một kết quả lý thuyết nền tảng khẳng định rằng mạng nơ-ron chỉ với một tầng ẩn, nếu có đủ số nơ-ron, có thể xấp xỉ bất kỳ hàm liên tục nào với độ chính xác tùy ý. Về mặt toán học, với mọi hàm liên tục $$g: \mathbb{R}^n \to \mathbb{R}^m$$ và mọi $$\epsilon > 0$$, tồn tại mạng nơ-ron $$f_\theta$$ sao cho:

$$\|f_\theta(\mathbf{x}) - g(\mathbf{x})\| < \epsilon \quad \text{for all } \mathbf{x}$$

Điều này đáng chú ý: mạng nơ-ron là **bộ xấp xỉ hàm phổ quát** (*universal function approximators*).

![Kết luận: universal function approximator](/deep-learning-self-learning/img/chapter_img/chapter01/nn_conclusion_universal_approx.jpg)
*Hình: Cốt lõi — mạng nơ-ron là bộ xấp xỉ hàm phổ quát. (Minh họa từ video về bản chất mạng nơ-ron)*

Tuy nhiên, định lý có những hạn chế quan trọng. Nó đảm bảo sự tồn tại nhưng không đảm bảo khả năng học được — việc tìm tham số $$\theta$$ bằng gradient descent không được đảm bảo. Nó có thể đòi hỏi số nơ-ron ở tầng ẩn tăng theo hàm mũ, điều không thực tế. Và nó áp dụng cho mạng nông, chưa giải thích vì sao mạng sâu thường hiệu quả hơn trong thực tiễn.

### Vì sao độ sâu quan trọng

Dù mạng nông về lý thuyết là đủ, **mạng sâu** thường hiệu quả hơn theo hàm mũ với nhiều hàm thực tế. Xét việc biểu diễn một hàm với $$k$$ mức hợp thành: $$f = f_k \circ f_{k-1} \circ \cdots \circ f_1$$. Một mạng nông có thể cần số nơ-ron tăng theo hàm mũ để biểu diễn, trong khi mạng sâu với $$k$$ tầng có thể biểu diễn tự nhiên với độ phức tạp đa thức.

Trực giác toán học là nhiều hàm trong tự nhiên mang cấu trúc hợp thành. Để nhận diện khuôn mặt, ta trước hết phát hiện cạnh, rồi kết hợp cạnh thành đặc trưng khuôn mặt (mắt, mũi, miệng), rồi kết hợp các đặc trưng thành biểu diễn khuôn mặt. Cấu trúc phân cấp này được biểu diễn tự nhiên bằng các phép biến đổi liên tiếp qua các tầng:

$$\mathbf{h}^{(1)} = \sigma(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{h}^{(2)} = \sigma(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
$$\vdots$$
$$\mathbf{y} = \mathbf{W}^{(L)}\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}$$

trong đó $$\sigma$$ là hàm kích hoạt phi tuyến (ReLU, sigmoid, tanh), $$\mathbf{W}^{(l)}$$ là ma trận trọng số và $$\mathbf{b}^{(l)}$$ là vector bias ở tầng $$l$$.

### Mục tiêu học

Huấn luyện mạng nơ-ron nghĩa là tìm tham số $$\theta = \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \ldots, \mathbf{W}^{(L)}, \mathbf{b}^{(L)}\}$$ cực tiểu hóa hàm mất mát $$\mathcal{L}(\theta)$$ đo lỗi dự đoán:

$$\theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$$

Với phân loại, ta thường dùng mất mát entropy chéo (*cross-entropy*):
$$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_j y_j \log \hat{y}_j$$

Với hồi quy, lỗi bình phương trung bình:
$$\mathcal{L}(\hat{y}, y) = \frac{1}{2}(y - \hat{y})^2$$

Ta tối ưu bằng **gradient descent**: cập nhật tham số lặp theo hướng làm giảm mất mát:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

trong đó $$\eta$$ là tốc độ học (*learning rate*). Việc tính gradient hiệu quả qua lan truyền ngược (*backpropagation*) — áp dụng quy tắc chuỗi tầng theo tầng — chính là điều khiến huấn luyện mạng sâu trở nên khả thi.

### Vì sao nó hoạt động: cân bằng bias–variance

Thành công của học sâu có thể hiểu qua cân bằng bias–variance cổ điển. Bias cao (underfitting) nghĩa là mô hình không nắm bắt được độ phức tạp của dữ liệu. Variance cao (overfitting) nghĩa là mô hình khớp nhiễu thay vì mẫu thật. Mạng sâu có dung lượng rất lớn (bias thấp) nhưng lại đáng ngạc nhiên kháng overfitting khi được chính quy hóa đúng cách, đạt variance thấp dù có hàng triệu tham số — thường nhiều hơn số mẫu huấn luyện!

Điều này dường như mâu thuẫn với lý thuyết học thống kê cổ điển, vốn gợi ý mô hình nên đơn giản hơn dữ liệu. Các công trình lý thuyết gần đây về "double descent" và "chính quy hóa ngầm" (*implicit regularization*) cho thấy mạng overparameterized huấn luyện bằng gradient descent ngầm ưu tiên các hàm đơn giản hơn, tạo ra một dạng chính quy hóa tự động mà lý thuyết cổ điển chưa tính đến.

## 3. Ví dụ / Trực giác

Để xây dựng trực giác về cách học sâu hoạt động, ta xét một ví dụ cụ thể: dạy mạng nhận dạng chữ số viết tay.

### Bài toán: nhận dạng chữ số MNIST

Giả sử ta có ảnh xám kích thước 28×28 của chữ số viết tay (0–9) và muốn một hệ thống xác định đúng chữ số trong mỗi ảnh. Mỗi ảnh chỉ là một vector 784 chiều (28×28 = 784 pixel, mỗi pixel có cường độ 0–255).

**Cách tiếp cận truyền thống**: thiết kế đặc trưng thủ công:
- Đếm vòng lặp (0, 6, 8, 9 có vòng; 1, 7 không)
- Phát hiện nét dọc/ngang
- Đo tỷ lệ chiều cao–chiều rộng
- Xác định điểm cuối và giao điểm

Cách này đòi hỏi chuyên môn sâu và khái quát kém (còn chữ viết hoa? font khác?).

**Cách tiếp cận học sâu**: đưa trực tiếp 784 giá trị pixel vào mạng nơ-ron:

```
Input (784 pixels) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (64 neurons) → Output (10 classes)
```

Mạng tự động học:
- **Tầng 1** khám phá bộ phát hiện cạnh — các nơ-ron kích hoạt với đường dọc, ngang, đường cong ở các vị trí khác nhau
- **Tầng 2** kết hợp cạnh thành mẫu nét — nét dọc dài (cho 1, 7), hình tròn (cho 0, 6, 8, 9), tổ hợp đường cong cụ thể
- **Tầng đầu ra** kết hợp các mẫu này để nhận diện chữ số hoàn chỉnh

### Học diễn ra như thế nào: ví dụ trực quan

Ban đầu, trọng số là ngẫu nhiên. Khi đưa vào một ảnh "3":
1. **Lan truyền xuôi** (*forward pass*): mạng đưa ra dự đoán ngẫu nhiên, ví dụ tin 70% đó là "7"
2. **Tính lỗi**: nhãn đúng là "3", dự đoán là "7" — lỗi lớn!
3. **Lan truyền ngược** (*backpropagation*):
   - Tầng đầu ra: "nên kích hoạt nơ-ron 3 mạnh hơn và nơ-ron 7 yếu hơn"
   - Các tầng ẩn: "kích hoạt nào đã góp phần vào dự đoán sai? điều chỉnh trọng số để sửa"
4. **Cập nhật trọng số**: điều chỉnh nhẹ mọi trọng số để giảm lỗi cụ thể này
5. **Lặp lại**: sau hàng nghìn ảnh "3" với nhiều kiểu viết tay, mạng học được các đặc trưng cốt yếu của "tính chất là 3"

Điều kỳ diệu là quy trình đơn giản này — lan truyền xuôi, tính lỗi, lan truyền ngược, cập nhật — khi lặp lại hàng triệu lần, khám phá được các đặc trưng phân cấp cần thiết cho nhận dạng.

### Vì sao học phân cấp quan trọng

Xét bài toán nhận diện khuôn mặt:
- **Đặc trưng mức thấp** (Tầng 1): bộ phát hiện cạnh ở nhiều hướng, vùng màu
- **Đặc trưng mức trung** (Tầng 2–3): kết hợp cạnh thành hình dạng đơn giản — đường cong, góc, texture
- **Đặc trưng mức cao** (Tầng 4–5): kết hợp hình dạng thành bộ phận khuôn mặt — mắt (cặp vòng tối có highlight), mũi (vùng tam giác có bóng), miệng (vùng tối ngang, có thể có răng)
- **Khái niệm hoàn chỉnh** (Đầu ra): kết hợp các bộ phận thành danh tính khuôn mặt cụ thể

Mỗi tầng học biểu diễn ngày càng trừu tượng, nắm bắt tự nhiên bản chất hợp thành của nhận dạng thị giác. Mạng phát hiện rằng mắt, mũi, miệng là các thành phần tái sử dụng xuất hiện ở mọi khuôn mặt, giống như nét và đường cong là thành phần tái sử dụng trong mọi chữ số.

Biểu diễn phân cấp, phân tán này cũng giải thích hiệu quả mẫu của học sâu. Một khi mạng đã học nơ-ron "phát hiện cạnh" và "phát hiện hình tròn" từ chữ số, chính các nơ-ron đó hỗ trợ nhận diện chữ cái, khuôn mặt và đối tượng — chuyển giao học (*transfer learning*) diễn ra tự nhiên nhờ đặc trưng mức thấp được chia sẻ.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Mạng nơ-ron là bộ xấp xỉ hàm](https://www.facebook.com/reel/720970114372332)
