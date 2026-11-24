---
layout: post
title: 24-01-01 Lý thuyết Diễn giải
chapter: '24'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter24
---

Nếu ta chỉ lấy đạo hàm $$\left|\frac{\partial f(x)}{\partial x_i}\right|$$ để đánh giá tầm quan trọng của đặc trưng $$x_i$$, một bẫy toán học tinh vi lập tức xuất hiện: tại vùng bão hòa của ReLU hay Sigmoid, đạo hàm có thể bằng **đúng 0** ngay cả khi đặc trưng đó đã đóng góp toàn bộ tín hiệu làm thay đổi dự đoán! Bài lý thuyết này trình bày cách **Integrated Gradients**, **Grad-CAM** và **SHAP (giá trị Shapley)** khắc phục bẫy bão hòa và thiết lập những tiên đề công bằng chặt chẽ cho việc quy kết đặc trưng.

## Bản đồ kiến thức nền: Nền tảng toán học của Saliency, Integrated Gradients, Grad-CAM và SHAP

- **Saliency Map & Integrated Gradients (IG):** Khắc phục bão hòa gradient bằng cách tích phân đạo hàm dọc đoạn thẳng từ điểm cơ sở $$\mathbf{x}'$$ tới $$\mathbf{x}$$: $$\text{IG}_i(\mathbf{x}) = (x_i - x'_i) \int_0^1 \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$, đảm bảo tiên đề *đầy đủ* (tổng đóng góp bằng $$f(\mathbf{x}) - f(\mathbf{x}')$$).
- **Grad-CAM trên tầng tích chập cuối:** Tính trọng số tầm quan trọng từng kênh bằng trung bình không gian của gradient $$\alpha_k^c = \frac{1}{Z}\sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}$$, rồi lấy tổ hợp tuyến tính qua ReLU để giữ các vùng đóng góp dương cho lớp $$c$$.
- **Giá trị Shapley trong lý thuyết trò chơi (SHAP):** Phương pháp quy kết *duy nhất* thỏa mãn đồng thời ba tiên đề: độ chính xác cục bộ, tính khuyết (missingness) và tính nhất quán (consistency).
- **LIME (Local Interpretable Model-Agnostic Explanations):** Lấy mẫu nhiễu quanh điểm $$\mathbf{x}$$, gán trọng số gần xa theo nhân khoảng cách, rồi khớp mô hình tuyến tính thưa giải thích hành vi cục bộ.

### Khung giải thích (nên đọc chậm): Tính tay giá trị Shapley (SHAP) cho một mô hình dự đoán có 2 đặc trưng

Giả sử một mô hình chấm điểm rủi ro tín dụng dùng 2 đặc trưng: `Thu nhập` ($$x_1$$) và `Lịch sử trễ hạn` ($$x_2$$). Giá trị dự đoán kỳ vọng của mô hình trên 4 tập con đặc trưng đã biết là:
- Không biết đặc trưng nào: $$v(\varnothing) = 50$$ (điểm cơ sở).
- Chỉ biết $$x_1$$: $$v(\{x_1\}) = 62$$.
- Chỉ biết $$x_2$$: $$v(\{x_2\}) = 70$$.
- Biết cả hai: $$v(\{x_1, x_2\}) = 90$$ (tổng mức tăng cần phân bổ là $$90 - 50 = +40$$).

Với 2 đặc trưng có $$2! = 2$$ thứ tự thêm vào ($$x_1 \to x_2$$ và $$x_2 \to x_1$$):
1. **Đóng góp biên của $$x_1$$:**
   - Khi thêm trước: $$v(\{x_1\}) - v(\varnothing) = 62 - 50 = +12$$.
   - Khi thêm sau $$x_2$$: $$v(\{x_1, x_2\}) - v(\{x_2\}) = 90 - 70 = +20$$.
   - Giá trị Shapley của $$x_1$$: $$\phi_1 = \frac{12 + 20}{2} = +16$$.
2. **Đóng góp biên của $$x_2$$:**
   - Khi thêm trước: $$70 - 50 = +20$$; khi thêm sau $$x_1$$: $$90 - 62 = +28$$.
   - Giá trị Shapley của $$x_2$$: $$\phi_2 = \frac{20 + 28}{2} = +24$$.
3. **Kiểm chứng tiên đề đầy đủ (Efficiency):** $$\phi_1 + \phi_2 = 16 + 24 = +40 = v(\{x_1, x_2\}) - v(\varnothing)$$ — chính xác tuyệt đối!

# Diễn giải Mô hình: Hiểu Cơ chế “Hộp đen”

## 1. Tổng quan khái niệm

Diễn giải mạng nơ-ron (*neural network interpretability*) giải quyết câu hỏi cốt lõi trong học sâu: vì sao mô hình lại đưa ra một dự đoán cụ thể? Dù mạng nơ-ron đạt độ chính xác cao trên nhiều tác vụ, quá trình suy luận bên trong thường khó quan sát vì hàng triệu tham số tương tác qua nhiều tầng biến đổi phi tuyến. Tính chất “hộp đen” này trở thành rào cản trong các lĩnh vực có mức độ rủi ro cao như y khoa (vì sao mô hình chẩn đoán bệnh này?), pháp lý (vì sao đánh giá rủi ro cao?) hay xe tự lái (vì sao xe phanh gấp?), nơi hệ thống cần không chỉ dự đoán đúng mà còn có căn cứ để kiểm toán, tin cậy và gỡ lỗi.

Cần phân biệt rõ giữa *khả năng diễn giải* (*interpretability*) và *khả năng giải thích* (*explainability*). Một mô hình có khả năng diễn giải khi cơ chế tính toán nội tại của nó minh bạch — chẳng hạn hồi quy tuyến tính hoặc cây quyết định nông, nơi ta nhìn trực tiếp cách các đặc trưng kết hợp thành đầu ra. Khả năng giải thích là việc cung cấp lời giải thích hậu kỳ (*post-hoc*) cho một dự đoán cụ thể, ngay cả khi bản thân mô hình không minh bạch. Mạng nơ-ron sâu hiếm khi diễn giải trực tiếp được toàn bộ tham số, nhưng có thể được giải thích thông qua các kỹ thuật làm nổi bật vùng đầu vào quan trọng, trực quan hóa đặc trưng đã học hoặc xấp xỉ quyết định cục bộ bằng một mô hình thay thế đơn giản hơn.

Nhu cầu diễn giải xuất phát từ nhiều mục đích thực tế: trong nghiên cứu khoa học và y học, quan sát đặc trưng mô hình dựa vào có thể gợi mở dấu hiệu mới; trong gỡ lỗi, diễn giải giúp phát hiện tương quan giả (như nhận diện chó husky dựa trên nền tuyết thay vì đặc điểm con chó); trong các hệ thống an toàn trọng yếu, nó hỗ trợ kiểm chứng hành vi trên nhiều kịch bản; và trong các ngành chịu sự quản lý chặt chẽ, giải thích quyết định có thể là yêu cầu pháp lý.

Các nhóm phương pháp diễn giải tiếp cận bài toán từ nhiều góc độ khác nhau: phương pháp *saliency* làm nổi bật vùng đầu vào tác động mạnh nhất đến dự đoán (“mô hình nhìn vào đâu?”); trực quan hóa kích hoạt cho thấy mẫu đầu vào nào làm một nơ-ron kích hoạt mạnh (“mạng học được đặc trưng gì?”); phương pháp quy kết (*attribution*) phân bổ giá trị dự đoán thành đóng góp của từng đặc trưng (“mỗi đặc trưng quan trọng đến mức nào?”); và giải thích theo khái niệm tìm kiếm các khái niệm bậc cao thay vì dừng ở từng pixel hay từ đơn lẻ. Trong thực tế, việc đánh giá toàn diện thường cần phối hợp nhiều kỹ thuật bổ trợ nhau.

Diễn giải mô hình cũng luôn đi kèm các sự đánh đổi: mô hình dễ diễn giải trực tiếp thường kém chính xác hơn mạng sâu; lời giải thích trung thực hoàn toàn với cơ chế nội tại có thể quá phức tạp để con người hiểu, trong khi lời giải thích đơn giản lại dễ lược bỏ chi tiết quan trọng. Vì vậy, nghiên cứu diễn giải tập trung cân bằng giữa độ trung thực, tính dễ hiểu và giá trị sử dụng cho từng bối cảnh cụ thể.

## 2. Nền tảng toán học

Các phương pháp quy kết hình thức hóa câu hỏi “đặc trưng đầu vào nào ảnh hưởng nhiều nhất đến dự đoán?” bằng một vectơ quy kết $$\mathbf{a}$$ cho đầu vào $$\mathbf{x}$$ và mô hình $$f$$, trong đó $$a_i$$ biểu thị mức độ quan trọng của thành phần đầu vào thứ $$i$$ đối với $$f(\mathbf{x})$$.

### Bản đồ nổi bật dựa trên Gradient (Gradient-based Saliency)

Cách quy kết đơn giản nhất sử dụng độ lớn đạo hàm theo đầu vào:

$$\mathbf{a} = \left|\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}\right|$$

Đại lượng này đo mức độ nhạy của đầu ra trước thay đổi nhỏ của từng chiều đầu vào. Trên dữ liệu ảnh, kết quả tạo thành bản đồ *saliency* làm nổi bật các pixel có ảnh hưởng lớn.

Tuy nhiên, gradient thuần túy có thể bị bão hòa (đạo hàm tiến về 0 ở vùng phẳng dù đặc trưng vẫn quan trọng) và không đối chiếu với một mốc tham chiếu (*baseline*). **Integrated Gradients** khắc phục điều này bằng cách tích phân gradient dọc theo đoạn thẳng nối từ mốc tham chiếu $$\mathbf{x}'$$ đến đầu vào $$\mathbf{x}$$:

$$\mathbf{a}_i = (x_i - x_i') \int_{\alpha=0}^1 \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

Phương pháp này thỏa mãn các tiên đề quan trọng như tính nhạy (*sensitivity* — đặc trưng không ảnh hưởng đầu ra có điểm quy kết bằng 0) và bất biến cài đặt (*implementation invariance* — hai mạng tương đương về hàm số cho cùng kết quả quy kết).

### SHAP (Shapley Additive Explanations)

SHAP dựa trên giá trị Shapley trong lý thuyết trò chơi hợp tác. Đóng góp của đặc trưng $$i$$ được định nghĩa là:

$$\phi_i = \sum_{S \subseteq \mathcal{F} \backslash \{i\}} \frac{|S|!(|\mathcal{F}|-|S|-1)!}{|\mathcal{F}|!} [f(S \cup \{i\}) - f(S)]$$

trong đó $$\mathcal{F}$$ là tập toàn bộ đặc trưng, $$S$$ là tập con đặc trưng và $$f(S)$$ là dự đoán của mô hình khi chỉ giữ các đặc trưng trong $$S$$ (các đặc trưng còn lại đặt về giá trị tham chiếu). Công thức tính trung bình có trọng số đóng góp biên của đặc trưng $$i$$ trên mọi tổ hợp con khả dĩ.

Vì tính chính xác giá trị Shapley đòi hỏi $$2^{|\mathcal{F}|}$$ lần đánh giá mô hình, thực tế thường dùng các thuật toán xấp xỉ như Kernel SHAP (hồi quy tuyến tính có trọng số) hoặc thuật toán chính xác trong thời gian đa thức như TreeSHAP dành cho mô hình cây.

### Lan truyền độ liên quan theo tầng (Layer-wise Relevance Propagation, LRP)

LRP phân bổ ngược độ liên quan từ đầu ra về đầu vào qua từng tầng:

$$R_i^{(l)} = \sum_j \frac{z_{ij}}{\sum_k z_{kj}} R_j^{(l+1)}$$

trong đó $$z_{ij} = a_i^{(l)} w_{ij}$$ là đóng góp của nơ-ron $$i$$ ở tầng $$l$$ vào nơ-ron $$j$$ ở tầng $$l+1$$. Xuất phát từ $$R_{\text{out}} = f(\mathbf{x})$$, phép lan truyền ngược bảo toàn tổng độ liên quan qua các tầng sao cho $$\sum_i R_i^{(0)} = f(\mathbf{x})$$.

## 3. Ví dụ và trực giác

Xét một mạng CNN phân loại một bức ảnh là “chó” với độ tin cậy 95%. Nếu không có công cụ diễn giải, ta không biết mô hình dựa vào khuôn mặt chó, dáng thân hay một yếu tố nền mang tính tương quan giả như bãi cỏ.

- **Gradient saliency** tính $$\partial p_{\text{dog}}/\partial \text{pixels}$$ để xác định các pixel mà thay đổi nhỏ tạo biến động lớn nhất cho xác suất lớp “chó”. Nếu bản đồ nhiệt tập trung vào mặt và tai chó, mô hình đang dùng đặc trưng hợp lý; nếu tập trung vào nền, mô hình có nguy cơ dựa vào tương quan giả.
- **Class Activation Mapping (CAM)** đối với CNN có tầng *global average pooling* tính tổ hợp tuyến tính của các bản đồ đặc trưng ở tầng tích chập cuối bằng trọng số của lớp “chó”:

$$\text{CAM} = \sum_k w_k^{\text{dog}} \cdot \text{FeatureMap}_k$$

Phóng to bản đồ nhiệt này lên kích thước ảnh gốc và chồng lên ảnh giúp thấy trực quan vùng không gian nào đóng góp vào dự đoán.

- **Giá trị SHAP** cho cùng dự đoán có thể cho kết quả minh họa như sau:
  - Vùng mặt chó: `+0.35` (đóng góp dương mạnh).
  - Vùng thân chó: `+0.28`.
  - Nền cỏ: `+0.08` (đóng góp nhỏ; nếu lớn sẽ là dấu hiệu tương quan giả).
  - Vùng bầu trời: `-0.02` (âm nhẹ, phù hợp với vùng không liên quan).

Nếu vùng nền cỏ có giá trị SHAP lớn, ta phát hiện được thiên lệch dữ liệu (chó thường xuất hiện trên cỏ) để bổ sung dữ liệu đa dạng hơn hoặc tăng cường dữ liệu.

- **Ví dụ đối kháng** (*adversarial example*) cung cấp một góc nhìn khác về độ vững chắc: tìm nhiễu đầu vào nhỏ nhất làm thay đổi dự đoán. Nếu một nhiễu rất nhỏ không thể nhận ra bằng mắt thường khiến ảnh “chó” bị phân loại thành “mèo”, biên quyết định của mô hình còn mong manh và có thể đang dựa vào các mẫu tín hiệu thiếu bền vững.

## Video tham khảo

- **Xem trước** [SHAP values for beginners | What they mean and their applications](https://www.youtube.com/watch?v=MQ6fFDwjuco) — A Data Odyssey, tiếng Anh. Giải thích giá trị SHAP như cách phân bổ đóng góp công bằng cho từng đặc trưng, tương ứng với định nghĩa trong bài.
- [Lecture 12 | Visualizing and Understanding](https://www.youtube.com/watch?v=6wcs6szJWMY) — Stanford University School of Engineering, tiếng Anh. Trình bày nhánh trực quan hóa đặc trưng (xem một nơ-ron phản ứng với mẫu ảnh nào).
- [Explainable AI explained! | #3 LIME](https://www.youtube.com/watch?v=d6j6bofhj2M) — DeepFindr, tiếng Anh. Giới thiệu LIME — phương pháp xấp xỉ tuyến tính cục bộ bổ sung cạnh SHAP.
