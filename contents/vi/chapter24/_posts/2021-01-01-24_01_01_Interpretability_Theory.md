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

# Diễn giải Mô hình: Hiểu Hộp đen

## 1. Tổng quan khái niệm
Diễn giải mạng neuron (*neural network interpretability*) giải quyết một trong những thách thức quan trọng nhất của học sâu: hiểu vì sao mô hình đưa ra dự đoán cụ thể. Dù mạng neuron đạt hiệu năng đáng kể trên nhiều tác vụ, quá trình ra quyết định của chúng thường mờ đục — hàng triệu tham số tương tác qua phép biến đổi phi tuyến khiến khó lần theo cách đầu vào ánh xạ sang đầu ra. Bản chất “hộp đen” này tạo ra vấn đề trong các miền rủi ro cao như y học (vì sao mô hình chẩn đoán tình trạng này?), luật (vì sao bị cáo này được phân loại rủi ro cao?), và xe tự lái (vì sao xe phanh đột ngột?), nơi ta cần không chỉ dự đoán chính xác mà còn biện minh có thể kiểm toán, tin cậy và gỡ lỗi.

Hiểu sự phân biệt giữa diễn giải (*interpretability*) và giải thích (*explainability*) làm rõ những gì ta tìm kiếm. Diễn giải nghĩa là hoạt động nội tại của mô hình trong suốt — ta có thể hiểu phép tính từ việc kiểm tra. Mô hình đơn giản như hồi quy tuyến tính hay cây quyết định vốn dễ diễn giải; ta thấy chính xác cách đặc trưng kết hợp để sinh dự đoán. Giải thích nghĩa là ta có thể cung cấp giải thích hậu kỳ về vì sao mô hình đưa ra dự đoán cụ thể, dù bản thân mô hình không trong suốt. Mạng neuron sâu hiếm khi dễ diễn giải (hiểu hàng triệu tham số là bất khả thi) nhưng có thể được làm dễ giải thích qua các kỹ thuật làm nổi bật đầu vào liên quan, trực quan hóa đặc trưng học được, hoặc xấp xỉ quyết định bằng surrogate dễ diễn giải.

Động lực cho diễn giải vượt xa việc thỏa mãn tò mò. Trong ứng dụng khoa học, hiểu đặc trưng mà mô hình dùng có thể sinh giả thuyết mới — nếu mô hình ảnh y khoa xác định mẫu tinh tế mà bác sĩ bỏ sót, nghiên cứu mẫu đó có thể hé lộ dấu hiệu chẩn đoán mới. Trong gỡ lỗi, diễn giải tiết lộ khi mô hình khai thác tương quan giả (phát hiện husky bằng tuyết nền thay vì đặc trưng chó) hoặc không dùng đặc trưng liên quan. Trong ứng dụng an toàn quan trọng, diễn giải cho phép kiểm chứng mô hình hành xử hợp lý trên các kịch bản đa dạng. Trong ngành được quy định, giải thích có thể là yêu cầu pháp lý cho quyết định ảnh hưởng cá nhân. Hiểu các động lực đa dạng này giúp đánh giá rằng diễn giải không phải một mục tiêu đơn lẻ mà nhiều mục tiêu liên quan đòi hỏi kỹ thuật khác nhau.

Bức tranh các phương pháp diễn giải rất rộng, phản ánh nhiều cách ta có thể hiểu mạng neuron. Phương pháp saliency làm nổi bật vùng đầu vào ảnh hưởng nhiều nhất đến dự đoán, trả lời “mô hình nhìn vào đâu?”. Trực quan hóa kích hoạt cho thấy mẫu nào kích hoạt neuron, tiết lộ “mạng đã học đặc trưng gì?”. Phương pháp gán công trạng (*attribution*) phân rã dự đoán thành đóng góp đặc trưng, giải thích “mỗi đặc trưng đầu vào quan trọng bao nhiêu?”. Giải thích dựa trên khái niệm xác định khái niệm cấp cao mà mô hình dùng, vượt gán công trạng cấp pixel hay từ sang hiểu biết ngữ nghĩa. Mỗi cách tiếp cận cung cấp góc nhìn khác, và diễn giải toàn diện thường đòi hỏi nhiều kỹ thuật bổ sung lẫn nhau.

Tuy nhiên, diễn giải có các căng thẳng cơ bản. Mô hình dễ diễn giải hơn thường kém chính xác hơn (mô hình tuyến tính so với mạng sâu). Giải thích trung thực (mô tả chính xác hành vi mô hình) có thể phức tạp và khó hiểu. Giải thích đơn giản có thể dễ hiểu nhưng không trung thực với hành vi thực của mô hình. Diễn giải hoàn hảo có thể đòi hỏi hiểu hàng triệu tham số — phức tạp ngang hiểu hiện tượng mà mô hình đã học. Các căng thẳng này nghĩa là nghiên cứu diễn giải liên quan đánh đổi cẩn thận giữa độ trung thực, tính đơn giản và tính hữu dụng, không có lời giải phổ quát thỏa mãn mọi tiêu chí cùng lúc.

## 2. Nền tảng toán học
Các phương pháp diễn giải thường hình thức hóa câu hỏi “đầu vào nào quan trọng nhất cho dự đoán này?” qua gán công trạng. Cho đầu vào $$\mathbf{x}$$ và mô hình $$f$$, tính gán công trạng $$\mathbf{a}$$ trong đó $$a_i$$ chỉ tầm quan trọng của chiều đầu vào $$i$$ đối với dự đoán $$f(\mathbf{x})$$.

### Saliency dựa trên Gradient

Gán công trạng đơn giản nhất dùng gradient:

$$\mathbf{a} = \left|\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}\right|$$

Phép này đo mức đầu ra sẽ thay đổi bao nhiêu với thay đổi nhỏ ở mỗi chiều đầu vào. Gradient lớn chỉ ra độ nhạy cao — chiều đầu vào đó ảnh hưởng mạnh đầu ra. Với ảnh, điều này sinh bản đồ saliency làm nổi bật pixel quan trọng.

Tuy nhiên, gradient có thể bão hòa trong mạng ReLU (gradient là 0 hoặc 1, không mang thông tin về độ lớn thay đổi) và không tính đến baseline (ta so sánh với cái gì?). Các cải tiến giải quyết vấn đề này:

**Integrated Gradients** tích lũy gradient dọc đường từ baseline $$\mathbf{x}'$$ đến đầu vào $$\mathbf{x}$$:

$$\mathbf{a}_i = (x_i - x_i') \int_{\alpha=0}^1 \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

Phép này thỏa các tiên đề mong muốn: độ nhạy (nếu đầu vào không ảnh hưởng đầu ra, gán công trạng bằng 0) và bất biến cài đặt (mạng tương đương cho cùng gán công trạng).

### SHAP: Shapley Additive Explanations

SHAP dùng giá trị Shapley từ lý thuyết trò chơi hợp tác. Đóng góp của đặc trưng $$i$$ là:

$$\phi_i = \sum_{S \subseteq \mathcal{F} \backslash \{i\}} \frac{|S|!(|\mathcal{F}|-|S|-1)!}{|\mathcal{F}|!} [f(S \cup \{i\}) - f(S)]$$

trong đó $$\mathcal{F}$$ là mọi đặc trưng, $$S$$ là tập con đặc trưng, $$f(S)$$ là đầu ra mô hình khi chỉ các đặc trưng trong $$S$$ hiện diện (các đặc trưng khác đặt về baseline). Phép này tính đóng góp biên trung bình của đặc trưng $$i$$ trên mọi liên minh đặc trưng khả dĩ — phân bổ công bằng dự đoán giữa các đặc trưng.

Tính giá trị Shapley chính xác đòi hỏi $$2^{|\mathcal{F}|}$$ lần đánh giá mô hình (mũ theo số đặc trưng), nên dùng xấp xỉ. Kernel SHAP xấp xỉ qua hồi quy tuyến tính có trọng số. Với mô hình cây, TreeSHAP tính chính xác trong thời gian đa thức.

### Lan truyền Độ liên quan theo Tầng (Layer-wise Relevance Propagation, LRP)

LRP lan truyền ngược độ liên quan từ đầu ra về đầu vào:

$$R_i^{(l)} = \sum_j \frac{z_{ij}}{\sum_k z_{kj}} R_j^{(l+1)}$$

trong đó $$z_{ij} = a_i^{(l)} w_{ij}$$ là đóng góp của neuron $$i$$ ở tầng $$l$$ cho neuron $$j$$ ở tầng $$l+1$$. Bắt đầu với $$R_{\text{out}} = f(\mathbf{x})$$ ở đầu ra, độ liên quan lan truyền ngược, phân rã dự đoán thành đóng góp đầu vào thỏa $$\sum_i R_i^{(0)} = f(\mathbf{x})$$ (bảo toàn).

## 3. Ví dụ / Trực giác

Xét CNN phân loại ảnh là “chó” với độ tin cậy 95%. Không có diễn giải, ta không biết vì sao. Là mặt chó, dáng thân, ngữ cảnh nền, hay mẫu giả như cỏ (nếu mọi chó trong huấn luyện có nền cỏ)?

**Gradient saliency** tính $$\partial p_{\text{dog}}/\partial \text{pixels}$$. Gradient lớn làm nổi bật pixel mà nếu thay đổi nhẹ sẽ ảnh hưởng mạnh nhất xác suất chó. Trực quan hóa dưới dạng bản đồ nhiệt phủ trên ảnh, ta có thể thấy giá trị cao quanh mặt và tai chó — tốt, mô hình dùng đặc trưng chó thực. Nếu giá trị cao xuất hiện ở nền, mô hình có thể đang khai thác tương quan giả.

**Class Activation Mapping (CAM)** cho CNN với global average pooling cho thấy vùng nào tầng tích chập cuối tìm thấy quan trọng. Với lớp “chó”, ta tính tổ hợp có trọng số các bản đồ đặc trưng tầng conv cuối dùng trọng số phân loại:

$$\text{CAM} = \sum_k w_k^{\text{dog}} \cdot \text{FeatureMap}_k$$

Phép này sinh bản đồ nhiệt ở độ phân giải bản đồ đặc trưng cho thấy vùng không gian nào đóng góp cho dự đoán “chó”. Nâng mẫu lên độ phân giải đầu vào và phủ trên ảnh tiết lộ mô hình tập trung vào đầu và thân chó — dễ diễn giải và đáng tin.

**Giá trị SHAP** cho một dự đoán cụ thể có thể cho thấy:
- Vùng pixel chứa mặt chó: +0.35 (đóng góp dương mạnh)
- Vùng pixel với thân chó: +0.28
- Cỏ nền: +0.08 (đóng góp nhỏ — đáng lo nếu cao)
- Vùng trời: −0.02 (âm nhẹ — kỳ vọng với vùng không liên quan)

Nếu cỏ có giá trị SHAP cao, ta đã phát hiện mô hình dùng tương quan giả (chó thường được chụp trên cỏ). Ta có thể thu thập dữ liệu huấn luyện đa dạng hơn hoặc dùng tăng cường dữ liệu để sửa.

**Ví dụ đối kháng** (*adversarial example*) cung cấp một lăng kính diễn giải khác. Bằng cách tìm nhiễu đầu vào tối thiểu thay đổi dự đoán, ta tiết lộ lỗ hổng mô hình. Nếu thêm nhiễu không nhận thức được vào ảnh chó gây dự đoán “mèo”, biểu diễn của mô hình mong manh — nó chưa học đặc trưng vững. Nghiên cứu các nhiễu đối kháng này tiết lộ đặc trưng nào quan trọng: nhiễu thường thêm mẫu mà mô hình liên kết mạnh với lớp đích, tiết lộ chỉ báo lớp học được (nhưng có thể giả).
