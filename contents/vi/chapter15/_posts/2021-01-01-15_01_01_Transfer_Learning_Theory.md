---
layout: post
title: 15-01-01 Lý thuyết Học Chuyển giao
chapter: '15'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter15
---

# Học Chuyển giao: Tận dụng Tri thức Tiền huấn luyện

## 1. Tổng quan khái niệm
Học chuyển giao (transfer learning) đại diện cho một trong những mô hình thực tiễn quan trọng nhất trong deep learning hiện đại, cho phép ta xây dựng mô hình hiệu quả cao với dữ liệu chuyên tác vụ hạn chế bằng cách tận dụng tri thức học được từ các tác vụ liên quan. Nguyên tắc cốt lõi tưởng chừng đơn giản: thay vì huấn luyện mạng neuron từ đầu với trọng số khởi tạo ngẫu nhiên, ta bắt đầu với trọng số tiền huấn luyện trên tập dữ liệu lớn cho tác vụ liên quan, rồi thích ứng các trọng số này cho bài toán cụ thể của ta. Cách tiếp cận này đã dân chủ hóa deep learning, khiến nó dễ tiếp cận với thực hành viên thiếu tập dữ liệu khổng lồ và tài nguyên tính toán cần để huấn luyện mô hình lớn từ đầu. Một ứng dụng ảnh y tế có thể tận dụng mạng tiền huấn luyện trên ImageNet. Một mô hình phân tích cảm xúc có thể bắt đầu từ BERT tiền huấn luyện trên văn bản web. Một hệ thống nhận dạng giọng nói có thể tinh chỉnh Wav2Vec học trên âm thanh không gán nhãn.

Hiểu vì sao học chuyển giao hoạt động đòi hỏi đánh giá điều mạng neuron học trong huấn luyện. Các tầng của mạng sâu xây dựng dần biểu diễn phân cấp. Tầng sớm học đặc trưng tổng quát, mức thấp — cạnh, texture, hình dạng đơn giản cho ảnh; phoneme cơ bản cho âm thanh; mẫu từ phổ biến cho văn bản. Các đặc trưng này nhất quán đáng chú ý qua các tác vụ và tập dữ liệu. Một mạng huấn luyện để phân loại xe hơi so với xe tải học bộ phát hiện cạnh gần như giống hệt mạng phân loại chó so với mèo, vì cạnh là nền tảng cho hiểu thị giác bất kể đối tượng cụ thể. Tầng giữa học đặc trưng mức trung — phần đối tượng, tổ hợp texture, cấu trúc hình dạng — phần nào chuyên tác vụ nhưng vẫn hữu ích rộng. Chỉ các tầng sâu nhất học đặc trưng chuyên tác vụ cao — "tổ hợp cụ thể này chỉ ra golden retriever" cho phân loại giống chó.

Việc tái sử dụng đặc trưng qua các tác vụ này là điều khiến học chuyển giao khả thi. Các tầng sớm và giữa, đã học đặc trưng tổng quát trên tập dữ liệu nguồn lớn, cung cấp điểm khởi đầu mạnh cho tác vụ đích. Ngay cả nếu tác vụ đích khác (phân loại ảnh y tế thay vì ảnh tự nhiên), các đặc trưng thị giác nền tảng — cạnh, texture, hình dạng — vẫn liên quan. Ta không cần hàng triệu ảnh y tế để học các cơ bản này; ta có thể chuyển giao chúng từ ImageNet và tập trung dữ liệu y tế hạn chế vào học đặc trưng chuyên tác vụ ở các tầng sâu hơn. Điều này tương tự cách con người học: đã học khái niệm thị giác cơ bản từ trải nghiệm hàng ngày, ta có thể nhanh chóng học nhận diện bệnh hiếm từ vài ví dụ, chuyển giao hiểu biết thị giác tổng quát thay vì học thị giác từ đầu.

Tác động thực tiễn không thể đánh giá quá cao. Trước khi học chuyển giao trở thành thực hành chuẩn, huấn luyện bộ phân loại ảnh tốt đòi hỏi hàng trăm nghìn ảnh gán nhãn. Với học chuyển giao từ mô hình tiền huấn luyện ImageNet, kết quả cạnh tranh khả thi với hàng nghìn hoặc thậm chí hàng trăm ảnh. Trong xử lý ngôn ngữ tự nhiên, tác động còn kịch tính hơn. Các mô hình ngôn ngữ tiền huấn luyện như BERT, huấn luyện trên hàng tỷ từ văn bản, có thể được tinh chỉnh cho tác vụ cụ thể (phân tích cảm xúc, nhận diện thực thể có tên, trả lời câu hỏi) với tập dữ liệu chỉ hàng nghìn ví dụ gán nhãn, đạt hiệu năng đòi hỏi hàng triệu nhãn nếu huấn luyện từ đầu. Điều này đã cho phép ứng dụng deep learning trong các miền nơi tập dữ liệu gán nhãn lớn không tồn tại: chẩn đoán y tế với dữ liệu bệnh nhân hạn chế, xử lý ngôn ngữ hiếm, hiểu tài liệu kỹ thuật chuyên biệt.

Tuy nhiên, học chuyển giao không phải phép màu, và hiểu khi nào nó hoạt động so với khi nào thất bại then chốt cho thực hành viên. Học chuyển giao giả định tác vụ nguồn và đích chia sẻ cấu trúc liên quan — cạnh học từ ImageNet giúp với ảnh y tế vì cả hai liên quan ảnh tự nhiên với cạnh, texture, và hình dạng. Nhưng đặc trưng ImageNet có thể không chuyển giao tốt sang ảnh radar (modal dữ liệu khác), ảnh vệ tinh (tỷ lệ và góc nhìn khác), hoặc nghệ thuật trừu tượng (tính chất thống kê khác). Phân phối nguồn và đích càng tương tự, đặc trưng chuyển giao càng hiệu quả. Nguyên tắc này hướng dẫn lựa chọn mô hình tiền huấn luyện: với ảnh y tế, mạng tiền huấn luyện trên X-quang ngực chuyển giao tốt hơn ImageNet, dù ImageNet vẫn hiệu quả đáng ngạc nhiên nhờ tính tổng quát của đặc trưng thị giác mức thấp và trung.

## 2. Nền tảng toán học
Khung toán học cho học chuyển giao nối với thích ứng miền (domain adaptation), học đa tác vụ (multi-task learning), và meta-learning. Hãy hình thức hóa điều ta làm khi chuyển giao tri thức và hiểu các nền tảng lý thuyết giải thích vì sao nó hoạt động.

Giả sử ta có miền nguồn với phân phối $$p_S(\mathbf{x}, y)$$ và dữ liệu gán nhãn dồi dào $$\mathcal{D}_S = \{(\mathbf{x}_i^S, y_i^S)\}_{i=1}^{N_S}$$, và miền đích với phân phối $$p_T(\mathbf{x}, y)$$ và dữ liệu gán nhãn hạn chế $$\mathcal{D}_T = \{(\mathbf{x}_j^T, y_j^T)\}_{j=1}^{N_T}$$ với $$N_T \ll N_S$$. Ta muốn học bộ dự đoán $$f_\theta(\mathbf{x})$$ hoạt động tốt trên miền đích.

Trong học giám sát chuẩn, ta sẽ tối thiểu hóa rủi ro thực nghiệm trên dữ liệu đích:

$$\theta^* = \arg\min_\theta \frac{1}{N_T}\sum_{j=1}^{N_T} \mathcal{L}(f_\theta(\mathbf{x}_j^T), y_j^T)$$

Nhưng với $$N_T$$ nhỏ, điều này dẫn đến overfitting nghiêm trọng — mô hình ghi nhớ ví dụ huấn luyện mà không học mẫu hình tổng quát hóa được.

Học chuyển giao thay vào đó thực hiện tối ưu hai giai đoạn:

**Giai đoạn 1 (Tiền huấn luyện)**: Huấn luyện trên miền nguồn
$$\theta_S^* = \arg\min_\theta \frac{1}{N_S}\sum_{i=1}^{N_S} \mathcal{L}(f_\theta(\mathbf{x}_i^S), y_i^S)$$

**Giai đoạn 2 (Tinh chỉnh)**: Khởi tạo với $$\theta_S^*$$, rồi huấn luyện trên miền đích
$$\theta_T^* = \arg\min_\theta \frac{1}{N_T}\sum_{j=1}^{N_T} \mathcal{L}(f_\theta(\mathbf{x}_j^T), y_j^T), \quad \text{bắt đầu từ } \theta_0 = \theta_S^*$$

Khởi tạo $$\theta_0 = \theta_S^*$$ then chốt — nó cung cấp điểm khởi đầu đã gần nghiệm tốt cho tác vụ đích (giả định các miền liên quan), cho phép tinh chỉnh hội tụ nhanh với dữ liệu hạn chế.

Ta có thể phân rã mô hình thành $$f_\theta = h_{\theta_h} \circ g_{\theta_g}$$ với $$g_{\theta_g}$$ là bộ trích xuất đặc trưng (tầng sớm/giữa) và $$h_{\theta_h}$$ là đầu chuyên tác vụ (tầng cuối). Các chiến lược học chuyển giao khác nhau về điều chúng chuyển giao và điều chúng thích ứng:

**Trích xuất đặc trưng (feature extraction)**: Đóng băng $$\theta_g = \theta_g^S$$ (dùng đặc trưng tiền huấn luyện), chỉ huấn luyện $$\theta_h$$ trên dữ liệu đích
$$\theta_h^* = \arg\min_{\theta_h} \frac{1}{N_T}\sum_{j=1}^{N_T} \mathcal{L}(h_{\theta_h}(g_{\theta_g^S}(\mathbf{x}_j^T)), y_j^T)$$

**Tinh chỉnh mọi tầng**: Khởi tạo cả $$\theta_g$$ và $$\theta_h$$ từ nguồn, huấn luyện cả hai trên đích
$$(\theta_g^*, \theta_h^*) = \arg\min_{\theta_g, \theta_h} \frac{1}{N_T}\sum_{j=1}^{N_T} \mathcal{L}(h_{\theta_h}(g_{\theta_g}(\mathbf{x}_j^T)), y_j^T)$$
bắt đầu từ $$(\theta_g^S, \theta_h^{\text{random}})$$

**Learning rate vi phân theo tầng**: Dùng learning rate khác nhau cho các tầng khác nhau
$$\theta_g \leftarrow \theta_g - \eta_g \nabla_{\theta_g} \mathcal{L}, \quad \theta_h \leftarrow \theta_h - \eta_h \nabla_{\theta_h} \mathcal{L}$$
thường với $$\eta_g < \eta_h$$ (learning rate nhỏ hơn cho tầng tiền huấn luyện, lớn hơn cho đầu mới)

Lựa chọn phụ thuộc kích thước tập dữ liệu và độ tương tự. Với dữ liệu đích rất nhỏ (hàng trăm ví dụ) và miền tương tự, trích xuất đặc trưng thường hoạt động tốt nhất — đặc trưng tiền huấn luyện đóng băng cung cấp biểu diễn robust, và ta chỉ cần học ánh xạ chuyên tác vụ. Với dữ liệu vừa phải (hàng nghìn) và độ tương tự vừa, tinh chỉnh với learning rate nhỏ thích ứng đặc trưng nhẹ trong khi tránh catastrophic forgetting. Với dữ liệu lớn (hàng chục nghìn+), tinh chỉnh đầy đủ hoặc thậm chí huấn luyện từ đầu có thể ưu tiên hơn.

### Lý thuyết Thích ứng Miền

Phân tích lý thuyết về khi nào chuyển giao hoạt động viện đến lý thuyết thích ứng miền. Định nghĩa không gian giả thuyết $$\mathcal{H}$$ (mọi hàm biểu diễn được bởi kiến trúc của ta). Lỗi trên miền đích cho giả thuyết $$h \in \mathcal{H}$$ có thể được chặn:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}}(D_S, D_T) + \lambda$$

trong đó:
- $$\epsilon_S(h)$$: lỗi trên miền nguồn (có thể tối thiểu hóa với dữ liệu nguồn dồi dào)
- $$d_{\mathcal{H}}(D_S, D_T)$$: khoảng cách giữa phân phối nguồn và đích (đo dịch miền)
- $$\lambda$$: lỗi của giả thuyết chung lý tưởng (lỗi tối thiểu có thể trên cả hai miền)

Cận này tiết lộ điều cần cho chuyển giao thành công: (1) lỗi nguồn thấp (tiền huấn luyện tốt), (2) khoảng cách miền nhỏ (nguồn và đích tương tự), (3) $$\lambda$$ nhỏ (tồn tại giả thuyết tối ưu chung). Khi các miền rất khác nhau, $$d_{\mathcal{H}}$$ lớn, và cận trở nên lỏng — không đảm bảo chuyển giao giúp. Điều này hình thức hóa trực giác rằng chuyển giao hoạt động khi các miền chia sẻ cấu trúc.

## 3. Ví dụ / Trực giác

Xem xét kịch bản cụ thể: xây dựng bộ phân loại loài chim với chỉ 500 ảnh gán nhãn trên 20 loài (25 ảnh mỗi loài). Huấn luyện ResNet-50 (25 triệu tham số) từ đầu trên dữ liệu này sẽ overfitting thảm họa — ta có nhiều tham số hơn nhiều so với ví dụ huấn luyện.

Cách tiếp cận học chuyển giao bắt đầu với ResNet-50 tiền huấn luyện trên ImageNet (1.2 triệu ảnh, 1000 lớp). Mạng này đã học:
- **Tầng 1**: Bộ phát hiện cạnh (ngang, dọc, chéo, cong)
- **Tầng 2**: Mẫu texture (lông, mỏ, nền)
- **Tầng 3**: Phần đối tượng (cánh, đầu, chân)
- **Tầng 4**: Cấu trúc đối tượng (toàn bộ chim, dù chuyên loài ImageNet)

Với tác vụ phân loại chim của ta, ta:

**Tùy chọn 1: Trích xuất Đặc trưng**
- Gỡ tầng phân loại cuối (1000 lớp)
- Đóng băng mọi tầng conv (giữ đặc trưng tiền huấn luyện)
- Thêm đầu phân loại mới (20 loài chim)
- Chỉ huấn luyện đầu mới này trên 500 ảnh của ta

Điều này hoạt động vì các tầng đóng băng cung cấp vectơ đặc trưng 2048 chiều phong phú cho mỗi ảnh, nắm cạnh, texture, và phần giống chim. Ta chỉ cần học tổ hợp nào của các đặc trưng này tương ứng loài nào trong 20 loài — bài toán đơn giản hơn nhiều đòi hỏi dữ liệu ít hơn xa.

**Tùy chọn 2: Tinh chỉnh**
- Bắt đầu với trọng số tiền huấn luyện mọi nơi
- Thay tầng cuối bằng đầu 20 lớp (khởi tạo ngẫu nhiên)
- Huấn luyện toàn bộ mạng với learning rate nhỏ (0.0001 so với điển hình 0.1)

Learning rate nhỏ then chốt. Đặc trưng tiền huấn luyện đã tốt; ta muốn thích ứng chúng nhẹ, không phá hủy chúng. Tầng sớm có thể hầu như không đổi (cạnh là phổ quát). Tầng giữa thích ứng nhiều hơn (texture chuyên chim). Tầng sâu thay đổi nhiều nhất (đặc trưng loài cụ thể của ta).

**Ví dụ số cụ thể**: Giả sử một bộ lọc conv tiền huấn luyện ở tầng 3 có trọng số phát hiện "cấu trúc cong" (hữu ích cho bất kỳ đối tượng nào có đường cong). Với loài chim, ta có thể muốn phát hiện "đường cong lông" cụ thể. Tinh chỉnh điều chỉnh nhẹ trọng số bộ lọc này:

Trọng số gốc: $$w_{\text{pre}} = 0.523$$  
Gradient trên dữ liệu chim: $$\nabla w = 0.015$$ (chỉ ra điều chỉnh nhỏ cần)  
Trọng số cập nhật: $$w_{\text{fine}} = 0.523 - 0.0001 \times 0.015 = 0.5229985$$

Thay đổi nhỏ (learning rate 0.0001) thích ứng đặc trưng nhẹ mà không phá hủy cấu trúc hữu ích học từ ImageNet. Trên hàng nghìn trọng số, các thích ứng nhỏ này tích lũy để chuyên hóa mạng cho chim trong khi bảo toàn hiểu biết thị giác tổng quát.

Kết quả: Với trích xuất đặc trưng, ta có thể đạt 85% độ chính xác phân loại chim. Với tinh chỉnh, 92% độ chính xác. Huấn luyện từ đầu với 500 ảnh của ta: có lẽ 60% độ chính xác (overfitting nghiêm trọng). Lợi thế học chuyển giao kịch tính và thực tiễn.
