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

Vì sao một mô hình học nhận diện chó, mèo và xe tải trên ImageNet lại có thể giúp phát hiện khối u trên phim X-quang phổi chính xác hơn hẳn một mô hình huấn luyện trực tiếp trên X-quang từ đầu? Câu trả lời không phải là phép màu, mà nằm trong **Lý thuyết Thích ứng Miền (Domain Adaptation Theory)** của Ben-David và cộng sự: sai số trên miền đích bị chặn trên bởi ba đại lượng đo được rất cụ thể.

## Bản đồ kiến thức nền: Cơ sở lý thuyết của học chuyển giao và thích ứng miền

- **Định lý cận tổng quát hóa miền đích (Ben-David Bound):** Ràng buộc sai số mục tiêu $$\epsilon_T(h) \le \epsilon_S(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$, tách rõ sai số nguồn, khoảng cách phân phối miền và sai số lý tưởng chung.
- **Tốc độ học phân tầng (Discriminative Learning Rates):** Vì gradient biến thiên mạnh ở tầng đầu phân loại mới khởi tạo ngẫu nhiên, ta gán $$\eta_{\text{head}} \gg \eta_{\text{backbone}}$$ để không phá hủy biểu diễn tiền huấn luyện (tránh *catastrophic forgetting*).
- **Hành vi thống kê của Batch Normalization khi chuyển miền:** Phân biệt rõ chế độ `eval()` (giữ trung bình/phương sai nguồn) và `train()` (cập nhật thống kê miền đích) khi tinh chỉnh trên batch nhỏ.
- **Tách biệt Covariate Shift và Concept Shift:** Nhận biết miền đích khác miền nguồn ở phân phối đầu vào $$P(X)$$ hay ở chính quy tắc gán nhãn $$P(Y\mid X)$$.

### Khung giải thích (nên đọc chậm): Định lượng cận sai số miền đích theo bất đẳng thức Ben-David

Giả sử ta chuyển giao một bộ phân loại ảnh y tế từ bệnh viện A (miền nguồn $$S$$) sang bệnh viện B (miền đích $$T$$) dùng máy chụp đời khác:
- Sai số trên miền nguồn: $$\epsilon_S(h) = 0{,}04$$ ($$4\%$$).
- Khoảng cách phân kỳ hai miền: $$d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) = 0{,}16$$.
- Sai số kết hợp tối ưu không thể tránh khỏi: $$\lambda^* = 0{,}03$$.

1. **Tính cận trên sai số miền đích trước khi thích ứng miền:**
   $$\epsilon_T(h) \le 0{,}04 + \frac{1}{2}(0{,}16) + 0{,}03 = 0{,}04 + 0{,}08 + 0{,}03 = 0{,}15\;(15\%).$$
2. **Áp dụng chuẩn hóa độ tương phản và trích xuất đặc trưng bất biến miền** để giảm $$d_{\mathcal{H}\Delta\mathcal{H}}$$ từ $$0{,}16$$ xuống $$0{,}04$$ (giữ nguyên $$\epsilon_S = 0{,}04, \lambda^* = 0{,}03$$):
   $$\epsilon_T(h) \le 0{,}04 + \frac{1}{2}(0{,}04) + 0{,}03 = 0{,}09\;(9\%).$$
   Con số này cho thấy bài học chiến lược: khi $$\epsilon_S$$ đã rất thấp, cố gắng ép loss trên miền nguồn giảm thêm gần như vô ích; muốn thắng trên miền đích phải tập trung kéo giảm **khoảng cách miền** $$d_{\mathcal{H}\Delta\mathcal{H}}$$.

# Học Chuyển giao: Tận dụng tri thức tiền huấn luyện

## 1. Tổng quan khái niệm

Học chuyển giao (*transfer learning*) là một trong những phương pháp thực tiễn quan trọng nhất của học sâu hiện đại, cho phép xây dựng mô hình có hiệu năng cao với lượng dữ liệu nhãn hạn chế bằng cách kế thừa tri thức học được từ một bài toán liên quan. Nguyên lý cốt lõi khá trực tiếp: thay vì khởi tạo trọng số ngẫu nhiên và huấn luyện từ đầu, ta xuất phát từ bộ trọng số đã được tiền huấn luyện trên một tập dữ liệu lớn, sau đó điều chỉnh cho bài toán mục tiêu. Nhờ vậy, học sâu trở nên khả thi ngay cả khi không có tập dữ liệu lớn hay nguồn lực tính toán dồi dào: bài toán ảnh y khoa có thể khởi đầu từ mạng tiền huấn luyện trên ImageNet, bài toán phân tích cảm xúc có thể tinh chỉnh từ BERT tiền huấn luyện trên kho văn bản web, và hệ thống nhận dạng giọng nói có thể tinh chỉnh từ Wav2Vec học trên dữ liệu âm thanh không nhãn.

Học chuyển giao phát huy hiệu quả nhờ cấu trúc biểu diễn phân cấp của mạng nơ-ron sâu. Các tầng sớm học đặc trưng mức thấp mang tính tổng quát — biên cạnh, kết cấu bề mặt và hình khối cơ bản đối với ảnh; âm vị cơ bản đối với âm thanh; các mẫu từ phổ biến đối với văn bản. Những đặc trưng này khá tương đồng giữa nhiều bài toán khác nhau: mạng phân loại xe hơi và xe tải học bộ lọc phát hiện cạnh tương tự mạng phân loại chó và mèo. Các tầng giữa học đặc trưng mức trung như bộ phận đối tượng hoặc tổ hợp kết cấu, vẫn có tính tái sử dụng cao. Chỉ những tầng sâu nhất mới chuyên biệt mạnh cho nhãn cụ thể của bài toán nguồn.

Nhờ tính tái sử dụng này, các tầng sớm và tầng giữa đã được tiền huấn luyện tạo thành điểm khởi đầu vững chắc cho bài toán đích. Ngay cả khi miền đích khác miền nguồn (chẳng hạn ảnh y khoa so với ảnh tự nhiên trong ImageNet), các đặc trưng thị giác cơ bản như biên cạnh và kết cấu vẫn còn giá trị, giúp mô hình tập trung dữ liệu đích hạn chế vào việc học các đặc trưng chuyên biệt ở tầng sâu hơn.

Về mặt thực tế, học chuyển giao giúp giảm mạnh nhu cầu dữ liệu gán nhãn. Trong thị giác máy tính, nhiều bài toán phân loại ảnh chỉ cần vài trăm đến vài nghìn ảnh khi tận dụng mô hình tiền huấn luyện ImageNet thay vì hàng trăm nghìn ảnh nếu huấn luyện từ đầu. Trong xử lý ngôn ngữ tự nhiên, các mô hình như BERT tiền huấn luyện trên kho văn bản lớn có thể tinh chỉnh hiệu quả cho phân tích cảm xúc, nhận diện thực thể hay hỏi đáp chỉ với vài nghìn ví dụ gán nhãn, mở rộng phạm vi ứng dụng sang các lĩnh vực khó thu thập nhãn như chẩn đoán y khoa hay tài liệu kỹ thuật chuyên ngành.

Tuy nhiên, học chuyển giao không phải lúc nào cũng hiệu quả. Phương pháp này dựa trên giả định miền nguồn và miền đích chia sẻ một phần cấu trúc dữ liệu. Đặc trưng học từ ImageNet thường chuyển giao kém hơn sang dữ liệu có tính chất cảm biến hoặc phân phối khác biệt mạnh như ảnh radar, ảnh vệ tinh đa phổ hoặc đồ họa trừu tượng. Hai miền càng gần nhau thì hiệu quả chuyển giao càng cao — vì thế, nếu có sẵn mô hình tiền huấn luyện trên miền gần hơn (như ảnh X-quang ngực cho bài toán y khoa), nó thường ưu thế hơn mô hình tổng quát từ ImageNet.

## 2. Nền tảng toán học

Cơ sở toán học của học chuyển giao liên hệ trực tiếp với thích ứng miền (*domain adaptation*), học đa tác vụ (*multi-task learning*) và học cách học (*meta-learning*).

Giả sử ta có miền nguồn với phân phối $$p_S(\mathbf{x}, y)$$ và tập dữ liệu gán nhãn lớn $$\mathcal{D}_S = \{(\mathbf{x}_i^S, y_i^S)\}_{i=1}^{N_S}$$, cùng miền đích với phân phối $$p_T(\mathbf{x}, y)$$ và tập dữ liệu gán nhãn nhỏ $$\mathcal{D}_T = \{(\mathbf{x}_j^T, y_j^T)\}_{j=1}^{N_T}$$ sao cho $$N_T \ll N_S$$. Mục tiêu là học bộ dự đoán $$f_\theta(\mathbf{x})$$ đạt hiệu năng tốt trên miền đích.

Nếu chỉ tối thiểu hóa rủi ro thực nghiệm trực tiếp trên dữ liệu đích:

$$\theta^* = \arg\min_\theta \frac{1}{N_T}\sum_{j=1}^{N_T} \mathcal{L}(f_\theta(\mathbf{x}_j^T), y_j^T)$$

khi $$N_T$$ nhỏ, mô hình rất dễ quá khớp và tổng quát hóa kém.

Học chuyển giao chia quá trình tối ưu thành hai giai đoạn:

**Giai đoạn 1 (Tiền huấn luyện):** tối ưu trên miền nguồn

$$\theta_S^* = \arg\min_\theta \frac{1}{N_S}\sum_{i=1}^{N_S} \mathcal{L}(f_\theta(\mathbf{x}_i^S), y_i^S)$$

**Giai đoạn 2 (Tinh chỉnh):** khởi tạo từ $$\theta_S^*$$ rồi tối ưu trên miền đích

$$\theta_T^* = \arg\min_\theta \frac{1}{N_T}\sum_{j=1}^{N_T} \mathcal{L}(f_\theta(\mathbf{x}_j^T), y_j^T), \quad \text{bắt đầu từ } \theta_0 = \theta_S^*$$

Điểm khởi tạo $$\theta_0 = \theta_S^*$$ đặt quá trình tối ưu gần vùng nghiệm tốt hơn so với khởi tạo ngẫu nhiên, giúp hội tụ nhanh và ít quá khớp hơn khi dữ liệu đích hạn chế.

Phân rã mô hình thành $$f_\theta = h_{\theta_h} \circ g_{\theta_g}$$, trong đó $$g_{\theta_g}$$ là phần trích xuất đặc trưng (các tầng sớm và giữa) và $$h_{\theta_h}$$ là tầng đầu ra chuyên biệt cho tác vụ. Ba chiến lược chuyển giao phổ biến là:

**Trích xuất đặc trưng (*feature extraction*):** giữ cố định (đóng băng) $$\theta_g = \theta_g^S$$ và chỉ huấn luyện $$\theta_h$$ trên dữ liệu đích:

$$\theta_h^* = \arg\min_{\theta_h} \frac{1}{N_T}\sum_{j=1}^{N_T} \mathcal{L}(h_{\theta_h}(g_{\theta_g^S}(\mathbf{x}_j^T)), y_j^T)$$

**Tinh chỉnh toàn bộ tầng (*full fine-tuning*):** khởi tạo $$(\theta_g^S, \theta_h^{\text{random}})$$ và cập nhật đồng thời cả hai phần trên dữ liệu đích:

$$(\theta_g^*, \theta_h^*) = \arg\min_{\theta_g, \theta_h} \frac{1}{N_T}\sum_{j=1}^{N_T} \mathcal{L}(h_{\theta_h}(g_{\theta_g}(\mathbf{x}_j^T)), y_j^T)$$

**Learning rate phân biệt theo tầng (*discriminative learning rates*):** áp dụng mức learning rate khác nhau cho từng nhóm tầng:

$$\theta_g \leftarrow \theta_g - \eta_g \nabla_{\theta_g} \mathcal{L}, \quad \theta_h \leftarrow \theta_h - \eta_h \nabla_{\theta_h} \mathcal{L}$$

thường chọn $$\eta_g < \eta_h$$ để cập nhật nhẹ các tầng tiền huấn luyện và cập nhật nhanh hơn cho tầng đầu ra mới.

Lựa chọn chiến lược phụ thuộc vào cỡ mẫu đích và mức độ tương đồng giữa hai miền. Với dữ liệu đích rất nhỏ (vài trăm mẫu) và miền tương đồng, trích xuất đặc trưng giúp hạn chế quá khớp; với dữ liệu cỡ trung bình (vài nghìn mẫu), tinh chỉnh với learning rate nhỏ giúp đặc trưng thích nghi tốt hơn mà tránh hiện tượng quên thảm họa (*catastrophic forgetting*); còn khi dữ liệu đích rất lớn (hàng chục nghìn mẫu trở lên), tinh chỉnh toàn mạng hoặc thậm chí huấn luyện từ đầu đều là phương án khả thi.

### Lý thuyết thích ứng miền

Trong lý thuyết thích ứng miền, với không gian giả thuyết $$\mathcal{H}$$, sai số trên miền đích của một giả thuyết $$h \in \mathcal{H}$$ được chặn trên bởi:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}}(D_S, D_T) + \lambda$$

trong đó:
- $$\epsilon_S(h)$$ là sai số trên miền nguồn (có thể làm nhỏ nhờ tập nguồn lớn).
- $$d_{\mathcal{H}}(D_S, D_T)$$ là khoảng cách phân phối giữa miền nguồn và miền đích (đo mức độ lệch miền).
- $$\lambda$$ là sai số của giả thuyết chung tối ưu trên cả hai miền.

Cận trên này cho thấy ba điều kiện để chuyển giao hiệu quả: sai số trên miền nguồn thấp, độ lệch giữa hai miền nhỏ và tồn tại một giả thuyết hoạt động tốt trên cả hai miền. Khi $$d_{\mathcal{H}}$$ lớn do hai miền quá khác biệt, cận trên trở nên lỏng và học chuyển giao không còn đảm bảo mang lại lợi ích.

## 3. Ví dụ và Trực giác

Xét bài toán phân loại 20 loài chim với chỉ 500 ảnh gán nhãn (25 ảnh mỗi loài). Huấn luyện một mạng ResNet-50 (khoảng 25 triệu tham số) từ đầu trên 500 ảnh chắc chắn sẽ quá khớp nghiêm trọng.

Thay vào đó, ta xuất phát từ ResNet-50 tiền huấn luyện trên ImageNet (1,2 triệu ảnh, 1000 lớp), vốn đã học sẵn hệ thống đặc trưng phân cấp:
- **Tầng 1:** bộ lọc phát hiện cạnh (ngang, dọc, chéo, cong).
- **Tầng 2:** kết cấu bề mặt (lông, mỏ, nền).
- **Tầng 3:** bộ phận đối tượng (cánh, đầu, chân).
- **Tầng 4:** cấu trúc đối tượng tổng thể (chim và các lớp ImageNet).

Ta có hai phương án chính:

**Phương án 1 — Trích xuất đặc trưng:** bỏ tầng phân loại 1000 lớp cuối, đóng băng toàn bộ các tầng tích chập, gắn tầng phân loại 20 lớp mới và chỉ huấn luyện tầng mới này trên 500 ảnh. Mỗi ảnh được biểu diễn bằng vector đặc trưng 2048 chiều đã học sẵn, giúp bài toán quy về học ánh xạ phân loại đơn giản cần ít dữ liệu hơn nhiều.

**Phương án 2 — Tinh chỉnh:** thay tầng cuối bằng đầu 20 lớp khởi tạo ngẫu nhiên và huấn luyện toàn bộ mạng với learning rate nhỏ (ví dụ 0,0001 thay vì 0,1 như khi huấn luyện từ đầu). Bước học nhỏ giúp các tầng sớm hầu như giữ nguyên đặc trưng cạnh tổng quát, trong khi các tầng giữa và tầng sâu điều chỉnh dần để bám sát đặc trưng của 20 loài chim.

Minh họa một bước cập nhật trọng số cụ thể ở tầng 3:

Trọng số tiền huấn luyện: $$w_{\text{pre}} = 0.523$$  
Gradient trên dữ liệu chim: $$\nabla w = 0.015$$  
Trọng số sau cập nhật: $$w_{\text{fine}} = 0.523 - 0.0001 \times 0.015 = 0.5229985$$

Những điều chỉnh rất nhỏ nhưng đồng loạt trên hàng nghìn trọng số giúp mạng chuyên biệt hóa dần cho bài toán mới mà không phá vỡ cấu trúc đặc trưng đã học từ ImageNet. Trong kịch bản minh họa 500 ảnh này, trích xuất đặc trưng có thể đạt quanh 85% độ chính xác và tinh chỉnh đạt quanh 92%, vượt xa mức khoảng 60% khi huấn luyện từ đầu.

## Video tham khảo

- **Xem trước:** [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, tiếng Anh. Giải thích trực tiếp câu hỏi trọng tâm của bài: nên đóng băng những tầng nào và tinh chỉnh những tầng nào.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, tiếng Anh. Làm rõ vì sao đặc trưng ở các tầng tích chập sớm dễ chuyển giao hơn các tầng sâu.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trực quan phân cấp đặc trưng trong CNN để xác định điểm cắt khi chuyển giao.
