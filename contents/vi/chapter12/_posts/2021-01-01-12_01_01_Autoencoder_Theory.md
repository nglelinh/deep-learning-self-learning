---
layout: post
title: 12-01-01 Lý thuyết Autoencoder
chapter: '12'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter12
---

# Autoencoder: Học Biểu diễn Hiệu quả

![Autoencoder Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Autoencoder_structure.png/600px-Autoencoder_structure.png)
*Hình ảnh: Kiến trúc Autoencoder với encoder và decoder. Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm
Autoencoder đại diện cho một mô hình huấn luyện mạng neuron khác biệt căn bản so với học giám sát mà ta đã nghiên cứu. Thay vì học ánh xạ đầu vào sang đầu ra có nhãn, autoencoder học tái tạo đầu vào của chính chúng qua một nút thắt thông tin (information bottleneck). Tác vụ tưởng chừng vòng tròn này — dự đoán đầu vào từ chính nó — trở nên có nghĩa khi ta ràng buộc mạng phải truyền thông tin qua một tầng ẩn chiều thấp hơn gọi là không gian ẩn (latent space) hay mã (code). Bằng cách buộc mạng nén rồi giải nén đầu vào, ta buộc nó học biểu diễn hiệu quả nắm cấu trúc thiết yếu của dữ liệu trong khi loại bỏ nhiễu và chi tiết không liên quan.

Sức mạnh của autoencoder không nằm ở chính tái tạo mà ở điều chúng học trong quá trình đó. Encoder học trích xuất các đặc trưng quan trọng nhất từ dữ liệu chiều cao và nén chúng thành biểu diễn gọn. Decoder học sinh dữ liệu thực tế từ các biểu diễn nén này. Không gian ẩn nổi lên có các tính chất đáng chú ý: các điểm lân cận trong không gian ẩn thường tương ứng đầu vào tương tự về ngữ nghĩa, và ta có thể nội suy mượt giữa các điểm để sinh ví dụ mới, thực tế. Các tính chất này khiến autoencoder có giá trị cho giảm chiều, khử nhiễu, phát hiện bất thường, học đặc trưng cho tác vụ downstream, và như khối xây dựng cho mô hình sinh tinh vi hơn.

Hiểu autoencoder đòi hỏi đánh giá sự tương tác giữa dung lượng và ràng buộc. Nếu chiều ẩn bằng hoặc vượt chiều đầu vào, mạng có thể đơn giản học hàm đồng nhất, sao chép đầu vào không đổi — vô dụng cho việc học cấu trúc có nghĩa. Nút thắt — làm chiều ẩn nhỏ hơn đầu vào — buộc mạng chọn lựa điều gì cần bảo toàn. Với ảnh đầu vào 784 chiều nén xuống 32 chiều ẩn, mạng không thể mã hóa mỗi pixel độc lập. Nó phải khám phá đặc trưng mức cao hơn như cạnh, hình dạng và texture biểu diễn gọn nội dung thiết yếu của ảnh. Việc nén này không tùy ý mà được học từ dữ liệu, thích ứng với cấu trúc cụ thể trong phân phối huấn luyện.

Ý nghĩa lịch sử của autoencoder mở rộng vượt ngoài ứng dụng thực tiễn. Chúng nằm trong số các phương pháp học không giám sát thành công đầu tiên trong deep learning, chứng tỏ mạng neuron có thể học biểu diễn có nghĩa không cần dữ liệu gán nhãn. Điều này ảnh hưởng sự phát triển các chiến lược tiền huấn luyện cho phép huấn luyện mạng sâu hơn ở kỷ nguyên trước ReLU. Học tự giám sát và phương pháp đối sánh (contrastive) hiện đại có thể được coi là hậu duệ của ý tưởng autoencoder — học biểu diễn bằng cách dự đoán hoặc tái tạo các phần của đầu vào từ các phần khác. Khung autoencoder cũng giới thiệu mẫu kiến trúc encoder–decoder đã chứng tỏ ảnh hưởng to lớn, xuất hiện trong mô hình sequence-to-sequence, variational autoencoder, và mạng sinh đối kháng.

Tuy nhiên, autoencoder có những hạn chế quan trọng thúc đẩy các mô hình sinh tinh vi hơn. Autoencoder chuẩn học nén và tái tạo dữ liệu huấn luyện nhưng không nhất thiết học một mô hình sinh tốt — không gian ẩn có thể có "lỗ hổng" nơi không có ví dụ huấn luyện nào ánh xạ tới, khiến lấy mẫu ngẫu nhiên tạo đầu ra phi thực tế. Chúng không mô hình hóa tường minh phân phối dữ liệu, hạn chế đảm bảo lý thuyết. Và mất mát tái tạo, dù trực quan, có thể không nắm tương tự cảm nhận (hai ảnh có thể khác theo pixel nhưng tương tự cảm nhận, hoặc tương tự theo pixel nhưng khác cảm nhận). Các hạn chế này dẫn đến variational autoencoder (mô hình hóa phân phối tường minh), mạng sinh đối kháng (dùng huấn luyện đối kháng thay vì mất mát tái tạo), và mất mát cảm nhận (đo tương tự trong không gian đặc trưng thay vì không gian pixel). Hiểu autoencoder vanilla cung cấp nền tảng để đánh giá các kỹ thuật nâng cao hơn này.

## 2. Nền tảng toán học
Khung toán học của autoencoder thanh lịch đơn giản nhưng giàu hệ quả. Một autoencoder gồm hai mạng neuron ghép nối tuần tự: encoder $$f_\phi$$ tham số hóa bởi $$\phi$$ và decoder $$g_\theta$$ tham số hóa bởi $$\theta$$. Cho đầu vào $$\mathbf{x} \in \mathbb{R}^{d}$$, encoder sinh biểu diễn ẩn:

$$\mathbf{z} = f_\phi(\mathbf{x}) \in \mathbb{R}^{k}$$

trong đó $$k < d$$ buộc nút thắt (dù ta sẽ thảo luận các trường hợp không bắt buộc nghiêm ngặt). Decoder tái tạo từ biểu diễn ẩn:

$$\hat{\mathbf{x}} = g_\theta(\mathbf{z}) = g_\theta(f_\phi(\mathbf{x})) \in \mathbb{R}^{d}$$

Mục tiêu huấn luyện tối thiểu hóa lỗi tái tạo:

$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

cho dữ liệu liên tục (mean squared error), hoặc:

$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = -\sum_{i=1}^{d} [x_i \log(\hat{x}_i) + (1-x_i)\log(1-\hat{x}_i)]$$

cho dữ liệu nhị phân (binary cross-entropy, coi mỗi chiều như Bernoulli độc lập).

Lựa chọn hàm mất mát thể hiện giả định về dữ liệu và mô hình nhiễu. MSE giả định nhiễu Gaussian: ta đang mô hình hóa $$p(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; g_\theta(\mathbf{z}), \sigma^2 I)$$, và tối thiểu hóa MSE tương đương hợp lý cực đại dưới giả định này. Binary cross-entropy giả định nhiễu Bernoulli: mỗi pixel độc lập nhị phân với xác suất $$\hat{x}_i$$. Với ảnh có giá trị liên tục trong [0,1], thực chất đang mô hình hóa mỗi pixel như xác suất, trông lạ nhưng hoạt động khá tốt trong thực tiễn. Các cách tiếp cận tinh vi hơn dùng mất mát cảm nhận dựa trên khoảng cách đặc trưng trong mạng tiền huấn luyện, nắm tốt hơn tương tự cảm nhận.

Chiều nút thắt $$k$$ là siêu tham số then chốt điều khiển đánh đổi nén–độ trung thực. $$k$$ rất nhỏ (như 2–3 chiều) tạo nén cực đoan, buộc mạng chỉ nắm các biến thiên thiết yếu nhất trong dữ liệu. Hữu ích cho trực quan hóa (ta có thể vẽ không gian ẩn 2D) nhưng có thể mất chi tiết quan trọng. $$k$$ vừa phải (32–128 chiều cho tập dữ liệu ảnh) cân bằng nén và chất lượng tái tạo. $$k$$ lớn (tiệm cận chiều đầu vào) giảm áp lực nén nhưng có thể không học cấu trúc thú vị.

Thú vị là, ngay cả với $$k \geq d$$ (không có nút thắt chiều), ta vẫn có thể buộc học có nghĩa qua các ràng buộc khác. Autoencoder thưa thêm phạt thưa lên các kích hoạt ẩn:

$$\mathcal{L}_{\text{sparse}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \sum_{j=1}^{k} KL(\rho \| \hat{\rho}_j)$$

trong đó $$\rho$$ là mức thưa mục tiêu (ví dụ 0.05) và $$\hat{\rho}_j$$ là kích hoạt trung bình của đơn vị ẩn $$j$$ trên tập huấn luyện. Phạt phân kỳ KL khuyến khích hầu hết đơn vị ẩn không hoạt động (gần zero) hầu hết thời gian, buộc các đơn vị khác nhau chuyên môn hóa vào các mẫu hình khác nhau. Điều này tạo biểu diễn thưa, phân tán ngay cả không có nút thắt chiều.

Autoencoder khử nhiễu làm hỏng đầu vào $$\mathbf{x}$$ bằng nhiễu để tạo $$\tilde{\mathbf{x}}$$ nhưng huấn luyện để tái tạo bản gốc:

$$\mathcal{L} = \|\mathbf{x} - g_\theta(f_\phi(\tilde{\mathbf{x}}))\|^2$$

Quá trình làm hỏng có thể thêm nhiễu Gaussian, che pixel ngẫu nhiên, hoặc thêm nhiễu muối-tiêu. Điều này buộc encoder học đặc trưng robust bất biến với loại nhiễu, và decoder học "điền" vùng hỏng dựa trên ngữ cảnh chưa hỏng. Autoencoder khử nhiễu thường học đặc trưng tốt hơn autoencoder vanilla vì tác vụ khử nhiễu đòi hỏi hiểu cấu trúc dữ liệu, không chỉ ghi nhớ ví dụ huấn luyện.

Hình học không gian ẩn xứng đáng phân tích cẩn thận. Trong autoencoder huấn luyện tốt trên dữ liệu ảnh, các điểm lân cận trong không gian ẩn thường tương ứng ảnh tương tự về cảm nhận. Ta có thể nội suy tuyến tính giữa hai mã ẩn $$\mathbf{z}_1$$ và $$\mathbf{z}_2$$:

$$\mathbf{z}_t = (1-t)\mathbf{z}_1 + t\mathbf{z}_2, \quad t \in [0,1]$$

và giải mã $$g_\theta(\mathbf{z}_t)$$ để sinh ảnh trung gian. Với autoencoder hành xử tốt, điều này tạo chuyển tiếp mượt (biến hình khuôn mặt này sang khuôn mặt khác, ví dụ). Tuy nhiên, autoencoder chuẩn không đảm bảo nội suy tốt — có thể có "lỗ hổng" trong không gian ẩn nơi không có ví dụ huấn luyện nào ánh xạ, và nội suy qua các lỗ hổng này tạo tái tạo phi thực tế. Variational autoencoder xử lý điều này bằng cách chính quy hóa tường minh không gian ẩn để liên tục và hành xử tốt.

## 3. Ví dụ / Trực giác

Để xây dựng trực giác cụ thể về cách autoencoder học biểu diễn, hãy đi qua huấn luyện trên chữ số MNIST. Giả sử ta nén ảnh 28×28=784 pixel xuống mã ẩn 32 chiều.

Ban đầu, với trọng số ngẫu nhiên, encoder sinh mã ẩn vô nghĩa và decoder tạo nhiễu ngẫu nhiên như tái tạo. Lỗi tái tạo khổng lồ — ta cố khớp 784 giá trị pixel nhưng nhận đầu ra về cơ bản ngẫu nhiên. Gradient qua backpropagation chỉ ra cách điều chỉnh trọng số encoder và decoder để giảm lỗi này.

Khi huấn luyện tiến triển, encoder học trích xuất đặc trưng ngày càng có nghĩa. Sớm, nó có thể học rằng một số pixel có xu hướng tối (nền) so với sáng (nét chữ số), mã hóa điều này thành các chiều ẩn biểu diễn độ sáng trung bình ở các vùng khác nhau. Mã hóa nguyên thủy này đã cho phép tái tạo tốt hơn nhiễu ngẫu nhiên — decoder học sinh ảnh với mẫu hình độ sáng tổng thể phù hợp.

Với thêm huấn luyện, encoder khám phá mẫu hình cạnh. Một số chiều ẩn trở nên hoạt động khi chữ số có nét dọc (1, 4, 7), chiều khác cho đường cong (0, 6, 8, 9), chiều khác cho đoạn ngang (2, 3, 5, 7). Decoder học tái tạo ảnh giống chữ số từ các chỉ báo cạnh này. Tái tạo nay nắm hình dạng tổng quát của chữ số, dù chi tiết có thể mờ.

Cuối cùng, 32 chiều ẩn tự tổ chức thành không gian biểu diễn có nghĩa. Các chiều có thể mã hóa: danh tính chữ số (đại khái chữ số nào), độ dày nét, độ nghiêng, kích thước, vị trí trong ảnh. Biểu diễn đã học này nổi lên thuần từ mục tiêu tái tạo — ta không bao giờ bảo mạng học đặc trưng gì, chỉ nén và tái tạo chính xác.

Xem xét điều gì xảy ra khi ta mã hóa một số "3" từ tập huấn luyện. Mã ẩn của chúng cụm lại trong không gian ẩn 32D vì chúng chia sẻ cấu trúc (cạnh, đường cong, topology tương tự). Các "3" khác nhau (dày, mỏng, nghiêng) ánh xạ tới các điểm ẩn hơi khác nhưng lân cận. Trong khi đó, các "8" cụm ở vùng khác của không gian ẩn — chúng chia sẻ cấu trúc topology (hai vòng) mà "3" thiếu. Không gian ẩn đã tự tổ chức để phản ánh các danh mục chữ số và biến thể trong danh mục, tất cả không cần nhãn.

Bây giờ cho kiểm tra nội suy. Mã hóa một "3" để được $$\mathbf{z}_3$$ và mã hóa một "8" để được $$\mathbf{z}_8$$. Giải mã các điểm trung gian:

$$\mathbf{z}_{0.0} = \mathbf{z}_3 \to$$ giải mã thành "3"  
$$\mathbf{z}_{0.25} = 0.75\mathbf{z}_3 + 0.25\mathbf{z}_8 \to$$ giải mã thành "3 với gợi ý của 8"  
$$\mathbf{z}_{0.5} = 0.5\mathbf{z}_3 + 0.5\mathbf{z}_8 \to$$ giải mã thành chữ số mơ hồ  
$$\mathbf{z}_{0.75} = 0.25\mathbf{z}_3 + 0.75\mathbf{z}_8 \to$$ giải mã thành "8 với gợi ý của 3"  
$$\mathbf{z}_{1.0} = \mathbf{z}_8 \to$$ giải mã thành "8"

Nếu nội suy mượt, ta thấy biến hình dần. Nếu có gián đoạn, ta có thể nhận đầu ra phi thực tế tại các điểm trung gian. Chất lượng nội suy này là chỉ báo chẩn đoán liệu không gian ẩn có cấu trúc tốt hay không.

Autoencoder khử nhiễu thêm một bước ngoặt thú vị. Giả sử ta làm hỏng một "7" bằng cách ngẫu nhiên đặt zero 20% pixel. Ảnh hỏng mơ hồ — có thể là "7" hỏng hoặc có thể là "1". Autoencoder khử nhiễu phải dùng ngữ cảnh (pixel chưa hỏng) để suy ra chữ số gốc khả dĩ nhất. Điều này đòi hỏi hiểu cấu trúc chữ số, không chỉ ghi nhớ mẫu pixel. Encoder học trích xuất đặc trưng robust dù nhiễu, và decoder học sinh chữ số hoàn chỉnh từ bằng chứng một phần. Biểu diễn học được thường hữu ích hơn cho tác vụ downstream so với autoencoder vanilla vì chúng bị buộc nắm cấu trúc ngữ nghĩa thay vì thống kê pixel mức thấp.
