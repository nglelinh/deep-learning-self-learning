---
layout: post
title: 11-01-01 Lý thuyết Mô hình Sinh
chapter: '11'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter11
---

Đằng sau mỗi bức ảnh khuôn mặt do AI tạo ra không hề có một chiếc máy ảnh nào — chỉ có một hàm mật độ xác suất trong không gian hàng trăm nghìn chiều đang thì thầm rằng "tổ hợp điểm ảnh này rất hợp lý". Nhưng làm thế nào để ta viết ra, tính toán và tối ưu được một hàm mật độ phức tạp như vậy mà không rơi vào bế tắc tích phân? Bài lý thuyết này mở khóa tấm bản đồ toán học phân tách mọi mô hình sinh thành hai trường phái lớn: mật độ tường minh và mật độ ẩn.

## Kiến thức cần nhớ: Nền tảng toán học phân loại các họ mô hình sinh

- **Quy tắc chuỗi và mô hình tự hồi quy:** Dùng đồng nhất thức $$p(\mathbf{x}) = \prod_{i=1}^d p(x_i \mid \mathbf{x}_{<i})$$ để biến mật độ đồng thời phức tạp thành chuỗi bài toán phân loại/hồi quy có điều kiện tính được chính xác.
- **Định lý đổi biến và chuẩn hóa luồng (Normalizing Flows):** Khai thác phép biến đổi khả nghịch $$\mathbf{x} = f(\mathbf{z})$$ cùng định thức ma trận Jacobian $$|\det J_{f^{-1}}(\mathbf{x})|$$ để vừa lấy mẫu nhanh vừa tính mật độ chính xác.
- **Suy diễn biến phân và cận dưới ELBO:** Khi tích phân biên $$\int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$$ bất khả tính, đưa vào phân phối xấp xỉ $$q_\phi(\mathbf{z}|\mathbf{x})$$ để tối ưu cận dưới khả tính.
- **Khoảng cách giữa hai phân phối (KL, FID, IS):** Liên kết lý thuyết thông tin (phân kỳ KL, entropy) với hình học xác suất (khoảng cách Fréchet giữa hai phân phối chuẩn) để đánh giá chất lượng sinh.

### Giải thích: Tính tay định thức Jacobian trong mô hình luồng và chỉ số FID trên đặc trưng 1D

**Phần A — Công thức đổi biến (Normalizing Flow 1D):** Giả sử biến ẩn $$z \sim \mathcal{N}(0, 1)$$ có mật độ $$p_z(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$, và phép biến đổi khả nghịch là $$x = f(z) = 2z + 3$$ (nên $$z = f^{-1}(x) = \frac{x - 3}{2}$$).
1. Tính đạo hàm ngược: $$\left|\frac{df^{-1}(x)}{dx}\right| = \frac{1}{2} = 0{,}5$$.
2. Tại điểm $$x = 5$$, ta có $$z = (5-3)/2 = 1$$ và $$p_z(1) = \frac{1}{\sqrt{2\pi}}e^{-0{,}5} \approx 0{,}2420$$.
3. Mật độ tại $$x = 5$$ là $$p_x(5) = 0{,}2420 \times 0{,}5 = 0{,}1210$$. Hệ số Jacobian $$0{,}5$$ chính là yếu tố bảo toàn tổng xác suất bằng $$1$$ khi phép biến đổi kéo giãn trục số gấp đôi.

**Phần B — Tính tay FID trên đặc trưng 1D:** Giả sử đặc trưng Inception 1 chiều của ảnh thật có trung bình $$\mu_r = 2{,}0$$, phương sai $$\sigma_r^2 = 4{,}0$$; ảnh sinh có $$\mu_g = 2{,}5$$, phương sai $$\sigma_g^2 = 1{,}0$$.
$$\text{FID} = (\mu_r - \mu_g)^2 + \left(\sigma_r^2 + \sigma_g^2 - 2\sqrt{\sigma_r^2 \sigma_g^2}\right) = (2{,}0 - 2{,}5)^2 + (4{,}0 + 1{,}0 - 2\sqrt{4{,}0 \cdot 1{,}0}) = 0{,}25 + (5 - 4) = 1{,}25.$$
Quan sát: Dù trung bình chỉ lệch $$0{,}5$$ (đóng góp $$0{,}25$$), việc phương sai bị thu hẹp từ $$4{,}0$$ xuống $$1{,}0$$ (dấu hiệu thiếu đa dạng / sụp đổ mode) làm FID vọt thêm $$1{,}00$$ đơn vị.

# Mô hình Sinh: Học cách tạo ra dữ liệu

## 1. Tổng quan khái niệm

Mô hình sinh (*generative models*) thể hiện một hướng tiếp cận khác căn bản so với mô hình phân biệt (*discriminative models*). Trong khi mô hình phân biệt học ánh xạ từ đầu vào sang đầu ra — chẳng hạn phân loại ảnh, dịch câu giữa hai ngôn ngữ hay dự báo chuỗi thời gian — mô hình sinh hướng tới việc nắm bắt và tái tạo cấu trúc xác suất của chính dữ liệu. Bài toán đặt ra tham vọng hơn: từ tập mẫu quan sát được, liệu mô hình có thể sinh ra các mẫu mới hợp lệ tuân theo cùng phân phối dữ liệu hay không? Khả năng này cho phép tổng hợp ảnh khuôn mặt chưa từng tồn tại, sáng tác nhạc theo phong cách cho trước, thiết kế cấu trúc phân tử đáp ứng tính chất mong muốn hoặc bổ sung dữ liệu tổng hợp cho các tập dữ liệu nhỏ.

Sự khác biệt cốt lõi nằm ở mức độ hiểu cấu trúc dữ liệu. Một bộ phân loại giống chó chỉ cần học các đặc trưng đủ để tách biệt các lớp — như hình dáng tai, họa tiết lông hay kích thước — mà không cần biết các bộ phận đó ghép lại thành một cơ thể hợp lý như thế nào. Ngược lại, một mô hình sinh phải học cấu trúc đầy đủ hơn: cách các điểm ảnh kết hợp thành kết cấu bề mặt, kết cấu tạo thành bộ phận, các bộ phận ghép thành đối tượng hoàn chỉnh, và tổ hợp nào là hợp lý so với phi thực tế. Chính vì vậy, mô hình sinh thường học được các biểu diễn dữ liệu giàu thông tin hơn, hữu ích ngay cả khi mục tiêu cuối cùng không phải là sinh mẫu.

Cơ sở toán học của mô hình sinh dựa trên lý thuyết xác suất và thống kê. Ta giả định dữ liệu $$\mathbf{x}$$ được lấy mẫu từ một phân phối thật chưa biết $$p_{\text{data}}(\mathbf{x})$$. Mục tiêu là học một phân phối tham số $$p_{\text{model}}(\mathbf{x}; \theta)$$ (với $$\theta$$ là trọng số mạng nơ-ron) sao cho gần với $$p_{\text{data}}$$. Nếu phép xấp xỉ đủ tốt, việc lấy mẫu từ $$p_{\text{model}}$$ sẽ tạo ra các mẫu khó phân biệt với dữ liệu thật. Khung tiếp cận này kết nối trực tiếp mô hình sinh với ước lượng hợp lý cực đại (*maximum likelihood estimation*), suy diễn biến phân (*variational inference*) và lý thuyết thông tin, đồng thời tận dụng khả năng biểu diễn hàm phi tuyến linh hoạt của mạng nơ-ron.

Mỗi họ mô hình sinh mang lại sự đánh đổi riêng giữa chất lượng mẫu, độ ổn định khi huấn luyện, khả năng tính xác suất và chi phí tính toán. Mô hình tự hồi quy (*autoregressive models*) như PixelCNN biểu diễn tường minh $$p(\mathbf{x}) = \prod_i p(x_i | x_{<i})$$, phân rã quá trình sinh thành chuỗi phân phối điều kiện; chúng cho phép tính chính xác hàm hợp lý và huấn luyện ổn định, nhưng tốc độ sinh chậm do phải tạo từng phần tử tuần tự. Variational Autoencoder (VAE) đưa vào biến ẩn $$\mathbf{z}$$ với $$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$$ và tối ưu cận dưới khả tính của log-hợp lý; VAE lấy mẫu nhanh và có cơ sở xác suất chặt chẽ, nhưng ảnh sinh ra thường hơi mờ. Mạng sinh đối kháng (GAN) không mô hình hóa mật độ tường minh mà dùng cơ chế đối kháng để học trực tiếp hàm lấy mẫu từ $$p_{\text{data}}$$; GAN thường tạo ảnh sắc nét hơn nhưng dễ gặp bất ổn khi huấn luyện và hiện tượng sụp đổ mode (*mode collapse*).

Trong thực tế, mô hình sinh được ứng dụng rộng rãi ở nhiều lĩnh vực. Trong thị giác máy tính, chúng hỗ trợ tăng cường dữ liệu, siêu phân giải (*super-resolution* — nâng độ phân giải ảnh), khôi phục vùng ảnh khuyết (*inpainting*) và chuyển đổi phong cách (*style transfer*). Trong xử lý ngôn ngữ tự nhiên, chúng là nền tảng cho sinh văn bản, dịch máy và tạo dữ liệu bổ sung cho ngôn ngữ ít tài nguyên. Trong hóa dược, mô hình sinh hỗ trợ đề xuất cấu trúc phân tử mới; trong phát hiện bất thường, chúng xác định điểm ngoại lai bằng cách đo mức độ phù hợp với phân phối đã học.

Tuy nhiên, mô hình hóa sinh về bản chất khó hơn học phân biệt. Không gian đầu ra có kích thước lớn hơn không gian nhãn theo cấp số mũ (chẳng hạn $$2^{784}$$ cấu hình ảnh nhị phân trên MNIST so với chỉ 10 nhãn lớp). Phân phối học được phải nắm bắt các phụ thuộc phức tạp giữa các chiều dữ liệu, vì các điểm ảnh lân cận luôn có tương quan chặt chẽ và các bộ phận đối tượng phải hợp lý về mặt cấu trúc. Việc đánh giá chất lượng mô hình cũng không đơn giản như tính độ chính xác phân loại. Những thách thức này khiến mô hình hóa sinh tiếp tục là một lĩnh vực nghiên cứu phát triển nhanh.

## 2. Nền tảng toán học

Cơ sở toán học của mô hình sinh dựa trên lý thuyết xác suất, ước lượng hợp lý và lý thuyết thông tin. Phần này trình bày các mục tiêu tối ưu chính và lý do vì sao các giả định khác nhau dẫn đến các họ thuật toán khác nhau.

### Mật độ xác suất và phân phối dữ liệu

Giả sử tập huấn luyện $$\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(m)}\}$$ gồm các mẫu độc lập rút ra từ phân phối chưa biết $$p_{\text{data}}(\mathbf{x})$$. Với dữ liệu ảnh, $$\mathbf{x}$$ có thể có $$28 \times 28 = 784$$ chiều (MNIST) hoặc $$224 \times 224 \times 3 = 150{,}528$$ chiều (ImageNet). Phân phối $$p_{\text{data}}$$ gán mật độ xác suất cao cho các ảnh hợp lệ (chữ số viết tay, ảnh chụp tự nhiên) và mật độ gần bằng 0 cho các cấu hình phi thực tế (nhiễu ngẫu nhiên, cấu trúc sai lệch).

Mục tiêu là tìm tham số $$\theta$$ của mô hình $$p_{\text{model}}(\mathbf{x}; \theta)$$ sao cho gán xác suất cao cho dữ liệu huấn luyện và tổng quát hóa tốt sang dữ liệu mới từ cùng phân phối. Cách tiếp cận chuẩn là ước lượng hợp lý cực đại (MLE):

$$\theta^* = \arg\max_\theta \prod_{i=1}^{m} p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$

Lấy logarit để thuận tiện cho việc tính toán và tránh tràn số dưới (*underflow*):

$$\theta^* = \arg\max_\theta \sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta) = \arg\max_\theta \frac{1}{m}\sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$

Giá trị log-hợp lý trung bình $$\frac{1}{m}\sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$ là ước lượng thực nghiệm của kỳ vọng log-hợp lý theo phân phối dữ liệu:

$$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\text{model}}(\mathbf{x}; \theta)]$$

Cực đại hóa kỳ vọng này tương đương với cực tiểu hóa phân kỳ Kullback–Leibler (KL) từ phân phối dữ liệu tới phân phối mô hình:

$$KL(p_{\text{data}} \| p_{\text{model}}) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\text{data}}(\mathbf{x}) - \log p_{\text{model}}(\mathbf{x}; \theta)]$$

Do $$p_{\text{data}}$$ cố định không phụ thuộc $$\theta$$, tối thiểu hóa phân kỳ KL tương đương với tối đa hóa kỳ vọng log-hợp lý. Kết quả này kết nối trực tiếp MLE với lý thuyết thông tin và cung cấp một thước đo rõ ràng về độ sai lệch giữa mô hình và phân phối thật.

### Mô hình mật độ tường minh và mô hình mật độ ẩn

Với dữ liệu nhiều chiều, khó khăn chính là thiết kế hàm mật độ $$p_{\text{model}}(\mathbf{x}; \theta)$$ sao cho vừa đủ linh hoạt để mô tả phân phối phức tạp, vừa có thể tính toán và tối ưu được (*tractable*).

**Mô hình mật độ tường minh (*explicit density models*)** tham số hóa trực tiếp hàm mật độ $$p_{\text{model}}(\mathbf{x}; \theta)$$:

*Mô hình tự hồi quy* dùng quy tắc chuỗi xác suất để phân rã mật độ đồng thời thành tích các phân phối điều kiện:

$$p(\mathbf{x}) = p(x_1) p(x_2|x_1) p(x_3|x_1, x_2) \cdots p(x_d|x_1, \ldots, x_{d-1}) = \prod_{i=1}^{d} p(x_i|\mathbf{x}_{<i})$$

Mỗi phân phối điều kiện $$p(x_i|\mathbf{x}_{<i})$$ được mô hình hóa bằng mạng nơ-ron. Cách này cho phép tính chính xác $$p(\mathbf{x})$$ cho mọi điểm $$\mathbf{x}$$, nhưng tốc độ sinh mẫu chậm do phải sinh tuần tự từng chiều và phụ thuộc vào thứ tự sắp xếp các biến.

*Mô hình dựa trên luồng (flow-based models)* sử dụng phép biến đổi khả nghịch $$\mathbf{x} = f(\mathbf{z})$$ với $$\mathbf{z} \sim p_{\mathbf{z}}$$ tuân theo phân phối đơn giản (như chuẩn hóa Gaussian). Theo công thức đổi biến xác suất:

$$p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f^{-1}(\mathbf{x})) \left|\det \frac{\partial f^{-1}}{\partial \mathbf{x}}\right|$$

Cách tiếp cận này vừa đánh giá được mật độ chính xác vừa cho phép lấy mẫu nhanh, nhưng đòi hỏi kiến trúc mạng được thiết kế đặc biệt để đảm bảo tính khả nghịch và tính được định thức ma trận Jacobian hiệu quả.

**Mô hình mật độ ẩn (*implicit density models*)** định nghĩa trực tiếp quy trình lấy mẫu ngẫu nhiên mà không biểu diễn tường minh $$p_{\text{model}}(\mathbf{x})$$:

*GAN* học một bộ sinh (*generator*) $$G: \mathcal{Z} \to \mathcal{X}$$ sao cho khi $$\mathbf{z} \sim p_{\mathbf{z}}$$ thì $$G(\mathbf{z})$$ có phân phối gần với $$p_{\text{data}}$$. Ta không tính tường minh $$p_{\text{model}}$$ mà huấn luyện thông qua mục tiêu đối kháng để đổi lấy tốc độ lấy mẫu nhanh.

*VAE* nằm ở nhóm trung gian (mật độ tường minh xấp xỉ): mô hình biểu diễn tường minh $$p(\mathbf{x}|\mathbf{z})$$, nhưng việc lấy tích phân biên theo biến ẩn $$\mathbf{z}$$ được xấp xỉ bằng suy diễn biến phân, tối ưu cận dưới ELBO thay vì log-hợp lý chính xác.

Việc chọn họ mô hình nào phụ thuộc vào yêu cầu bài toán: cần tính xác suất chính xác (cho phát hiện bất thường, nén dữ liệu), cần lấy mẫu nhanh theo thời gian thực, hay ưu tiên độ sắc nét của mẫu hơn tính ổn định khi huấn luyện.

### Mô hình biến ẩn

Nhiều mô hình sinh đưa thêm biến ẩn $$\mathbf{z}$$ để biểu diễn các nhân tố nền tảng chi phối sự biến thiên của dữ liệu. Quy trình sinh gồm hai bước:

1. Lấy mẫu mã ẩn từ phân phối tiên nghiệm: $$\mathbf{z} \sim p(\mathbf{z})$$ (thường chọn $$\mathcal{N}(0, I)$$).
2. Sinh dữ liệu có điều kiện: $$\mathbf{x} \sim p(\mathbf{x}|\mathbf{z}; \theta)$$.

Phân phối biên của dữ liệu khi đó là:

$$p(\mathbf{x}; \theta) = \int p(\mathbf{x}|\mathbf{z}; \theta) p(\mathbf{z}) d\mathbf{z}$$

Khung mô hình này rất hữu ích vì các chiều của biến ẩn có thể tương ứng với những thuộc tính diễn giải được (chẳng hạn góc mặt, ánh sáng, biểu cảm đối với ảnh khuôn mặt), đồng thời biểu diễn đa tạp dữ liệu nhiều chiều trong không gian thấp chiều hơn. Khó khăn nằm ở chỗ tích phân biên theo biến liên tục $$\mathbf{z}$$ không thể tính trực tiếp. Các họ mô hình giải quyết vấn đề này theo cách khác nhau:

VAE áp dụng suy diễn biến phân, giới thiệu bộ mã hóa (*encoder*) $$q(\mathbf{z}|\mathbf{x}; \phi)$$ để xấp xỉ phân phối hậu nghiệm $$p(\mathbf{z}|\mathbf{x})$$ và tối đa hóa cận dưới bằng chứng (*Evidence Lower Bound* — ELBO):

$$\log p(\mathbf{x}; \theta) \geq \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}; \phi)}[\log p(\mathbf{x}|\mathbf{z}; \theta)] - KL(q(\mathbf{z}|\mathbf{x}; \phi) \| p(\mathbf{z}))$$

Cận dưới này có thể ước lượng bằng lấy mẫu và tối ưu qua lan truyền ngược nhờ kỹ thuật tái tham số hóa (*reparameterization trick*).

GAN bỏ qua hoàn toàn tích phân hợp lý và huấn luyện trực tiếp bộ sinh $$G(\mathbf{z}; \theta)$$ thông qua trò chơi đối kháng với bộ phân biệt để học cách lấy mẫu ẩn từ phân phối dữ liệu.

### Các chỉ số đánh giá

Đánh giá mô hình sinh khó hơn đánh giá phân loại vì mục tiêu là đo mức độ trùng khớp giữa hai phân phối nhiều chiều:

**Log-hợp lý (*log-likelihood*)** (khi tính được): đo xác suất mà mô hình gán cho tập kiểm tra; giá trị càng cao càng tốt. Tuy nhiên, log-hợp lý cao không luôn đồng nghĩa với chất lượng thị giác tốt, và một mô hình ghi nhớ dữ liệu huấn luyện có thể đạt log-hợp lý rất cao trên tập huấn luyện nhưng vô dụng khi sinh mẫu mới.

**Inception Score (IS)**: đưa các mẫu sinh qua mạng phân loại Inception tiền huấn luyện rồi tính:

$$IS = \exp(\mathbb{E}_{\mathbf{x} \sim p_G}[KL(p(y|\mathbf{x}) \| p(y))])$$

IS phản ánh đồng thời độ rõ nét (phân phối có điều kiện $$p(y|\mathbf{x})$$ tập trung vào một lớp) và độ đa dạng (phân phối biên $$p(y)$$ trải đều các lớp). Giá trị cao hơn là tốt hơn, nhưng IS bị phụ thuộc vào các lớp của ImageNet và không phát hiện được hiện tượng ghi nhớ dữ liệu.

**Fréchet Inception Distance (FID)**: so sánh phân phối đặc trưng trích từ mạng Inception của tập mẫu thật và tập mẫu sinh dưới giả định phân phối chuẩn nhiều chiều:

$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

FID càng thấp chứng tỏ phân phối mẫu sinh càng gần phân phối dữ liệu thật. Chỉ số này ổn định hơn IS nhưng vẫn bị giới hạn bởi giả định đặc trưng tuân theo phân phối chuẩn.

Vì không có thước đo tự động nào hoàn hảo, đánh giá trực quan bằng mắt người vẫn rất cần thiết. Ngoài ra, với các ứng dụng chuyên biệt, những chỉ số theo miền (như độ bảo toàn danh tính khuôn mặt hay tính hợp lệ hóa học của phân tử) thường mang lại thông tin thiết thực hơn chỉ số tổng quát.

## 3. Ví dụ và Trực giác

Để hình dung cách mô hình sinh hoạt động, hãy xét bài toán sinh ảnh chữ số viết tay. Nếu bạn chưa từng nhìn thấy chữ số "3", bạn khó có thể vẽ ra một chữ số "3" hợp lệ. Nhưng sau khi quan sát hàng nghìn ví dụ về các chữ số, bạn nắm được quy luật chung: chữ số "3" có hai nét cong nối nhau theo chiều dọc với nét bút liên tục. Từ đó bạn có thể viết ra nhiều chữ số "3" mới — không sao chép nguyên mẫu nào nhưng vẫn đúng quy luật. Mô hình sinh thực hiện chính quá trình học cấu trúc đó một cách tự động từ dữ liệu.

Cùng trên bài toán này, ba họ mô hình hành xử rất khác nhau:

**Mô hình tự hồi quy:** Sinh ảnh lần lượt từng điểm ảnh theo thứ tự trái sang phải, trên xuống dưới. Mỗi điểm ảnh mới được dự đoán dựa trên toàn bộ các điểm ảnh đã sinh trước đó, giúp đảm bảo tính nhất quán cục bộ nhưng mất nhiều bước tuần tự (784 bước cho một ảnh MNIST).

**VAE:** Học một không gian ẩn trong đó các vùng khác nhau tương ứng với các chữ số và phong cách viết khác nhau. Để sinh chữ số "3", chỉ cần lấy mẫu một điểm từ vùng tương ứng và giải mã trong một lượt truyền xuôi. Không gian ẩn mượt cũng cho phép nội suy liên tục giữa các chữ số, nhưng hàm mất mát tái tạo theo điểm ảnh (như MSE) thường khiến ảnh đầu ra hơi mờ.

**GAN:** Huấn luyện bộ sinh để đánh lừa bộ phân biệt phân tách ảnh thật và ảnh giả. Vì không bị ép khớp từng điểm ảnh theo nghĩa trung bình bình phương, bộ sinh có xu hướng tạo ra ảnh sắc nét hơn, nhưng đổi lại quá trình huấn luyện kém ổn định hơn và dễ xảy ra sụp đổ mode (chẳng hạn chỉ sinh lặp đi lặp lại một vài kiểu viết chữ số "3").

Xét thêm một ví dụ đơn giản trong không gian 2D: phân phối dữ liệu thật $$p_{\text{data}}$$ là hỗn hợp hai phân phối chuẩn tạo thành hai cụm tách biệt:

$$p_{\text{data}}(\mathbf{x}) = 0.5 \mathcal{N}(\mathbf{x}; [2, 2], I) + 0.5 \mathcal{N}(\mathbf{x}; [-2, -2], I)$$

- **Mô hình tự hồi quy** phân rã thành $$p(x_2|x_1)p(x_1)$$. Tại cụm thứ nhất có tâm [2, 2], thành phần tương ứng xấp xỉ $$p(x_1) \approx \mathcal{N}(2, 1)$$ và phân phối điều kiện $$p(x_2|x_1) \approx \mathcal{N}(2, 1)$$ (tương tự quanh [-2, -2] cho cụm thứ hai). Khi sinh mẫu, mô hình lấy mẫu $$x_1 \sim p(x_1)$$ rồi lấy mẫu $$x_2 \sim p(x_2|x_1)$$.
- **VAE** sử dụng biến ẩn $$z \in \mathbb{R}$$, học cách ánh xạ vùng $$z < 0$$ sang cụm [-2, -2] và vùng $$z > 0$$ sang cụm [2, 2]. Khi sinh mẫu, chỉ cần lấy $$z \sim \mathcal{N}(0, 1)$$ và đưa qua bộ giải mã để thu được $$\mathbf{x}$$.
- **GAN** huấn luyện bộ sinh ánh xạ nhiễu 1D $$z$$ sang điểm 2D sao cho bộ phân biệt không thể tách biệt với mẫu thật, ví dụ ánh xạ đoạn $$z \in [-3, 0]$$ sang cụm thứ nhất và $$z \in [0, 3]$$ sang cụm thứ hai.

Nếu được huấn luyện phù hợp, cả ba cách tiếp cận đều có thể bao phủ đủ hai cụm dữ liệu, nhưng khác nhau rõ rệt về cách biểu diễn phân phối, độ ổn định khi tối ưu và quy trình lấy mẫu.

## Video tham khảo

- **Xem trước:** [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, tiếng Anh. Giải thích mục tiêu khớp $$p_{\text{model}}$$ với $$p_{\text{data}}$$ xuyên suốt phần lý thuyết của bài.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, tiếng Anh. Trình bày ba hướng tiếp cận chính: tự hồi quy, biến ẩn và huấn luyện đối kháng.
- [In Statistics, Probability is not Likelihood.](https://www.youtube.com/watch?v=pYxNSUDSFH4) — StatQuest with Josh Starmer, tiếng Anh. Phân biệt rõ xác suất và hàm hợp lý, giúp hiểu chính xác mục tiêu MLE trong bài.
