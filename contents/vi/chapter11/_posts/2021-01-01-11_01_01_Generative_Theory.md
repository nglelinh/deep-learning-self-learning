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

# Mô hình Sinh: Học Cách Tạo ra Dữ liệu

## 1. Tổng quan khái niệm
Mô hình sinh (generative models) đại diện cho một sự chuyển hướng căn bản trong cách chúng ta nghĩ về học máy. Trong khi các mô hình phân biệt (discriminative models) học ánh xạ từ đầu vào sang đầu ra — phân loại ảnh thành các danh mục, dịch câu giữa các ngôn ngữ, hoặc dự báo giá cổ phiếu từ dữ liệu lịch sử — thì mô hình sinh học hiểu và tái tạo cấu trúc nền tảng của chính dữ liệu. Chúng đặt ra một câu hỏi tham vọng hơn: cho các ví dụ từ một phân phối dữ liệu, liệu ta có thể học để sinh ra các mẫu mới, thực tế từ phân phối đó? Năng lực này mở ra những khả năng đáng chú ý: tạo ảnh chân thực của người không tồn tại, soạn nhạc theo phong cách Bach, thiết kế phân tử với các tính chất mong muốn, hoặc tăng cường tập dữ liệu hạn chế bằng các ví dụ tổng hợp.

Hiểu vì sao mô hình hóa sinh quan trọng đòi hỏi đánh giá những gì khác biệt căn bản giữa sinh và phân biệt. Một bộ phân loại phân biệt cho giống chó học các đặc trưng đủ để phân biệt các giống — hình dạng tai, họa tiết lông, kích thước. Nó không cần hiểu cách các đặc trưng này kết hợp thành một con chó mạch lạc, hay điều gì khiến một con chó hợp giải phẫu so với bất khả. Một mô hình sinh phải học cấu trúc sâu hơn: cách pixel tổ chức thành texture, cách texture tạo thành đối tượng, cách đối tượng hợp thành cảnh, và quan trọng là tổ hợp nào thực tế so với phi thực tế. Sự hiểu sâu hơn này khiến mô hình sinh thường học được biểu diễn phong phú hơn mô hình phân biệt, khiến chúng có giá trị ngay cả khi sinh không phải mục tiêu cuối cùng.

Khung toán học cho mô hình sinh bám rễ trong lý thuyết xác suất và mô hình thống kê. Ta giả định dữ liệu $$\mathbf{x}$$ đến từ một phân phối chưa biết $$p_{\text{data}}(\mathbf{x})$$. Mục tiêu là học một phân phối mô hình $$p_{\text{model}}(\mathbf{x}; \theta)$$ được tham số hóa bởi $$\theta$$ (trọng số mạng neuron) xấp xỉ $$p_{\text{data}}$$. Nếu thành công, lấy mẫu từ $$p_{\text{model}}$$ sẽ sinh ra dữ liệu không thể phân biệt với mẫu từ $$p_{\text{data}}$$. Khung xác suất này nối mô hình sinh với ước lượng hợp lý cực đại (maximum likelihood estimation), suy diễn biến phân (variational inference), và các khái niệm nền tảng khác trong thống kê, trong khi việc dùng mạng neuron cho mô hình mang lại sự linh hoạt chưa từng có về các dạng hàm có thể biểu diễn.

Các cách tiếp cận mô hình hóa sinh khác nhau đưa ra các đánh đổi khác nhau giữa chất lượng mẫu, ổn định huấn luyện, đảm bảo lý thuyết và yêu cầu tính toán. Các mô hình tự hồi quy (autoregressive models) như PixelCNN mô hình hóa tường minh $$p(\mathbf{x}) = \prod_i p(x_i | x_{<i})$$, phân rã quá trình sinh thành các phân phối điều kiện tuần tự. Chúng cung cấp hợp lý chính xác và huấn luyện ổn định nhưng sinh chậm (từng pixel một). Variational autoencoder đưa vào biến ẩn $$\mathbf{z}$$ và mô hình hóa $$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$$, tối ưu một cận dưới tractable của hợp lý. Chúng cho phép lấy mẫu nhanh và cung cấp khung xác suất có nguyên tắc nhưng thường sinh mẫu hơi mờ. Mạng sinh đối kháng (generative adversarial networks) tránh hoàn toàn mô hình hóa mật độ tường minh, dùng huấn luyện đối kháng để học một generator lấy mẫu ẩn từ $$p_{\text{data}}$$. Chúng thường tạo mẫu sắc nét, thực tế nhất nhưng chịu bất ổn huấn luyện và sụp đổ mode (mode collapse).

Các ứng dụng thực tiễn của mô hình sinh mở rộng xa hơn tính mới lạ. Trong thị giác máy tính, chúng cho phép tăng cường dữ liệu (sinh thêm ví dụ huấn luyện), siêu phân giải (siêu phân giải / super-resolution — phóng to ảnh độ phân giải thấp), inpainting (điền vùng thiếu), và chuyển phong cách (style transfer — áp dụng phong cách nghệ thuật lên ảnh). Trong xử lý ngôn ngữ tự nhiên, chúng cung cấp năng lực sinh văn bản, dịch máy qua mô hình seq2seq sinh, và tăng cường dữ liệu cho ngôn ngữ ít tài nguyên. Trong khám phá thuốc, chúng sinh cấu trúc phân tử với các tính chất mong muốn. Trong ứng dụng sáng tạo, chúng hỗ trợ nghệ sĩ và nhà thiết kế. Trong phát hiện bất thường, chúng nhận diện outlier bằng cách đo mức độ khớp với phân phối đã học. Hiểu mô hình sinh mở ra không gian ứng dụng rộng lớn này đồng thời cung cấp hiểu biết về cấu trúc dữ liệu có ích ngay cả cho các tác vụ thuần phân biệt.

Tuy nhiên, mô hình hóa sinh về bản chất khó hơn học phân biệt theo nhiều khía cạnh. Không gian đầu ra có thể có kích thước theo hàm mũ lớn hơn không gian nhãn ($$2^{784}$$ ảnh MNIST có thể so với 10 nhãn). Phân phối đã học phải nắm các phụ thuộc phức tạp giữa các chiều đầu ra (pixel không độc lập — pixel lân cận tương quan, các phần của đối tượng phải hợp giải phẫu). Đánh giá là thách thức — ta không thể đơn giản tính độ chính xác như với phân loại. Và sinh đòi hỏi hiểu không chỉ điều gì tách các lớp mà còn điều gì khiến ví dụ thực tế — một ngưỡng hiểu cao hơn. Những thách thức này khiến mô hình hóa sinh trở thành lĩnh vực nghiên cứu sôi động, nơi các đổi mới lớn tiếp tục xuất hiện thường xuyên.

## 2. Nền tảng toán học
Nền tảng toán học của mô hình sinh dựa trên lý thuyết xác suất, ước lượng hợp lý và lý thuyết thông tin. Hãy xây dựng các khái niệm này một cách có hệ thống để hiểu ta tối ưu cái gì khi huấn luyện mô hình sinh và vì sao các cách tiếp cận khác nhau dẫn đến các thuật toán khác nhau.

### Mật độ Xác suất và Phân phối Dữ liệu

Ta giả định dữ liệu huấn luyện $$\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(m)}\}$$ gồm các mẫu độc lập từ một phân phối chưa biết $$p_{\text{data}}(\mathbf{x})$$. Với ảnh, $$\mathbf{x}$$ có thể có $$28 \times 28 = 784$$ chiều (MNIST) hoặc $$224 \times 224 \times 3 = 150{,}528$$ chiều (ImageNet). Phân phối $$p_{\text{data}}$$ gán mật độ xác suất cho mỗi $$\mathbf{x}$$ có thể, với mật độ cao cho ảnh thực tế (chữ số thật, ảnh chụp đối tượng) và mật độ thấp hoặc bằng không cho ảnh phi thực tế (nhiễu ngẫu nhiên, cảnh bất khả giải phẫu).

Mục tiêu là học một mô hình tham số $$p_{\text{model}}(\mathbf{x}; \theta)$$ xấp xỉ $$p_{\text{data}}$$. Các tham số $$\theta$$ (trọng số mạng neuron) cần được đặt sao cho mô hình gán xác suất cao cho các ví dụ huấn luyện và, nhờ tổng quát hóa, cho các ví dụ giữ lại từ cùng phân phối. Cách tiếp cận chuẩn là ước lượng hợp lý cực đại:

$$\theta^* = \arg\max_\theta \prod_{i=1}^{m} p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$

Lấy logarit (để ổn định số và thuận tiện toán học):

$$\theta^* = \arg\max_\theta \sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta) = \arg\max_\theta \frac{1}{m}\sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$

Log-hợp lý trung bình $$\frac{1}{m}\sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$ xấp xỉ kỳ vọng log-hợp lý dưới phân phối dữ liệu:

$$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\text{model}}(\mathbf{x}; \theta)]$$

Tối đa hóa kỳ vọng này tương đương với tối thiểu hóa phân kỳ Kullback-Leibler giữa phân phối dữ liệu và mô hình:

$$KL(p_{\text{data}} \| p_{\text{model}}) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\text{data}}(\mathbf{x}) - \log p_{\text{model}}(\mathbf{x}; \theta)]$$

Vì $$p_{\text{data}}$$ cố định, tối thiểu hóa phân kỳ KL tương đương tối đa hóa log-hợp lý kỳ vọng. Điều này nối hợp lý cực đại với lý thuyết thông tin và cung cấp một thước đo có nguyên tắc về mức độ mô hình xấp xỉ phân phối thật.

### Mô hình Mật độ Tường minh so với Ẩn

Thách thức trong mô hình hóa sinh là với dữ liệu chiều cao, việc định nghĩa tường minh $$p_{\text{model}}(\mathbf{x}; \theta)$$ vừa linh hoạt (có thể xấp xỉ phân phối phức tạp) vừa tractable (thực sự tính được và tối ưu được) là khó.

**Mô hình mật độ tường minh (explicit density models)** tham số hóa trực tiếp $$p_{\text{model}}(\mathbf{x}; \theta)$$:

*Mô hình tự hồi quy* dùng quy tắc chuỗi để phân rã mật độ:

$$p(\mathbf{x}) = p(x_1) p(x_2|x_1) p(x_3|x_1, x_2) \cdots p(x_d|x_1, \ldots, x_{d-1}) = \prod_{i=1}^{d} p(x_i|\mathbf{x}_{<i})$$

Mỗi điều kiện $$p(x_i|\mathbf{x}_{<i})$$ được mô hình hóa bằng mạng neuron. Đây là chính xác — ta có thể tính $$p(\mathbf{x})$$ cho bất kỳ $$\mathbf{x}$$ nào — nhưng sinh chậm (phải sinh các chiều tuần tự) và các giả định độc lập điều kiện có thể hạn chế.

*Mô hình dựa trên luồng (flow-based models)* dùng các phép biến đổi khả nghịch $$\mathbf{x} = f(\mathbf{z})$$ trong đó $$\mathbf{z} \sim p_{\mathbf{z}}$$ đơn giản (Gaussian). Công thức đổi biến cho:

$$p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f^{-1}(\mathbf{x})) \left|\det \frac{\partial f^{-1}}{\partial \mathbf{x}}\right|$$

Điều này chính xác và cho phép cả đánh giá mật độ lẫn lấy mẫu nhanh, nhưng đòi hỏi kiến trúc được thiết kế cẩn thận để đảm bảo khả nghịch và định thức Jacobian tractable.

**Mô hình mật độ ẩn (implicit density models)** định nghĩa một thủ tục ngẫu nhiên để lấy mẫu mà không chỉ rõ tường minh $$p_{\text{model}}(\mathbf{x})$$:

*GAN* học một generator $$G: \mathcal{Z} \to \mathcal{X}$$ sao cho nếu $$\mathbf{z} \sim p_{\mathbf{z}}$$ thì $$G(\mathbf{z})$$ có phân phối xấp xỉ $$p_{\text{data}}$$. Ta không bao giờ tính $$p_{\text{model}}$$ nhưng có thể lấy mẫu hiệu quả. Huấn luyện dùng mục tiêu đối kháng thay vì hợp lý.

*VAE* một phần tường minh: chúng mô hình hóa $$p(\mathbf{x}|\mathbf{z})$$ tường minh nhưng biên hóa trên biến ẩn $$\mathbf{z}$$ bằng xấp xỉ biến phân. Chúng tối đa hóa một cận dưới của log-hợp lý (ELBO) thay vì chính hợp lý.

Lựa chọn giữa tường minh và ẩn, giữa các họ mô hình khác nhau, phụ thuộc ưu tiên: ta có cần hợp lý chính xác (cho phát hiện bất thường, nén)? Có cần lấy mẫu nhanh (cho sinh thời gian thực)? Có ưu tiên chất lượng mẫu hơn ổn định huấn luyện? Hiểu các đánh đổi này hướng dẫn chọn mô hình cho ứng dụng cụ thể.

### Mô hình Biến ẩn

Nhiều mô hình sinh đưa vào biến ẩn $$\mathbf{z}$$ biểu diễn các nhân tố ẩn của biến thiên. Quá trình sinh trở thành:

1. Lấy mẫu mã ẩn: $$\mathbf{z} \sim p(\mathbf{z})$$ (thường là $$\mathcal{N}(0, I)$$)
2. Sinh dữ liệu: $$\mathbf{x} \sim p(\mathbf{x}|\mathbf{z}; \theta)$$

Phân phối biên là:

$$p(\mathbf{x}; \theta) = \int p(\mathbf{x}|\mathbf{z}; \theta) p(\mathbf{z}) d\mathbf{z}$$

Khung này mạnh vì biến ẩn có thể biểu diễn các nhân tố diễn giải được (với khuôn mặt: tư thế, ánh sáng, biểu cảm, danh tính) và không gian ẩn chiều thấp có thể nắm manifold dữ liệu chiều cao. Thách thức là tính tích phân cho hợp lý chính xác đòi hỏi tích phân trên mọi mã ẩn có thể, không tractable với $$\mathbf{z}$$ liên tục. Các mô hình sinh khác nhau xử lý điều này khác nhau:

VAE dùng suy diễn biến phân, đưa vào encoder $$q(\mathbf{z}|\mathbf{x}; \phi)$$ xấp xỉ hậu nghiệm $$p(\mathbf{z}|\mathbf{x})$$ và tối ưu Evidence Lower BOund (ELBO):

$$\log p(\mathbf{x}; \theta) \geq \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}; \phi)}[\log p(\mathbf{x}|\mathbf{z}; \theta)] - KL(q(\mathbf{z}|\mathbf{x}; \phi) \| p(\mathbf{z}))$$

Cận dưới này tractable — ta có thể ước lượng qua lấy mẫu và tối ưu qua backpropagation nhờ reparameterization trick.

GAN bỏ qua hoàn toàn việc tính hợp lý, huấn luyện trực tiếp generator $$G(\mathbf{z}; \theta)$$ để tạo mẫu không thể phân biệt với dữ liệu qua huấn luyện đối kháng. Ta không bao giờ tính $$p(\mathbf{x})$$ nhưng học ẩn cách lấy mẫu từ nó.

### Chỉ số Đánh giá

Đánh giá mô hình sinh là thách thức vì ta quan tâm đến khớp phân phối, không chỉ hiệu năng trên các ví dụ cụ thể. Một số chỉ số đã được đề xuất:

**Log-hợp lý** (khi tính được): Đo mức độ mô hình gán xác suất cho dữ liệu kiểm tra. Cao hơn thì tốt hơn. Tuy nhiên, log-hợp lý cao không đảm bảo mẫu tốt (một mô hình ghi nhớ dữ liệu huấn luyện có log-hợp lý hoàn hảo trên tập huấn luyện).

**Inception Score (IS)**: Sinh mẫu, phân loại bằng mạng Inception, tính:

$$IS = \exp(\mathbb{E}_{\mathbf{x} \sim p_G}[KL(p(y|\mathbf{x}) \| p(y))])$$

Đo cả chất lượng (mẫu nên được phân loại tự tin) và đa dạng (nên phủ mọi lớp). Cao hơn thì tốt hơn, nhưng IS có vấn đề (thiên lệch về các lớp ImageNet, không phát hiện ghi nhớ).

**Fréchet Inception Distance (FID)**: So sánh thống kê của mẫu thật và mẫu sinh trong không gian đặc trưng Inception, coi chúng như Gaussian và tính:

$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

FID thấp hơn chỉ ra phân phối gần hơn. Đáng tin cậy hơn IS nhưng vẫn không hoàn hảo (giả định đặc trưng Gaussian).

Hiểu hạn chế của các chỉ số này quan trọng ngang việc dùng chúng. Chúng tương quan với chất lượng cảm nhận nhưng không hoàn hảo. Kiểm tra bằng mắt vẫn then chốt. Với ứng dụng cụ thể, chỉ số chuyên ngành (bảo toàn danh tính khuôn mặt cho sinh mặt, tính hợp lệ phân tử cho thiết kế thuốc) thường quan trọng hơn chỉ số chung.

## 3. Ví dụ / Trực giác

Để xây dựng trực giác về mô hình sinh, hãy nghĩ về việc học sinh chữ số viết tay. Tưởng tượng bạn chưa từng thấy chữ số "3" nhưng đã thấy hàng nghìn chữ số khác. Bạn có thể bịa ra một "3" hợp lý không? Có lẽ không — bạn thiếu hiểu biết về điều gì tạo nên một chữ số hợp lệ, "3" trông như thế nào, nét nối ra sao.

Bây giờ giả sử bạn thấy hàng nghìn ví dụ của mỗi chữ số kể cả "3". Bạn có thể học: "3" có hai phần cong, thường nối với nhau, hướng thẳng đứng, nét mượt. Với hiểu biết này, bạn có thể sinh các "3" mới — không phải bản sao ví dụ huấn luyện mà các biến thể mới theo mẫu đã học. Đây là điều mô hình sinh làm, nhưng được khám phá tự động từ dữ liệu thay vì được mô tả bằng lời.

Xem xét các cách tiếp cận khác nhau cho tác vụ này:

**Cách tiếp cận tự hồi quy**: Sinh chữ số từng pixel, trái-sang-phải, trên-xuống-dưới. Tại mỗi vị trí, dự đoán giá trị pixel có điều kiện trên tất cả pixel trước. Điều này đảm bảo mỗi pixel nhất quán với các pixel trước (nếu phần trên đã trông như "3", tiếp tục mẫu đó). Sinh tuần tự cung cấp hướng dẫn mạnh nhưng chậm — 784 quyết định tuần tự cho MNIST.

**Cách tiếp cận VAE**: Học một không gian ẩn nơi các vùng khác nhau tương ứng các chữ số và biến thể khác nhau. Để sinh một "3", lấy mẫu mã ẩn từ "vùng 3" (đã học trong huấn luyện) và giải mã qua mạng decoder. Không gian ẩn cung cấp sinh hiệu quả (lấy mẫu một lần, giải mã một lần) và cho phép nội suy (biến hình mượt giữa các chữ số). Tuy nhiên, huấn luyện dựa trên tái tạo có thể tạo mẫu mờ vì MSE theo pixel không nắm tốt chất lượng cảm nhận.

**Cách tiếp cận GAN**: Huấn luyện generator để đánh lừa discriminator đang cố phát hiện giả. Generator học bất kỳ ánh xạ nào từ nhiễu sang ảnh khiến discriminator không thể phát hiện giả. Huấn luyện đối kháng này không đòi hỏi tái tạo tường minh theo pixel, cho phép generator ưu tiên tính thực tế cảm nhận hơn khớp pixel chính xác. Kết quả thường là mẫu sắc nét, thực tế hơn, dù huấn luyện có thể bất ổn và sụp đổ mode có thể xảy ra (generator chỉ học tạo một số kiểu "3").

Hãy đi qua một ví dụ cụ thể với tập dữ liệu toy đơn giản: các điểm 2D tạo thành hai cụm (biểu diễn hai mode của một phân phối). Phân phối thật $$p_{\text{data}}$$ là hỗn hợp hai Gaussian:

$$p_{\text{data}}(\mathbf{x}) = 0.5 \mathcal{N}(\mathbf{x}; [2, 2], I) + 0.5 \mathcal{N}(\mathbf{x}; [-2, -2], I)$$

**Mô hình tự hồi quy**: Mô hình hóa $$p(x_2|x_1)p(x_1)$$. Với mode thứ nhất tâm tại [2, 2], nó học $$p(x_1) \approx \mathcal{N}(2, 1)$$ và $$p(x_2|x_1) \approx \mathcal{N}(2, 1)$$ (xấp xỉ độc lập vì dùng Gaussian, nhưng có thể học tương quan). Sinh: lấy mẫu $$x_1 \sim p(x_1)$$, rồi $$x_2 \sim p(x_2|x_1)$$.

**VAE**: Đưa vào biến ẩn $$z \in \mathbb{R}$$. Học rằng $$z < 0$$ ánh xạ tới mode tại [-2, -2] và $$z > 0$$ ánh xạ tới mode tại [2, 2]. Để sinh, lấy mẫu $$z \sim \mathcal{N}(0, 1)$$, giải mã thành $$\mathbf{x}$$. Không gian ẩn biến thiên mượt từ mode này sang mode kia.

**GAN**: Generator học ánh xạ nhiễu 1D $$z$$ thành điểm 2D sao cho discriminator (thấy cả mẫu thật từ hai Gaussian và mẫu sinh) không thể phân biệt thật/giả. Generator có thể học một hàm phi tuyến ánh xạ $$z \in [-3, 0]$$ tới mode thứ nhất và $$z \in [0, 3]$$ tới mode thứ hai.

Mỗi cách tiếp cận thành công sinh từ cả hai mode nếu huấn luyện đúng, nhưng chúng khác nhau về cách biểu diễn phân phối, ổn định huấn luyện và thủ tục sinh. Hiểu các khác biệt này qua ví dụ đơn giản xây dựng trực giác cho hành vi của chúng trên dữ liệu phức tạp như ảnh.
