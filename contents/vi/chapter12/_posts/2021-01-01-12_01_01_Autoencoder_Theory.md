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

Một câu hỏi hóc búa từng khiến nhiều người mới học phải khựng lại: nếu mục tiêu của Autoencoder chỉ là tái tạo lại đúng đầu vào $$\mathbf{x}$$, vậy tại sao mạng không đơn giản là "chép nguyên xi" đầu vào ra đầu ra để đạt loss bằng 0? Câu trả lời nằm ở hình học của những ràng buộc: chính việc *tước bỏ năng lực ghi nhớ tầm thường* mới buộc mạng phải học ra các đặc trưng bậc cao. Bài lý thuyết này phân tích chặt chẽ từ toán học tuyến tính (PCA) đến các biến thể Denoising và Sparse Autoencoder.

## Bản đồ kiến thức nền: Nền tảng toán học của Autoencoder và các cơ chế chính quy hóa

- **Autoencoder thiếu hoàn chỉnh (Undercomplete) và định lý tương đương PCA:** Chứng minh khi kích hoạt tuyến tính và loss MSE, nghiệm tối ưu của Autoencoder trải cùng không gian con với $$d$$ vector riêng hàng đầu của ma trận hiệp phương sai.
- **Autoencoder dư thừa (Overcomplete) và nguy cơ ánh xạ đồng nhất:** Giải thích vì sao khi $$d \ge D$$, mục tiêu tái tạo thuần túy trở nên vô nghĩa nếu thiếu ràng buộc cấu trúc.
- **Denoising Autoencoder (DAE) và hình học trường vector:** Làm nhiễu đầu vào $$\tilde{\mathbf{x}} \sim q(\tilde{\mathbf{x}}|\mathbf{x})$$ rồi bắt mạng khôi phục $$\mathbf{x}$$, thực chất dạy mạng học vector kéo điểm nhiễu quay về đa tạp dữ liệu.
- **Sparse & Contractive Autoencoder:** Dùng phạt phân kỳ KL lên độ kích hoạt trung bình $$\hat{\rho}_j$$ hoặc chuẩn Frobenius của Jacobian $$\|J_f(\mathbf{x})\|_F^2$$ để tạo biểu diễn chọn lọc và bền vững.

### Khung giải thích (nên đọc chậm): Tính tay số hạng phạt thưa KL trong Sparse Autoencoder

Trong Sparse Autoencoder, ta mong muốn mỗi nơ-ron ẩn chỉ "bật" trung bình khoảng $$\rho = 0{,}05$$ ($$5\%$$ số mẫu trong batch). Hàm phạt phân kỳ KL giữa phân phối Bernoulli($$\rho$$) và độ kích hoạt thực tế $$\hat{\rho}_j$$ là:
$$\Omega_{\text{sparse}}(\hat{\rho}_j) = \rho \ln\frac{\rho}{\hat{\rho}_j} + (1 - \rho)\ln\frac{1 - \rho}{1 - \hat{\rho}_j}.$$

So sánh hai nơ-ron ẩn trên cùng một batch:
1. **Nơ-ron A hoạt động đúng mục tiêu** ($$\hat{\rho}_A = 0{,}05$$):
   $$\Omega(0{,}05) = 0{,}05\ln(1) + 0{,}95\ln(1) = 0.$$
2. **Nơ-ron B bị "bật tràn lan"** ($$\hat{\rho}_B = 0{,}50$$):
   $$\Omega(0{,}50) = 0{,}05\ln\frac{0{,}05}{0{,}50} + 0{,}95\ln\frac{0{,}95}{0{,}50} = 0{,}05(-2{,}3026) + 0{,}95(0{,}6419) = -0{,}1151 + 0{,}6098 = 0{,}4947.$$
3. **Hiệu ứng tối ưu:** Nơ-ron B chịu mức phạt gần $$0{,}50$$ nhân hệ số $$\beta$$, tạo gradient dương ép bias và trọng số của nó giảm xuống cho đến khi nó chỉ phản ứng mạnh với một mẫu đặc trưng thực sự chuyên biệt.

# Autoencoder: Học biểu diễn hiệu quả

![Autoencoder Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Autoencoder_structure.png/600px-Autoencoder_structure.png)
*Hình ảnh: Kiến trúc Autoencoder với bộ mã hóa (encoder) và bộ giải mã (decoder). Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm

Autoencoder đại diện cho một cách huấn luyện mạng nơ-ron khác hẳn học có giám sát. Thay vì học ánh xạ từ đầu vào sang nhãn mục tiêu, autoencoder học cách tái tạo chính đầu vào của nó đi qua một nút thắt thông tin (*information bottleneck*). Nhiệm vụ tưởng như đơn giản này trở nên hữu ích khi mạng bị ràng buộc phải truyền dữ liệu qua một tầng ẩn có số chiều nhỏ hơn gọi là không gian ẩn (*latent space*) hay mã (*code*). Để nén rồi giải nén thành công, mạng buộc phải giữ lại cấu trúc cốt lõi của dữ liệu và loại bỏ nhiễu cũng như chi tiết dư thừa.

Giá trị chính của autoencoder không nằm ở bản thân việc tái tạo lại ảnh gốc mà ở biểu diễn dữ liệu được học trong quá trình đó. Bộ mã hóa (*encoder*) học cách rút trích các đặc trưng quan trọng từ dữ liệu nhiều chiều thành mã gọn, còn bộ giải mã (*decoder*) học cách khôi phục dữ liệu từ mã đó. Trong một autoencoder huấn luyện tốt, các điểm gần nhau trong không gian ẩn thường tương ứng với các đầu vào tương đồng về mặt ngữ nghĩa, cho phép nội suy mượt giữa các mẫu. Nhờ vậy, autoencoder được ứng dụng trong giảm chiều, khử nhiễu, phát hiện bất thường, trích xuất đặc trưng cho tác vụ sau và làm nền tảng cho các mô hình sinh phức tạp hơn.

Hoạt động của autoencoder phụ thuộc trực tiếp vào sự cân bằng giữa dung lượng mạng và ràng buộc nén. Nếu số chiều ẩn lớn hơn hoặc bằng số chiều đầu vào mà không có ràng buộc bổ sung, mạng có thể học hàm đồng nhất — sao chép nguyên xi đầu vào — và không rút trích được cấu trúc nào. Nút thắt chiều buộc mạng phải chọn lọc thông tin cần giữ: khi nén ảnh 784 chiều xuống 32 chiều ẩn, mạng không thể lưu từng điểm ảnh độc lập mà phải khám phá các đặc trưng bậc cao hơn như biên cạnh, hình dạng và kết cấu phù hợp với phân phối dữ liệu huấn luyện.

Về mặt lịch sử, autoencoder là một trong những hướng học không giám sát thành công sớm nhất của học sâu, chứng minh mạng nơ-ron có thể học biểu diễn hữu ích mà không cần nhãn. Ý tưởng này từng đóng vai trò quan trọng trong tiền huấn luyện nhiều tầng trước thời kỳ ReLU, đồng thời ảnh hưởng trực tiếp tới các phương pháp học tự giám sát hiện đại — nơi mô hình dự đoán hoặc tái tạo một phần dữ liệu từ phần còn lại. Kiến trúc encoder–decoder cũng trở thành mẫu thiết kế phổ biến trong mô hình chuỗi-sang-chuỗi, VAE và nhiều họ mô hình sinh khác.

Tuy vậy, autoencoder chuẩn có những giới hạn rõ rệt khi dùng làm mô hình sinh. Vì mục tiêu chỉ là tái tạo dữ liệu huấn luyện, không gian ẩn có thể tồn tại các "khoảng trống" không tương ứng với mẫu huấn luyện nào, khiến việc lấy mẫu ngẫu nhiên dễ tạo đầu ra bất hợp lý. Ngoài ra, autoencoder không mô hình hóa tường minh phân phối dữ liệu, và sai số tái tạo theo điểm ảnh (như MSE) không luôn phản ánh đúng cảm nhận thị giác của con người. Những hạn chế này dẫn tới sự ra đời của Variational Autoencoder (chính quy hóa phân phối không gian ẩn), GAN (dùng mục tiêu đối kháng) và hàm mất mát cảm nhận (*perceptual loss*).

## 2. Nền tảng toán học

Cấu trúc toán học của autoencoder tương đối gọn gàng nhưng mang nhiều hệ quả quan trọng. Mô hình gồm hai mạng nơ-ron nối tiếp: bộ mã hóa $$f_\phi$$ với tham số $$\phi$$ và bộ giải mã $$g_\theta$$ với tham số $$\theta$$. Với đầu vào $$\mathbf{x} \in \mathbb{R}^{d}$$, bộ mã hóa tạo ra biểu diễn ẩn:

$$\mathbf{z} = f_\phi(\mathbf{x}) \in \mathbb{R}^{k}$$

trong đó thường chọn $$k < d$$ để tạo nút thắt (trừ các biến thể dùng ràng buộc chính quy hóa thay cho nút thắt chiều). Bộ giải mã tái tạo đầu vào từ mã ẩn:

$$\hat{\mathbf{x}} = g_\theta(\mathbf{z}) = g_\theta(f_\phi(\mathbf{x})) \in \mathbb{R}^{d}$$

Mục tiêu huấn luyện là tối thiểu hóa sai số tái tạo:

$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

cho dữ liệu liên tục (sai số bình phương trung bình — MSE), hoặc:

$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = -\sum_{i=1}^{d} [x_i \log(\hat{x}_i) + (1-x_i)\log(1-\hat{x}_i)]$$

cho dữ liệu nhị phân (binary cross-entropy, xem mỗi chiều là một biến Bernoulli độc lập).

Lựa chọn hàm mất mát phản ánh giả định về mô hình nhiễu của dữ liệu. Tối thiểu hóa MSE tương đương với ước lượng hợp lý cực đại dưới giả định nhiễu chuẩn $$p(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; g_\theta(\mathbf{z}), \sigma^2 I)$$. Binary cross-entropy giả định mỗi điểm ảnh tuân theo phân phối Bernoulli với xác suất $$\hat{x}_i$$; trong thực tế, hàm này cũng thường hoạt động tốt với ảnh có giá trị chuẩn hóa về đoạn [0, 1]. Ở mức cao hơn, có thể dùng hàm mất mát cảm nhận đo khoảng cách trong không gian đặc trưng của mạng tiền huấn luyện.

Số chiều nút thắt $$k$$ điều chỉnh sự đánh đổi giữa mức độ nén và độ trung thực tái tạo. Giá trị $$k$$ rất nhỏ (2–3 chiều) thuận tiện cho việc trực quan hóa không gian ẩn nhưng dễ làm mất chi tiết quan trọng; giá trị trung bình (32–128 chiều đối với ảnh cỡ nhỏ) cân bằng tốt giữa nén và chất lượng tái tạo; còn $$k$$ quá lớn làm giảm áp lực nén.

Ngay cả khi $$k \geq d$$ (không tạo nút thắt chiều), ta vẫn có thể tránh hàm đồng nhất bằng các ràng buộc chính quy hóa khác. **Autoencoder thưa (*sparse autoencoder*)** bổ sung thành phần phạt độ thưa lên giá trị kích hoạt ẩn:

$$\mathcal{L}_{\text{sparse}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \sum_{j=1}^{k} KL(\rho \| \hat{\rho}_j)$$

trong đó $$\rho$$ là mức độ kích hoạt mục tiêu nhỏ (chẳng hạn 0,05) và $$\hat{\rho}_j$$ là mức kích hoạt trung bình của nơ-ron ẩn thứ $$j$$ trên tập huấn luyện. Thành phần phân kỳ KL khuyến khích phần lớn đơn vị ẩn giữ giá trị gần 0 ở hầu hết thời điểm, buộc mỗi đơn vị chuyên biệt hóa vào một nhóm mẫu hình riêng.

**Autoencoder khử nhiễu (*denoising autoencoder*)** chủ động làm nhiễu đầu vào $$\mathbf{x}$$ thành $$\tilde{\mathbf{x}}$$ (bằng nhiễu Gaussian, che ngẫu nhiên điểm ảnh hoặc nhiễu muối-tiêu) nhưng yêu cầu mạng tái tạo bản gốc sạch:

$$\mathcal{L} = \|\mathbf{x} - g_\theta(f_\phi(\tilde{\mathbf{x}}))\|^2$$

Cách huấn luyện này buộc bộ mã hóa học các đặc trưng bền vững trước nhiễu, còn bộ giải mã học cách khôi phục phần bị hỏng dựa trên ngữ cảnh còn lại. Do đó, autoencoder khử nhiễu thường học được biểu diễn hữu ích hơn autoencoder chuẩn vì không thể chỉ ghi nhớ điểm ảnh thô.

Về hình học không gian ẩn, trong một autoencoder được huấn luyện tốt, ta có thể nội suy tuyến tính giữa hai mã ẩn $$\mathbf{z}_1$$ và $$\mathbf{z}_2$$:

$$\mathbf{z}_t = (1-t)\mathbf{z}_1 + t\mathbf{z}_2, \quad t \in [0,1]$$

rồi giải mã $$g_\theta(\mathbf{z}_t)$$ để thu được các mẫu trung gian. Tuy nhiên, autoencoder chuẩn không có cơ chế đảm bảo không gian ẩn liên tục khắp mọi nơi; nếu đường nội suy đi qua vùng trống không có mẫu huấn luyện, ảnh giải mã có thể bị méo hoặc phi thực tế. Đây là lý do Variational Autoencoder bổ sung ràng buộc trực tiếp lên phân phối không gian ẩn.

## 3. Ví dụ và Trực giác

Hãy theo dõi quá trình huấn luyện một autoencoder nén ảnh MNIST kích thước 28×28 = 784 điểm ảnh xuống mã ẩn 32 chiều.

Ở những bước đầu với trọng số ngẫu nhiên, bộ mã hóa tạo ra các vector vô nghĩa và bộ giải mã xuất ra ảnh gần như nhiễu, dẫn đến sai số tái tạo rất lớn. Lan truyền ngược dần điều chỉnh trọng số để giảm sai số này.

Trong giai đoạn sớm, bộ mã hóa học các thông tin thô như vùng nền tối và vùng nét sáng ở từng khu vực ảnh. Khi tiếp tục huấn luyện, mạng nhận ra các thành phần hình học rõ hơn: một số chiều ẩn phản ứng mạnh với nét dọc (như chữ số 1, 4, 7), chiều khác gắn với đường cong (0, 6, 8, 9) hoặc nét ngang (2, 3, 5, 7). Đến khi hội tụ, 32 chiều ẩn tự tổ chức để mã hóa kiểu chữ số, độ dày nét, độ nghiêng, kích thước và vị trí — hoàn toàn phát sinh từ mục tiêu tái tạo mà không cần nhãn.

Khi chiếu các mẫu kiểm tra vào không gian ẩn 32 chiều, các biến thể của chữ số "3" (nét dày, mảnh, nghiêng) thường tập trung gần nhau do chia sẻ cấu trúc hình học tương tự, trong khi các chữ số "8" (gồm hai vòng kín) tụ thành một cụm khác.

Ta có thể kiểm tra cấu trúc không gian ẩn bằng phép nội suy giữa mã ẩn $$\mathbf{z}_3$$ của một chữ số "3" và $$\mathbf{z}_8$$ của một chữ số "8":

$$\mathbf{z}_{0.0} = \mathbf{z}_3 \to$$ giải mã thành "3"  
$$\mathbf{z}_{0.25} = 0.75\mathbf{z}_3 + 0.25\mathbf{z}_8 \to$$ giải mã thành "3" bắt đầu xuất hiện nét của "8"  
$$\mathbf{z}_{0.5} = 0.5\mathbf{z}_3 + 0.5\mathbf{z}_8 \to$$ giải mã thành hình dạng trung gian giữa "3" và "8"  
$$\mathbf{z}_{0.75} = 0.25\mathbf{z}_3 + 0.75\mathbf{z}_8 \to$$ giải mã thành "8" còn lưu nét của "3"  
$$\mathbf{z}_{1.0} = \mathbf{z}_8 \to$$ giải mã thành "8"

Quá trình biến đổi mượt cho thấy không gian ẩn có cấu trúc liên tục tốt; ngược lại, các điểm nhảy đột ngột hoặc ảnh mờ nhiễu cho thấy tồn tại khoảng trống trong không gian ẩn.

Với autoencoder khử nhiễu, giả sử ta xóa ngẫu nhiên 20% điểm ảnh của chữ số "7". Dựa trên phần nét còn lại, mạng phải suy luận bối cảnh tổng thể để phân biệt đó là "7" chứ không phải "1" và khôi phục trọn vẹn chữ số gốc. Chính yêu cầu suy luận từ thông tin thiếu hụt này giúp biểu diễn ẩn nắm bắt cấu trúc ngữ nghĩa tốt hơn so với chỉ sao chép điểm ảnh.

## Video tham khảo

- **Xem trước:** [What are Autoencoders?](https://www.youtube.com/watch?v=qiUEgSCyY5o) — IBM Technology, tiếng Anh. Minh họa cách tín hiệu tái tạo tự đóng vai trò mục tiêu huấn luyện không cần nhãn.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, tiếng Anh. Phân biệt kiến trúc *undercomplete* và *overcomplete* được đề cập trong phần lý thuyết.
- [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, tiếng Anh. Góc nhìn nút thắt thông tin đối với cùng mục tiêu tái tạo.
