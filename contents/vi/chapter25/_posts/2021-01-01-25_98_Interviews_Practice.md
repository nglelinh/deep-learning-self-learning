---
layout: post
title: 25-98 Luyện phỏng vấn (Bayesian và bất định)
chapter: '25'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter25
lesson_type: optional
---

# Tùy chọn: Luyện phỏng vấn — Học sâu Bayes và độ bất định

> Bài này là **tùy chọn**, **không** thay thế phần tổng quan hướng phát triển tương lai của chương. Sau khi đọc bài tổng quan đó (và phần xác suất ở Chương 00), hãy luyện tập với các câu hỏi dưới đây rồi tham khảo thêm *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để xem đầy đủ các câu hỏi có lời giải về lập trình xác suất và học sâu Bayes.

**Chủ đề đọc thêm trong tài liệu PDF:** suy luận Bayes, phân bố tiên nghiệm (*prior*) và hậu nghiệm (*posterior*) ở mức khái niệm, cùng cách định lượng độ bất định trong mạng sâu. Vì khóa học không tách riêng một chương về học sâu Bayes, nội dung này được đặt trong Chương 25.

Trang tổng hợp lộ trình luyện tập: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu hỏi tự luyện

### Q1. Hai loại độ bất định

Một mô hình y khoa trả về $$p(\text{bệnh}=1\mid\mathbf{x})=0.51$$. Hãy nêu một tình huống trong đó giá trị này phản ánh độ bất định *aleatoric* và một tình huống phản ánh độ bất định *epistemic*. Thử nghiệm nào giúp phân biệt hai trường hợp?

**Gợi ý.** *Aleatoric* là nhiễu nội tại không mất đi ngay cả khi có vô hạn dữ liệu; *epistemic* là sự thiếu hiểu biết của mô hình và sẽ giảm khi bổ sung thêm dữ liệu có nhãn đúng miền.

**Thảo luận.** Với *aleatoric*, bản thân ảnh đầu vào vốn mơ hồ (ví dụ bị mờ do chuyển động hoặc hai dấu hiệu bệnh chồng lấn nhau), nên việc thu thập thêm dữ liệu cùng loại không đẩy xác suất về 0 hay 1. Với *epistemic*, mẫu đến từ máy quét hoặc bệnh viện mới mà mô hình chưa từng thấy nên bộ trọng số chưa đủ thông tin; bổ sung dữ liệu có nhãn trong miền mới sẽ làm phân bố hậu nghiệm tập trung hơn. Khi chạy nhiều lượt suy luận bằng *MC dropout* hoặc *deep ensemble*, các dự đoán sẽ ít biến động trong trường hợp *aleatoric* nhưng phân tán mạnh trong trường hợp *epistemic*.

### Q2. Chính quy hóa L2 dưới góc nhìn tiên nghiệm Gauss

Hãy chứng minh rằng việc cực đại hóa hậu nghiệm $$p(\mathbf{w}\mid\mathcal{D}) \propto p(\mathcal{D}\mid\mathbf{w})\,p(\mathbf{w})$$ với phân bố tiên nghiệm $$p(\mathbf{w})=\mathcal{N}(\mathbf{0},\lambda^{-1}I)$$ và mô hình quan sát Gauss (sai khác hằng số) tương đương với cực tiểu hóa MSE cộng số hạng phạt $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$.

**Gợi ý.** Lấy $$-\log$$ hai vế của phân bố hậu nghiệm; phân bố tiên nghiệm Gauss sẽ cho ra một số hạng bậc hai theo $$\mathbf{w}$$.

**Thảo luận.** Ta có $$-\log p(\mathbf{w}) = \frac{\lambda}{2}\|\mathbf{w}\|_2^2 + \mathrm{const}$$. Dưới giả định hợp lý (*likelihood*) Gauss, $$-\log p(\mathcal{D}\mid\mathbf{w})$$ chính là MSE (sai khác một hệ số tỉ lệ và hằng số). Do đó, ước lượng cực đại hậu nghiệm (MAP) với tiên nghiệm Gauss tương đương hoàn toàn với hồi quy có chính quy hóa L2. Đây là cầu nối trực tiếp từ Chương 00 và Chương 09 sang ngôn ngữ Bayes. Tuy nhiên, MAP chỉ trả về một ước lượng điểm nên tự nó **không** cung cấp khoảng tin cậy.

### Q3. Vì sao mạng ước lượng điểm thường quá tự tin?

Một mạng dùng hàm softmax được huấn luyện bằng cross-entropy trên tập dữ liệu sạch nhưng lại gán xác suất $$0.99$$ cho một lớp sai trên đầu vào ngoài phân bố (OOD). Dưới góc nhìn Bayes, quy trình huấn luyện đã bỏ qua yếu tố nào?

**Gợi ý.** Huấn luyện chuẩn chỉ tìm một bộ trọng số $$\mathbf{w}_{\mathrm{MAP}}$$ (hoặc một điểm nghiệm SGD) rồi dùng $$p(y\mid\mathbf{x},\mathbf{w}_{\mathrm{MAP}})$$ thay cho $$p(y\mid\mathbf{x},\mathcal{D})$$.

**Thảo luận.** Phân bố dự đoán Bayes đầy đủ phải lấy tích phân $$p(y\mid\mathbf{x},\mathbf{w})$$ theo phân bố hậu nghiệm $$p(\mathbf{w}\mid\mathcal{D})$$. Việc chỉ giữ lại một điểm $$\mathbf{w}$$ duy nhất bỏ qua sự bất đồng giữa nhiều bộ trọng số khác nhau cùng khớp tốt tập huấn luyện. Trên dữ liệu OOD, mức bất đồng này thường lớn, nên trung bình hỗn hợp sẽ phẳng hơn nhiều so với đầu ra softmax quá nhọn của một mạng đơn lẻ. Lưu ý rằng *temperature scaling* có thể cải thiện hiệu chỉnh xác suất *trong miền*, nhưng không khắc phục một cách hệ thống sự quá tự tin trên dữ liệu OOD.

### Q4. MC Dropout như phép xấp xỉ hậu nghiệm chi phí thấp

Giả sử bạn giữ bật *dropout* lúc kiểm thử, chạy cùng một mạng $$T=20$$ lần và lấy trung bình các vectơ softmax. Cách làm này xấp xỉ không chính thức đối tượng Bayes nào, và có những hạn chế gì?

**Gợi ý.** Mỗi mặt nạ *dropout* tạo ra một hàm dự đoán khác nhau; trung bình của chúng tạo thành một phân bố hỗn hợp.

**Thảo luận.** Về mặt trực giác, kết quả là phân bố dự đoán hỗn hợp $$\frac1T\sum_t p(y\mid\mathbf{x},\mathbf{w}\odot m_t)$$, đôi khi được diễn giải như một phép suy luận biến phân trên mặt nạ *dropout*. Tuy nhiên, phương pháp này **không** đảm bảo khớp chính xác với phân bố hậu nghiệm thật và cũng không miễn phí vì tốn $$T$$ lượt lan truyền xuôi khi suy luận. Một phương án cạnh tranh phổ biến khác là *deep ensemble* (huấn luyện độc lập nhiều mạng khác nhau).

### Q5. Lựa chọn phân bố tiên nghiệm hợp lý cho hệ số chặn

Khi cần đặt phân bố tiên nghiệm cho tham số *bias* trong một mô hình hồi quy logistic đã xác định đúng dạng (Chương 03), vì sao $$\mathcal{N}(0,10^2)$$ thường hợp lý hơn một phân bố điểm tập trung tại $$0$$, và khi nào thì phương sai quá lớn lại gây bất lợi?

**Gợi ý.** Phân bố Dirac tại 0 hàm ý bạn khẳng định chắc chắn hệ số chặn phải bằng đúng 0.

**Thảo luận.** Tiên nghiệm Gauss rộng thể hiện giả định “chưa biết chính xác hệ số chặn nhưng giá trị của nó nằm trong khoảng hợp lý”, thay vì ép cứng biên quyết định đi qua một giá trị log-odds cố định khi đặc trưng bằng 0. Tuy nhiên, phương sai quá lớn trở nên bất lợi khi các đặc trưng đầu vào chưa được chuẩn hóa đơn vị, khiến mức độ ảnh hưởng của tiên nghiệm phụ thuộc tùy ý vào thang đo. Các chủ đề sâu hơn về tiên nghiệm yếu thông tin (*weakly informative prior*) và mô hình phân cấp được trình bày chi tiết trong sách tham khảo; bài học cốt lõi ở đây là ngay cả việc “dùng thiết lập mặc định” cũng ngầm tương ứng với một lựa chọn tiên nghiệm.

## Ghi công

Kashani, S. và Ivry, A., *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Có thể tải bản PDF từ arXiv để xem đầy đủ bộ câu hỏi và lời giải; các câu hỏi trên là phần tự luyện biên soạn riêng cho khóa học, không sao chép nguyên văn từ sách.

## Video tham khảo

- **Xem trước** [MaDL - Bayesian Decision Theory](https://www.youtube.com/watch?v=j4yxsEQqPMI) — Tübingen Machine Learning, tiếng Anh. Trình bày lý thuyết quyết định Bayes — nền tảng trực tiếp cho bộ câu hỏi luyện tập này.
- [Bayes theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM) — 3Blue1Brown, tiếng Anh. Minh họa trực quan định lý Bayes, xuất phát điểm của các câu hỏi trên.
- [MaDL - Kullback-Leibler Divergence](https://www.youtube.com/watch?v=vPvnpC0PNVQ) — Tübingen Machine Learning, tiếng Anh. Giải thích phân kỳ KL, công cụ thường dùng khi so sánh các phân bố bất định.
