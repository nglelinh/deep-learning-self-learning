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

# Tùy chọn: luyện phỏng vấn — học sâu Bayesian và bất định

> Bài này là **tùy chọn**. Nó **không** viết lại khảo sát hướng tương lai của chương. Sau khảo sát đó (và ghi chú xác suất Chương 00), dùng bài luyện này, rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để làm các câu đã giải về lập trình xác suất và học sâu Bayesian.

**Chủ đề sách cho chương này (đọc trong PDF, không phải ở đây):** suy luận Bayesian, prior / posterior ở mức khái niệm, và bất định trong mô hình sâu. Khóa không có chương Bayesian riêng; Chương 25 là nhà cho chủ đề nâng cao.

Hub: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu luyện (gốc)

### Q1. Hai loại bất định

Một mô hình y khoa ra $$p(\text{bệnh}=1\mid\mathbf{x})=0.51$$. Kể một câu chuyện trong đó số đó là *aleatoric* và một câu chuyện *epistemic*. Thí nghiệm thêm nào phân biệt chúng?

**Gợi ý.** Aleatoric: nhiễu còn lại dù có vô hạn dữ liệu. Epistemic: thu hẹp nếu bạn thu thêm nhãn đúng miền.

**Thảo luận.** Aleatoric: ảnh vốn mơ hồ (mờ chuyển động, hai dấu hiệu chồng); thêm dữ liệu cùng loại không đẩy xác suất về 0 hoặc 1. Epistemic: máy quét / bệnh viện mới và trọng số chưa xác định; thêm nhãn trong miền hoặc prior tốt hơn sẽ làm posterior tập trung. Lặp lượt tới với MC dropout hoặc ensemble sâu (dưới) ít đổi ở câu chuyện thứ nhất và đổi nhiều ở câu chuyện thứ hai.

### Q2. L2 như prior Gauss

Chứng minh cực đại $$ p(\mathbf{w}\mid\mathcal{D}) \propto p(\mathcal{D}\mid\mathbf{w})\,p(\mathbf{w}) $$ với $$p(\mathbf{w})=\mathcal{N}(\mathbf{0},\lambda^{-1}I)$$ và mô hình quan sát Gauss (sai khác hằng số) giống cực tiểu MSE cộng $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$.

**Gợi ý.** Lấy $$-\log$$ của posterior; prior đóng góp một hạng bậc hai.

**Thảo luận.** $$-\log p(\mathbf{w}) = \frac{\lambda}{2}\|\mathbf{w}\|_2^2 + \mathrm{const}$$. Dưới hợp lý Gauss, $$-\log p(\mathcal{D}\mid\mathbf{w})$$ là MSE (hoặc square loss có scale). Vậy MAP với prior đó *chính là* hồi quy chính quy L2. Đây là cầu từ Chương 00 / 09 sang “ngôn ngữ Bayesian.” Nó **không** tự cho bạn khoảng tin cậy — MAP vẫn là một điểm.

### Q3. Vì sao mạng ước lượng điểm trông quá tự tin

Một softmax huấn luyện bằng CE trên tập sạch gán $$0.99$$ cho lớp sai trên đầu vào OOD. Theo ngôn ngữ Bayesian, quy trình huấn luyện đã bỏ gì?

**Gợi ý.** Bạn tối ưu $$\mathbf{w}_{\mathrm{MAP}}$$ (hoặc một điểm SGD) rồi coi $$p(y\mid\mathbf{x},\mathbf{w}_{\mathrm{MAP}})$$ như $$p(y\mid\mathbf{x},\mathcal{D})$$.

**Thảo luận.** Phân phối dự đoán phải tích $$p(y\mid\mathbf{x},\mathbf{w})$$ theo $$p(\mathbf{w}\mid\mathcal{D})$$. Một $$\mathbf{w}$$ bỏ qua sự bất đồng giữa các bộ trọng số đều khớp tập train. Trên $$\mathbf{x}$$ OOD, bất đồng đó thường lớn, nên hỗn hợp gần đều hơn bất kỳ softmax nhọn nào. Temperature scaling có thể sửa hiệu chỉnh *trong miền* mà không cần Bayesian; nó không sửa OOD một cách hệ thống.

### Q4. MC dropout như phác thảo posterior rẻ

Bạn chạy cùng mạng dropout $$T=20$$ lần lúc test và trung bình các softmax. Đối tượng Bayesian nào việc này xấp xỉ *không chính thức*, và nêu một điều nó không phải?

**Gợi ý.** Mỗi mask là một hàm khác; trung bình là hỗn hợp.

**Thảo luận.** Không chính thức đó là dự đoán hỗn hợp $$\frac1T\sum_t p(y\mid\mathbf{x},\mathbf{w}\odot m_t)$$, đôi khi được biện minh như suy diễn biến phân trên mask dropout. Nó **không** bảo đảm hỗn hợp khớp posterior thật, và không miễn phí: bạn trả $$T$$ lượt tới. Ensemble sâu (nhiều mạng huấn luyện độc lập) là phác thảo cạnh tranh; chúng nằm ở ghi chú ensemble trên hub, không phải chương mới.

### Q5. Một prior bạn bảo vệ được

Bạn phải đặt prior lên tham số bias trong mô hình logistic đặc tả đúng (Chương 03). Vì sao $$\mathcal{N}(0,10^2)$$ thường đáng bảo vệ hơn khối điểm tại $$0$$, và khi nào bạn *không* muốn phương sai rất lớn?

**Gợi ý.** Dirac tại 0 nghĩa là “tôi đã biết hệ số chặn.”

**Thảo luận.** Gauss rộng nói “hệ số chặn chưa biết nhưng không điên.” Không cứng buộc biên quyết định đi qua một log-odds cụ thể khi đặc trưng bằng không — một khẳng định khoa học mạnh. Phương sai rất lớn thành vấn đề nếu mã hóa đặc trưng tùy ý (đầu vào chưa chuẩn hóa): prior khi đó phụ thuộc đơn vị. Prior phân cấp / yếu thông tin thuộc các chương Bayesian của sách; ở đây điểm mức khóa là “tôi dùng mặc định” vẫn là một prior.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Tải PDF từ arXiv để xem toàn bộ Q&A đã giải. Trang này là bài luyện gốc của khóa, không phải bản in lại.
