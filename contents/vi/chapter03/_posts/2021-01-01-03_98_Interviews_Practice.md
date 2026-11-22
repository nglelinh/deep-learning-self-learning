---
layout: post
title: 03-98 Luyện phỏng vấn (autodiff, Hessian, logistic)
chapter: '03'
order: 15
owner: Deep Learning Course
lang: vi
categories:
- chapter03
lesson_type: optional
---

# Tùy chọn: Luyện phỏng vấn — Autodiff, Hessian và Hồi quy Logistic

> Bài này là **tùy chọn** và **không** thay thế phần lý thuyết về hàm mất mát, gradient descent hay lan truyền ngược. Sau khi học các bài chính trong chương, bạn có thể luyện với bộ câu hỏi dưới đây rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để tham khảo thêm các câu hỏi kèm lời giải đầy đủ về vi phân tự động, độ cong và hồi quy logistic.

**Chủ đề tham khảo trong sách cho chương này (đọc trực tiếp trong bản PDF):** vi phân tự động, trực giác về ma trận Hessian và phân loại logistic / log-odds.

Trang tổng hợp: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu hỏi luyện tập

### Q1. Gradient của hồi quy logistic

Mô hình hồi quy logistic nhị phân dự đoán $$\hat y=\sigma(\mathbf{w}^\top\mathbf{x})$$ và dùng hàm mất mát $$\ell= -y\log\hat y-(1-y)\log(1-\hat y)$$. Hãy chứng minh:

$$\nabla_{\mathbf{w}}\ell = (\hat y-y)\,\mathbf{x}.$$

**Gợi ý.** Sử dụng $$\sigma'(z)=\sigma(z)(1-\sigma(z))$$ và quy tắc dây chuyền từ Chương 00 hoặc bài lan truyền ngược.

**Thảo luận.** Đặt $$z=\mathbf{w}^\top\mathbf{x}$$. Khi đó $$\partial\ell/\partial z = -\frac{y}{\hat y}\sigma'(z)+\frac{1-y}{1-\hat y}\sigma'(z)$$. Thay $$\sigma'(z)=\hat y(1-\hat y)$$ vào, các thừa số ở mẫu triệt tiêu và biểu thức rút gọn thành $$\hat y-y$$. Nhân tiếp với $$\partial z/\partial\mathbf{w}=\mathbf{x}$$ ta thu được kết quả cần chứng minh. Đây cũng là lý do phép kết hợp "sigmoid + BCE" ổn định hơn về mặt số học khi được gộp chung (*fused*): ta không cần tính tách rời $$\sigma'(z)$$ vốn dễ bị tràn số dưới (*underflow*).

### Q2. Odds, log-odds và ý nghĩa của hệ số

Một mô hình hồi quy logistic học được $$z = -1 + 2\,x_{\mathrm{smoke}}$$ với $$x_{\mathrm{smoke}}\in\{0,1\}$$. Tỷ lệ odds của biến cố $$y=1$$ đối với người không hút thuốc và người hút thuốc là bao nhiêu? Hệ số $$2$$ có ý nghĩa gì trên thang log-odds?

**Gợi ý.** Odds được định nghĩa là $$\hat y/(1-\hat y)=e^{z}$$.

**Thảo luận.** Với người không hút thuốc ($$x_{\mathrm{smoke}}=0$$): $$z=-1$$, odds bằng $$e^{-1}\approx 0.37$$. Với người hút thuốc ($$x_{\mathrm{smoke}}=1$$): $$z=1$$, odds bằng $$e^{1}\approx 2.72$$. Hệ số $$2$$ chính là *log của tỷ số odds* (*log-odds ratio*) gắn với đặc trưng nhị phân này: việc hút thuốc làm odds nhân lên $$e^{2}\approx 7.4$$ lần. Câu hỏi này thường dùng để kiểm tra xem ứng viên có tránh được sai lầm coi trọng số logistic là tác động cộng tuyến tính lên xác suất hay không.

### Q3. Chế độ xuôi và chế độ ngược trên đồ thị nhỏ

Giả sử cần tính $$\partial f/\partial x$$ và $$\partial f/\partial y$$ cho hàm $$f=(x y)+ \sin(x)$$. Hãy so sánh chi phí tính toán (số phép nhân vô hướng và phép gọi hàm lượng giác) giữa (a) hai lượt chế độ xuôi (*forward mode*) và (b) một lượt chế độ ngược (*reverse mode*). Nếu $$f$$ là hàm mất mát của một mạng 50 lớp với hàng triệu trọng số, bạn sẽ chọn chế độ nào?

**Gợi ý.** Chế độ xuôi khởi tạo đạo hàm ở một đầu vào; chế độ ngược khởi tạo đạo hàm ở đầu ra.

**Thảo luận.** Với hai lượt chế độ xuôi, ta chạy riêng một lượt cho $$x$$ và một lượt cho $$y$$, mỗi lượt đều phải đánh giá lại $$xy$$ và $$\sin x$$. Với một lượt chế độ ngược, ta chỉ cần một lượt xuôi lưu các giá trị trung gian rồi truyền ngược tín hiệu đạo hàm (*adjoint*) qua các phép toán $$+$$, $$\times$$ và $$\sin$$ để thu đồng thời cả hai đạo hàm. Khi hàm mất mát là một số vô hướng còn số tham số $$\mathbf{w}$$ rất lớn, chế độ ngược (lan truyền ngược) là lựa chọn thực tế duy nhất — đây là trọng tâm của Chương 03. Chế độ xuôi chỉ có lợi khi số đầu vào rất ít nhưng số đầu ra rất nhiều (ví dụ tính ma trận Jacobian của một ánh xạ từ ít biến sang vector nhiều chiều).

### Q4. Ma trận Hessian của hồi quy logistic và tính lồi

Với một mẫu dữ liệu, xét $$\ell(\mathbf{w})= -y\log\sigma(\mathbf{w}^\top\mathbf{x})-(1-y)\log\bigl(1-\sigma(\mathbf{w}^\top\mathbf{x})\bigr)$$. Hãy chứng minh ma trận Hessian:

$$\nabla^2_{\mathbf{w}}\ell = \hat y(1-\hat y)\,\mathbf{x}\mathbf{x}^\top$$

là nửa xác định dương (*positive semi-definite*). Điều này cho biết gì về cực tiểu cục bộ của hồi quy logistic không có chính quy hóa?

**Gợi ý.** Với mọi vector $$\mathbf{v}$$, ta có $$\mathbf{v}^\top(\mathbf{x}\mathbf{x}^\top)\mathbf{v}=(\mathbf{v}^\top\mathbf{x})^2\ge 0$$ và $$\hat y(1-\hat y)\in(0,1)$$.

**Thảo luận.** Vì Hessian là tích của một số dương với ma trận nửa xác định dương hạng 1 nên $$\ell(\mathbf{w})$$ là hàm lồi theo $$\mathbf{w}$$. Tổng các hàm lồi trên toàn tập dữ liệu vẫn là hàm lồi, do đó mọi cực tiểu cục bộ đều là cực tiểu toàn cục. Nếu cộng thêm hạng tử chính quy hóa L2 $$\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$$ với $$\lambda>0$$, Hessian trở thành xác định dương chặt và bài toán có nghiệm cực tiểu duy nhất. Khi ghép thêm các lớp ẩn phi tuyến trong mạng sâu, tính lồi này không còn được bảo toàn.

### Q5. Ma trận Hessian dưới góc độ chẩn đoán

Hãy nêu một hiện tượng khi huấn luyện có thể giải thích bằng phần tử *đường chéo* của ma trận Hessian (hoặc moment bậc hai theo từng tham số), và một lý do khiến ta không chạy phương pháp Newton đầy đủ trên các mạng CNN hiện đại.

**Gợi ý.** Liên hệ khái niệm "hướng dốc và hướng phẳng" từ Q5 Chương 00 cùng độ phức tạp lưu trữ $$n_{\mathrm{params}}^2$$.

**Thảo luận.** Một phần tử đường chéo lớn cho thấy hàm mất mát có độ cong rất gắt theo tọa độ đó: một tốc độ học phù hợp cho các hướng phẳng hơn sẽ dễ gây dao động hoặc vượt quá cực tiểu theo hướng này — đây là động lực cho các bộ tối ưu thích nghi như RMSprop và Adam (Chương 10). Phương pháp Newton đầy đủ đòi hỏi tạo và giải hệ phương trình tuyến tính với ma trận Hessian kích thước $$P\times P$$; với hàng triệu tham số, ngay cả việc lưu trữ ma trận này cũng bất khả thi, và kể cả phép nhân Hessian–vector cũng chỉ được dùng trong các tình huống chuyên biệt chứ không làm bước cập nhật mặc định.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Bạn có thể tải bản PDF từ arXiv để xem đầy đủ bộ câu hỏi và lời giải. Trang này là bài luyện tập gốc của khóa học, không phải bản sao chép từ sách.

## Video tham khảo

- **Xem trước:** [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, tiếng Anh. Hữu ích cho các câu hỏi về vi phân tự động (autodiff).
- [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown, tiếng Anh. Ôn lại quy tắc dây chuyền vô hướng cần dùng khi chứng minh các công thức đạo hàm.
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) — StatQuest with Josh Starmer, tiếng Anh. Ôn lại hồi quy logistic — nhóm chủ đề thứ ba trong bài luyện này.
