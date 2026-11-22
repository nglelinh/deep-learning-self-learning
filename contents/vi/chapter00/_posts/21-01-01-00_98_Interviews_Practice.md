---
layout: post
title: 00-98 Luyện phỏng vấn (lý thuyết thông tin và giải tích)
chapter: '00'
order: 25
owner: Deep Learning Course
lang: vi
categories:
- chapter00
lesson_type: optional
---

# Tùy chọn: Luyện phỏng vấn — Lý thuyết thông tin và giải tích

> Bài này là phần **tùy chọn**, **không** thay thế các bài học cốt lõi về giải tích, đại số tuyến tính và xác suất. Sau khi hoàn thành những bài đó, bạn có thể luyện tập với các câu hỏi dưới đây, rồi tham khảo thêm *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để xem thêm ngân hàng câu hỏi đã có lời giải cùng chủ đề.

**Chủ đề tương ứng trong sách (đọc trực tiếp trong bản PDF):** lý thuyết thông tin — entropy, phân kỳ KL, thông tin tương hỗ — cùng giải tích và trực giác về vi phân tự động (*automatic differentiation*).

Trang tổng hợp lộ trình: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Bộ câu hỏi luyện tập (biên soạn gốc)

### Q1. Phân tách cross-entropy

Cho $$p$$ là nhãn one-hot trên $$K$$ lớp và $$q$$ là phân phối dự đoán từ softmax. Hãy chứng minh

$$\mathrm{CE}(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$$

và giải thích vì sao $$H(p) = 0$$ trong thiết lập học có giám sát thông thường.

**Gợi ý.** Khai triển $$D_{\mathrm{KL}}(p\|q) = \sum_k p_k \log(p_k/q_k)$$ và áp dụng quy ước $$0\log 0 := 0$$.

**Thảo luận.** Ta có $$\mathrm{CE}(p,q) = -\sum_k p_k\log q_k$$. Phân kỳ KL chênh lệch với cross-entropy đúng một lượng $$H(p) = -\sum_k p_k\log p_k$$, và lượng này bằng 0 khi $$p$$ là phân phối one-hot (tập trung toàn bộ xác suất tại một lớp). Do đó, dù ta cài đặt hàm mất mát là cross-entropy, quá trình huấn luyện thực chất đang cực tiểu hóa phân kỳ KL giữa phân phối dự đoán và phân phối thực nghiệm one-hot. Khi dùng nhãn mềm (như *label smoothing*), $$H(p)$$ là một hằng số dương — không ảnh hưởng đến gradient theo $$q$$, nhưng cần lưu ý khi đọc giá trị cross-entropy báo cáo.

### Q2. Thông tin tương hỗ của bộ phân loại hoàn hảo

Cho $$Y$$ là nhãn nhị phân cân bằng và $$\hat Y$$ là nhãn dự đoán cứng. Giá trị $$I(Y;\hat Y)$$ bằng bao nhiêu nếu bộ phân loại dự đoán hoàn hảo, và bằng bao nhiêu nếu nó luôn dự đoán lớp đa số?

**Gợi ý.** Dùng đẳng thức $$I(Y;\hat Y) = H(Y) - H(Y\mid\hat Y)$$.

**Thảo luận.** Khi bộ phân loại dự đoán hoàn hảo, biết $$\hat Y$$ xác định hoàn toàn $$Y$$ nên $$H(Y\mid\hat Y) = 0$$, suy ra $$I(Y;\hat Y) = H(Y) = 1$$ bit (vì $$Y$$ cân bằng). Ngược lại, bộ dự đoán hằng độc lập với $$Y$$ nên thông tin tương hỗ bằng $$0$$. Trên tập dữ liệu mất cân bằng, độ chính xác (*accuracy*) vẫn có thể rất cao dù $$I(Y;\hat Y)$$ gần bằng 0 — đây là tình huống đối lập thường gặp trong phỏng vấn.

### Q3. Tính gradient bằng quy tắc dây chuyền

Cho $$f(\mathbf{w}) = \sigma(\mathbf{w}^\top\mathbf{x})$$ với $$\sigma(z) = (1+e^{-z})^{-1}$$ và $$\mathbf{x}$$ cố định. Viết biểu thức $$\nabla_{\mathbf{w}} f$$ dưới dạng gọn để cài đặt trực tiếp trong một dòng mã.

**Gợi ý.** Sử dụng $$\sigma'(z) = \sigma(z)\bigl(1-\sigma(z)\bigr)$$.

**Thảo luận.** Ta có $$\nabla_{\mathbf{w}} f = \sigma(\mathbf{w}^\top\mathbf{x})\bigl(1-\sigma(\mathbf{w}^\top\mathbf{x})\bigr)\,\mathbf{x}$$. Hệ số đạo hàm cục bộ này cũng xuất hiện trong hồi quy logistic (Chương 03). Vi phân tự động chế độ ngược (*reverse-mode autodiff*) tính gradient đi ngược từ đầu ra vô hướng $$f$$ mà không cần dựng tường minh ma trận Jacobian theo $$\mathbf{w}$$.

### Q4. Vì sao chế độ ngược hiệu quả hơn khi hàm mất mát là vô hướng

Xét một mạng nơ-ron ánh xạ $$\mathbb{R}^n\to\mathbb{R}$$ (một giá trị mất mát duy nhất). Vì sao vi phân tự động chế độ ngược chỉ tốn chi phí cỡ “một lượt xuôi + một lượt ngược” và gần như không phụ thuộc $$n$$, trong khi chế độ xuôi có chi phí tỉ lệ với $$n$$?

**Gợi ý.** Chế độ xuôi lan truyền từng hướng đầu vào một; chế độ ngược lan truyền từ đầu ra vô hướng duy nhất.

**Thảo luận.** Chế độ xuôi (*forward-mode*) tính tích Jacobian–vector $$J\mathbf{v}$$. Để thu đủ mọi đạo hàm riêng của một hàm vô hướng theo $$n$$ biến đầu vào, cần thực hiện $$n$$ lượt tính với các vector cơ sở chuẩn. Ngược lại, chế độ ngược (*reverse-mode*) tính tích vector–Jacobian $$\mathbf{u}^\top J$$; với $$\mathbf{u}=1$$, chỉ cần một lượt quét ngược là thu được toàn bộ vector gradient. Đây là lý do các framework học sâu cài đặt chế độ ngược (lan truyền ngược). Chế độ xuôi chỉ lợi thế hơn khi số đầu ra lớn hơn nhiều so với số đầu vào.

### Q5. Ma trận Hessian và độ cong cục bộ

Với $$f(\mathbf{x}) = \tfrac12\mathbf{x}^\top A\mathbf{x}$$ và $$A$$ là ma trận đối xứng xác định dương, ma trận Hessian $$\nabla^2 f$$ bằng gì? Số điều kiện $$\kappa(A)$$ lớn ảnh hưởng thế nào đến thuật toán gradient descent?

**Gợi ý.** Liên hệ với khai triển Taylor bậc hai trong bài giải tích: Hessian là ma trận của số hạng bậc hai.

**Thảo luận.** Ta có $$\nabla^2 f = A$$. Khi số điều kiện $$\kappa(A)=\lambda_{\max}/\lambda_{\min}$$ lớn, các đường mức bị kéo dài thành thung lũng hẹp khiến gradient descent dao động zic-zac: độ dài bước đủ an toàn để không phân kỳ theo trục dốc lại quá nhỏ để tiến nhanh dọc theo trục phẳng. Trong phỏng vấn, câu trả lời mang trực giác hình học này thường là bước đệm trước khi chuyển sang các câu hỏi về Adam hoặc phương pháp Newton.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Bạn có thể tải bản PDF từ arXiv để xem đầy đủ bộ câu hỏi và lời giải. Trang này là bài luyện tập biên soạn riêng cho khóa học, không phải bản sao lại từ sách.

## Video tham khảo

- **Xem trước:** [MaDL - Information and Entropy](https://www.youtube.com/watch?v=pXh53lqPkHs) — Tübingen Machine Learning, tiếng Anh. Trình bày khái niệm entropy dùng trực tiếp trong các câu hỏi lý thuyết thông tin của bài.
- [MaDL - Kullback-Leibler Divergence](https://www.youtube.com/watch?v=vPvnpC0PNVQ) — Tübingen Machine Learning, tiếng Anh. Giải thích phân kỳ KL — đại lượng lý thuyết thông tin trọng tâm thứ hai trong bài luyện tập.
- [The paradox of the derivative | Chapter 2, Essence of calculus](https://www.youtube.com/watch?v=9vKqVkMQHKk) — 3Blue1Brown, tiếng Anh. Củng cố trực giác về đạo hàm, nền tảng cho các câu hỏi giải tích và vi phân tự động.
