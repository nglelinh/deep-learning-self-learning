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

# Tùy chọn: luyện phỏng vấn — lý thuyết thông tin và giải tích

> Bài này là **tùy chọn**. Nó **không** thay thế các ghi chú giải tích, đại số tuyến tính, hay xác suất. Sau những bài đó, dùng bài luyện này, rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để làm thêm bộ đã giải cùng chủ đề.

**Chủ đề sách cho chương này (đọc trong PDF, không phải ở đây):** lý thuyết thông tin — entropy, phân kỳ KL, thông tin tương hỗ — và giải tích / trực giác vi phân tự động.

Hub: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu luyện (gốc)

### Q1. Tách cross-entropy

Cho $$p$$ là nhãn one-hot trên $$K$$ lớp và $$q$$ là dự đoán softmax. Chứng minh

$$\mathrm{CE}(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$$

và giải thích vì sao $$H(p) = 0$$ trong thiết lập giám sát thông thường.

**Gợi ý.** Khai triển $$D_{\mathrm{KL}}(p\|q) = \sum_k p_k \log(p_k/q_k)$$ và dùng $$0\log 0 := 0$$.

**Thảo luận.** $$\mathrm{CE}(p,q) = -\sum_k p_k\log q_k$$. KL cộng thêm $$H(p) = -\sum_k p_k\log p_k$$, bằng không khi $$p$$ là khối điểm. Huấn luyện vì thế cực tiểu KL tới one-hot thực nghiệm, dù ta cài CE. Với mục tiêu không one-hot (làm mượt nhãn), $$H(p)$$ là hằng dương có thể bỏ qua khi lấy gradient nhưng không nên bỏ qua khi đọc số CE báo cáo.

### Q2. Thông tin tương hỗ sau bộ phân loại hoàn hảo

Cho $$Y$$ là nhãn nhị phân cân bằng và $$\hat Y$$ là dự đoán cứng. $$I(Y;\hat Y)$$ ra sao nếu bộ phân loại hoàn hảo? Nếu nó luôn ra lớp đa số?

**Gợi ý.** $$I(Y;\hat Y) = H(Y) - H(Y\mid\hat Y)$$.

**Thảo luận.** Dự đoán hoàn hảo khiến $$H(Y\mid\hat Y) = 0$$, nên $$I(Y;\hat Y) = H(Y) = 1$$ bit khi $$Y$$ công bằng. Bộ dự đoán hằng độc lập với $$Y$$, thông tin tương hỗ bằng $$0$$. Accuracy có thể “cao” trên tập lệch trong khi $$I(Y;\hat Y)$$ gần không — một đối lập hay gặp khi phỏng vấn.

### Q3. Một gradient theo quy tắc chuỗi

Cho $$f(\mathbf{w}) = \sigma(\mathbf{w}^\top\mathbf{x})$$ với $$\sigma(z) = (1+e^{-z})^{-1}$$ và $$\mathbf{x}$$ cố định. Viết $$\nabla_{\mathbf{w}} f$$ dạng cài được trong một dòng.

**Gợi ý.** $$\sigma'(z) = \sigma(z)\bigl(1-\sigma(z)\bigr)$$.

**Thảo luận.** $$\nabla_{\mathbf{w}} f = \sigma(\mathbf{w}^\top\mathbf{x})\bigl(1-\sigma(\mathbf{w}^\top\mathbf{x})\bigr)\,\mathbf{x}$$. Đây là hệ số địa phương cũng xuất hiện trong hồi quy logistic (Chương 03). Autodiff chiều ngược tính nó từ vô hướng $$f$$ đi lùi; không cần lập Jacobian của $$\mathbf{w}$$ tường minh.

### Q4. Vì sao chiều ngược thắng khi loss vô hướng

Một mạng ánh xạ $$\mathbb{R}^n\to\mathbb{R}$$ (một loss). Vì sao autodiff chiều ngược khoảng “một lượt tới + một lượt lùi,” gần như không phụ thuộc $$n$$, trong khi chiều xuôi tỉ lệ với $$n$$?

**Gợi ý.** Chiều xuôi gieo một hướng đầu vào mỗi lần; chiều ngược gieo đầu ra vô hướng một lần.

**Thảo luận.** Chiều xuôi tính tích Jacobian–vector $$J\mathbf{v}$$. Để lấy mọi đạo hàm riêng của một vô hướng cần $$n$$ tích như vậy (cơ sở chuẩn). Chiều ngược tính $$\mathbf{u}^\top J$$; với $$\mathbf{u}=1$$ bạn được cả gradient trong một lần quét. Đó là lý do framework học sâu cài chiều ngược (backprop). Chiều xuôi vẫn thắng khi nhiều đầu ra và ít đầu vào.

### Q5. Hessian như độ cong địa phương

Với $$f(\mathbf{x}) = \tfrac12\mathbf{x}^\top A\mathbf{x}$$ và $$A$$ đối xứng xác định dương, $$\nabla^2 f$$ là gì? Số điều kiện lớn $$\kappa(A)$$ hiện ra thế nào trong gradient descent?

**Gợi ý.** Hình ảnh Taylor trong ghi chú giải tích: Hessian là số hạng bậc hai.

**Thảo luận.** $$\nabla^2 f = A$$. Gradient descent zigzag dọc thung lũng hẹp khi $$\kappa(A)=\lambda_{\max}/\lambda_{\min}$$ lớn: bước an toàn trên trục dốc thì quá nhỏ trên trục phẳng. Người phỏng vấn thường muốn câu hình học này trước khi hỏi Adam hay Newton.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Tải PDF từ arXiv để xem toàn bộ Q&A đã giải. Trang này là bài luyện gốc của khóa, không phải bản in lại.
