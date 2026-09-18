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

# Tùy chọn: luyện phỏng vấn — autodiff, Hessian, logistic

> Bài này là **tùy chọn**. Nó **không** thay thế lý thuyết loss, gradient descent, hay lan truyền ngược. Sau các ghi chú đó, dùng bài luyện này, rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để làm các câu đã giải về vi phân thuật toán, độ cong, và hồi quy logistic.

**Chủ đề sách cho chương này (đọc trong PDF, không phải ở đây):** vi phân thuật toán (tự động), trực giác Hessian, và phân loại logistic / log-odds.

Hub: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu luyện (gốc)

### Q1. Gradient logistic trong một dòng

Hồi quy logistic nhị phân dự đoán $$\hat y=\sigma(\mathbf{w}^\top\mathbf{x})$$ và dùng $$\ell= -y\log\hat y-(1-y)\log(1-\hat y)$$. Chứng minh

$$\nabla_{\mathbf{w}}\ell = (\hat y-y)\,\mathbf{x}.$$

**Gợi ý.** Dùng $$\sigma'(z)=\sigma(z)(1-\sigma(z))$$ và quy tắc chuỗi từ Chương 00 / ghi chú backprop.

**Thảo luận.** Đặt $$z=\mathbf{w}^\top\mathbf{x}$$. Khi đó $$\partial\ell/\partial z = -\frac{y}{\hat y}\sigma'(z)+\frac{1-y}{1-\hat y}\sigma'(z)$$. Thay $$\sigma'=\hat y(1-\hat y)$$ thu gọn thành $$\hat y-y$$. Nhân $$\partial z/\partial\mathbf{w}=\mathbf{x}$$ được điều phải chứng minh. Đó là lý do “sigmoid + BCE” số học ổn hơn khi fuse: bạn không bao giờ materialize một $$\sigma'$$ đơn lẻ bị underflow.

### Q2. Odds, log-odds, và một hệ số

Mô hình logistic khớp được $$z = -1 + 2\,x_{\mathrm{smoke}}$$ với $$x_{\mathrm{smoke}}\in\{0,1\}$$. Odds của $$y=1$$ cho người không hút so với người hút là bao nhiêu? Số $$2$$ nghĩa là gì trên thang log-odds?

**Gợi ý.** Odds là $$\hat y/(1-\hat y)=e^{z}$$.

**Thảo luận.** Không hút: $$z=-1$$, odds $$e^{-1}\approx 0.37$$. Hút: $$z=1$$, odds $$e^{1}\approx 2.72$$. Hệ số $$2$$ là *tỉ số log-odds* của đặc trưng nhị phân: hút nhân odds với $$e^{2}\approx 7.4$$. Câu này được thích vì nó kiểm tra bạn không coi trọng số logistic như hiệu ứng xác suất tuyến tính.

### Q3. Chiều xuôi và chiều ngược trên đồ thị nhỏ

Bạn cần $$\partial f/\partial x$$ và $$\partial f/\partial y$$ với $$f=(x y)+ \sin(x)$$. Đếm phép nhân vô hướng / gọi hàm siêu việt trong (a) hai lượt xuôi và (b) một lượt ngược. Bạn chọn gì nếu sau đó $$f$$ thành mạng 50 tầng với hàng triệu trọng số?

**Gợi ý.** Chiều xuôi gieo một đầu vào; chiều ngược gieo đầu ra.

**Thảo luận.** Hai lượt xuôi: gieo $$x$$ rồi $$y$$, mỗi lần phát lại $$xy$$ và $$\sin x$$. Một lượt ngược lưu giá trị tới và gửi adjoint ngược qua $$+$$, $$\times$$, và $$\sin$$. Với loss vô hướng và $$\mathbf{w}$$ khổng lồ, chiều ngược (backprop) là lựa chọn thực tế duy nhất — đó là cả ý của Chương 03. Chiều xuôi vẫn hữu ích khi ít đầu vào và nhiều đầu ra (ví dụ Jacobian của một tầng vector).

### Q4. Hessian logistic và tính lồi

Với một mẫu, $$\ell(\mathbf{w})= -y\log\sigma(\mathbf{w}^\top\mathbf{x})-(1-y)\log\bigl(1-\sigma(\mathbf{w}^\top\mathbf{x})\bigr)$$. Chứng minh

$$\nabla^2_{\mathbf{w}}\ell = \hat y(1-\hat y)\,\mathbf{x}\mathbf{x}^\top$$

nửa xác định dương. Điều đó nói gì về cực tiểu địa phương của hồi quy logistic không chính quy?

**Gợi ý.** $$\mathbf{v}^\top(\mathbf{x}\mathbf{x}^\top)\mathbf{v}=(\mathbf{v}^\top\mathbf{x})^2\ge 0$$ và $$\hat y(1-\hat y)\in(0,1)$$.

**Thảo luận.** Hessian là bội không âm của ma trận PSD hạng 1, nên $$\ell$$ lồi theo $$\mathbf{w}$$. Tổng trên tập dữ liệu vẫn lồi. Mọi cực tiểu địa phương là toàn cục. Thêm $$\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$$ làm Hessian xác định dương khi $$\lambda>0$$. Mạng sâu mất bảo đảm này ngay khi ghép các tầng ẩn phi tuyến.

### Q5. Hessian như chẩn đoán, không phải bộ giải

Nêu một triệu chứng huấn luyện mà Hessian *đường chéo* (hoặc moment bậc hai theo từng tham số) giải thích được, và một lý do ta vẫn không chạy Newton đầy đủ trên CNN hiện đại.

**Gợi ý.** Nghĩ “hướng dốc và hướng phẳng” từ Q5 Chương 00, và nghĩ $$n_{\mathrm{params}}^2$$.

**Thảo luận.** Phần tử đường chéo lớn nghĩa là tọa độ đó sắc: learning rate ổn chỗ khác sẽ vượt ở đây — câu chuyện của RMSprop / Adam (Chương 10). Newton đầy đủ cần Hessian $$P\times P$$ (hoặc giải tuyến tính với nó). Với hàng triệu tham số không thể lưu; thậm chí tích Hessian–vector chỉ dùng dè dặt, không phải bước mặc định.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Tải PDF từ arXiv để xem toàn bộ Q&A đã giải. Trang này là bài luyện gốc của khóa, không phải bản in lại.
