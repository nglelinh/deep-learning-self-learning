---
layout: post
title: 02-98 Luyện phỏng vấn (neuron và hàm kích hoạt)
chapter: '02'
order: 11
owner: Deep Learning Course
lang: vi
categories:
- chapter02
lesson_type: optional
---

# Tùy chọn: luyện phỏng vấn — neuron và hàm kích hoạt

> Bài này là **tùy chọn**. Nó **không** viết lại lý thuyết perceptron, kiến trúc, hay hàm kích hoạt. Sau các ghi chú đó, dùng bài luyện này, rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để làm các câu đã giải về perceptron, kích hoạt, và phần kiến trúc “deep learning mở rộng.”

**Chủ đề sách cho chương này (đọc trong PDF, không phải ở đây):** perceptron, hàm kích hoạt, và cấu trúc MLP sơ cấp trong vùng chủ đề học sâu kéo dài.

Hub: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu luyện (gốc)

### Q1. Độ sâu khi không có phi tuyến

Bạn xếp ba tầng $$\mathbf{h} = W_3W_2W_1\mathbf{x}$$ không hàm kích hoạt. Chứng minh tồn tại $$W$$ sao cho $$\mathbf{h}=W\mathbf{x}$$. Điều đó nói gì về “thêm tầng” như cách làm giàu lớp giả thuyết?

**Gợi ý.** Nhân ma trận kết hợp được.

**Thảo luận.** $$W := W_3W_2W_1$$ là một ánh xạ tuyến tính. Thêm tầng tuyến tính không mở rộng tập hàm biểu diễn được; chúng chỉ phân tích cùng một ma trận (và có thể làm tối ưu khó hơn). Phi tuyến mới tạo mô hình sâu thật sự. Người phỏng vấn dùng câu này để kiểm tra “nhiều tầng hơn” không tự động là “mạnh hơn.”

### Q2. Bất biến tịnh tiến của softmax

Chứng minh $$\mathrm{softmax}(\mathbf{z}+c\mathbf{1}) = \mathrm{softmax}(\mathbf{z})$$ với mọi vô hướng $$c$$. Vì sao framework thường trừ $$\max_i z_i$$ trước hàm mũ?

**Gợi ý.** Tách $$e^{c}$$ ra khỏi mọi số hạng.

**Thảo luận.** Tử và mẫu đều nhân cùng $$e^{c}$$, tỉ số không đổi. Trừ max chính là đồng nhất thức với $$c=-\max z_i$$; giữ $$e^{z_i+c}\le 1$$ và tránh tràn. Bất biến này cũng nghĩa là một bias chung mọi logit không xác định được — chỉ *hiệu* logit mới có ý nghĩa.

### Q3. Nơi sigmoid bão hòa

Với $$z=10$$ và $$z=-10$$, ước lượng $$\sigma'(z)$$ khi $$\sigma(z)=(1+e^{-z})^{-1}$$. Vì sao điều đó thành vấn đề trong chồng sigmoid sâu?

**Gợi ý.** $$\sigma'(z)=\sigma(z)(1-\sigma(z))$$ đạt đỉnh $$1/4$$ tại $$z=0$$.

**Thảo luận.** $$\sigma(10)\approx 1$$ nên $$\sigma'(10)\approx 0$$; tương tự tại $$-10$$. Backprop nhân các hệ số này. Một chuỗi sigmoid bão hòa đẩy gradient ẩn về không (câu chuyện vanishing-gradient cổ điển). ReLU tránh bão hòa phía dương, đánh đổi bằng không cứng phía âm (câu sau).

### Q4. ReLU chết

Một đơn vị ẩn tính $$\mathrm{ReLU}(\mathbf{w}^\top\mathbf{x}+b)$$. Đưa điều kiện đơn giản trên $$(\mathbf{w},b)$$ và đám mây dữ liệu để đơn vị luôn bằng $$0$$ trên mọi điểm huấn luyện, và vẫn thế sau mọi bước SGD.

**Gợi ý.** Nếu tiền kích hoạt âm trên cả batch, gradient theo $$(\mathbf{w},b)$$ bằng không.

**Thảo luận.** Nếu $$\mathbf{w}^\top\mathbf{x}+b<0$$ với mọi $$\mathbf{x}$$ huấn luyện, ReLU và gradient địa phương đồng nhất không, nên SGD không bao giờ dịch đơn vị đó. Bias âm lớn hoặc khởi tạo xấu cộng learning rate cao có thể đẩy nhiều đơn vị vào trạng thái này. Leaky ReLU / GELU là cách giảm nhẹ thông dụng; bước nhỏ hơn sau khi loss nhảy cũng vậy.

### Q5. Đếm tham số MLP rộng

Một MLP ánh xạ $$784\to 256\to 256\to 10$$ có bias. Bao nhiêu vô hướng được huấn luyện? Điều gì đổi nếu bạn chèn một khối residual $$256$$ đơn vị vẫn là affine-cộng-ReLU?

**Gợi ý.** Tầng $$\mathbb{R}^{m}\to\mathbb{R}^{n}$$ có $$nm+n$$ tham số.

**Thảo luận.** Tầng một $$784\cdot 256+256$$, hai tầng ẩn $$256\cdot 256+256$$ mỗi tầng, tầng cuối $$256\cdot 10+10$$. Tổng $$784\cdot256 + 2\cdot256^{2} + 256\cdot10 + (256+256+10)$$. Khối residual cùng độ rộng thêm $$256^{2}+256$$ (plus skip, không thêm trọng số nếu chiều khớp). Phỏng vấn thường muốn con số *và* nhận xét rằng skip identity tự nó không thêm tham số.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Tải PDF từ arXiv để xem toàn bộ Q&A đã giải. Trang này là bài luyện gốc của khóa, không phải bản in lại.
