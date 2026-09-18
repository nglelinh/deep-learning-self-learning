---
layout: post
title: 04-98 Luyện phỏng vấn (CNN và đặc trưng sớm)
chapter: '04'
order: 15
owner: Deep Learning Course
lang: vi
categories:
- chapter04
lesson_type: optional
---

# Tùy chọn: luyện phỏng vấn — CNN và đặc trưng sớm

> Bài này là **tùy chọn**. Nó **không** viết lại toán tích chập, pooling, hay ghi chú LeNet–ResNet. Sau các bài đó, dùng bài luyện này, rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để làm các câu đã giải về tích chập, kiến trúc CNN, và tách đặc trưng ban đầu.

**Chủ đề sách cho chương này (đọc trong PDF, không phải ở đây):** hình học tích chập, kiến trúc CNN, và nửa đầu chủ đề tách đặc trưng. Luyện học chuyển giao tiếp tục ở **15-98**.

Hub: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu luyện (gốc)

### Q1. Đếm tham số conv so với dense

Một tầng ánh xạ tensor $$32\times32\times3$$ thành đặc trưng $$32\times32\times16$$. So (a) tầng dày đặc làm phẳng đầu vào và (b) tích chập $$3\times3$$ với 16 kênh ra, same padding, có bias. Số nào bạn nên nêu trước khi phỏng vấn?

**Gợi ý.** Conv: $$k_h k_w c_{\mathrm{in}} c_{\mathrm{out}} + c_{\mathrm{out}}$$.

**Thảo luận.** Dense: đầu vào $$32\cdot32\cdot3=3072$$, đầu ra $$32\cdot32\cdot16=16384$$, khoảng $$3072\cdot16384+16384$$ trọng số — hàng chục triệu. Conv: $$3\cdot3\cdot3\cdot16+16=448$$. Nêu số conv trước, rồi *lý do*: kết nối cục bộ và chia sẻ trọng số. Nhiều ứng viên trượt vì chỉ nói “CNN ít tham số hơn” mà không có công thức.

### Q2. Hình dạng đầu ra tính tay

Đầu vào $$H\times W=28\times28$$, kernel $$5\times5$$, stride $$s=1$$, padding $$p=0$$, $$c_{\mathrm{out}}=8$$. Shape tensor đầu ra? Nếu $$s=2$$ và $$p=2$$?

**Gợi ý.** $$H'=\bigl\lfloor(H+2p-k)/s\bigr\rfloor+1$$ (tương tự $$W$$).

**Thảo luận.** Trường hợp một: $$H'=28-5+1=24$$, shape $$24\times24\times8$$. Trường hợp hai: $$(28+4-5)/2+1=14$$, shape $$14\times14\times8$$. Luôn nêu layout (channels-last hay channels-first); các framework không thống nhất.

### Q3. Hai $$3\times3$$ so với một $$5\times5$$

Hai tích chập $$3\times3$$ xếp chồng (không pooling, stride 1, padding giữ kích thước) và một tích chập $$5\times5$$ có cùng trường tiếp nhận lý thuyết trên đầu vào. Đưa một lý do đếm tham số và một lý do biểu đạt để chọn chồng. Giả sử $$c$$ kênh suốt.

**Gợi ý.** Trường tiếp nhận cộng $$(k-1)$$ mỗi tầng khi stride bằng 1.

**Thảo luận.** Một $$5\times5$$: $$25c^2$$ trọng số. Hai $$3\times3$$: $$18c^2$$ trọng số, cộng thêm một phi tuyến ở giữa, nên ánh xạ hợp thành không còn là một bộ lọc tuyến tính. Chồng kiểu VGG dùng đúng lập luận này. Nếu người hỏi chuyển sang depthwise-separable hay bottleneck $$1\times1$$, đó là câu nối, không bắt buộc cho chương này.

### Q4. Equivariance và invariance

Một bộ lọc cạnh dọc được áp lên ảnh, rồi ảnh được cuốn hai pixel sang phải và lọc lại. Feature map phải đổi thế nào nếu cài đặt là tích chập đúng? Bạn thêm phép gì nếu *bộ phân loại* không được quan tâm tới độ dịch đó?

**Gợi ý.** Tích chập giao hoán với tịnh tiến; pooling toàn cục bỏ vị trí không gian.

**Thảo luận.** Feature map phải cuốn cùng hai pixel (*equivariance* với tịnh tiến). Global average / max pool cuối (hoặc downsample đủ mạnh) làm *vector* đưa vào đầu tuyến tính gần *bất biến* tịnh tiến hơn. Trộn hai từ này là lỗi phỏng vấn phổ biến.

### Q5. Skip đồng nhất để làm gì

Trong khối residual $$ \mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}) $$, viết $$\partial\mathbf{y}/\partial\mathbf{x}$$ dạng sơ đồ. Vì sao điều này giúp CNN 50 tầng hơn là “chỉ thêm ReLU”?

**Gợi ý.** Số $$+1$$ từ skip là câu chuyện; $$\mathcal{F}$$ có thể nhỏ lúc khởi tạo.

**Thảo luận.** $$\frac{\partial\mathbf{y}}{\partial\mathbf{x}} = I + \frac{\partial\mathcal{F}}{\partial\mathbf{x}}$$. Hạng identity giữ một đường có hệ số không biến mất khi $$\mathcal{F}$$ gần không, nên gradient đi được nhiều khối. Thêm ReLU không skip vẫn nhân nhiều Jacobian có thể co. Đây là câu trả lời kiến trúc, không phải tuyên bố “ResNet luôn thắng accuracy.”

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Tải PDF từ arXiv để xem toàn bộ Q&A đã giải. Trang này là bài luyện gốc của khóa, không phải bản in lại.
