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

# Tùy chọn: Luyện phỏng vấn — CNN và Đặc trưng Sớm

> Bài này là **tùy chọn** và **không** lặp lại phần toán tích chập, tầng gộp hay lịch sử LeNet–ResNet. Sau khi học các bài chính trong chương, bạn có thể luyện tập với bộ câu hỏi dưới đây rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để tham khảo thêm các câu hỏi kèm lời giải đầy đủ về tích chập, kiến trúc CNN và trích xuất đặc trưng ban đầu.

**Chủ đề tham khảo trong sách cho chương này (đọc trực tiếp trong bản PDF):** hình học tích chập, kiến trúc CNN và phần đầu của chủ đề trích xuất đặc trưng. Phần luyện về học chuyển giao tiếp tục ở **15-98**.

Trang tổng hợp: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu hỏi luyện tập

### Q1. So sánh số tham số: tầng tích chập và tầng liên kết đầy đủ

Một tầng ánh xạ tensor kích thước $$32\times32\times3$$ thành bản đồ đặc trưng $$32\times32\times16$$. Hãy so sánh số tham số giữa (a) một tầng liên kết đầy đủ (*dense*) nhận đầu vào đã làm phẳng và (b) một tầng tích chập $$3\times3$$ với 16 kênh đầu ra, `same` padding, có bias. Trong buổi phỏng vấn, bạn nên nêu điểm nào trước?

**Gợi ý.** Công thức số tham số của tầng tích chập là $$k_h k_w c_{\mathrm{in}} c_{\mathrm{out}} + c_{\mathrm{out}}$$.

**Thảo luận.** Với tầng liên kết đầy đủ: đầu vào có $$32\cdot32\cdot3=3072$$ phần tử, đầu ra có $$32\cdot32\cdot16=16384$$ phần tử, nên tổng tham số là $$3072\cdot16384+16384$$ (hơn 50 triệu). Với tầng tích chập: chỉ cần $$3\cdot3\cdot3\cdot16+16=448$$ tham số. Khi trả lời phỏng vấn, nên nêu trực tiếp con số của tầng tích chập cùng công thức, sau đó giải thích *lý do* (kết nối cục bộ và chia sẻ trọng số) thay vì chỉ nói chung chung rằng CNN ít tham số hơn.

### Q2. Tính tay kích thước đầu ra

Cho đầu vào kích thước $$H\times W=28\times28$$, nhân $$5\times5$$, bước nhảy $$s=1$$, đệm $$p=0$$, và $$c_{\mathrm{out}}=8$$. Kích thước tensor đầu ra là bao nhiêu? Kết quả thay đổi thế nào nếu $$s=2$$ và $$p=2$$?

**Gợi ý.** Áp dụng công thức $$H'=\bigl\lfloor(H+2p-k)/s\bigr\rfloor+1$$ (tương tự cho $$W'$$).

**Thảo luận.** Trường hợp thứ nhất: $$H'=28-5+1=24$$, kích thước đầu ra là $$24\times24\times8$$. Trường hợp thứ hai: $$H'=\lfloor(28+4-5)/2\rfloor+1=\lfloor 27/2\rfloor+1=14$$, kích thước đầu ra là $$14\times14\times8$$. Khi trả lời, luôn ghi rõ quy ước sắp xếp chiều (*channels-last* hay *channels-first*) vì các framework dùng quy ước khác nhau.

### Q3. Hai tầng $$3\times3$$ so với một tầng $$5\times5$$

Hai tầng tích chập $$3\times3$$ xếp chồng (không pooling, stride 1, padding giữ kích thước) và một tầng tích chập $$5\times5$$ có cùng trường tiếp nhận lý thuyết trên đầu vào. Giả sử số kênh luôn là $$c$$, hãy nêu một lý do về số tham số và một lý do về năng lực biểu diễn để chọn cấu trúc xếp chồng.

**Gợi ý.** Khi stride bằng 1, mỗi tầng cộng thêm $$(k-1)$$ vào trường tiếp nhận.

**Thảo luận.** Một tầng $$5\times5$$ cần $$25c^2$$ trọng số, trong khi hai tầng $$3\times3$$ chỉ cần $$2 \times 9c^2 = 18c^2$$ trọng số và bổ sung thêm một hàm kích hoạt phi tuyến ở giữa, khiến phép biến đổi hợp thành không còn là một bộ lọc tuyến tính đơn thuần. Thiết kế của VGG dựa đúng trên lập luận này. (Nếu người phỏng vấn hỏi mở rộng sang depthwise-separable hay bottleneck $$1\times1$$, đó là câu hỏi nâng cao ngoài trọng tâm cơ bản của chương.)

### Q4. Đồng biến tịnh tiến (*equivariance*) và bất biến tịnh tiến (*invariance*)

Áp dụng một bộ lọc dò cạnh đứng lên ảnh, sau đó dịch ảnh sang phải 2 pixel rồi lọc lại. Bản đồ đặc trưng đầu ra sẽ thay đổi thế nào nếu phép toán là tích chập đúng nghĩa? Cần thêm phép toán gì nếu muốn *bộ phân loại* không bị ảnh hưởng bởi độ dịch đó?

**Gợi ý.** Tích chập giao hoán với phép tịnh tiến; gộp toàn cục (*global pooling*) loại bỏ thông tin vị trí không gian.

**Thảo luận.** Bản đồ đặc trưng đầu ra cũng sẽ dịch sang phải đúng 2 pixel — đây là tính *đồng biến tịnh tiến* (*translation equivariance*). Khi bổ sung tầng global average/max pooling (hoặc giảm mẫu đủ mạnh) ở cuối mạng, vector đặc trưng đưa vào lớp phân loại trở nên gần *bất biến tịnh tiến* (*translation invariance*) hơn. Nhầm lẫn giữa hai thuật ngữ này là lỗi rất phổ biến khi phỏng vấn.

### Q5. Vai trò của kết nối tắt đồng nhất

Trong một khối dư $$\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$$, hãy viết dạng tổng quát của đạo hàm $$\partial\mathbf{y}/\partial\mathbf{x}$$. Vì sao điều này giúp huấn luyện CNN 50 tầng hiệu quả hơn so với chỉ dùng ReLU?

**Gợi ý.** Chú ý hạng tử ma trận đơn vị $$I$$ từ nhánh tắt; $$\mathcal{F}$$ có thể nhỏ khi mới khởi tạo.

**Thảo luận.** Ta có $$\frac{\partial\mathbf{y}}{\partial\mathbf{x}} = I + \frac{\partial\mathcal{F}}{\partial\mathbf{x}}$$. Thành phần đơn vị $$I$$ duy trì một đường truyền không bị suy giảm cho gradient ngay cả khi $$\partial\mathcal{F}/\partial\mathbf{x}$$ nhỏ, nhờ đó tín hiệu đạo hàm đi qua được nhiều khối liên tiếp. Ngược lại, mạng thuần không có kết nối tắt (dù dùng ReLU) vẫn phải nhân liên tiếp nhiều ma trận Jacobian có thể làm gradient co nhỏ. Đây là lý do về mặt tối ưu kiến trúc, không phải khẳng định tuyệt đối rằng ResNet luôn cho độ chính xác cao nhất trong mọi bài toán.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Bạn có thể tải bản PDF từ arXiv để tham khảo đầy đủ bộ câu hỏi và lời giải. Trang này là bộ bài luyện gốc của khóa học, không phải bản sao chép từ sách.

## Video tham khảo

- **Xem trước:** [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Nên xem trước khi làm các câu hỏi tính toán về tích chập.
- [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, tiếng Anh. Ôn lại công thức tính kích thước đầu ra dùng cho câu Q2.
- [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, tiếng Anh. Hữu ích khi cần giải thích trực quan bộ lọc đang phát hiện đặc trưng gì.
