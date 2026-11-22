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

# Tùy chọn: Luyện phỏng vấn — Neuron và Hàm kích hoạt

> Bài này là **tùy chọn** và **không** lặp lại phần lý thuyết về perceptron, kiến trúc hay hàm kích hoạt. Sau khi học các bài chính trong chương, bạn có thể dùng bộ câu hỏi dưới đây để tự kiểm tra, rồi tham khảo thêm *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để xem các câu hỏi kèm lời giải đầy đủ về perceptron, hàm kích hoạt và kiến trúc MLP.

**Chủ đề tham khảo trong sách cho chương này (đọc trực tiếp trong bản PDF):** perceptron, hàm kích hoạt và cấu trúc MLP cơ bản.

Trang tổng hợp: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu hỏi luyện tập

### Q1. Độ sâu khi không có tính phi tuyến

Giả sử bạn xếp ba lớp tuyến tính $$\mathbf{h} = W_3W_2W_1\mathbf{x}$$ không kèm hàm kích hoạt. Hãy chứng minh tồn tại một ma trận $$W$$ sao cho $$\mathbf{h}=W\mathbf{x}$$. Kết quả này nói lên điều gì về việc "thêm lớp" để tăng năng lực biểu diễn?

**Gợi ý.** Phép nhân ma trận có tính chất kết hợp.

**Thảo luận.** Đặt $$W := W_3W_2W_1$$, ta thu được một ánh xạ tuyến tính duy nhất. Việc thêm các lớp tuyến tính không mở rộng tập hàm mà mạng có thể biểu diễn — chúng chỉ phân tích cùng một phép biến đổi tuyến tính thành tích nhiều ma trận (và đôi khi còn làm việc tối ưu khó hơn). Chính hàm kích hoạt phi tuyến mới tạo nên sức mạnh của mạng sâu. Trong phỏng vấn, câu hỏi này thường dùng để kiểm tra nhận thức rằng "nhiều lớp hơn" không đồng nghĩa với "mô hình mạnh hơn" nếu thiếu phi tuyến.

### Q2. Tính bất biến tịnh tiến của softmax

Chứng minh $$\mathrm{softmax}(\mathbf{z}+c\mathbf{1}) = \mathrm{softmax}(\mathbf{z})$$ với mọi hằng số vô hướng $$c$$. Vì sao các framework thường trừ $$\max_i z_i$$ trước khi tính hàm mũ?

**Gợi ý.** Đưa thừa số $$e^{c}$$ ra ngoài ở cả tử và mẫu.

**Thảo luận.** Cả tử số và mẫu số đều nhân thêm cùng một hằng số dương $$e^{c}$$ nên tỷ số không thay đổi. Việc trừ đi giá trị lớn nhất chính là chọn $$c=-\max_i z_i$$, giúp đảm bảo mọi số hạng $$e^{z_i+c}\le 1$$ và tránh tràn số (*overflow*). Tính bất biến này cũng cho thấy một độ lệch cộng đều vào mọi logit không làm thay đổi xác suất — chỉ *hiệu* giữa các logit mới có ý nghĩa.

### Q3. Vùng bão hòa của sigmoid

Với $$z=10$$ và $$z=-10$$, hãy ước lượng $$\sigma'(z)$$ khi $$\sigma(z)=(1+e^{-z})^{-1}$$. Vì sao hiện tượng này gây vấn đề trong mạng sâu dùng nhiều lớp sigmoid?

**Gợi ý.** $$\sigma'(z)=\sigma(z)(1-\sigma(z))$$ đạt giá trị lớn nhất là $$1/4$$ tại $$z=0$$.

**Thảo luận.** Tại $$z=10$$, $$\sigma(10)\approx 1$$ nên $$\sigma'(10)\approx 0$$; tương tự tại $$z=-10$$, $$\sigma(-10)\approx 0$$ nên $$\sigma'(-10)\approx 0$$. Khi lan truyền ngược, các hệ số đạo hàm này bị nhân liên tiếp qua nhiều lớp. Một chuỗi các lớp sigmoid rơi vào vùng bão hòa sẽ kéo gradient ở các lớp đầu về gần 0 (hiện tượng gradient biến mất kinh điển). ReLU khắc phục tình trạng bão hòa ở miền dương, đổi lại có vùng bằng 0 cứng ở miền âm (xem câu sau).

### Q4. Hiện tượng ReLU chết

Một neuron ẩn tính $$\mathrm{ReLU}(\mathbf{w}^\top\mathbf{x}+b)$$. Hãy nêu một điều kiện đơn giản trên $$(\mathbf{w},b)$$ và tập dữ liệu huấn luyện để neuron này luôn cho ra $$0$$ trên mọi mẫu và giữ nguyên trạng thái đó sau mọi bước SGD.

**Gợi ý.** Nếu giá trị tiền kích hoạt âm trên toàn bộ batch thì gradient theo $$(\mathbf{w},b)$$ đều bằng 0.

**Thảo luận.** Nếu $$\mathbf{w}^\top\mathbf{x}+b<0$$ với mọi mẫu $$\mathbf{x}$$ trong tập huấn luyện, cả đầu ra ReLU lẫn đạo hàm cục bộ của nó đều bằng 0, nên SGD sẽ không bao giờ cập nhật $$(\mathbf{w},b)$$. Độ lệch khởi tạo quá âm, khởi tạo trọng số kém hoặc tốc độ học quá lớn có thể đẩy nhiều neuron vào trạng thái này. Cách giảm thiểu thường dùng là chuyển sang Leaky ReLU hoặc GELU, đồng thời kiểm soát tốc độ học khi hàm mất mát tăng đột biến.

### Q5. Đếm số tham số của MLP

Một mạng MLP có cấu trúc $$784\to 256\to 256\to 10$$ (có độ lệch ở mọi lớp). Hỏi có bao nhiêu tham số vô hướng cần huấn luyện? Con số này thay đổi thế nào nếu chèn thêm một khối residual cùng độ rộng $$256$$ (gồm một lớp affine kèm ReLU và đường nối tắt đồng nhất)?

**Gợi ý.** Một lớp ánh xạ $$\mathbb{R}^{m}\to\mathbb{R}^{n}$$ có $$nm+n$$ tham số.

**Thảo luận.** Lớp thứ nhất ($$784 \to 256$$) có $$784\cdot 256+256$$ tham số; lớp thứ hai ($$256 \to 256$$) có $$256\cdot 256+256$$ tham số; lớp đầu ra ($$256 \to 10$$) có $$256\cdot 10+10$$ tham số. Tổng cộng là $$784\cdot 256 + 256^{2} + 256\cdot 10 + (256+256+10) = 269{,}322$$ tham số. Nếu chèn thêm một khối residual cùng độ rộng $$256$$ gồm một lớp tuyến tính cộng đường nối tắt đồng nhất, số tham số tăng thêm đúng $$256^{2}+256 = 65{,}792$$ (vì đường nối tắt đồng nhất không thêm trọng số khi hai đầu cùng kích thước). Trong phỏng vấn, người hỏi thường muốn nghe cả cách tính cụ thể lẫn nhận xét rằng nhánh skip đồng nhất không làm tăng tham số.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Bạn có thể tải bản PDF từ arXiv để tham khảo đầy đủ các câu hỏi và lời giải. Trang này là bộ bài tập luyện gốc của khóa học, không phải bản sao chép từ sách.

## Video tham khảo

- **Xem trước:** [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Nên xem trước khi làm các câu hỏi về neuron và hàm kích hoạt.
- [Neural Networks Pt. 3: ReLU In Action!!!](https://www.youtube.com/watch?v=68BZ5f7P94E) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trực quan điểm gãy của ReLU — chủ đề rất hay xuất hiện khi phỏng vấn.
- [Why Non-linear Activation Functions (C1W3L07)](https://www.youtube.com/watch?v=NkOv_k7r6no) — DeepLearningAI, tiếng Anh. Giải thích rõ vì sao hàm kích hoạt phải phi tuyến (liên quan trực tiếp tới câu Q1).
