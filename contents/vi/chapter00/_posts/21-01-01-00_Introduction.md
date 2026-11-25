---
layout: post
title: 00 Giới thiệu
chapter: '00'
order: 1
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Nếu ví một mạng nơ-ron sâu như một chiếc phi thuyền liên hành tinh, thì toán học nền tảng chính là bản thiết kế động cơ và hệ thống dẫn đường giúp phi thuyền không bao giờ lạc quỹ đạo. Nhiều người học Deep Learning vội vàng lao ngay vào gọi hàm `.backward()` trong PyTorch, để rồi lúng túng mỗi khi hàm mất mát đột ngột bùng nổ thành `NaN` hay mô hình đứng yên không chịu hội tụ. Chương 00 được thiết kế như tấm bản đồ sinh tồn toán học, kết nối trực giác hình học của giải tích, đại số tuyến tính, tôpô và xác suất với những dòng code thực chiến bạn sẽ viết xuyên suốt khóa học.

## Kiến thức cần nhớ: Bốn trụ cột toán học nâng đỡ mọi mô hình Học sâu

1. **Giải tích đa biến và tối ưu hóa (Calculus)**:
   Cung cấp ngôn ngữ để đo độ nhạy của sai số theo hàng triệu trọng số thông qua đạo hàm riêng, vector gradient và chuỗi Taylor—trái tim của thuật toán lan truyền ngược (*backpropagation*) và Gradient Descent.

2. **Đại số tuyến tính (Linear Algebra)**:
   Biến dữ liệu (ảnh, câu chữ, âm thanh) thành vector và tensor, đồng thời mô tả mỗi lớp mạng nơ-ron như một phép biến đổi tuyến tính bằng phép nhân ma trận tốc độ cao trên GPU.

3. **Giải tích thực và Tôpô (Real Analysis & Topology)**:
   Đảm bảo các khái niệm tưởng chừng trừu tượng như tập compact, tính liên tục hay hội tụ thực sự bảo chứng cho sự tồn tại của nghiệm tối ưu và tính ổn định số học khi huấn luyện.

4. **Xác suất và Thống kê (Probability & Statistics)**:
   Giúp mạng nơ-ron không chỉ đoán bừa mà biết định lượng độ bất định, từ đó giải thích vì sao hàm Cross-Entropy hay MSE lại tự nhiên xuất phát từ nguyên lý Ước lượng Hợp lý Cực đại (MLE).

### Giải thích: Một bước học của nơ-ron đơn nhìn qua cả bốn nhánh toán học

Hãy theo dõi một nơ-ron cực nhỏ dự đoán giá trị $$y = 1$$ từ đầu vào $$x = 2$$ với trọng số khởi tạo $$w = 0.2$$ và hàm mất mát bình phương $$L(w) = \frac{1}{2}(wx - y)^2$$:

1. **Đại số tuyến tính (Lan truyền xuôi)**: Nhân vô hướng đầu vào với trọng số: $$\hat{y} = wx = 0.2 \times 2 = 0.4$$.
2. **Xác suất (Ý nghĩa hàm mất mát)**: Sai số bình phương $$L(0.2) = \frac{1}{2}(0.4 - 1)^2 = \frac{1}{2}(-0.6)^2 = 0.18$$ tương ứng với âm log-likelihood của phân phối chuẩn có phương sai bằng $$1$$.
3. **Giải tích (Tính gradient)**: Đạo hàm theo $$w$$ là $$\frac{dL}{dw} = (wx - y)x = (0.4 - 1) \times 2 = -1.2$$. Dấu âm cho biết cần tăng $$w$$ để giảm sai số.
4. **Tôpô & Hội tụ (Bước cập nhật)**: Với tốc độ học $$\eta = 0.1$$, ta cập nhật $$w_{\text{mới}} = 0.2 - 0.1 \times (-1.2) = 0.32$$. Khi đó $$\hat{y}_{\text{mới}} = 0.64$$ và $$L(0.32) = \frac{1}{2}(0.64 - 1)^2 = 0.0648 < 0.18$$. Chỉ một dòng code `w -= lr * grad`, nhưng cả bốn trụ cột toán học đều đang vận hành đồng bộ phía sau!

Tối ưu hóa giữ vai trò trung tâm trong khoa học dữ liệu. Dù bạn đang huấn luyện mạng nơ-ron, cực tiểu hóa sai số trong mô hình hồi quy hay phân bổ tài nguyên trong hệ thống gợi ý, về bản chất bạn đều đang giải bài toán tìm nghiệm tốt nhất trong một không gian khả năng rất lớn. Để làm việc hiệu quả với các bài toán này, bạn cần nắm vững ngôn ngữ toán học nền tảng. Chương này ôn tập các khái niệm cốt lõi từ đại số tuyến tính, lý thuyết tập hợp, giải tích và xác suất, giúp bạn xử lý tốt gradient, ma trận, ràng buộc và yếu tố bất định trong các tác vụ tối ưu hóa.

**Đọc thêm (tùy chọn, 2022–2026):** Sau khi học xong phần toán nền tảng, bạn có thể xem bài **00-99 Ứng dụng và cập nhật hiện đại** để tìm hiểu về trình biên dịch vi phân tự động (JAX, `torch.compile`) cũng như cách gradient, ma trận và hàm hợp lý xuất hiện trong học máy khoa học (*scientific ML*) và huấn luyện theo quy luật mở rộng (*scaling law*).

**Luyện phỏng vấn (tùy chọn):** Sau các bài về xác suất và giải tích, bạn có thể xem **00-98 Luyện phỏng vấn** với các câu hỏi gốc về entropy, phân kỳ KL và vi phân tự động; bộ ngân hàng câu hỏi có lời giải đầy đủ được cung cấp miễn phí dưới dạng PDF trong *Deep Learning Interviews* ([arXiv:2201.00650](https://arxiv.org/abs/2201.00650)).

## Video tham khảo

- **Xem trước:** [MaDL - Introduction](https://www.youtube.com/watch?v=gq0m_2Glz0g) — Tübingen Machine Learning, tiếng Anh. Video giới thiệu ngắn gọn các kiến thức đại số tuyến tính, giải tích và xác suất mà chương này giả định trước khi đi vào mạng nơ-ron.
- [Vectors | Chapter 1, Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs) — 3Blue1Brown, tiếng Anh. Cách tiếp cận vector trong video này cũng là cách các bài sau biểu diễn trọng số, giá trị kích hoạt và điểm dữ liệu.
- [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, tiếng Anh. Ba tiên đề xác suất trong bài giảng này là nền tảng xuyên suốt nửa sau của chương.
