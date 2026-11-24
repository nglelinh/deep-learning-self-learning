---
layout: post
title: 00-04 Xác Suất và Thống Kê
chapter: '00'
order: 14
owner: AI Assistant
categories:
- chapter00
lang: vi
---

Thế giới thực không bao giờ sạch sẽ như sách giáo khoa: cảm biến camera luôn có nhiễu hạt, chữ viết tay của hai người cho cùng số "7" có thể hoàn toàn khác nhau, và cùng một câu hỏi có thể có nhiều câu trả lời đúng. Nếu chỉ dùng hàm tất định cứng nhắc, AI sẽ gãy vụn trước sự bất định ấy. **Xác suất và Thống kê** là ngôn ngữ giúp mạng nơ-ron không chỉ đưa ra dự đoán, mà còn biết mức độ tự tin của mình—đồng thời biến bài toán huấn luyện thành bài toán tìm phân phối xác suất khớp nhất với dữ liệu quan sát được.

## Bản đồ kiến thức nền: Bốn trụ cột Xác suất & Thống kê trong Học sâu

1. **Không gian mẫu, Tiên đề và Định lý Bayes**:
   Đặt nền móng logic chặt chẽ cho việc cập nhật niềm tin (*posterior*) khi có thêm dữ liệu quan sát mới (*likelihood*) kết hợp với tri thức ban đầu (*prior*).

2. **Biến ngẫu nhiên và Các phân phối kinh điển**:
   Bernoulli/Categorical mô hình hóa bài toán phân loại; phân phối Chuẩn (Gaussian) mô hình hóa nhiễu hồi quy và khởi tạo trọng số (Xavier/He); phân phối đa biến mô tả tương quan giữa các đặc trưng.

3. **Ước lượng Hợp lý Cực đại (MLE) và MAP**:
   Cầu nối kỳ diệu nối xác suất với tối ưu hóa: tối đa hóa log-likelihood của dữ liệu cho ra ngay hàm mất mát MSE (với nhiễu Gaussian) và Cross-Entropy (với phân phối Categorical); thêm prior Gaussian cho ra ngay $$L_2$$ Regularization!

4. **Kỳ vọng, Phương sai và Luật số lớn**:
   Giải thích vì sao Mini-Batch Stochastic Gradient Descent (SGD) chỉ dùng 32 hoặc 64 mẫu mỗi bước vẫn cho ước lượng không chệch của gradient toàn tập dữ liệu hàng triệu mẫu.

### Khung giải thích (nên đọc chậm): Kiểm chứng bằng tay ước lượng không chệch của Mini-Batch

Giả sử toàn tập dữ liệu chỉ có 4 mẫu với gradient riêng lẻ tại tham số hiện tại là: $$g_1 = 2, g_2 = 6, g_3 = -2, g_4 = 10$$.

1. **Gradient thực trên toàn tập (Batch GD)**:
   $$g_{\text{true}} = \frac{2 + 6 + (-2) + 10}{4} = \frac{16}{4} = 4.$$
2. **Lấy ngẫu nhiên mini-batch cỡ 2** (có $$\binom{4}{2} = 6$$ cặp đồng khả năng):
   - $$\{1, 2\} \to \frac{2+6}{2} = 4$$; $$\{1, 3\} \to \frac{2-2}{2} = 0$$; $$\{1, 4\} \to \frac{2+10}{2} = 6$$;
   - $$\{2, 3\} \to \frac{6-2}{2} = 2$$; $$\{2, 4\} \to \frac{6+10}{2} = 8$$; $$\{3, 4\} \to \frac{-2+10}{2} = 4$$.
3. **Tính kỳ vọng của ước lượng mini-batch**:
   $$\mathbb{E}[\hat{g}] = \frac{4 + 0 + 6 + 2 + 8 + 4}{6} = \frac{24}{6} = 4 = g_{\text{true}}.$$

Mỗi bước mini-batch có thể dao động quanh $$4$$ (từ $$0$$ đến $$8$$), nhưng **trung bình đúng bằng gradient thật**—chính tính chất thống kê này cho phép huấn luyện LLM trên hàng nghìn tỷ token mà không bao giờ phải cộng dồn toàn bộ dataset trong một bước!

## Xác suất và thống kê cho tối ưu hóa lồi

Xác suất và thống kê là nền tảng quan trọng để hiểu nhiều bài toán tối ưu hóa, đặc biệt trong học máy và khoa học dữ liệu. Phần này giới thiệu các khái niệm xác suất thiết yếu thường xuất hiện trong tối ưu hóa lồi, từ ước lượng hợp lý cực đại đến tối ưu hóa Bayes.

### Vì sao xác suất quan trọng trong tối ưu hóa

Nhiều bài toán tối ưu hóa bắt nguồn trực tiếp từ mô hình hóa thống kê:

- **Ước lượng hợp lý cực đại (MLE):** tìm tham số cực đại hóa độ hợp lý của dữ liệu quan sát.
- **Tối ưu hóa Bayes:** dùng mô hình xác suất để định hướng tìm kiếm nghiệm tối ưu.
- **Tối ưu hóa ngẫu nhiên:** xử lý yếu tố bất định và nhiễu ngẫu nhiên trong hàm mục tiêu.
- **Điều chuẩn (*Regularization*):** đưa phân phối tiên nghiệm (*prior*) vào mô hình nhằm hạn chế quá khớp (*overfitting*).
- **Cực tiểu hóa rủi ro:** cực tiểu hóa giá trị kỳ vọng của hàm mất mát trên một phân phối xác suất.

### Các chủ đề chính

1. **Lý thuyết xác suất cơ bản:** không gian mẫu, biến cố và tiên đề xác suất.
2. **Các phân phối xác suất thông dụng:** phân phối chuẩn, phân phối mũ và các phân phối quan trọng khác.
3. **Kỳ vọng và phương sai:** cách tính và tối ưu hóa các đại lượng đặc trưng của biến ngẫu nhiên.
4. **Định lý Bayes:** nền tảng cho suy luận và tối ưu hóa Bayes.
5. **Ước lượng thống kê:** cầu nối giữa mô hình xác suất và bài toán tối ưu hóa.

### Liên hệ với tối ưu hóa lồi

Nắm vững xác suất giúp bạn:

- **Mô hình hóa bài toán:** chuyển yếu tố bất định thực tế thành bài toán tối ưu hóa toán học.
- **Chọn hàm mục tiêu:** lựa chọn hàm mất mát phù hợp dựa trên giả định xác suất của dữ liệu.
- **Diễn giải kết quả:** hiểu khoảng tin cậy và ý nghĩa thống kê của nghiệm thu được.
- **Xử lý nhiễu:** làm việc hiệu quả với sai số đo lường và quá trình ngẫu nhiên.
- **Thiết kế thuật toán:** xây dựng phương pháp tối ưu ổn định trước yếu tố bất định.

Nền tảng này sẽ được sử dụng xuyên suốt khi ta khảo sát cách các mô hình xác suất dẫn đến bài toán tối ưu hóa lồi trong học máy, thống kê và các ứng dụng kỹ thuật.

<div style="background: #e8f4fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0;">
<strong>💡 Lộ trình học:</strong> Hãy bắt đầu từ các khái niệm xác suất cơ bản, sau đó tìm hiểu cách chúng gắn với tối ưu hóa qua ước lượng hợp lý cực đại và suy luận Bayes. Mỗi bài học đều hướng tới việc làm rõ cách yếu tố bất định hình thành nên bài toán tối ưu hóa.
</div>

## Video tham khảo

- **Xem trước:** [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, tiếng Anh. Các tiên đề xác suất là nền tảng cho toàn bộ các phân phối trong phần này.
- [MaDL - Random Variables and Probability Distributions](https://www.youtube.com/watch?v=bvBukLjHZh8) — Tübingen Machine Learning, tiếng Anh. Trình bày biến ngẫu nhiên và phân phối xác suất theo ký hiệu chuẩn dùng trong học sâu.
- [MaDL - Common Probability Distributions](https://www.youtube.com/watch?v=NbzVwvE_syQ) — Tübingen Machine Learning, tiếng Anh. Giới thiệu các họ phân phối thông dụng (từ Bernoulli đến Gauss) sẽ được phân tích kỹ ở các bài tiếp theo.
