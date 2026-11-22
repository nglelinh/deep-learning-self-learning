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
