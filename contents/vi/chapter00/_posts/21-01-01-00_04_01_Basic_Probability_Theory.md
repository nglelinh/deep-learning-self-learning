---
layout: post
title: 00-04-01 Lý Thuyết Xác Suất Cơ Bản
chapter: '00'
order: 15
owner: AI Assistant
lang: vi
categories:
- chapter00
---

Trước khi làm quen với những mô hình xác suất phức tạp như Variational Autoencoder hay Diffusion Models, mọi thứ đều bắt đầu từ ba câu hỏi rất đời thường: *những kết cục nào có thể xảy ra?* (không gian mẫu), *biết thêm manh mối mới thì xác suất thay đổi ra sao?* (xác suất có điều kiện), và *làm sao gán con số cho các kết cục ngẫu nhiên để tính toán?* (biến ngẫu nhiên). Mục tổng quan này dẫn lối qua ba bài học nền tảng giúp bạn chuyển hóa trực giác xác suất thành công cụ toán học sắc bén cho Deep Learning.

## Kiến thức cần nhớ: Ba tầng kiến thức của Lý thuyết Xác suất Cơ bản

1. **Không gian mẫu và Tiên đề Kolmogorov**:
   Thiết lập luật chơi tối thiểu (không âm, tổng bằng 1, cộng tính trên biến cố xung khắc) để mọi công thức xác suất sau này không bao giờ mâu thuẫn—chính xác là điều kiện mà hàm Softmax ở tầng cuối mạng phân loại luôn phải thỏa mãn.

2. **Xác suất có điều kiện, Độc lập và Định lý Bayes**:
   Nền tảng của mô hình ngôn ngữ tự hồi quy $$P(w_1, \dots, w_T) = \prod_t P(w_t \mid w_{<t})$$ và giả định dữ liệu i.i.d. (độc lập, cùng phân phối) khi tính tổng hàm mất mát trên tập huấn luyện.

3. **Biến ngẫu nhiên, Kỳ vọng và Phương sai**:
   Cho phép định lượng rủi ro kỳ vọng (*expected risk*), kiểm soát phương sai của tín hiệu qua các tầng mạng (BatchNorm, LayerNorm) và phân tích độ nhiễu của gradient ngẫu nhiên.

### Giải thích: Tách một xác suất đồng thời bằng Quy tắc Chuỗi Xác suất

Xét một mô hình ngôn ngữ mini sinh câu 3 từ `"Tôi học AI"`:
1. Xác suất từ đầu tiên: $$P(\text{"Tôi"}) = 0.20$$.
2. Biết từ đầu là `"Tôi"`, xác suất từ kế tiếp là `"học"`: $$P(\text{"học"} \mid \text{"Tôi"}) = 0.30$$.
3. Biết hai từ đầu là `"Tôi học"`, xác suất từ cuối là `"AI"`: $$P(\text{"AI"} \mid \text{"Tôi học"}) = 0.50$$.
4. **Áp dụng quy tắc chuỗi xác suất**:
   $$P(\text{"Tôi học AI"}) = 0.20 \times 0.30 \times 0.50 = 0.03.$$
5. **Chuyển sang log-xác suất (để tránh tràn số dưới *underflow* trên máy tính)**:
   $$\ln P = \ln(0.20) + \ln(0.30) + \ln(0.50) \approx -1.609 - 1.204 - 0.693 = -3.506.$$
Chỉ từ một quy tắc xác suất cơ bản, bạn đã nhìn thấy trọn vẹn cơ chế tính điểm của mọi mô hình ngôn ngữ lớn (LLM) hiện nay!

Bài học này trình bày lý thuyết xác suất cơ bản phục vụ cho học sâu (*deep learning*), gồm ba phần chính:

1. **Không gian mẫu và tiên đề xác suất**
2. **Xác suất có điều kiện và tính độc lập**
3. **Biến ngẫu nhiên và học sâu**

Mỗi phần có kèm minh họa tương tác (nếu có).

## Video tham khảo

- **Xem trước:** [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, tiếng Anh. Trình bày các tiên đề xác suất mở đầu cho cụm bài này.
- [Conditional Probabilities, Clearly Explained!!!](https://www.youtube.com/watch?v=_IgyaD7vOOA) — StatQuest with Josh Starmer, tiếng Anh. Giải thích rõ ràng về xác suất có điều kiện, nội dung trọng tâm của bài thứ hai.
- [MaDL - Random Variables and Probability Distributions](https://www.youtube.com/watch?v=bvBukLjHZh8) — Tübingen Machine Learning, tiếng Anh. Giới thiệu biến ngẫu nhiên và phân phối, gắn với cách mô hình học sâu biểu diễn dữ liệu.
