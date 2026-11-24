---
layout: post
title: 00-03 Giải tích thực và Lý thuyết tập hợp
chapter: '00'
order: 11
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Khi viết một bài toán tối ưu như $$\min_{\theta \in \Theta} L(\theta)$$, chúng ta thường mặc nhiên giả định rằng nghiệm nhỏ nhất *chắc chắn tồn tại* và chuỗi cập nhật $$\theta_1, \theta_2, \dots$$ sẽ tiến về đâu đó có nghĩa. Nhưng trên thực tế, một hàm số trên tập mở hoàn toàn có thể không bao giờ đạt cực tiểu (ví dụ $$f(x) = x$$ trên $$(0, 1)$$ không có giá trị nhỏ nhất!). Giải tích thực và lý thuyết tập hợp cung cấp khung ngôn ngữ chính xác về tập đóng, tập mở, tính compact và sự hội tụ—những bảo chứng thầm lặng đằng sau mọi định lý tối ưu hóa và học máy lý thuyết.

## Bản đồ kiến thức nền: Ba trụ cột của Lý thuyết Tập hợp và Giải tích Thực

1. **Lý thuyết tập hợp và Ánh xạ (Sets & Functions)**:
   Xây dựng ngôn ngữ chuẩn hóa để biểu diễn không gian mẫu, tập dữ liệu huấn luyện/kiểm thử, miền tham số khả thi và các ánh xạ tầng mạng nơ-ron $$f: \mathbb{R}^d \to \mathbb{R}^k$$.

2. **Không gian Metric và Tôpô trong $$\mathbb{R}^n$$ (Open, Closed, Boundary)**:
   Định nghĩa khái niệm "gần nhau" thông qua khoảng cách, hình cầu mở và lân cận—cần thiết để phát biểu cực trị địa phương, miền hội tụ và lân cận nhiễu đối kháng (*adversarial ball* $$\|\delta\|_p \le \varepsilon$$).

3. **Tính Compact và Định lý Weierstrass**:
   Trong $$\mathbb{R}^n$$, tập vừa đóng vừa bị chặn là tập compact; mọi hàm liên tục trên tập compact đều đạt giá trị lớn nhất và nhỏ nhất—bảo chứng cốt lõi cho sự tồn tại nghiệm của bài toán tối ưu có ràng buộc.

### Khung giải thích (nên đọc chậm): So sánh sự tồn tại cực tiểu trên tập mở và tập đóng

Xét hàm đơn giản $$f(x) = \frac{1}{x} + x$$ trên hai miền khác nhau:

1. **Trên tập mở $$(0, 2)$$**:
   - Đạo hàm $$f'(x) = 1 - \frac{1}{x^2} = 0 \iff x = 1 \in (0, 2)$$, cho cực tiểu $$f(1) = 2$$.
   - Tuy nhiên, nếu hỏi **cực đại** trên $$(0, 2)$$: khi $$x \to 0^+$$ thì $$f(x) \to +\infty$$ nên không có cực đại!
2. **Xét hàm $$g(x) = x^2$$ trên tập mở $$(1, 3)$$ so với tập đóng $$[1, 3]$$**:
   - Trên $$(1, 3)$$: giá trị $$g(x)$$ có thể tiến sát $$1$$ tùy ý (như $$g(1.001) = 1.002001$$), nhưng không tồn tại điểm $$x^* \in (1, 3)$$ nào đạt đúng $$1$$ vì điểm biên $$1$$ không thuộc tập.
   - Trên tập compact $$[1, 3]$$: điểm biên nằm trong tập, nên $$g$$ chắc chắn đạt cực tiểu tại $$x^* = 1$$ ($$g(1) = 1$$) và cực đại tại $$x^* = 3$$ ($$g(3) = 9$$).

Khi bạn thêm ràng buộc chuẩn trọng số $$\|\theta\|_2 \le R$$ trong huấn luyện mạng nơ-ron, bạn đang biến miền tìm kiếm thành một **hình cầu đóng và bị chặn (compact)** để bài toán tối ưu luôn có nghiệm xác định!

Phần này trình bày các khái niệm nền tảng từ giải tích thực và lý thuyết tập hợp phục vụ cho tối ưu hóa, được chia thành hai bài học chính.

## Video tham khảo

- **Xem trước:** [Intro to Sets | Examples, Notation & Properties](https://www.youtube.com/watch?v=B1v2-nGXNzs) — Dr. Trefor Bazett, tiếng Anh. Giới thiệu tập hợp, phần tử và hàm số — ngôn ngữ cơ bản dùng trong bài lý thuyết tập hợp.
- [Intro to Open Sets (with Examples) | Real Analysis](https://www.youtube.com/watch?v=pnWgj8jjs3w) — Wrath of Math, tiếng Anh. Giải thích khái niệm tập mở, đối tượng tôpô đầu tiên của bài thứ hai.
- [Limits, L'Hôpital's rule, and epsilon delta definitions | Chapter 7, Essence of calculus](https://www.youtube.com/watch?v=kfF40MiS7zA) — 3Blue1Brown, tiếng Anh. Trình bày định nghĩa giới hạn và liên tục kiểu $$\varepsilon\text{-}\delta$$ trong phần giải tích thực.
