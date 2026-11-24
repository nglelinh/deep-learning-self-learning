---
layout: post
title: 03-02 Gradient Descent
chapter: '03'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

Đã có bản đồ sai số (hàm mất mát), bước tiếp theo là trả lời câu hỏi cốt lõi nhất của tối ưu hóa: *làm thế nào để điều chỉnh hàng triệu trọng số cùng lúc sao cho hàm mất mát tụt xuống đáy nhanh và ổn định nhất?* Câu trả lời mang tính biểu tượng của cả ngành Học sâu là **Gradient Descent**—thuật toán xuống dốc từng bước ngược hướng vector gradient. Mục tổng quan này phác họa bức tranh chung về quy tắc cập nhật, ba biến thể theo kích thước batch và những cạm bẫy hình học sẽ được mổ xẻ sâu ở hai bài kế tiếp.

## Bản đồ kiến thức nền: Ba trụ cột của Thuật toán Gradient Descent

1. **Quy tắc Cập nhật và Vai trò của Learning Rate ($$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$)**:
   $$\eta$$ quá nhỏ khiến huấn luyện rùa bò và dễ mắc kẹt; $$\eta$$ quá lớn khiến tham số bật qua lại hai bờ thung lũng và phân kỳ về vô cực.

2. **Ba chế độ ước lượng Gradient (Batch, SGD, Mini-Batch)**:
   Đánh đổi giữa độ chính xác của hướng đi (Batch GD), tốc độ cập nhật và nhiễu giúp thoát cực tiểu nông (SGD), và hiệu suất tận dụng song song GPU (Mini-Batch GD—chuẩn mực thực tế).

3. **Địa hình Bề mặt Mất mát Phi lồi (Non-Convex Landscape)**:
   Khác với bài toán lồi chỉ có 1 đáy duy nhất, bề mặt mất mát của mạng sâu đầy rẫy điểm yên ngựa (*saddle points*), thung lũng hẹp dài (*ill-conditioned ravines*) và vùng phẳng (*plateaus*).

### Khung giải thích (nên đọc chậm): Thấy ngay tác động của Learning Rate trên parabol $$L(w) = w^2$$

Xét hàm đơn giản $$L(w) = w^2$$ (đáy tại $$w^* = 0$$), có đạo hàm $$L'(w) = 2w$$. Xuất phát từ $$w_0 = 4$$, công thức cập nhật là $$w_{t+1} = w_t - \eta(2w_t) = (1 - 2\eta)w_t$$:

| Learning rate $$\eta$$ | Hệ số co $$(1 - 2\eta)$$ | Chuỗi $$w_0 \to w_1 \to w_2 \to w_3$$ | Hành vi hội tụ |
| :---: | :---: | :--- | :--- |
| **$$\eta = 0.05$$ (Nhỏ)** | $$0.9$$ | $$4.0 \to 3.6 \to 3.24 \to 2.916$$ | Hội tụ êm nhưng chậm |
| **$$\eta = 0.50$$ (Lý tưởng)** | $$0.0$$ | $$4.0 \to 0.0 \to 0.0 \to 0.0$$ | Chạm đáy trong đúng **1 bước**! |
| **$$\eta = 0.90$$ (Hơi lớn)** | $$-0.8$$ | $$4.0 \to -3.2 \to +2.56 \to -2.048$$ | Dao động nhảy qua lại hai bờ |
| **$$\eta = 1.10$$ (Quá lớn)** | $$-1.2$$ | $$4.0 \to -4.8 \to +5.76 \to -6.912$$ | **Phân kỳ nổ tung!** ($$\lvert 1-2\eta \rvert > 1$$) |

Bất đẳng thức vàng rút ra ngay từ bảng trên: để hội tụ trên parabol độ cong $$H = 2$$, điều kiện bắt buộc là $$0 < \eta < \frac{2}{H} = 1$$!

Phần này trình bày thuật toán gradient descent và các biến thể thực tế:

1. **Các biến thể Gradient Descent và Learning Rate** — Batch GD, SGD, Mini-batch GD và lịch trình điều chỉnh tốc độ học
2. **Triển khai và thách thức** — mã nguồn cài đặt, tiêu chí hội tụ và các khó khăn thường gặp khi tối ưu

## Video tham khảo

- **Xem trước:** [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown, tiếng Anh. Minh họa trực giác quá trình đi xuống dốc của gradient descent.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, tiếng Anh. Ví dụ tính từng bước cập nhật bằng số kèm cách chọn tốc độ học.
- [Stochastic Gradient Descent, Clearly Explained!!!](https://www.youtube.com/watch?v=vMh0zPT0tLI) — StatQuest with Josh Starmer, tiếng Anh. Giải thích SGD để so sánh trực tiếp với Batch Gradient Descent.
