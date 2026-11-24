---
layout: post
title: 03-01 Hàm Mất mát
chapter: '03'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

Trong tối ưu hóa, mạng nơ-ron không hề hiểu khái niệm "ảnh đẹp", "dự đoán thông minh" hay "câu trả lời hữu ích"—nó chỉ trung thành tuyệt đối với một mục tiêu duy nhất: **làm cho hàm mất mát $$L(\theta)$$ nhỏ nhất có thể**. Nếu bạn chọn sai hàm mất mát (chẳng hạn dùng MSE trên dữ liệu giá nhà đầy ngoại lai, hay dùng độ chính xác 0–1 không khả vi), mô hình sẽ tối ưu rất chăm chỉ theo đúng... hướng sai! Mục tổng quan này phác thảo bản đồ hai họ hàm mất mát lớn nhất—Hồi quy và Phân loại—trước khi đi vào chi tiết toán học từng công thức.

## Bản đồ kiến thức nền: Ba nguyên tắc chọn Hàm Mất mát chuẩn xác

1. **Khớp với Bản chất Đầu ra và Phân phối Xác suất (Góc nhìn MLE)**:
   Nhiễu chuẩn trên số thực $$\to$$ MSE; nhiễu đuôi dày nhiều ngoại lai $$\to$$ MAE hoặc Huber; nhãn nhị phân $$\to$$ Binary Cross-Entropy; nhãn đa lớp $$\to$$ Categorical Cross-Entropy.

2. **Khả vi và Hành vi Gradient tốt**:
   Hàm mất mát phải cung cấp tín hiệu gradient đủ mạnh khi mô hình đoán sai nặng (Cross-Entropy phạt theo $$-\log p \to \infty$$ khi $$p \to 0$$) và mượt dần khi tiến sát nghiệm đúng.

3. **Phân biệt Loss (tối ưu hóa) vs. Metric (đánh giá nghiệp vụ)**:
   Ta huấn luyện bằng Cross-Entropy (trơn, khả vi) nhưng đánh giá bằng Accuracy, F1-score hay ROC-AUC (những đại lượng bậc thang không đạo hàm được trực tiếp).

### Khung giải thích (nên đọc chậm): So sánh phản ứng của MSE và Cross-Entropy khi mô hình đoán sai tự tin

Giả sử nhãn thật là $$y = 1$$ nhưng mô hình phân loại nhị phân đoán rất sai và rất tự tin: $$\hat{y} = 0.01$$ (xác suất lớp 1 chỉ 1%):

1. **Nếu dùng MSE** $$L_{\text{MSE}} = \frac{1}{2}(\hat{y} - y)^2$$ với $$\hat{y} = \sigma(z)$$:
   - Mất mát: $$\frac{1}{2}(0.01 - 1)^2 = 0.490$$ (khá nhỏ!).
   - Gradient theo logit $$z$$: $$(\hat{y} - y)\hat{y}(1 - \hat{y}) = (-0.99)(0.01)(0.99) \approx -0.0098$$ (cực nhỏ do Sigmoid bão hòa $$\to$$ mạng học chậm rì rì dù đang sai nghiêm trọng!).
2. **Nếu dùng Binary Cross-Entropy** $$L_{\text{BCE}} = -[y\ln\hat{y} + (1-y)\ln(1-\hat{y})]$$:
   - Mất mát: $$-\ln(0.01) \approx 4.605$$ (phạt rất nặng!).
   - Gradient theo logit $$z$$: $$\hat{y} - y = 0.01 - 1 = -0.99$$ (khử sạch thừa số bão hòa $$\sigma'(z)$$, tạo lực sửa sai mạnh gấp **100 lần** MSE!).

Phần này trình bày các hàm mất mát (*loss functions*) thường dùng khi huấn luyện mạng neuron:

1. **Hàm mất mát cho hồi quy** — khái niệm hàm mất mát, MSE, MAE và Huber
2. **Hàm mất mát phân loại và cách lựa chọn** — mất mát cho phân loại nhị phân/đa lớp, ổn định số học và hướng dẫn chọn hàm phù hợp

## Video tham khảo

- **Xem trước:** [Lecture 3 | Loss Functions and Optimization](https://www.youtube.com/watch?v=h7iBpEHGVNc) — Stanford University School of Engineering, tiếng Anh. Giới thiệu vai trò của hàm mất mát trước khi phân tách sang hồi quy và phân loại.
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) — StatQuest with Josh Starmer, tiếng Anh. Trình bày hồi quy logistic gắn với hàm mất mát phân loại nhị phân ở bài thứ hai.
- [But what is cross-entropy? | Compression is Intelligence Part 2](https://www.youtube.com/watch?v=GlYgs6v2YfU) — 3Blue1Brown, tiếng Anh. Diễn giải trực quan về cross-entropy — hàm mất mát trung tâm cho phân loại đa lớp và mô hình ngôn ngữ.
