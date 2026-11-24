---
layout: post
title: 10-01 Thuật toán Tối ưu Nâng cao
chapter: '10'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter10
---

Chọn đúng bộ tối ưu không chỉ quyết định mô hình hội tụ sau 2 giờ hay 2 ngày, mà còn ảnh hưởng trực tiếp tới **khả năng tổng quát hóa** trên dữ liệu thực tế: tại sao ResNet trên ImageNet thường đạt đỉnh bằng SGD + Momentum, còn Transformer/LLM gần như bắt buộc phải dùng AdamW? Mục này hệ thống hóa trọn vẹn từ nền tảng toán học, cài đặt từ đầu cho đến bản đồ so sánh và cạm bẫy thực chiến.

## Bản đồ kiến thức nền: Ba trụ cột học phần Thuật toán Tối ưu Nâng cao

1. **Phả hệ toán học của các bộ tối ưu**:
   Hiểu rõ bước tiến hóa logic: SGD $$\to$$ Momentum / Nesterov (thêm moment bậc 1 $$m_t$$) và AdaGrad $$\to$$ RMSprop (thêm moment bậc 2 $$v_t$$ có suy giảm mũ) $$\to$$ Adam / AdamW (hợp nhất cả hai).
2. **Tự cài đặt từ đầu và trực quan hóa quỹ đạo**:
   Viết tay lớp `Momentum`, `RMSprop`, `Adam` trong vài chục dòng Python/NumPy và vẽ quỹ đạo di chuyển của chúng trên bề mặt thung lũng hẹp 2D (như hàm Rosenbrock) để thấy tận mắt sự khác biệt.
3. **Chiến lược chọn Bộ tối ưu và Lịch Learning Rate**:
   Biết phối hợp bộ tối ưu với Warmup + Cosine Annealing, hiểu lý do tách rời Weight Decay (AdamW) và tránh bẫy tuning sai siêu tham số $$(\beta_1, \beta_2, \epsilon)$$.

### Khung giải thích (nên đọc chậm): Bảng tóm tắt 4 phương trình cập nhật cốt lõi để đối chiếu trong một cái nhìn

Gọi $$g_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$$ là gradient tại bước $$t$$ và $$\eta$$ là tốc độ học:

| Bộ tối ưu | Biến trạng thái tích lũy | Quy tắc cập nhật tham số $$\theta_t$$ | Bộ nhớ trạng thái thêm / 1 tham số |
| :--- | :--- | :--- | :---: |
| **SGD** | Không có | $$\theta_{t-1} - \eta g_t$$ | **0** số |
| **SGD + Momentum** | $$v_t = \beta v_{t-1} + g_t$$ | $$\theta_{t-1} - \eta v_t$$ | **1** số ($$v$$) |
| **RMSprop** | $$s_t = \rho s_{t-1} + (1-\rho) g_t^2$$ | $$\theta_{t-1} - \frac{\eta}{\sqrt{s_t} + \epsilon} g_t$$ | **1** số ($$s$$) |
| **Adam** | $$m_t$$ (bậc 1) & $$v_t$$ (bậc 2) kèm hiệu chỉnh $$\hat{m}_t, \hat{v}_t$$ | $$\theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$ | **2** số ($$m, v$$) |

Chú ý cột cuối cùng rất quan trọng khi huấn luyện mô hình lớn: với mô hình 7 tỷ tham số (FP32 = 4 byte/số), riêng hai bảng trạng thái $$(m, v)$$ của Adam đã ngốn thêm **56 GB VRAM**!

Bài học này trình bày các bộ tối ưu hiện đại cải tiến từ hạ gradient thuần túy:

1. **Lý thuyết Bộ tối ưu** — Momentum, NAG, AdaGrad, RMSprop, Adam, AdamW và trực giác hình học.
2. **Cài đặt Bộ tối ưu** — mã cài đặt so sánh, bài báo nền tảng, các cạm bẫy thường gặp và điểm then chốt.

## Video tham khảo

- **Xem trước:** [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, tiếng Anh. Tổng quan nhóm thuật toán được phân tích trong phần lý thuyết và phần cài đặt.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, tiếng Anh. Giải thích hai ước lượng moment của Adam, đúng với phép cập nhật bạn sẽ tự cài đặt.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering, tiếng Anh. Bài giảng về tối ưu hóa đặt cùng nhóm thuật toán này trong bối cảnh huấn luyện mạng nơ-ron đầy đủ.
