---
layout: post
title: 10-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '10'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter10
lesson_type: optional
---

# Tùy chọn: AdamW và các bộ tối ưu giai đoạn 2023–2025 (Lion, Sophia, Muon)

> Bài này mang tính **tùy chọn** và **không** thay thế lý thuyết về momentum, RMSprop hay Adam. Các bài chính đã trình bày AdamW; bài này tóm tắt những phương pháp nổi bật ra đời *sau* AdamW, đồng thời làm rõ khi nào vẫn nên tiếp tục dùng AdamW.

AdamW hiện vẫn là lựa chọn mặc định trong Hugging Face và phần lớn các hệ thống huấn luyện LLM:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,$$

$$\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} - \eta\lambda \theta.$$

Số hạng cuối là weight decay **tách rời** (không gộp L2 vào bước chuẩn hóa thích nghi). Các phương pháp dưới đây là những lựa chọn thay thế hoặc chuyên biệt hóa gần đây.

## 1. Lion (2023)

[Chen et al., 2023](https://arxiv.org/abs/2302.06675) (phát hiện thông qua tìm kiếm ký hiệu tại Google) chỉ giữ lại **dấu** của momentum nội suy:

$$\theta \leftarrow \theta - \eta\,\mathrm{sign}(\beta_1 m + (1-\beta_1)g) - \eta\lambda\theta.$$

Thuật toán tốn ít bộ nhớ hơn Adam vì không phải lưu moment bậc hai. Lion được áp dụng trong một số bài toán thị giác và chưng cất mô hình, nhưng ít phổ biến hơn AdamW khi huấn luyện LLM.

## 2. Sophia (2023)

[Liu et al., 2023](https://arxiv.org/abs/2305.14342) ước lượng đường chéo của ma trận Hessian (bằng ước lượng Hutchinson hoặc Gauss–Newton có cắt ngưỡng) để tiền điều kiện bước cập nhật:

$$\theta \leftarrow \theta - \eta \cdot \mathrm{clip}\!\left(\frac{\hat{m}_t}{\max\{\gamma \hat{h}_t, \varepsilon\}}\right).$$

Nhóm tác giả báo cáo giúp giảm đáng kể số bước huấn luyện trên mô hình kiểu GPT so với AdamW ở cùng mức chất lượng. Dù vậy, trong phần lớn các thư viện huấn luyện mã nguồn mở, Sophia vẫn chủ yếu dừng ở mức nghiên cứu.

## 3. Muon (2024)

[Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/) áp dụng SGD kết hợp momentum lên các **ma trận trọng số ẩn**, sau đó trực giao hóa bước cập nhật bằng vài vòng lặp Newton–Schulz (xấp xỉ thừa số cực *polar factor* với chi phí thấp). Các tầng embedding và đầu ra ngôn ngữ thường vẫn giữ AdamW. [Moonshot / Kimi, 2025](https://arxiv.org/abs/2502.16982) báo cáo đã huấn luyện LLM quy mô lớn bằng Muon. Đây là bộ tối ưu ở cấp độ ma trận được quan tâm nhiều nhất trong giai đoạn 2024–2025.

## 4. Ứng dụng và thư viện

- **Tiếp tục dùng AdamW** cho các lần chạy đầu tiên với LLM hoặc ConvNeXt (`torch.optim.AdamW`).
- **Lion**: [lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch) và một số công thức huấn luyện trong `timm`.
- **Sophia**: mã nguồn tham chiếu đi kèm bài báo.
- **Muon**: [KellerJordan/Muon](https://github.com/KellerJordan/Muon) và các thử nghiệm tốc độ trong [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
- **Schedule-free AdamW** ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682)) — cho phép bỏ qua lịch trình warmup và cosine tường minh trong một số bài toán.

## 5. Tài liệu tham khảo (2022–2026)

- [Lion (Chen et al., 2023)](https://arxiv.org/abs/2302.06675).
- [Sophia (Liu et al., 2023)](https://arxiv.org/abs/2305.14342).
- [Muon (Jordan et al., 2024)](https://kellerjordan.github.io/posts/muon/).
- [Muon is Scalable for LLM Training (Liu et al., 2025)](https://arxiv.org/abs/2502.16982).
- [The Road Less Scheduled (Defazio et al., 2024)](https://arxiv.org/abs/2405.15682).

## 6. Phần bổ sung so với nội dung cốt lõi

Bạn nên nắm vững Adam và phần so sánh trên hàm Rosenbrock trước. Bài này chỉ bổ sung các phương pháp **sau AdamW** xuất hiện trong nghiên cứu và thử nghiệm giai đoạn 2023–2026, không thay thế phần cài đặt bộ tối ưu cốt lõi của chương.

## Video tham khảo

- **Xem trước:** [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, tiếng Anh. Lion, Sophia và Muon là các bước mở rộng tiếp theo từ nhóm momentum/Adam được trình bày trong video này.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, tiếng Anh. Adam là mốc so sánh chuẩn để đánh giá các phương pháp giai đoạn 2023–2024.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering, tiếng Anh. Đặt vai trò của bộ tối ưu trong bức tranh huấn luyện tổng thể cùng dữ liệu và lịch trình học.
