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

Suốt gần một thập kỷ, AdamW gần như không có đối thủ trong việc huấn luyện các mô hình nền tảng quy mô lớn — nhưng chi phí bộ nhớ khổng lồ của nó (lưu 2 bản sao trạng thái FP32 cho hàng trăm tỷ tham số) và khát vọng rút ngắn hàng triệu giờ GPU đã thúc đẩy một làn sóng bộ tối ưu mới đầy ấn tượng giai đoạn 2023–2025: từ **Lion** (do AI tự tiến hóa tìm ra!), **Sophia** (tận dụng thông tin bậc hai Hessian), cho tới **Muon** (tối ưu hóa trực giao hóa ma trận theo phổ giá trị kỳ dị).

## Bản đồ kiến thức nền: Bốn đột phá Bộ tối ưu giai đoạn 2023–2025

1. **Lion (EvoLved Sign Momentum, Google Brain 2023)**:
   Được tìm ra bằng thuật toán tìm kiếm tiến hóa chương trình! Lion bỏ hoàn toàn moment bậc 2 $$v_t$$ (tiết kiệm ngay **50% bộ nhớ trạng thái optimizer** so với AdamW) và cập nhật tham số chỉ bằng **dấu** `sign(...)` của tổ hợp quán tính — nên mọi tham số có độ lớn bước nhảy đồng đều tuyệt đối.
2. **Sophia (Second-Order Clipped Stochastic Optimization, Stanford 2023)**:
   Thay vì dùng bình phương gradient $$g_t^2$$ (vốn đo độ lớn gradient chứ không phải độ cong thật), Sophia ước lượng nhẹ đường chéo **ma trận Hessian** $$h_t$$ định kỳ (mỗi $$k=10$$ bước) kết hợp cắt ngưỡng `clip` để tăng tốc huấn luyện LLM gấp ~2 lần so với AdamW.
3. **Muon (MomentUm Orthogonalized by Newton–Schulz, 2024–2025)**:
   Nhận ra rằng trọng số trong mạng sâu là **ma trận** chứ không phải một túi các số vô hướng rời rạc! Muon dùng vài vòng lặp Newton–Schulz siêu nhanh trên GPU để **trực giao hóa** ma trận cập nhật quán tính (đưa mọi giá trị kỳ dị về 1), lập kỷ lục tốc độ huấn luyện NanoGPT và được ứng dụng trong các LLM thế hệ mới (như Kimi K2).
4. **Kỹ thuật tiết kiệm bộ nhớ hệ thống (8-bit AdamW, Adafactor, GaLore)**:
   Lượng tử hóa trạng thái optimizer xuống 8-bit (`bitsandbytes`) hoặc chiếu gradient xuống không gian con hạng thấp (GaLore) giúp tinh chỉnh mô hình 7B–70B ngay trên một GPU tiêu dùng.

### Khung giải thích (nên đọc chậm): Tính tay bước cập nhật bằng dấu của Lion và so sánh bộ nhớ trạng thái với AdamW

**Phần A — Một bước cập nhật của Lion** (cho $$w = 1{,}0, \; m_0 = 0{,}5, \; g_1 = -2{,}0, \; \beta_1 = 0{,}9, \; \beta_2 = 0{,}99, \; \eta = 0{,}01, \; \lambda = 0{,}1$$):
1. Tính hướng cập nhật tạm (nội suy bằng $$\beta_1$$):
   $$c_1 = \beta_1 m_0 + (1 - \beta_1) g_1 = 0{,}9(0{,}5) + 0{,}1(-2{,}0) = 0{,}45 - 0{,}20 = \mathbf{+0{,}25}$$
2. Cập nhật tham số chỉ dùng **dấu** của $$c_1$$ kèm weight decay tách rời:
   $$w_1 = w_0 - \eta \big(\operatorname{sign}(c_1) + \lambda w_0\big) = 1{,}0 - 0{,}01\big(+1 + 0{,}1(1{,}0)\big) = \mathbf{0{,}989}$$
3. Cập nhật bộ nhớ quán tính duy nhất (bằng $$\beta_2$$):
   $$m_1 = \beta_2 m_0 + (1 - \beta_2) g_1 = 0{,}99(0{,}5) + 0{,}01(-2{,}0) = \mathbf{0{,}475}$$

**Phần B — So sánh bộ nhớ trạng thái Optimizer cho mô hình 10 tỷ tham số (FP32 = 4 byte/số)**:
- **AdamW** (lưu cả $$m$$ và $$v$$): $$2 \times 10^{10} \times 4\text{ B} = \mathbf{80\text{ GB VRAM}}$$.
- **Lion / Muon** (chỉ lưu duy nhất 1 bảng quán tính $$m$$): $$1 \times 10^{10} \times 4\text{ B} = \mathbf{40\text{ GB VRAM}}$$ (cắt giảm đúng một nửa!).

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
