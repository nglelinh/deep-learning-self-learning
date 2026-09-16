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

# Tùy chọn: AdamW và làn sóng optimizer 2023–2025 (Lion, Sophia, Muon)

> Bài này là **tùy chọn**. Nó **không** thay lý thuyết momentum / RMSprop / Adam. Ghi chú cốt lõi đã nhắc AdamW; bài này phủ những gì thành chuẩn *sau* AdamW, và khi nào nên ở lại với AdamW.

AdamW vẫn là mặc định trong Hugging Face và hầu hết trainer LLM:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,$$

$$\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} - \eta\lambda \theta.$$

Hạng cuối là weight decay **tách** (không phải L2 trong bước thích nghi). Mọi thứ dưới đây là phương án thay hoặc nâng cấp chuyên biệt.

## 1. Lion (2023)

[Chen et al., 2023](https://arxiv.org/abs/2302.06675) (tìm kiếm ký hiệu tại Google) chỉ giữ **dấu** của momentum nội suy:

$$\theta \leftarrow \theta - \eta\,\mathrm{sign}(\beta_1 m + (1-\beta_1)g) - \eta\lambda\theta.$$

Bộ nhớ thấp hơn Adam (không có moment hai). Dùng trong một số việc thị giác và chưng cất; ít phổ quát hơn AdamW cho LLM.

## 2. Sophia (2023)

[Liu et al., 2023](https://arxiv.org/abs/2305.14342) ước lượng Hessian đường chéo (Hutchinson hoặc Gauss–Newton cắt) rồi tiền điều kiện:

$$\theta \leftarrow \theta - \eta \cdot \mathrm{clip}\!\left(\frac{\hat{m}_t}{\max\{\gamma \hat{h}_t, \varepsilon\}}\right).$$

Báo cáo cắt số bước kiểu GPT so với AdamW ở chất lượng tương tự. Vẫn là optimizer nghiên cứu trong hầu hết trainer mở.

## 3. Muon (2024)

[Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/) áp SGD-momentum lên **ma trận ẩn**, rồi trực giao hóa bước cập nhật bằng vài vòng Newton–Schulz (xấp xỉ nhân tử cực rẻ). Embedding và đầu LM thường ở lại AdamW. [Moonshot / Kimi, 2025](https://arxiv.org/abs/2502.16982) báo cáo huấn luyện LLM quy mô với Muon. Đây là optimizer ma trận *mới* được bàn nhiều nhất 2024–2025.

## 4. Ứng dụng và phần mềm

- **Ở lại AdamW** cho lần chạy LLM hoặc ConvNeXt đầu (`torch.optim.AdamW`).
- **Lion**: [lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch); một số công thức `timm`.
- **Sophia**: mã theo paper.
- **Muon**: [KellerJordan/Muon](https://github.com/KellerJordan/Muon) và speedrun [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
- **Schedule-free AdamW** ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682)) — bỏ warmup/cosine tường minh trên một số tải.

## 5. Trích dẫn (2022–2026)

- [Lion (Chen et al., 2023)](https://arxiv.org/abs/2302.06675).
- [Sophia (Liu et al., 2023)](https://arxiv.org/abs/2305.14342).
- [Muon (Jordan et al., 2024)](https://kellerjordan.github.io/posts/muon/).
- [Muon is Scalable for LLM Training (Liu et al., 2025)](https://arxiv.org/abs/2502.16982).
- [The Road Less Scheduled (Defazio et al., 2024)](https://arxiv.org/abs/2405.15682).

## 6. Bài này bổ sung gì cho ghi chú cốt lõi

Học Adam và so sánh Rosenbrock trước. Bài này chỉ thêm phương pháp **sau-AdamW** dùng trong paper và speedrun 2023–2026 — không thay các cài đặt optimizer trong chương.
