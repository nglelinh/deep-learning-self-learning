---
layout: post
title: 21-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '21'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter21
lesson_type: optional
---

# Tùy chọn: RL sâu sau DQN/PPO — RLHF, DPO, và GRPO

> Bài này là **tùy chọn**. Nó **không** thay lý thuyết DQN, replay, hay PPO (ghi chú cốt lõi đã nhắc RLHF thoáng qua). Đây là stack **căn chỉnh** 2022–2026 khiến RL sâu lộ diện ngoài game.

PPO vẫn cực đại hóa surrogate đã cắt. Căn chỉnh ngôn ngữ thêm **mô hình thưởng** rồi, nhanh chóng, các phương pháp **bỏ vòng RL**.

## 1. RLHF với PPO

[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) và [Bai et al., 2022](https://arxiv.org/abs/2204.05862) (Anthropic HH):

1. SFT trên demo,
2. huấn luyện $$r_\phi$$ trên preference từng cặp,
3. tối ưu chính sách bằng PPO đối với $$r_\phi$$ cộng KL về mô hình SFT.

Đó là actor–critic Chương 21 áp lên token. Đắt (mô hình giá trị, lấy mẫu, bất ổn).

## 2. DPO: preference không cần mô hình thưởng

[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) viết lại tối ưu RLHF thành loss **phân loại** trên cặp $$(y_w, y_l)$$:

$$\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Big).$$

Không vòng lấy mẫu lúc train. DPO và họ hàng (IPO, KTO, ORPO) thành công thức căn chỉnh mã nguồn mở mặc định 2024–2025 ([trl](https://github.com/huggingface/trl)).

## 3. Ứng dụng cụ thể

### GRPO và huấn luyện mô hình lập luận

[Shao et al., 2024](https://arxiv.org/abs/2402.03300) (DeepSeekMath) đưa **Group Relative Policy Optimization**: vài mẫu mỗi prompt, lợi thế từ thưởng của chính nhóm — không mạng giá trị học được. Các mô hình lập luận sau (họ DeepSeek-R1, 2025) phổ biến hóa dòng này.

### World model

[Hafner et al., 2023](https://arxiv.org/abs/2301.04104) (DreamerV3) học MDP ẩn (kiểu VAE) và lập kế hoạch bên trong. Bổ sung cho LLM-RL: vẫn RL sâu, không phải chat.

### Phần mềm

- [huggingface/trl](https://github.com/huggingface/trl) — trainer PPO, DPO, GRPO.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) / [verl](https://github.com/volcengine/verl) — RLHF phân tán.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) và [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — RL sâu cổ điển, không phải LLM.

## 4. Trích dẫn (2022–2026)

- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155).
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290).
- [DeepSeekMath — GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300).
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Cài ý tưởng DQN/PPO của chương trước. Bài này chỉ thêm **tối ưu preference cho LM** (RLHF/DPO/GRPO) — không phải backup Bellman mới.
