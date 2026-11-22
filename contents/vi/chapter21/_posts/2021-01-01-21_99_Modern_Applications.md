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

# Tùy chọn: RL sâu sau DQN/PPO — RLHF, DPO và GRPO

> Bài này là **tùy chọn**, **không** thay thế lý thuyết về DQN, bộ đệm phát lại hay PPO trong bài cốt lõi. Nội dung cập nhật hệ thống **căn chỉnh mô hình ngôn ngữ** giai đoạn 2022–2026 — lĩnh vực đưa học tăng cường sâu vào ứng dụng rộng rãi ngoài phạm vi trò chơi.

PPO vẫn tối đa hóa hàm mục tiêu thay thế có cắt ngưỡng. Trong bài toán căn chỉnh mô hình ngôn ngữ, quy trình ban đầu bổ sung **mô hình phần thưởng**, sau đó nhanh chóng xuất hiện các phương pháp **loại bỏ vòng lặp RL tường minh**.

## 1. RLHF với PPO

Theo [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) và [Bai et al., 2022](https://arxiv.org/abs/2204.05862) (Anthropic HH), quy trình RLHF gồm ba bước:

1. Tinh chỉnh có giám sát (SFT) trên dữ liệu minh họa.
2. Huấn luyện mô hình phần thưởng $$r_\phi$$ từ các cặp so sánh ưu tiên của con người.
3. Tối ưu chính sách bằng PPO theo phần thưởng $$r_\phi$$ kèm số hạng phạt KL giữ mô hình gần với bản SFT ban đầu.

Đây chính là mô hình *actor–critic* của Chương 21 áp dụng lên chuỗi token, nhưng chi phí huấn luyện cao (cần duy trì thêm mô hình giá trị, phải lấy mẫu trực tuyến và dễ mất ổn định).

## 2. DPO: Tối ưu sở thích không cần mô hình phần thưởng

[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) biến đổi bài toán tối ưu RLHF thành hàm mất mát **phân loại** trực tiếp trên các cặp phản hồi thắng/thua $$(y_w, y_l)$$:

$$\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Big).$$

Phương pháp này không cần vòng lặp lấy mẫu trực tuyến khi huấn luyện. DPO cùng các biến thể (IPO, KTO, ORPO) trở thành công thức căn chỉnh phổ biến trong hệ sinh thái mã nguồn mở giai đoạn 2024–2025 (ví dụ thư viện [trl](https://github.com/huggingface/trl)).

## 3. Ứng dụng cụ thể

### GRPO và huấn luyện mô hình suy luận

[Shao et al., 2024](https://arxiv.org/abs/2402.03300) (DeepSeekMath) đề xuất **Group Relative Policy Optimization (GRPO)**: với mỗi prompt, mô hình sinh ra một nhóm nhiều câu trả lời rồi ước lượng lợi thế (*advantage*) tương đối từ phần thưởng của chính nhóm đó, không cần huấn luyện mạng giá trị (*critic*) riêng. Các mô hình suy luận sau đó (như dòng DeepSeek-R1 năm 2025) đã phổ biến hóa hướng tiếp cận này.

### Mô hình thế giới (World Model)

[Hafner et al., 2023](https://arxiv.org/abs/2301.04104) (DreamerV3) học một MDP trong không gian ẩn (kiểu VAE) và lập kế hoạch trực tiếp bên trong mô hình đó. Đây là nhánh bổ sung quan trọng của deep RL ngoài bài toán hội thoại LLM.

### Phần mềm phổ biến

- [huggingface/trl](https://github.com/huggingface/trl) — bộ công cụ huấn luyện PPO, DPO và GRPO.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) và [verl](https://github.com/volcengine/verl) — khung huấn luyện RLHF phân tán.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) và [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — thư viện deep RL cổ điển (ngoài LLM).

## 4. Trích dẫn (2022–2026)

- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155).
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290).
- [DeepSeekMath — GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300).
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Trước hết cần nắm vững cơ chế DQN và PPO trong bài cốt lõi. Bài này bổ sung các kỹ thuật **tối ưu theo sở thích cho mô hình ngôn ngữ** (RLHF, DPO, GRPO), không thay đổi các phương trình Bellman nền tảng. Phần tổng hợp lộ trình căn chỉnh LLM (SFT → mô hình phần thưởng / PPO → DPO) nằm ở bài **26-03**.

## Video tham khảo

- **Xem trước** [Proximal Policy Optimization (PPO) for LLMs Explained Intuitively](https://www.youtube.com/watch?v=8jtAzxUwDj0) — Julia Turc, tiếng Anh. Giải thích trực quan cách PPO được áp dụng trong quy trình RLHF cho mô hình ngôn ngữ lớn.
- [Direct Preference Optimization (DPO) - How to fine-tune LLMs directly without reinforcement learning](https://www.youtube.com/watch?v=k2pD3k1485A) — Luis Serrano Academy, tiếng Anh. Trình bày DPO, mục tiêu tối ưu sở thích được đặt cạnh PPO trong bài.
- [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained!!!](https://www.youtube.com/watch?v=qPN_XZcJf_s) — StatQuest with Josh Starmer, tiếng Anh. Minh họa vòng phản hồi con người mà cả PPO và DPO đều hướng tới tối ưu.
