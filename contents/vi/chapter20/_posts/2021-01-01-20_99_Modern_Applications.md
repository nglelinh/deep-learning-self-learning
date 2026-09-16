---
layout: post
title: 20-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '20'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter20
lesson_type: optional
---

# Tùy chọn: nền tảng RL trong production — bandit, recsys, và MDP của RLHF

> Bài này là **tùy chọn**. Nó **không** thay MDP, lợi tức, hay định nghĩa chính sách/giá trị. Nó cho thấy các đối tượng ấy xuất hiện ở đâu trong hệ *khoa học dữ liệu* 2022–2026 — đặc biệt huấn luyện ngôn ngữ theo preference.

MDP $$(S,A,P,R,\gamma)$$ vẫn là mô hình. Ứng dụng bất ngờ: **căn chỉnh chat** là MDP mà “môi trường” là con người (hoặc mô hình thưởng) chấm cả bản ghi.

## 1. RLHF như MDP bạn đã biết

[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) (InstructGPT) coi mỗi prompt là trạng thái đầu, token là hành động, và thưởng học được $$r_\phi$$ là $$R$$. PPO (Chương 21) rồi cực đại hóa

$$J(\theta) = \mathbb{E}_{\pi_\theta}[r_\phi] - \beta\, D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}}).$$

Hạng KL là cùng ý “đừng rời chính sách hành vi” như RL bảo thủ. Chi tiết PPO/DPO chờ bài tùy chọn Chương 21; ở đây điểm là **từ vựng Chương 20 là thứ ngành đã dùng**.

## 2. Ứng dụng cụ thể

### Contextual bandit trong xếp hạng

Bảng tin và thông báo thường là **bandit**, không phải MDP đầy đủ: một hành động, thưởng tức thì (click). Thư viện: [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit), [Ray RLlib](https://github.com/ray-project/ray). Phần khám phá–khai thác của chương này là lý thuyết.

### RL offline / batch trên log

Khi không reset được nhà máy hay thị trường, bạn học từ $$(s,a,r,s')$$ đã ghi. [Levine et al., 2020](https://arxiv.org/abs/2005.01643) khảo sát RL offline; việc ứng dụng 2022–2025 dùng conservative Q-learning và fitted policy trên log gợi ý.

### Mô phỏng trước thưởng thật

Robot và quảng cáo vẫn dựng simulator để MDP rẻ. Định nghĩa $$P$$ và $$R$$ trong chương này là checklist cho “simulator có đang nói dối?”

## 3. Phần mềm phổ biến

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) — fork 2023+ của Gym.
- [huggingface/trl](https://github.com/huggingface/trl) — trainer RLHF (xem Chương 21).
- [vw](https://github.com/VowpalWabbit/vowpal_wabbit) — bandit production.

## 4. Trích dẫn (2022–2026)

- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155).
- [Gymnasium](https://gymnasium.farama.org/) — API MDP được duy trì, dùng trong khóa 2023–2026.
- [Offline RL survey (Levine et al., 2020)](https://arxiv.org/abs/2005.01643) — vẫn là bản đồ cho việc DS dựa log.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Học trạng thái, lợi tức, và phương trình Bellman trước. Bài này chỉ đặt tên **các MDP đã giao hàng** (bandit, log offline, RLHF) — không phải định nghĩa mới của hàm giá trị.
