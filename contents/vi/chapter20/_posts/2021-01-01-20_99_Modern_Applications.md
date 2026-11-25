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

Khi nghe đến Học Tăng cường, nhiều người chỉ nghĩ tới robot đi bộ hay AI chơi cờ vây; nhưng trên thực tế, phần lớn hệ thống RL đang tạo ra giá trị kinh tế khổng lồ mỗi giây hôm nay lại nằm ở **hệ thống gợi ý (Contextual Bandits)**, **RL ngoại tuyến trên dữ liệu log (Offline RL)** và đặc biệt là **khung MDP của RLHF** giúp căn chỉnh các mô hình ngôn ngữ lớn như ChatGPT, Claude và Gemini!

## Kiến thức cần nhớ: Ba ứng dụng thực chiến của nền tảng RL hiện đại (2022–2026)

- **Mô hình hóa sinh văn bản LLM dưới dạng MDP (RLHF):** Coi chuỗi token đã sinh là trạng thái $$s_t$$, token kế tiếp là hành động $$a_t \in V$$, chính sách là chính LLM $$\pi_\theta(a_t\mid s_t)$$, và phần thưởng nằm ở cuối câu kèm phạt KL để không trôi xa mô hình gốc.
- **Contextual Multi-Armed Bandits trong xếp hạng & quảng cáo:** Bài toán RL một bước (không có chuyển trạng thái dài hạn) dùng UCB hoặc Thompson Sampling để vừa khai thác nội dung có tỷ lệ nhấp cao vừa khám phá nội dung mới.
- **RL ngoại tuyến (Offline / Batch RL — CQL, IQL):** Huấn luyện chính sách hoàn toàn từ nhật ký lịch sử cố định (y tế, tài chính, công nghiệp) mà không được phép tương tác mạo hiểm với môi trường thật.
- **Bẫy phân phối lệch (Distributional Shift) trong Offline RL:** Hiểu lý do Q-Learning tiêu chuẩn ước lượng quá cao các hành động ngoài tập log (OOD actions) và cần ràng buộc bảo thủ (conservative penalty).

### Giải thích: Tính điểm chọn cánh tay bằng thuật toán UCB1 (Upper Confidence Bound) trong Bandit

Một hệ thống gợi ý tin tức có 2 bài viết ứng viên sau tổng cộng $$t = 100$$ lượt hiển thị ($$\ln 100 \approx 4{,}605$$):
- **Bài A:** Đã hiển thị $$N_A = 80$$ lần, tỷ lệ nhấp trung bình $$\hat{\mu}_A = 0{,}20$$.
- **Bài B:** Mới hiển thị $$N_B = 5$$ lần, tỷ lệ nhấp trung bình $$\hat{\mu}_B = 0{,}15$$.

Áp dụng công thức UCB1 (hệ số khám phá $$c = 0{,}5$$): $$\text{UCB}(a) = \hat{\mu}_a + c\sqrt{\frac{\ln t}{N_a}}$$:
1. **Bài A:** $$0{,}20 + 0{,}5\sqrt{\frac{4{,}605}{80}} = 0{,}20 + 0{,}5(0{,}240) = 0{,}320$$.
2. **Bài B:** $$0{,}15 + 0{,}5\sqrt{\frac{4{,}605}{5}} = 0{,}15 + 0{,}5(0{,}960) = 0{,}630$$.
3. **Quyết định:** Dù trung bình của B đang thấp hơn A ($$0{,}15 < 0{,}20$$), phần thưởng cho độ bất định của B lớn gấp 4 lần A nên UCB1 chọn **khám phá Bài B** trước — ngay khi $$N_B$$ tăng lên, khoảng tin cậy tự thu hẹp lại mà không cần tinh chỉnh $$\epsilon$$ thủ công!

# Tùy chọn: Nền tảng RL trong thực tế — Bandit, hệ thống gợi ý và MDP của RLHF

> Bài này là **tùy chọn**, **không** thay thế định nghĩa về MDP, tổng phần thưởng chiết khấu hay hàm chính sách/giá trị. Nội dung cho thấy các khái niệm đó xuất hiện ở đâu trong các hệ thống khoa học dữ liệu giai đoạn 2022–2026, đặc biệt là huấn luyện mô hình ngôn ngữ theo sở thích con người.

MDP $$(S,A,P,R,\gamma)$$ vẫn là khung mô hình nền tảng. Một ứng dụng đáng chú ý là **căn chỉnh mô hình hội thoại** (*chat alignment*), trong đó “môi trường” chính là con người (hoặc mô hình phần thưởng) đánh giá toàn bộ câu trả lời sinh ra.

## 1. RLHF dưới góc nhìn MDP

[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) (InstructGPT) mô hình hóa mỗi prompt là trạng thái ban đầu, mỗi token sinh ra là một hành động, và điểm số từ mô hình phần thưởng đã học $$r_\phi$$ đóng vai trò hàm phần thưởng $$R$$. Sau đó, PPO (xem Chương 21) tối đa hóa mục tiêu:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[r_\phi] - \beta\, D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}}).$$

Số hạng phạt KL giữ cho chính sách mới không lệch quá xa chính sách tham chiếu ban đầu, tương tự nguyên lý bảo thủ trong RL. Chi tiết về PPO và DPO sẽ được trình bày ở bài tùy chọn Chương 21; điểm cốt lõi ở đây là hệ thuật ngữ của Chương 20 được sử dụng trực tiếp trong thực tế công nghiệp.

## 2. Ứng dụng cụ thể

### Contextual Bandit trong bài toán xếp hạng

Các hệ thống bảng tin và gửi thông báo thường được mô hình hóa bằng **contextual bandit** thay vì MDP đầy đủ: chỉ có một bước hành động và phần thưởng phản hồi ngay lập tức (như lượt nhấp chuột). Các thư viện phổ biến gồm [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) và [Ray RLlib](https://github.com/ray-project/ray), dựa trực tiếp trên lý thuyết đánh đổi khám phá–khai thác của chương này.

### RL ngoại tuyến (Offline / Batch RL) trên dữ liệu log

Khi không thể thử nghiệm tự do hoặc đặt lại môi trường thực (như nhà máy hay thị trường tài chính), mô hình phải học từ tập dữ liệu tương tác $$(s,a,r,s')$$ đã ghi nhận sẵn. [Levine et al., 2020](https://arxiv.org/abs/2005.01643) tổng hợp nền tảng của RL ngoại tuyến; các ứng dụng giai đoạn 2022–2025 thường sử dụng *conservative Q-learning* và *fitted policy* trên dữ liệu log của hệ thống gợi ý.

### Mô phỏng trước khi triển khai thực tế

Trong robot và hệ thống quảng cáo, môi trường mô phỏng (*simulator*) vẫn được dùng để giảm chi phí tương tác. Định nghĩa chặt chẽ về xác suất chuyển trạng thái $$P$$ và hàm phần thưởng $$R$$ trong chương này là cơ sở để kiểm tra xem bộ mô phỏng có phản ánh đúng thực tế hay không.

## 3. Phần mềm phổ biến

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) — nhánh kế thừa từ năm 2023 trở đi của OpenAI Gym.
- [huggingface/trl](https://github.com/huggingface/trl) — thư viện huấn luyện RLHF (xem thêm Chương 21).
- [vw](https://github.com/VowpalWabbit/vowpal_wabbit) — công cụ triển khai bandit trong sản phẩm thực tế.

## 4. Trích dẫn (2022–2026)

- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155).
- [Gymnasium](https://gymnasium.farama.org/) — giao diện MDP chuẩn được duy trì và dùng phổ biến trong các khóa học giai đoạn 2023–2026.
- [Offline RL survey (Levine et al., 2020)](https://arxiv.org/abs/2005.01643) — tài liệu tổng quan cho các bài toán học tăng cường từ dữ liệu log.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Trước hết cần nắm vững trạng thái, tổng phần thưởng chiết khấu và hệ phương trình Bellman. Bài này chỉ minh họa **các bài toán MDP đã được triển khai rộng rãi** (bandit, RL ngoại tuyến trên log và RLHF), không thay đổi định nghĩa của hàm giá trị.

## Video tham khảo

- **Xem trước** [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained!!!](https://www.youtube.com/watch?v=qPN_XZcJf_s) — StatQuest with Josh Starmer, tiếng Anh. Giải thích RLHF như một bài toán MDP với phần thưởng học từ phản hồi con người, bám sát nội dung bài.
- [Reinforcement Learning from Human Feedback (RLHF) Explained](https://www.youtube.com/watch?v=T_X4XFwKX8k) — IBM Technology, tiếng Anh. Tóm tắt ngắn gọn vòng lặp RLHF theo góc nhìn ứng dụng doanh nghiệp.
- [RL Course by David Silver - Lecture 2: Markov Decision Process](https://www.youtube.com/watch?v=lfHX2hHRMVQ) — Google DeepMind, tiếng Anh. Ôn lại hệ khái niệm MDP làm nền tảng cho các ứng dụng trên.
