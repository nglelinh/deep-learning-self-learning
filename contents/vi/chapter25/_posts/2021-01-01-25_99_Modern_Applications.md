---
layout: post
title: 25-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '25'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter25
lesson_type: optional
---

# Tùy chọn: thứ thực sự đã giao — compute lúc suy luận, agent, đa phương thức

> Bài này là **tùy chọn**. Nó **không** viết lại khảo sát scaling-law / đa phương thức / an toàn của chương. Ghi chú ấy đã nhìn tới; bài này liệt kê **triển khai cụ thể 2024–2026** để phần “tương lai” gắn với hệ bạn chạy được.

Luật lũy thừa của chương cốt lõi

$$L \propto N^{-\alpha}$$

vẫn mô tả pretrain. Núm công nghiệp mới là **compute lúc suy luận**: tốn thêm FLOP *mỗi truy vấn*.

## 1. Scale lúc inference

[OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/) (2024) và các mô hình lập luận kế tiếp (họ DeepSeek-R1, 2025) huấn luyện mô hình phát chuỗi suy nghĩ dài và tìm kiếm. Thực nghiệm, độ chính xác có thể tăng theo thêm mẫu hoặc vết dài hơn, gần

$$\text{error} \approx c \cdot C_{\mathrm{test}}^{-\gamma}$$

hơn là chỉ tăng $$N$$ pretrained ([Snell et al., 2024](https://arxiv.org/abs/2408.03314), “Scaling LLM Test-Time Compute”). Bổ sung cho scale lúc *huấn luyện* của Chinchilla (bài tùy chọn Chương 00/01).

## 2. Ứng dụng cụ thể

### Agent dùng công cụ

“Agent” production (2024–2026) là vòng: LM → công cụ (SQL, trình duyệt, mã) → quan sát. Phần mềm: công cụ Agents / Responses của OpenAI, [LangGraph](https://github.com/langchain-ai/langgraph), [smolagents](https://github.com/huggingface/smolagents). Rủi ro nghiên cứu vẫn là độ tin cậy, không phải neuron mới.

### Đa phương thức gốc

Hệ lớp GPT-4V, Gemini, và VLM mở (LLaVA, Qwen2-VL, PaliGemma) nhận ảnh/audio trong một stack. Con trỏ CLIP/Flamingo của chương là hạt giống nghiên cứu; đây là các API.

### Bền vững như ràng buộc

Phục vụ mô hình 70B cho mọi cú nhấp không còn là “vấn đề tương lai”: lượng tử (bài tùy chọn Chương 23), định tuyến sang mô hình nhỏ, và batching là câu trả lời đã triển khai cho đoạn năng lượng của chương.

## 3. Phần mềm phổ biến

- [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) — thông lượng cao + decode suy diễn / có cấu trúc.
- Model card mô hình lập luận trên Hugging Face (chưng cất DeepSeek-R1, v.v.).
- Đánh giá mở: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## 4. Trích dẫn (2022–2026)

- [Scaling LLM Test-Time Compute (Snell et al., 2024)](https://arxiv.org/abs/2408.03314).
- [DeepSeek-R1 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948) — lập luận quy mô lớn với RL (xem bài tùy chọn Chương 21).
- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — công thức VLM mở được chép rộng 2024–2026.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Giữ thảo luận scaling, NAS, học liên tục, và an toàn của chương. Bài này chỉ ghim **compute lúc suy luận, agent, và VLM mở** như thứ đã chuyển từ “tương lai” sang “sản phẩm mặc định” giữa 2024 và 2026.
