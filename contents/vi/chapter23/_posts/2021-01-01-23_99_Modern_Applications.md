---
layout: post
title: 23-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '23'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter23
lesson_type: optional
---

# Tùy chọn: LLM hiệu quả — GPTQ, AWQ, GGUF, và speculative decoding

> Bài này là **tùy chọn**. Nó **không** thay pruning, toán lượng tử, chưng cất, hay MobileNet/EfficientNet trong ghi chú cốt lõi. Những thứ ấy vẫn quan trọng cho CNN và thị giác biên. Bài này là stack hiệu quả **serving LLM** 2022–2026.

INT8

$$w_q = \mathrm{round}\big((w-z)/s\big)$$

cùng ý tưởng. Việc mới là **4-bit chỉ trọng số** cho Transformer và **thuật toán decode** cắt bước tuần tự.

## 1. Lượng tử sau huấn luyện cho LM

- [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323) — PTQ bậc hai theo lớp; nền của nhiều mô hình `gptq` trên Hub.
- [AWQ (Lin et al., 2024)](https://arxiv.org/abs/2306.00978) — bảo vệ kênh *nổi* (activation-aware) trước khi làm tròn 4-bit.
- [GGUF / llama.cpp](https://github.com/ggerganov/llama.cpp) — k-quant (`Q4_K_M`, …) cho CPU và Apple Silicon.

Chúng bổ sung, không trùng, bộ lượng tử tổng quát của chương: đây là **công thức + kernel** cho khối attention.

## 2. Ứng dụng cụ thể

### LoRA + lượng tử (QLoRA)

Đã phác trong bài tùy chọn Chương 15: base 4-bit, adapter 16-bit. Serving rồi merge hoặc đổi adapter. Hiệu quả *và* chuyển giao.

### Speculative decoding

[Leviathan et al., 2023](https://arxiv.org/abs/2211.17192): mô hình nháp rẻ đề xuất $$K$$ token; mô hình lớn chấp nhận một tiền tố trong một lượt thuận song song. Tăng tốc đồng hồ tường mà không đổi $$\pi$$. vLLM và nhiều API giao việc này.

### Định tuyến MoE như hiệu quả

[Mixtral (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) kích hoạt một tập con expert mỗi token: nhiều tham số hơn, FLOP tương tự. Bổ sung cho nén: *tính thưa*, không phải ít trọng số lưu hơn.

## 3. Phần mềm phổ biến

- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) và [vllm-project/vllm](https://github.com/vllm-project/vllm).

## 4. Trích dẫn (2022–2026)

- [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323).
- [AWQ (Lin et al., 2024)](https://arxiv.org/abs/2306.00978).
- [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192).
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Giữ suy diễn pruning / KD / MobileNet. Bài này chỉ thêm **PTQ riêng LLM, GGUF, speculative decoding, và MoE** — công cụ khi “mô hình” là Transformer 7B–70B.

## Video tham khảo

- **xem trước** [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, tiếng Anh. Lượng tử hóa sau huấn luyện cho mô hình ngôn ngữ là ghi chú này.
- [QLoRA: Efficient Finetuning of Quantized LLMs | Tim Dettmers](https://www.youtube.com/watch?v=fQirE9N5q_Y) — AMLD Intelligence Summit, tiếng Anh. QLoRA, lần tinh chỉnh giữ ở 4-bit, của Tim Dettmers.
- [Knowledge Distillation: How LLMs train each other](https://www.youtube.com/watch?v=jrJKRYAdh7I) — Julia Turc, tiếng Anh. Chưng cất như cách khác để có một mô hình triển khai nhỏ hơn.
