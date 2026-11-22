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

# Tùy chọn: LLM hiệu quả — GPTQ, AWQ, GGUF và Speculative Decoding

> Bài này là **tùy chọn**, **không** thay thế lý thuyết về cắt tỉa, lượng tử hóa, chưng cất hay kiến trúc MobileNet/EfficientNet trong bài cốt lõi (vốn vẫn quan trọng với CNN và thị giác biên). Nội dung cập nhật hệ thống **tối ưu phục vụ LLM** giai đoạn 2022–2026.

Công thức lượng tử hóa INT8 cơ bản

$$w_q = \mathrm{round}\big((w-z)/s\big)$$

vẫn giữ nguyên nguyên lý. Điểm mới nằm ở **lượng tử hóa 4-bit chỉ cho trọng số** trên mô hình Transformer và các **thuật toán giải mã** giúp giảm số bước sinh tuần tự.

## 1. Lượng tử hóa sau huấn luyện (PTQ) cho mô hình ngôn ngữ

- [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323) — phương pháp PTQ dùng thông tin đạo hàm bậc hai theo từng lớp, là nền tảng của nhiều trọng số định dạng `gptq` trên Hugging Face Hub.
- [AWQ (Lin et al., 2024)](https://arxiv.org/abs/2306.00978) — lượng tử hóa nhận biết kích hoạt (*activation-aware*), bảo vệ các kênh trọng số quan trọng trước khi làm tròn xuống 4-bit.
- [GGUF / llama.cpp](https://github.com/ggerganov/llama.cpp) — định dạng nén nhiều mức bit (*k-quant* như `Q4_K_M`) tối ưu cho suy luận trên CPU và Apple Silicon.

Các phương pháp này bổ sung cho lý thuyết lượng tử hóa tổng quát của chương bằng những công thức và nhân tính toán (*kernel*) chuyên biệt cho khối *attention*.

## 2. Ứng dụng cụ thể

### LoRA kết hợp lượng tử hóa (QLoRA)

Như đã giới thiệu ở bài tùy chọn Chương 15, QLoRA giữ mô hình gốc ở 4-bit và chỉ huấn luyện các bộ thích ứng (*adapter*) 16-bit. Khi triển khai phục vụ, có thể gộp trực tiếp *adapter* vào trọng số hoặc hoán đổi nhanh giữa nhiều *adapter*, kết hợp lợi ích của cả nén mô hình và học chuyển giao.

### Speculative Decoding

[Leviathan et al., 2023](https://arxiv.org/abs/2211.17192) sử dụng một mô hình nháp nhỏ để đề xuất nhanh $$K$$ token liên tiếp, sau đó mô hình lớn kiểm định song song trong một lượt lan truyền xuôi và chấp nhận tiền tố phù hợp. Kỹ thuật này giảm thời gian suy luận thực tế mà vẫn bảo toàn chính xác phân bố đầu ra $$\pi$$ của mô hình lớn, và đã được tích hợp trong vLLM cùng nhiều dịch vụ API.

### Định tuyến hỗn hợp chuyên gia (MoE)

[Mixtral (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) chỉ kích hoạt một tập con các mạng chuyên gia (*expert*) cho mỗi token: tổng số tham số lưu trữ lớn hơn nhưng lượng FLOP trên mỗi token tương đương một mô hình dày đặc nhỏ hơn. MoE bổ sung cho nén mô hình theo hướng tận dụng *tính thưa khi tính toán* thay vì chỉ giảm số lượng trọng số lưu trữ.

## 3. Phần mềm phổ biến

- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) và [vllm-project/vllm](https://github.com/vllm-project/vllm).

## 4. Trích dẫn (2022–2026)

- [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323).
- [AWQ (Lin et al., 2024)](https://arxiv.org/abs/2306.00978).
- [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192).
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Phần lý thuyết cắt tỉa, chưng cất và MobileNet trong bài cốt lõi vẫn là nền tảng chung. Bài này bổ sung **PTQ chuyên biệt cho LLM, GGUF, speculative decoding và MoE** — những công cụ cần thiết khi đối tượng triển khai là các mô hình Transformer cỡ 7B–70B tham số.

## Video tham khảo

- **Xem trước** [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, tiếng Anh. Giải thích lượng tử hóa sau huấn luyện cho mô hình ngôn ngữ lớn, bám sát trọng tâm của bài.
- [QLoRA: Efficient Finetuning of Quantized LLMs | Tim Dettmers](https://www.youtube.com/watch?v=fQirE9N5q_Y) — AMLD Intelligence Summit, tiếng Anh. Tim Dettmers trình bày QLoRA — kỹ thuật tinh chỉnh trực tiếp trên mô hình nền 4-bit.
- [Knowledge Distillation: How LLMs train each other](https://www.youtube.com/watch?v=jrJKRYAdh7I) — Julia Turc, tiếng Anh. Giới thiệu chưng cất tri thức như một hướng khác để tạo mô hình triển khai nhỏ gọn hơn.
