---
layout: post
title: 15-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '15'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter15
lesson_type: optional
---

# Tùy chọn: Học chuyển giao theo hướng PEFT — LoRA, QLoRA và Adapter

> Bài này mang tính **tùy chọn** và **không** thay thế phần lý thuyết so sánh trích xuất đặc trưng với tinh chỉnh toàn phần. Các bài chính xuất phát từ ResNet và BERT; bài này trình bày cách tiếp cận mặc định giai đoạn 2022–2026 cho các mô hình **hàng tỷ tham số**.

Tinh chỉnh toàn phần cập nhật mọi tham số $$\theta$$. Tinh chỉnh tiết kiệm tham số (PEFT) đóng băng $$\theta$$ và chỉ học một phần bổ sung $$\Delta$$ có kích thước nhỏ:

$$W' = W + \frac{\alpha}{r} BA, \quad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r\ll \min(d,k).$$

Đây chính là công thức của [LoRA (Hu et al., 2021/2022)](https://arxiv.org/abs/2106.09685). Về bản chất vẫn là *học chuyển giao*: đặc trưng tiền huấn luyện được thích nghi cho tác vụ đích, nhưng bị ràng buộc trong một không gian con hạng thấp.

## 1. QLoRA và chuyển giao trên mô hình lượng tử hóa

[Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) giữ nguyên bộ trọng số nền ở định dạng lượng tử hóa 4-bit (NF4) và chỉ huấn luyện các ma trận LoRA ở độ chính xác 16-bit, giúp tinh chỉnh mô hình 65B trên một GPU 48 GB. Sự kết hợp giữa [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) và [Hugging Face PEFT](https://github.com/huggingface/peft) đã biến QLoRA thành lựa chọn phổ biến trong khoa học dữ liệu ứng dụng.

## 2. Một số ứng dụng tiêu biểu

### Adapter theo miền và theo chỉ dẫn

Thay vì lưu một bản sao mô hình hoàn chỉnh cho từng khách hàng hay lĩnh vực, hệ thống chỉ cần triển khai **một mô hình nền kết hợp nhiều adapter LoRA** (pháp lý, y tế, SQL). Các nền tảng phục vụ suy luận như vLLM hay SGLang hỗ trợ hoán đổi adapter linh hoạt theo từng yêu cầu.

### PEFT trong thị giác và đa phương thức

Áp dụng LoRA lên các tầng attention của U-Net hoặc DiT (như DreamBooth-LoRA, SDXL LoRA) cho phép học thêm một phong cách hoặc đối tượng mới chỉ từ khoảng 10 ảnh — sử dụng cùng công thức hạng thấp như LoRA ngôn ngữ nhưng đặt trên các tensor thị giác.

### Khi nào *không* nên dùng LoRA

Với mạng CNN cỡ nhỏ, mô hình dữ liệu bảng huấn luyện từ đầu hoặc trường hợp miền đích lệch mạnh nhưng có nhiều dữ liệu gán nhãn, tinh chỉnh toàn phần hoặc thay mới nhiều tầng vẫn phù hợp hơn — tương đồng với nguyên tắc chọn giữa trích xuất đặc trưng và tinh chỉnh trong bài lý thuyết.

## 3. Thư viện và mã nguồn phổ biến

- [huggingface/peft](https://github.com/huggingface/peft) — hỗ trợ LoRA, AdaLoRA, IA3 và prefix-tuning.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) kết hợp [QLoRA](https://github.com/artidoro/qlora).
- [unsloth](https://github.com/unslothai/unsloth) — thư viện tăng tốc huấn luyện LoRA phổ biến trong giai đoạn 2024–2026.

## 4. Tài liệu tham khảo (2022–2026)

- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685).
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314).
- [Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691) — một hướng tiếp cận khác trong nhóm PEFT.

## 5. Phần bổ sung so với nội dung cốt lõi

Bạn nên hoàn thành phần thực hành tinh chỉnh ResNet/BERT trước. Bài này bổ sung cách tiếp cận chuyển giao **hạng thấp và lượng tử hóa**, thay thế cho cách "mở băng vài tầng cuối" truyền thống khi làm việc với LLM.

## Video tham khảo

- **Xem trước:** [QLoRA: Efficient Finetuning of Quantized LLMs | Tim Dettmers](https://www.youtube.com/watch?v=fQirE9N5q_Y) — AMLD Intelligence Summit, tiếng Anh. Bài thuyết trình của chính tác giả Tim Dettmers về QLoRA, phương pháp tinh chỉnh hạng thấp trên mô hình lượng tử hóa được trình bày trong bài.
- [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, tiếng Anh. Giải thích nền tảng số học độ chính xác thấp mà QLoRA dựa vào.
- [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, tiếng Anh. Ôn lại tinh chỉnh truyền thống để thấy QLoRA thay đổi phương thức tính toán chứ không thay đổi mục tiêu chuyển giao.
