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

# Tùy chọn: học chuyển giao như PEFT — LoRA, QLoRA, và adapter

> Bài này là **tùy chọn**. Nó **không** thay lý thuyết tách đặc trưng so với fine-tune toàn phần. Ghi chú cốt lõi vẫn bắt đầu từ đầu ResNet/BERT; bài này là mặc định 2022–2026 cho mô hình **tỷ tham số**.

Fine-tune đầy đủ cập nhật mọi $$\theta$$. Fine-tune tiết kiệm tham số (PEFT) đóng băng $$\theta$$ và học một $$\Delta$$ nhỏ:

$$W' = W + \frac{\alpha}{r} BA, \quad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r\ll \min(d,k).$$

Đó là [LoRA (Hu et al., 2021/2022)](https://arxiv.org/abs/2106.09685). Vẫn là *học chuyển giao*: đặc trưng pretrained dịch chuyển, nhưng chỉ trong không gian con hạng thấp.

## 1. QLoRA và chuyển giao lượng tử

[Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) giữ trọng số gốc 4-bit (NF4) và huấn luyện LoRA 16-bit. Mô hình 65B fine-tune được trên một GPU 48 GB. [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) cộng [Hugging Face PEFT](https://github.com/huggingface/peft) biến đây thành mặc định DS ứng dụng.

## 2. Ứng dụng cụ thể

### Adapter chỉ dẫn và miền

Thay vì BERT mới cho mỗi khách, đội giao **một base + nhiều LoRA** (pháp lý, y tế, SQL). Stack serving (vLLM, SGLang) đổi adapter theo request.

### PEFT thị giác và đa phương thức

LoRA trên attention U-Net / DiT (DreamBooth-LoRA, SDXL LoRA) là đường “huấn luyện phong cách trên 10 ảnh.” Cùng toán với LoRA ngôn ngữ; tensor khác.

### Khi *không* LoRA

CNN nhỏ, mạng bảng từ đầu, và miền lệch nặng có nhiều nhãn vẫn muốn fine-tune đầy đủ hoặc đầu mới — đúng tách feature-extractor vs fine-tune của bài lý thuyết.

## 3. Phần mềm phổ biến

- [huggingface/peft](https://github.com/huggingface/peft) — LoRA, AdaLoRA, IA3, prefix-tuning.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) + [QLoRA](https://github.com/artidoro/qlora).
- [unsloth](https://github.com/unslothai/unsloth) — trainer LoRA nhanh trong tutorial 2024–2026.

## 4. Trích dẫn (2022–2026)

- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685).
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314).
- [Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691) — cực PEFT kia.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Làm bài fine-tune ResNet/BERT trước. Bài này chỉ thêm công thức chuyển giao **hạng thấp / lượng tử** đã thay “mở đóng ba lớp cuối” cho LLM.
