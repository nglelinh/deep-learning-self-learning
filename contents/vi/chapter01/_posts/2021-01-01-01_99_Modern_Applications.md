---
layout: post
title: 01-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '01'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter01
lesson_type: optional
---

# Tùy chọn: “làm deep learning” sau thời foundation model

> Bài này là **tùy chọn**. Nó **không** thay phần giới thiệu deep learning, MNIST hay hướng dẫn khóa học. Nó cập nhật *cách làm* của ngành sau khi foundation model trở thành điểm xuất phát mặc định.

Ý tưởng chương 01: mô hình sâu học đặc trưng phân cấp từ dữ liệu. Từ 2022, hầu hết việc ứng dụng bắt đầu từ mô hình **đã pretrained** và một stack phục vụ, không phải mạng MNIST trống — nhưng cùng câu hỏi (khi nào DL có ích, cần dữ liệu gì, sai ở đâu) vẫn quyết định thành công.

## 1. Quy trình mặc định mới

Một dự án điển hình 2024–2026:

1. Chọn foundation model (thị giác, ngôn ngữ, tiếng nói, hoặc đa phương thức).
2. Thích nghi bằng prompting, retrieval, hoặc fine-tune tiết kiệm tham số (Chương 15).
3. Đánh giá trên tập giữ lại *và* trên ràng buộc an toàn / chi phí.

Vẫn là học có giám sát hoặc tự giám sát. Cái đổi là **khởi tạo** và **giao diện** (API, tokenizer, chat template), không phải định nghĩa mạng neuron.

## 2. Ứng dụng cụ thể

### Copilot cho khoa học dữ liệu

Các pipeline bảng và SQL bọc LLM quanh pandas, scikit-learn, truy vấn kho. Mô hình không thay loss hay cách chia tập; nó viết mã và lời giải thích. [transformers](https://github.com/huggingface/transformers) và [datasets](https://github.com/huggingface/datasets) của Hugging Face là keo nghiên cứu phổ biến.

### Huấn luyện tối ưu theo compute

[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) chỉ ra nhiều mô hình quy mô GPT-3 bị **undertrain**: với ngân sách FLOP $$C$$ cố định, số tham số tối ưu $$N$$ và số token $$D$$ tăng cùng nhau. Đội ứng dụng giờ hỏi “bao nhiêu token mỗi tham số?” trước khi mua GPU.

### Hệ sinh thái mô hình mở

[LLaMA](https://arxiv.org/abs/2302.13971) / [Llama 2](https://arxiv.org/abs/2307.09288) / Llama 3 của Meta và các họ mở sau đó (Mistral, Qwen, Gemma) khiến việc chạy LM khá trên một máy trạm với [llama.cpp](https://github.com/ggerganov/llama.cpp) hoặc [vLLM](https://github.com/vllm-project/vllm) trở thành bình thường. Bài MNIST vẫn dạy vòng lặp; các stack này là nơi vòng lặp được triển khai.

## 3. Phần mềm phổ biến

- [PyTorch](https://pytorch.org/) 2.x với `torch.compile` là đường tăng tốc mặc định.
- [Hugging Face Hub](https://huggingface.co/) — model card, tokenizer, harness đánh giá.
- [JAX](https://github.com/google/jax) / Flax — huấn luyện nghiên cứu quy mô lớn.

## 4. Trích dẫn (2022–2026)

- [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) — Chinchilla: scale dữ liệu cùng kích thước mô hình.
- [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) — LM nền tảng mô tả công khai, làm reset hệ sinh thái.
- [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — trọng số mở tinh chỉnh hội thoại / RLHF.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Các bài “Học sâu là gì” và MNIST vẫn định nghĩa lĩnh vực. Bài này chỉ cập nhật *chỗ* người mới bắt đầu năm 2026: trọng số pretrained, hub, và ngân sách theo scaling law — không phải định nghĩa mới của mạng neuron.
