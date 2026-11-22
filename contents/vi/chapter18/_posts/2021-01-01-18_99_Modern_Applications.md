---
layout: post
title: 18-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '18'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter18
lesson_type: optional
---

# Tùy chọn: NLP sau nhúng từ tĩnh — LLM, RAG và Tokenizer

> Bài này là **tùy chọn**, **không** thay thế phần hình học của Word2Vec hay GloVe. Bài trước đã giới thiệu vectơ theo ngữ cảnh của BERT; bài này cập nhật **hệ sinh thái ứng dụng** được dùng rộng rãi trong các sản phẩm NLP giai đoạn 2022–2026.

Nhúng từ tĩnh vẫn là mô hình dẫn nhập trực quan cho nguyên lý “gần nhau là tương tự”. Tuy nhiên, hầu hết hệ thống NLP mới không dừng ở đó mà sử dụng **mô hình ngôn ngữ chỉ-decoder** kết hợp với truy xuất thông tin, gọi công cụ hoặc trích xuất dữ liệu có cấu trúc.

## 1. Từ một vector cho mỗi từ đến câu trả lời được sinh ra

Một dịch vụ NLP trong giai đoạn 2024–2026 thường gồm ba thành phần chính:

1. Tokenizer tách dưới từ (BPE / Unigram: [tiktoken](https://github.com/openai/tiktoken), SentencePiece).
2. Mô hình Transformer tinh chỉnh theo chỉ dẫn (xem bài tùy chọn Chương 08).
3. **RAG** (tùy chọn): nhúng câu truy vấn, truy xuất các đoạn tài liệu liên quan rồi đưa vào ngữ cảnh của mô hình ngôn ngữ.

Bảng nhúng $$E\in\mathbb{R}^{V\times d}$$ đã học trong chương này vẫn là tầng đầu tiên (tầng 0) của mô hình ngôn ngữ đó; phần triển khai thực tế xoay quanh các lớp xử lý phía trên.

## 2. Ứng dụng cụ thể

### Sinh tăng cường truy hồi (RAG)

[Lewis et al., 2020](https://arxiv.org/abs/2005.11401) (RAG, được áp dụng rộng rãi từ sau 2022) cùng các bộ nhúng truy xuất như Dense Passage, [E5](https://arxiv.org/abs/2212.03533) và [GTE](https://arxiv.org/abs/2308.03281) biến mô hình “truy xuất + sinh văn bản” thành kiến trúc phổ biến trong NLP doanh nghiệp. Khi đánh giá hệ thống loại này, cần kiểm tra mức độ bám sát nguồn (*groundedness*) thay vì chỉ dựa vào các chỉ số như BLEU.

### Trích xuất có cấu trúc và phân loại

Thay vì xây dựng mô hình BiLSTM-CRF riêng cho từng tác vụ, nhiều nhóm kỹ sư dùng prompt hoặc tinh chỉnh LoRA cho mô hình ngôn ngữ để xuất trực tiếp định dạng JSON (phục vụ NER, phân tích cảm xúc, định tuyến yêu cầu hỗ trợ). Dù vậy, các bộ mã hóa nhỏ như BERT hay DeBERTa vẫn được ưu tiên khi độ trễ thấp và độ hiệu chỉnh xác suất (*calibration*) quan trọng hơn một mô hình decoder 7B.

### Tokenization trong thực hành

Cách chia từ vựng dưới từ (*subword*) ảnh hưởng trực tiếp đến chi phí suy luận và chất lượng xử lý đa ngôn ngữ. Dù thuật toán BPE không mới, việc hiểu và sử dụng thành thạo các tokenizer như `tiktoken` hoặc tokenizer của Llama trên dữ liệu thực tế nhiều nhiễu là kỹ năng quan trọng trong giai đoạn 2022–2026.

## 3. Phần mềm phổ biến

- [huggingface/transformers](https://github.com/huggingface/transformers) và [datasets](https://github.com/huggingface/datasets).
- [langchain](https://github.com/langchain-ai/langchain) và [llama_index](https://github.com/run-llama/llama_index) — khung tích hợp RAG (thuật toán cốt lõi vẫn là nhúng–truy hồi–sinh).
- Sentence-Transformers và E5 — các mô hình nhúng phục vụ truy hồi.

## 4. Trích dẫn (2022–2026)

- [E5 (Wang et al., 2022)](https://arxiv.org/abs/2212.03533).
- [RAG (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) — kiến trúc được triển khai lại trong hầu hết hệ thống giai đoạn 2023–2026.
- [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — mô hình ngôn ngữ mở tinh chỉnh theo chỉ dẫn dùng làm bộ sinh.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Trước hết cần nắm vững số học vectơ của Word2Vec và sự khác biệt giữa nhúng tĩnh với nhúng theo ngữ cảnh. Bài này chỉ bổ sung bối cảnh ứng dụng gồm **RAG, bộ sinh tinh chỉnh theo chỉ dẫn và tokenizer hiện đại**, không thay đổi phần suy diễn skip-gram cốt lõi.

## Video tham khảo

- **Xem trước** [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) — 3Blue1Brown, tiếng Anh. Tổng quan ngắn gọn về cách các mô hình ngôn ngữ lớn sinh câu trả lời, vượt ra ngoài nhúng từ tĩnh.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, tiếng Anh. Giải thích cơ chế biểu diễn theo ngữ cảnh bên trong Transformer thay cho một vectơ cố định cho mỗi từ.
- [\[1hr Talk\] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Andrej Karpathy, tiếng Anh. Trình bày cách ứng dụng và vận hành các mô hình ngôn ngữ lớn trong thực tế.
