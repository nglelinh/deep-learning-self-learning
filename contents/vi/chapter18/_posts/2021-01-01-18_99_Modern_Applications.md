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

# Tùy chọn: NLP sau embedding tĩnh — LLM, RAG, và tokenizer

> Bài này là **tùy chọn**. Nó **không** thay hình học Word2Vec / GloVe. Bài hiện có đã chỉ vector BERT ngữ cảnh; ghi chú này phủ **stack ứng dụng** chiếm sản phẩm NLP 2022–2026.

Embedding tĩnh vẫn là mô hình đầu đúng của “gần = giống.” Hầu như không sản phẩm mới *dừng* ở đó: chúng dùng **LM decoder** cộng retrieval, công cụ, hoặc trích xuất có cấu trúc.

## 1. Từ một vector mỗi từ loại tới câu trả lời sinh ra

Một dịch vụ NLP 2024–2026 thường là:

1. Tokenizer (BPE / Unigram: [tiktoken](https://github.com/openai/tiktoken), SentencePiece).
2. Transformer tinh chỉnh theo chỉ dẫn (bài tùy chọn Chương 08).
3. **RAG** tùy chọn: nhúng truy vấn, lấy đoạn, điều kiện hóa LM.

Bảng embedding $$E\in\mathbb{R}^{V\times d}$$ của chương này vẫn là lớp 0 của LM đó. Thứ bạn giao là phần còn lại của stack.

## 2. Ứng dụng cụ thể

### Sinh tăng cường truy hồi

[Lewis et al., 2020](https://arxiv.org/abs/2005.11401) (RAG; phổ biến sau 2022) và các embedder Dense Passage / [E5](https://arxiv.org/abs/2212.03533) / [GTE](https://arxiv.org/abs/2308.03281) biến “tìm + sinh” thành mẫu NLP doanh nghiệp mặc định. Đánh giá theo groundedness, không chỉ BLEU.

### Trích xuất có cấu trúc và phân loại

Thay vì BiLSTM-CRF riêng, đội prompt hoặc LoRA-tune LM để xuất JSON (NER, cảm xúc, định tuyến ticket). Giữ encoder nhỏ (BERT/DeBERTa) khi latency và hiệu chỉnh thắng decoder 7B.

### Tokenization như kỹ năng ứng dụng

Từ vựng subword quyết định chi phí và chất lượng đa ngữ. BPE là cũ; **dùng** tokenizer tiktoken / Llama trên văn bản DS lộn xộn là kỹ năng 2022–2026.

## 3. Phần mềm phổ biến

- [huggingface/transformers](https://github.com/huggingface/transformers) + [datasets](https://github.com/huggingface/datasets).
- [langchain](https://github.com/langchain-ai/langchain) / [llama_index](https://github.com/run-llama/llama_index) — keo RAG (thuật toán vẫn là nhúng–truy hồi–sinh).
- Sentence-Transformers / E5 — embedding truy hồi.

## 4. Trích dẫn (2022–2026)

- [E5 (Wang et al., 2022)](https://arxiv.org/abs/2212.03533).
- [RAG (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) — mẫu mọi stack 2023–2026 tái triển khai.
- [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — LM mở tinh chỉnh chỉ dẫn dùng làm bộ sinh.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Xong số học Word2Vec và ghi chú “tĩnh vs ngữ cảnh” trước. Bài này chỉ thêm **RAG, bộ sinh tinh chỉnh chỉ dẫn, và tokenizer hiện đại** — không phải suy diễn skip-gram mới.
