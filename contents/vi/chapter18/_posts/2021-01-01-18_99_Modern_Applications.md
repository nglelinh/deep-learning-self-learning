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

Nếu ở năm 2015 kỹ sư NLP chủ yếu tải vector Word2Vec/GloVe về để đưa vào LSTM phân loại cảm xúc, thì ngày nay nhúng từ tĩnh đã nhường sân khấu cho **Mô hình Ngôn ngữ Lớn (LLM)** và **nhúng ngữ nghĩa cấp đoạn văn (Dense Embeddings)** làm trái tim của hệ thống **RAG (Retrieval-Augmented Generation)**. Bài cập nhật này giải thích mạch chuyển tiếp liền lạc từ một vector cho mỗi từ sang hệ thống vừa truy hồi tri thức vừa sinh câu trả lời có căn cứ.

## Bản đồ kiến thức nền: Bốn trụ cột của NLP hiện đại (2022–2026)

- **Subword Tokenization (BPE / Unigram) và nhúng theo ngữ cảnh:** Biến văn bản thành chuỗi token con rồi đi qua nhiều tầng Self-Attention để cùng một từ mang vector khác nhau hoàn toàn theo ngữ cảnh.
- **Truy hồi dày (Dense Retrieval) và RAG:** Nhúng cả câu hỏi lẫn các đoạn tài liệu thành vector chuẩn hóa, tìm top-$$k$$ đoạn gần nhất bằng tích vô hướng (hoặc kết hợp BM25), rồi nạp vào cửa sổ ngữ cảnh của LLM để giảm ảo giác.
- **Trích xuất có cấu trúc (Structured Outputs / JSON Schema):** Dùng giải mã có ràng buộc ngữ pháp để ép LLM trả về dữ liệu JSON hợp lệ 100% cho các hệ thống phần mềm.
- **Đánh giá hệ thống NLP ứng dụng:** Tách biệt rõ hai tầng lỗi trong RAG — lỗi truy hồi (Recall@k, MRR) và lỗi sinh câu trả lời trung thực (Faithfulness) — thay vì đánh giá mơ hồ.

### Khung giải thích (nên đọc chậm): Tính tay điểm truy hồi lai (Hybrid Search: BM25 + Dense Cosine bằng RRF) trong RAG

Trong hệ thống RAG thực tế, ta thường kết hợp xếp hạng từ khóa (BM25) với xếp hạng vector ngữ nghĩa (Dense Embedding) bằng công thức **Reciprocal Rank Fusion (RRF)** với hằng số làm mượt $$k = 60$$:
$$\text{RRF}(d) = \frac{1}{k + r_{\text{BM25}}(d)} + \frac{1}{k + r_{\text{dense}}(d)}.$$

Xét 3 đoạn tài liệu ứng viên cho câu hỏi về mã lỗi kỹ thuật `ERR_409`:
- **Đoạn A** (chứa đúng mã `ERR_409` và giải thích đúng nghĩa): hạng BM25 $$= 1$$, hạng Dense $$= 2$$.
- **Đoạn B** (diễn đạt đồng nghĩa rất tốt nhưng không ghi mã lỗi): hạng BM25 $$= 15$$, hạng Dense $$= 1$$.
- **Đoạn C** (lặp mã `ERR_409` trong log rác): hạng BM25 $$= 2$$, hạng Dense $$= 20$$.

Tính điểm RRF:
1. $$\text{RRF}(A) = \frac{1}{61} + \frac{1}{62} \approx 0{,}01639 + 0{,}01613 = 0{,}03252$$ (**Hạng 1**).
2. $$\text{RRF}(B) = \frac{1}{75} + \frac{1}{61} \approx 0{,}01333 + 0{,}01639 = 0{,}02972$$ (**Hạng 2**).
3. $$\text{RRF}(C) = \frac{1}{62} + \frac{1}{80} \approx 0{,}01613 + 0{,}01250 = 0{,}02863$$ (**Hạng 3**).
RRF tự động đưa đoạn A (tốt cả từ khóa lẫn ngữ nghĩa) lên đầu mà không cần cân chỉnh thang đo khó nhằn giữa điểm BM25 không bị chặn và cosin trong $$[-1, 1]$$!

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
