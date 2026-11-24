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

Nếu năm 2016 câu hỏi mặc định khi bắt đầu một dự án AI là *"Mình nên thu thập bao nhiêu nghìn mẫu để huấn luyện mạng từ đầu?"*, thì trong giai đoạn 2022–2026 câu hỏi mặc định đã hoàn toàn thay đổi: *"Mình nên bắt đầu từ mô hình nền tảng (Foundation Model) nào, thử prompting/RAG trước hay tinh chỉnh nhẹ bằng LoRA?"* Sự chuyển dịch này không hề làm kiến thức nền tảng của bạn trở nên lỗi thời—ngược lại, nó giúp bạn đi xa hơn gấp nhiều lần nếu hiểu rõ quy trình thực hành hiện đại và biết đánh giá rủi ro của từng lựa chọn.

## Bản đồ kiến thức nền: Bốn tầng của Quy trình Thực hành Học sâu Hiện đại

1. **Bậc thang Thích ứng (Adaptation Ladder: Prompting $$\to$$ RAG $$\to$$ LoRA $$\to$$ Pretraining)**:
   Luôn đi từ phương án chi phí thấp, lặp nhanh nhất trước khi đầu tư tinh chỉnh trọng số hay huấn luyện từ đầu—chỉ nâng bậc khi đo lường thực nghiệm chứng minh bậc dưới đã chạm trần.

2. **Đánh giá Hệ thống (Evals-Driven Development)**:
   Trong kỷ nguyên mô hình sinh, độ chính xác đơn thuần không còn đủ; kỹ sư phải xây dựng bộ *evals* tự động đo độ trung thực (chống ảo giác), độ tuân thủ định dạng, độ trễ và chi phí mỗi truy vấn.

3. **Suy luận Hiệu quả và Triển khai (Quantization, KV-Cache, Serving)**:
   Hiểu các nút thắt bộ nhớ băng thông (memory-bandwidth bound) khi sinh token, cách lượng tử hóa xuống `INT8`/`FP8`/`INT4` cắt giảm VRAM gấp 2–4 lần mà hầu như giữ nguyên chất lượng.

4. **Sự kết hợp giữa Mô hình Nền tảng và Mô hình Chuyên biệt**:
   Dùng mô hình lớn làm "giáo viên" sinh dữ liệu tổng hợp hoặc gán nhãn tự động, rồi chưng cất (*distill*) sang mô hình nhỏ chạy siêu nhanh và tiết kiệm chi phí trong sản xuất.

### Khung giải thích (nên đọc chậm): Tính tay lượng VRAM cần thiết để nạp một LLM 7 tỷ tham số ở các độ chính xác khác nhau

Giả sử bạn muốn chạy suy luận một mô hình nền tảng cỡ $$7\text{ tỷ}$$ tham số ($$7 \times 10^9$$) trên GPU cá nhân có 16 GB VRAM:

| Định dạng số học | Số byte / tham số | Bộ nhớ chỉ riêng trọng số (ước tính) | Chạy vừa trên GPU 16 GB VRAM? |
| :--- | :---: | :---: | :--- |
| **FP32 (32-bit)** | $$4\text{ bytes}$$ | $$7 \times 4 = 28\text{ GB}$$ | **Không** (tràn VRAM ngay lập tức) |
| **FP16 / BF16 (16-bit)** | $$2\text{ bytes}$$ | $$7 \times 2 = 14\text{ GB}$$ | **Sát ngưỡng** (chỉ còn 2 GB cho KV-cache) |
| **INT8 (8-bit)** | $$1\text{ byte}$$ | $$7 \times 1 = 7\text{ GB}$$ | **Thoải mái** (còn 9 GB cho ngữ cảnh dài) |
| **INT4 / Q4 (4-bit)** | $$0.5\text{ byte}$$ | $$7 \times 0.5 = 3.5\text{ GB}$$ | **Rất nhẹ** (chạy tốt cả trên laptop 8 GB VRAM) |

Chỉ bằng một phép nhân đơn giản ($$\text{Số tham số} \times \text{Số byte}$$), bạn đã có thể ước tính tức thì cấu hình phần cứng và quyết định chiến lược triển khai mà không cần đoán mò!

# Tùy chọn: Thực hành học sâu trong kỷ nguyên mô hình nền tảng

> Bài này là phần **tùy chọn**, **không** thay thế các bài cốt lõi về giới thiệu học sâu, MNIST hay hướng dẫn khóa học. Mục đích của bài là cập nhật *quy trình thực hành* trong ngành kể từ khi mô hình nền tảng (*foundation model*) trở thành điểm xuất phát mặc định.

Ý tưởng trọng tâm của Chương 01 là mô hình sâu học các đặc trưng phân cấp từ dữ liệu. Từ năm 2022 trở đi, phần lớn dự án thực tế không còn bắt đầu từ một mạng MNIST khởi tạo ngẫu nhiên mà xuất phát từ mô hình **đã được tiền huấn luyện** (*pretrained*) cùng một hạ tầng phục vụ (*serving stack*). Tuy vậy, những câu hỏi cốt lõi — khi nào học sâu thực sự hữu ích, cần loại dữ liệu nào và mô hình dễ sai ở đâu — vẫn quyết định hiệu quả của hệ thống.

## 1. Quy trình mặc định mới

Một dự án ứng dụng điển hình giai đoạn 2024–2026 thường đi qua ba bước:

1. Chọn mô hình nền tảng phù hợp (thị giác, ngôn ngữ, tiếng nói hoặc đa phương thức).
2. Thích nghi mô hình cho bài toán đích bằng kỹ thuật *prompting*, truy xuất thông tin (*retrieval*) hoặc tinh chỉnh tiết kiệm tham số (Chương 15).
3. Đánh giá trên tập kiểm thử giữ lại, *đồng thời* kiểm tra các ràng buộc về an toàn và chi phí vận hành.

Bản chất vẫn dựa trên học có giám sát hoặc học tự giám sát; điểm thay đổi nằm ở **trạng thái khởi tạo** và **giao diện tương tác** (API, *tokenizer*, mẫu hội thoại *chat template*), chứ không phải định nghĩa của mạng nơ-ron.

## 2. Một số ứng dụng cụ thể

### Trợ lý cho khoa học dữ liệu

Các quy trình xử lý dữ liệu dạng bảng và SQL hiện thường tích hợp LLM quanh `pandas`, `scikit-learn` và truy vấn kho dữ liệu. Mô hình không thay thế hàm mất mát hay quy tắc chia tập dữ liệu, mà hỗ trợ viết mã nguồn và giải thích kết quả. Hai thư viện [transformers](https://github.com/huggingface/transformers) và [datasets](https://github.com/huggingface/datasets) của Hugging Face đóng vai trò kết nối phổ biến trong nghiên cứu và phát triển.

### Huấn luyện tối ưu theo ngân sách tính toán

Nghiên cứu [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) chỉ ra rằng nhiều mô hình cỡ GPT-3 trước đây bị **huấn luyện thiếu dữ liệu** (*undertrained*): với ngân sách tính toán FLOP $$C$$ cố định, số tham số tối ưu $$N$$ và số token huấn luyện $$D$$ cần tăng đồng thời. Nhờ đó, các nhóm kỹ thuật hiện thường đặt câu hỏi "tỷ lệ bao nhiêu token trên mỗi tham số?" trước khi đầu tư cụm GPU.

### Hệ sinh thái mô hình trọng số mở

Các thế hệ [LLaMA](https://arxiv.org/abs/2302.13971), [Llama 2](https://arxiv.org/abs/2307.09288) và Llama 3 của Meta, cùng nhiều họ mô hình mở khác (Mistral, Qwen, Gemma), đã giúp việc chạy mô hình ngôn ngữ chất lượng tốt ngay trên một máy trạm với [llama.cpp](https://github.com/ggerganov/llama.cpp) hoặc [vLLM](https://github.com/vllm-project/vllm) trở nên phổ biến. Bài toán MNIST vẫn dạy đúng vòng lặp huấn luyện nền tảng, còn các công cụ trên là môi trường triển khai thực tế của vòng lặp đó.

## 3. Công cụ phần mềm phổ biến

- [PyTorch](https://pytorch.org/) 2.x với `torch.compile` là lựa chọn tăng tốc mặc định.
- [Hugging Face Hub](https://huggingface.co/) — nơi lưu trữ *model card*, *tokenizer* và bộ công cụ đánh giá chuẩn hóa.
- [JAX](https://github.com/google/jax) / Flax — được dùng rộng rãi trong huấn luyện nghiên cứu quy mô lớn.

## 4. Tài liệu tham khảo (2022–2026)

- [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) — quy luật Chinchilla về việc mở rộng đồng thời lượng dữ liệu và kích thước mô hình.
- [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) — mô hình ngôn ngữ nền tảng công bố chi tiết kỹ thuật, tạo bước chuyển lớn cho hệ sinh thái mô hình mở.
- [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — mô hình trọng số mở được tinh chỉnh hội thoại và căn chỉnh bằng RLHF.

## 5. Vai trò của bài này đối với phần cốt lõi

Các bài *Học sâu là gì* và *Bước đầu với MNIST* vẫn là nền tảng định nghĩa lĩnh vực. Bài tùy chọn này chỉ cập nhật *điểm khởi đầu thực tế* vào năm 2026 — gồm trọng số tiền huấn luyện, kho mô hình và cách phân bổ ngân sách theo quy luật mở rộng — chứ không thay đổi khái niệm cốt lõi của mạng nơ-ron.

## Video tham khảo

- **Xem trước:** [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) — 3Blue1Brown, tiếng Anh. Giới thiệu ngắn gọn về tiền huấn luyện, Transformer và RLHF, khớp với quy trình mặc định giai đoạn 2022–2026 trong bài.
- [\[1hr Talk\] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Andrej Karpathy, tiếng Anh. Trình bày chi tiết hơn về toàn bộ quy trình từ dữ liệu, huấn luyện đến triển khai mô hình ngôn ngữ lớn.
- [MIT 6.S191 (2024): Language Models and New Frontiers](https://www.youtube.com/watch?v=N1fbskTpwZ0) — Alexander Amini, tiếng Anh. Bài giảng năm 2024 của MIT về mô hình ngôn ngữ và các hướng phát triển mới, bổ sung góc nhìn thực tiễn cho các công trình được trích dẫn trong bài.
