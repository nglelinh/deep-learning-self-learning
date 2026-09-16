---
layout: post
title: 08-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '08'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter08
lesson_type: optional
---

# Tùy chọn: stack Transformer trở thành LLM

> Bài này là **tùy chọn**. Nó **không** viết lại lý thuyết encoder/decoder hay mã hóa vị trí. Nó ánh xạ khối 2017 sang stack LLM 2022–2026: RoPE, KV-cache, trọng số mở, và serving.

Một lớp Transformer vẫn là “giao tiếp (attention) rồi tính (MLP).” Sau ChatGPT (cuối 2022), khối đó là **máy tính mặc định** cho ngôn ngữ, mã, và nhiều hệ đa phương thức.

## 1. Khối thực sự đổi gì

Hầu hết LM chỉ decoder (dòng GPT-2 → Llama / Mistral / Qwen / Gemma) dùng:

- **RoPE** ([Su et al., 2021/2023](https://arxiv.org/abs/2104.09864)) thay sin–cos tuyệt đối cộng vào embedding,
- **pre-norm + RMSNorm + SwiGLU** (bài tùy chọn Chương 02),
- **GQA** và FlashAttention (bài tùy chọn Chương 07),
- **KV-cache** lúc decode: key/value của token $$1..t-1$$ được tái sử dụng nên mỗi token mới tốn $$O(t)$$, không phải nhân $$O(t^2)$$ từ đầu.

RoPE xoay từng cặp chiều một góc $$\theta_i t$$:

$$\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
\leftarrow
\begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}.$$

PE sin–cos trong bài lý thuyết vẫn là mô hình đầu đúng; RoPE là thứ bạn thấy trong checkpoint 2025.

## 2. Ứng dụng cụ thể

### LM nền tảng mở

[LLaMA](https://arxiv.org/abs/2302.13971), [Llama 2](https://arxiv.org/abs/2307.09288), Llama 3, [Mistral 7B](https://arxiv.org/abs/2310.06825), Qwen2, Gemma 2 là stack decoder tham chiếu. Biến thể tinh chỉnh theo chỉ dẫn là thứ sản phẩm gọi.

### Serving và suy luận cục bộ

- [vLLM](https://github.com/vllm-project/vllm) — batch liên tục + PagedAttention.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — suy luận lượng tử GGUF trên CPU và Apple Silicon.
- Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) — mô hình nhỏ đề xuất token; mô hình lớn kiểm song song.

### Transformer đa phương thức

ViT cộng decoder ngôn ngữ (LLaVA, 2023; các hệ lớp GPT-4V sau đó) tái sử dụng cùng khối trên patch ảnh. Bài tập kiến trúc ở lại chương này; sản phẩm vision-language là ứng dụng.

## 3. Phần mềm phổ biến

- [Hugging Face transformers](https://github.com/huggingface/transformers) — định nghĩa mô hình và `generate()`.
- [vLLM](https://github.com/vllm-project/vllm) và [SGLang](https://github.com/sgl-project/sglang) — serving thông lượng cao.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF cục bộ.

## 4. Trích dẫn (2022–2026)

- [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) và [Llama 2 (2023)](https://arxiv.org/abs/2307.09288).
- [Mistral 7B (Jiang et al., 2023)](https://arxiv.org/abs/2310.06825) — cửa sổ trượt + GQA.
- [RoFormer / RoPE (Su et al.)](https://arxiv.org/abs/2104.09864).
- [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Hoàn thành suy diễn encoder–decoder và Transformer PyTorch đồ chơi trước. Bài này chỉ đặt tên **stack sản phẩm LLM** xây trên suy diễn đó.
