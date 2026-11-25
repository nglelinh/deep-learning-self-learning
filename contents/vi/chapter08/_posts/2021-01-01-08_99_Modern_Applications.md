---
layout: post
title: 08-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '08'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter08
lesson_type: optional
---

Nếu đem sơ đồ khối Transformer gốc năm 2017 đặt cạnh một mô hình ngôn ngữ lớn hiện đại giai đoạn 2024–2026 như **Llama 3**, **Qwen 2.5**, **DeepSeek** hay **Gemma 2**, bạn sẽ thấy bộ khung tổng thể vẫn gần như nguyên vẹn — nhưng gần như **mọi linh kiện bên trong** đều đã được tinh chỉnh sắc bén hơn để huấn luyện ổn định hơn ở quy mô nghìn tỷ token và suy luận nhanh hơn trên phần cứng thực tế.

## Kiến thức cần nhớ: Bốn nâng cấp tiêu chuẩn của Khối Transformer Hiện đại (2022–2026)

1. **RMSNorm + Pre-LN thay cho Post-LN LayerNorm**:
   RMSNorm bỏ bước trừ giá trị trung bình (chỉ chia cho căn trung bình bình phương $$\text{RMS}(x)$$), giảm ~10–15% chi phí tính toán chuẩn hóa mà chất lượng và độ ổn định tương đương LayerNorm.
2. **RoPE (Rotary Position Embedding) thay cho Sinusoidal/Learned PE**:
   Thay vì cộng vector vị trí ở tầng đầu, RoPE **xoay** trực tiếp các cặp tọa độ của vector $$Q$$ và $$K$$ theo góc tỷ lệ với vị trí $$m, n$$ — khiến tích vô hướng $$q_m^T k_n$$ chỉ phụ thuộc vào **khoảng cách tương đối** $$(m - n)$$ và dễ ngoại suy sang ngữ cảnh dài hơn!
3. **SwiGLU thay cho FFN + ReLU cổ điển**:
   Thêm nhánh cổng nhân (gating): $$\text{SwiGLU}(x) = (\text{SiLU}(x W_1) \odot (x W_3)) W_2$$. Vì có 3 ma trận thay vì 2, chiều ẩn trung gian thường lấy $$\frac{8}{3} d_{\text{model}}$$ thay vì $$4 d_{\text{model}}$$ để giữ nguyên tổng số tham số mà độ chính xác tăng rõ rệt.
4. **Mixture of Experts (MoE) và Transformer Đa phương thức (Multimodal)**:
   Thay 1 khối FFN đặc bằng nhiều chuyên gia (ví dụ 8–64 experts) nhưng chỉ kích hoạt Top-2 chuyên gia cho mỗi token (như Mixtral, DeepSeek-V3), cho phép mở rộng dung lượng tri thức lên hàng trăm tỷ tham số trong khi chi phí suy luận chỉ bằng mô hình cỡ nhỏ.

### Giải thích: Tính tay phép xoay vị trí RoPE 2 chiều để thấy tính bất biến khoảng cách tương đối

Xét vector 2D $$\mathbf{q} = [1, 0]^T$$ đặt ở vị trí $$m$$ và $$\mathbf{k} = [1, 0]^T$$ đặt ở vị trí $$n$$, với góc xoay cơ sở $$\theta = 30^\circ$$. Ma trận xoay RoPE ở vị trí $$t$$ là $$R(t\theta) = \begin{bmatrix} \cos t\theta & -\sin t\theta \\ \sin t\theta & \cos t\theta \end{bmatrix}$$:

1. **Trường hợp 1 ($$m = 1, \; n = 2$$, khoảng cách $$n - m = 1$$)**:
   - $$\tilde{\mathbf{q}}_1 = R(30^\circ)\mathbf{q} = [\cos 30^\circ, \; \sin 30^\circ]^T = [0{,}866, \; 0{,}500]^T$$
   - $$\tilde{\mathbf{k}}_2 = R(60^\circ)\mathbf{k} = [\cos 60^\circ, \; \sin 60^\circ]^T = [0{,}500, \; 0{,}866]^T$$
   - Tích vô hướng: $$\tilde{\mathbf{q}}_1^T \tilde{\mathbf{k}}_2 = (0{,}866)(0{,}500) + (0{,}500)(0{,}866) = \mathbf{0{,}866} = \cos(30^\circ)$$
2. **Trường hợp 2 (dịch cả hai từ xuống cuối đoạn văn: $$m = 10, \; n = 11$$, vẫn cách nhau $$1$$)**:
   - $$\tilde{\mathbf{q}}_{10}^T \tilde{\mathbf{k}}_{11} = \mathbf{q}^T R(10\theta)^T R(11\theta) \mathbf{k} = \mathbf{q}^T R((11 - 10)\theta) \mathbf{k} = \cos(30^\circ) = \mathbf{0{,}866}$$!

Dù hai từ đứng ở đầu hay ở tận cuối một tài liệu dài 100.000 từ, điểm chú ý RoPE giữa chúng vẫn **chỉ phụ thuộc vào khoảng cách tương đối $$n - m$$** — một tính chất hình học tuyệt đẹp!

# Tùy chọn: Từ khối Transformer gốc đến hệ thống LLM hiện đại

> Bài này mang tính **tùy chọn** và **không** thay thế phần lý thuyết encoder/decoder hay mã hóa vị trí. Nội dung đối chiếu khối Transformer năm 2017 với cấu trúc LLM giai đoạn 2022–2026 gồm RoPE, KV-cache, mô hình trọng số mở và hạ tầng phục vụ.

Mỗi tầng Transformer vẫn giữ cấu trúc cốt lõi: trao đổi thông tin qua attention rồi biến đổi cục bộ qua MLP. Từ cuối năm 2022 sau sự xuất hiện của ChatGPT, khối kiến trúc này trở thành nền tảng phổ biến cho xử lý ngôn ngữ, sinh mã nguồn và nhiều hệ thống đa phương thức.

## 1. Những thay đổi chính trong khối kiến trúc

Phần lớn mô hình ngôn ngữ chỉ-decoder hiện nay (từ GPT-2 đến Llama, Mistral, Qwen, Gemma) cập nhật một số chi tiết quan trọng:

- **RoPE** ([Su et al., 2021/2023](https://arxiv.org/abs/2104.09864)) thay thế mã hóa vị trí sin–cos tuyệt đối cộng ở đầu vào.
- **Pre-norm kết hợp RMSNorm và SwiGLU** (xem bài tùy chọn Chương 02).
- **GQA và FlashAttention** (xem bài tùy chọn Chương 07).
- **KV-cache khi giải mã**: lưu lại key/value của các token $$1..t-1$$ để mỗi token mới chỉ tốn $$O(t)$$ thay vì tính lại $$O(t^2)$$ từ đầu.

RoPE xoay từng cặp chiều của query và key một góc $$\theta_i t$$ theo vị trí $$t$$:

$$\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
\leftarrow
\begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}.$$

Mã hóa vị trí sin–cos trong bài lý thuyết vẫn là nền tảng đúng về mặt nguyên lý; RoPE là biến thể chuẩn trong các mô hình LLM giai đoạn 2024–2026.

## 2. Các ứng dụng và hệ sinh thái tiêu biểu

### Mô hình ngôn ngữ nền tảng trọng số mở

[LLaMA](https://arxiv.org/abs/2302.13971), [Llama 2](https://arxiv.org/abs/2307.09288), Llama 3, [Mistral 7B](https://arxiv.org/abs/2310.06825), Qwen2 và Gemma 2 là các kiến trúc chỉ-decoder tham chiếu phổ biến; các phiên bản tinh chỉnh theo chỉ dẫn (*instruction-tuned*) thường được dùng trực tiếp trong ứng dụng thực tế.

### Hệ thống phục vụ và suy luận cục bộ

- [vLLM](https://github.com/vllm-project/vllm) — ghép batch liên tục (*continuous batching*) kết hợp PagedAttention.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — suy luận mô hình lượng tử hóa định dạng GGUF trên CPU và Apple Silicon.
- Giải mã suy đoán (*Speculative decoding*, [Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) — mô hình nhỏ dự đoán trước nhiều token để mô hình lớn kiểm chứng song song.

### Transformer đa phương thức

Các hệ thống kết hợp Vision Transformer (ViT) với decoder ngôn ngữ (như LLaVA năm 2023 và các mô hình dòng GPT-4V) áp dụng cùng cấu trúc khối Transformer lên chuỗi mảng ảnh (*image patches*).

## 3. Thư viện và phần mềm phổ biến

- [Hugging Face transformers](https://github.com/huggingface/transformers) — định nghĩa kiến trúc mô hình và giao diện `generate()`.
- [vLLM](https://github.com/vllm-project/vllm) và [SGLang](https://github.com/sgl-project/sglang) — hệ thống phục vụ suy luận thông lượng cao.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — chạy mô hình GGUF cục bộ.

## 4. Trích dẫn (2022–2026)

- [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) và [Llama 2 (2023)](https://arxiv.org/abs/2307.09288).
- [Mistral 7B (Jiang et al., 2023)](https://arxiv.org/abs/2310.06825) — kết hợp sliding-window attention và GQA.
- [RoFormer / RoPE (Su et al.)](https://arxiv.org/abs/2104.09864).
- [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192).

## 5. Vai trò bổ trợ cho nội dung cốt lõi

Bạn nên hoàn thành phần suy diễn encoder–decoder và cài đặt Transformer nhỏ bằng PyTorch trước. Bài đọc thêm này kết nối các khái niệm đó với **hệ thống LLM thực tế**. Chi tiết công thức, sơ đồ khối RMSNorm → attention + RoPE → SwiGLU cùng ước lượng tham số và FLOP được trình bày trong bài **26-07**.

## Video tham khảo

- **Xem trước:** [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown (tiếng Anh). Ôn lại khung khối Transformer chuẩn mà các mô hình hiện đại vẫn kế thừa.
- [How might LLMs store facts | Deep Learning Chapter 7](https://www.youtube.com/watch?v=9-Jl0dxWQs8) — 3Blue1Brown (tiếng Anh). Góc nhìn về vai trò lưu trữ tri thức của khối MLP trong các mô hình decoder hiện đại.
- [How FlashAttention Accelerates Generative AI Revolution](https://www.youtube.com/watch?v=gBMO1JZav44) — Jia-Bin Huang (tiếng Anh). Giải thích FlashAttention dưới góc độ tối ưu hóa hệ thống mà không thay đổi công thức attention.
