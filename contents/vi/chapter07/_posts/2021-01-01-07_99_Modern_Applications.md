---
layout: post
title: 07-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '07'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter07
lesson_type: optional
---

Công thức Attention chuẩn có một điểm yếu chí mạng ở cấp độ phần cứng: để tính `softmax(QK^T / sqrt(d))`, GPU phải ghi cả ma trận điểm số $$L \times L$$ ra bộ nhớ chính HBM chậm chạp rồi lại đọc ngược vào — khiến Attention bị nghẽn băng thông bộ nhớ (*memory-bound*) ngay cả khi nhân tính toán Tensor Core đang rảnh rỗi! Giai đoạn 2022–2026 chứng kiến cuộc cách mạng kỹ thuật hệ thống với **FlashAttention**, **GQA/MQA** và **PagedAttention**, giúp kéo dài ngữ cảnh LLM từ 4K lên hàng triệu token.

## Bản đồ kiến thức nền: Bốn đột phá hệ thống về Attention (2022–2026)

1. **FlashAttention (Tri Dao et al., 2022–2024) — Chính xác 100%, bộ nhớ $$O(L)$$**:
   Không hề xấp xỉ toán học! FlashAttention chia nhỏ $$Q, K, V$$ thành các khối (tiles) vừa khít bộ nhớ siêu nhanh SRAM trên chip GPU, dùng thuật toán *online softmax* để tính trọn vẹn Attention mà **không bao giờ** phải ghi ma trận $$L \times L$$ xuống HBM.
2. **Multi-Query (MQA) và Grouped-Query Attention (GQA)**:
   Trong suy luận LLM, nút thắt lớn nhất là kích thước **KV-Cache**. MQA cho tất cả các Query head dùng chung 1 cặp K/V; GQA (dùng trong Llama 3, Mistral, Qwen) chia $$H_q$$ đầu query thành $$G$$ nhóm dùng chung K/V — giảm bộ nhớ KV-cache tới 4–8 lần mà chất lượng gần như không đổi.
3. **PagedAttention (vLLM) — Quản lý bộ nhớ ảo cho KV-Cache**:
   Mượn ý tưởng phân trang (virtual memory paging) của hệ điều hành, PagedAttention chia KV-cache thành các khối nhỏ không cần liên tục trong VRAM, loại bỏ tình trạng phân mảnh bộ nhớ và tăng thông lượng phục vụ LLM gấp nhiều lần.
4. **Sliding-Window và Sparse/Linear Attention**:
   Giới hạn mỗi token chỉ chú ý vào cửa sổ cục bộ $$W$$ token gần nhất (Mistral) hoặc kết hợp các token neo (*attention sinks*), giảm độ phức tạp từ $$O(L^2)$$ xuống $$O(L \cdot W)$$.

### Khung giải thích (nên đọc chậm): Tính tay mức tiết kiệm VRAM của GQA và FlashAttention ở ngữ cảnh $$L = 8{,}192$$

**Phần A — FlashAttention vs. Standard Attention (1 layer, 1 batch, $$H=32$$ head, FP16 = 2 byte)**:
- Ma trận điểm số $$S = Q K^T$$ nếu lưu ra HBM chiếm:
  $$32 \times 8{,}192 \times 8{,}192 \times 2\text{ bytes} \approx \mathbf{4{,}29\text{ GB}}$$ (chỉ riêng 1 tầng cho 1 câu!).
- **FlashAttention**: không lưu ma trận $$L \times L$$, chỉ lưu đầu ra và thống kê softmax $$O(L) \Rightarrow$$ giảm bộ nhớ trung gian từ **4,29 GB xuống vài MB**!

**Phần B — Tiết kiệm KV-Cache nhờ GQA (kiểu Llama-3 8B: $$H_q = 32, \; H_{kv} = 8, \; d_k = 128$$, 32 tầng, $$L = 8{,}192$$, FP16)**:
- **MHA chuẩn** ($$H_{kv} = 32$$): $$32\text{ tầng} \times 2\text{ (K,V)} \times 32 \times 128 \times 8{,}192 \times 2\text{ B} = \mathbf{4{,}29\text{ GB}}$$.
- **GQA** ($$H_{kv} = 8$$, tỷ lệ 4:1): giảm đúng **4 lần**, chỉ còn **1,07 GB** — giúp phục vụ đồng thời gấp 4 lần số người dùng trên cùng một GPU!

# Tùy chọn: Kernel attention, hệ thống phục vụ và biến thể multi-query

> Bài này mang tính **tùy chọn** và **không** thay thế phần toán học scaled dot-product hay multi-head attention. Nội dung tập trung vào các kỹ thuật hệ thống và biến thể kiến trúc giúp attention vận hành hiệu quả ở quy mô LLM.

Công thức cốt lõi vẫn giữ nguyên:

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V.$$

Trong giai đoạn 2022–2026, trọng tâm tối ưu hóa chuyển sang **băng thông bộ nhớ (IO)**, **cách quản lý KV-cache** và **giảm số lượng head key/value**.

## 1. FlashAttention: Tính attention chính xác với lưu lượng bộ nhớ thấp

[Dao et al., 2022](https://arxiv.org/abs/2205.14135) chia khối (*tiling*) các ma trận $$Q, K, V$$ trong bộ nhớ đệm nhanh (SRAM) để không phải ghi toàn bộ ma trận điểm số $$T\times T$$ ra bộ nhớ chính GPU (HBM). [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023) tối ưu thêm mức độ song song hóa, còn [FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024) khai thác phần cứng Hopper với thực thi bất đồng bộ và định dạng FP8. Hàm `scaled_dot_product_attention` của PyTorch tự động điều hướng sang các kernel này khi kích thước tensor tương thích.

Kết quả toán học vẫn là phép softmax chính xác như phần lý thuyết; cải tiến nằm ở **cách tổ chức thuật toán theo phân cấp bộ nhớ**.

## 2. Ứng dụng và biến thể tiêu biểu

### PagedAttention và phục vụ LLM

[Kwon et al., 2023](https://arxiv.org/abs/2309.06180) (vLLM) quản lý KV-cache theo từng khối phân trang tương tự bộ nhớ ảo trong hệ điều hành, giúp nhiều chuỗi giải mã chia sẻ bộ nhớ GPU mà không bị phân mảnh hay lãng phí vùng cấp phát trước. Đây là kỹ thuật tối ưu tầng *phục vụ (serving)* chứ không thay đổi hàm chấm điểm.

### GQA và MQA

Multi-query attention ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) và grouped-query attention ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245), GQA) cho phép nhiều query head dùng chung một hoặc một nhóm nhỏ KV head. Các mô hình như Llama 2/3 và Mistral áp dụng GQA để giảm mạnh kích thước KV-cache:

$$\text{KV bytes} \propto T \cdot n_{\text{kv}} \cdot d, \quad n_{\text{kv}} \ll n_{\text{heads}}.$$

### Sliding-window và linear attention

Attention **cửa sổ trượt** (*sliding-window*, như trong Mistral) và các biến thể linear attention thay thế ma trận tương tác đầy đủ $$T\times T$$ bằng độ phức tạp $$O(T)$$, hữu ích khi chi phí bậc hai của attention chuẩn trở thành rào cản triển khai thực tế.

## 3. Thư viện và phần mềm phổ biến

- PyTorch `F.scaled_dot_product_attention` (SDPA) / [FlashAttention](https://github.com/Dao-AILab/flash-attention).
- [vLLM](https://github.com/vllm-project/vllm) — hệ thống phục vụ LLM sử dụng PagedAttention.
- [xFormers](https://github.com/facebookresearch/xformers) — thư viện toán tử attention tiết kiệm bộ nhớ.

## 4. Trích dẫn (2022–2026)

- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135).
- [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691).
- [FlashAttention-3 (Shah et al., 2024)](https://arxiv.org/abs/2407.08608).
- [PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180).
- [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245).

## 5. Vai trò bổ trợ cho nội dung cốt lõi

Bạn nên hoàn thành phần cài đặt attention bằng NumPy và bài về các bẫy multi-head trước. Bài đọc thêm này bổ sung góc nhìn về **kernel tối ưu, KV-cache và GQA** thường gặp trong mã nguồn LLM hiện đại. Bạn có thể xem thêm Chương **26-07** về cơ chế chia khối kết hợp softmax trực tuyến của FlashAttention cùng GQA trong khối decoder, và Chương **26-04** về quản lý KV-cache khi phục vụ mô hình.

## Video tham khảo

- **Xem trước:** [How FlashAttention Accelerates Generative AI Revolution](https://www.youtube.com/watch?v=gBMO1JZav44) — Jia-Bin Huang (tiếng Anh). Giải thích trực quan cách FlashAttention giữ nguyên kết quả attention trong khi giảm mạnh lưu lượng truy cập bộ nhớ.
- [FlashAttention - Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4) — Stanford MLSys Seminars (tiếng Anh). Buổi trình bày gốc của tác giả Tri Dao về thuật toán FlashAttention (nên xem sau video ngắn).
- [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Ôn lại công thức attention chính xác mà FlashAttention tính toán không cần xấp xỉ.
