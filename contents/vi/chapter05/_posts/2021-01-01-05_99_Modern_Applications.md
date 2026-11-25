---
layout: post
title: 05-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '05'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter05
lesson_type: optional
---

Từng có thời điểm ngay sau khi Transformer bùng nổ, nhiều người cho rằng tư tưởng hồi quy (recurrence) đã khép lại sứ mệnh lịch sử. Nhưng khi ngữ cảnh LLM kéo dài lên hàng trăm nghìn rồi hàng triệu token, nút thắt bộ nhớ KV-cache và chi phí $$O(L^2)$$ của Attention đã làm sống dậy một thế hệ **mô hình hồi quy tuyến tính hiện đại** — tiêu biểu là **Mamba (SSM)** và **RWKV** — kết hợp khả năng huấn luyện song song với tốc độ suy luận $$O(1)$$ cho mỗi token.

## Kiến thức cần nhớ: Bốn hướng phát triển của Tư duy Hồi quy giai đoạn 2022–2026

1. **Vì sao hồi quy hồi sinh — Bài toán KV-Cache và Suy luận $$O(1)$$**:
   Transformer phải giữ toàn bộ lịch sử khóa/giá trị trong bộ nhớ GPU (tăng tuyến tính theo độ dài chuỗi), trong khi mô hình hồi quy hiện đại chỉ cần duy trì một trạng thái kích thước **cố định** bất kể chuỗi dài bao nhiêu.
2. **Mô hình Không gian Trạng thái có cấu trúc (SSM) và Mamba**:
   Xuất phát từ phương trình vi phân tuyến tính $$h'(t) = A h(t) + B x(t)$$ được rời rạc hóa, Mamba (2023–2024) bổ sung cơ chế **chọn lọc theo đầu vào** ($$B_t, C_t, \Delta_t$$ phụ thuộc $$x_t$$) và thuật toán quét song song trên SRAM GPU.
3. **RWKV (Receptance Weighted Key Value)**:
   Viết lại attention thành dạng hồi quy tuyến tính cho phép "chế độ Transformer" (song song) khi huấn luyện và "chế độ RNN" (bộ nhớ hằng số) khi suy luận.
4. **Kiến trúc lai (Hybrid SSM–Attention) và thiết bị biên**:
   Các mô hình thực tiễn như Jamba, Zamba hay Nemotron-H xen kẽ các tầng Mamba (rẻ, dài) với một ít tầng Attention (truy xuất chính xác), đồng thời RNN nhỏ vẫn giữ vị thế trong xử lý âm thanh streaming độ trễ thấp.

### Giải thích: So sánh bằng số độ phức tạp suy luận ở độ dài ngữ cảnh $$L = 100{,}000$$ token

Giả sử một mô hình 32 tầng với kích thước trạng thái/ẩn $$d = 4{,}096$$ sinh ra **1 token mới** khi ngữ cảnh đã đạt $$L = 100{,}000$$ token (dùng FP16 = 2 byte/số):

| Kiến trúc | Bộ nhớ trạng thái cần đọc cho 1 token mới | Số phép tính để sinh thêm 1 token | Tính chất mở rộng khi $$L$$ tăng gấp 10 lần |
| :--- | :--- | :--- | :--- |
| **Transformer chuẩn (KV-Cache)** | $$32 \times 2 \times L \times d \times 2 \approx \mathbf{52{,}4\text{ GB}}$$ | Tỷ lệ với $$O(L \cdot d)$$ | Bộ nhớ và độ trễ **tăng gấp 10 lần** |
| **Mô hình hồi quy hiện đại (Mamba / RWKV)** | Trạng thái cố định $$32 \times (N_{\text{state}} \cdot d) \times 2 \approx \mathbf{4\text{–}32\text{ MB}}$$ | Cố định $$O(N_{\text{state}} \cdot d)$$ | Bộ nhớ và độ trễ **không đổi ($$O(1)$$)** |

Con số 52 GB so với vài chục MB giải thích rõ ràng vì sao nguyên lý trạng thái ẩn hồi quy của Chương 05 không hề lỗi thời, mà đang tái sinh mạnh mẽ trong các mô hình ngữ cảnh siêu dài.

# Tùy chọn: Mô hình chuỗi hồi quy sau thời đại Transformer

> Bài này mang tính **tùy chọn** và **không** thay thế phần lý thuyết RNN gốc hay bài cài đặt mô hình ngôn ngữ cấp ký tự. Nội dung tập trung làm rõ những bối cảnh mà cơ chế hồi quy vẫn được sử dụng, cũng như các kiến trúc giai đoạn 2023–2025 đưa ý tưởng "quét theo thời gian" trở lại.

Phương trình của RNN gốc:

$$\mathbf{h}_t = \sigma(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b}).$$

Transformer loại bỏ vòng lặp từ trái sang phải trong quá trình *huấn luyện*. Từ sau năm 2022, các **mô hình không gian trạng thái (state-space models) và hồi quy tuyến tính** đưa cơ chế hồi quy trở lại dưới dạng *quét song song (parallel scan)* với chi phí tính toán tuyến tính theo độ dài chuỗi.

## 1. Vì sao RNN không hoàn toàn biến mất

Transformer tiêu tốn $$O(T^2)$$ bộ nhớ và thời gian theo độ dài chuỗi $$T$$ (khi chưa áp dụng các kỹ thuật xấp xỉ). Các bài toán như nhận dạng giọng nói trực tuyến (speech streaming), điều khiển thời gian thực và phân tích trình tự gen rất dài vẫn cần độ phức tạp $$O(T)$$ cùng trạng thái bộ nhớ gọn nhẹ. Đó chính là động lực ban đầu của RNN, nay được cải tiến bằng cách tham số hóa hiệu quả hơn.

## 2. Các ứng dụng và kiến trúc tiêu biểu

### Mô hình không gian trạng thái có cấu trúc (SSM) và Mamba

[Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) đề xuất Mamba sử dụng SSM có cơ chế chọn lọc, trong đó bước cập nhật trạng thái rời rạc có dạng hồi quy:

$$\mathbf{h}_t = \overline{A}_t \mathbf{h}_{t-1} + \overline{B}_t \mathbf{x}_t, \quad \mathbf{y}_t = C_t \mathbf{h}_t,$$

với $$\overline{A}_t$$ phụ thuộc trực tiếp vào đầu vào (cơ chế *selection*). Quá trình huấn luyện được thực hiện bằng *parallel scan* thay vì vòng lặp `for` tuần tự trong Python. Đây là một trong những câu trả lời nổi bật giai đoạn 2023–2025 cho câu hỏi liệu kiến trúc hồi quy có thể cạnh tranh với attention trong xử lý ngôn ngữ hay không.

### RWKV

[Peng et al., 2023](https://arxiv.org/abs/2305.13048) kết hợp khối trộn theo thời gian (*time-mix*) theo kiểu linear attention với khối trộn theo kênh (*channel-mix*). Nhờ đó, RWKV suy luận như một RNN với kích thước trạng thái cố định, trong khi vẫn huấn luyện song song được. Kiến trúc này được ứng dụng trong các mô hình hội thoại mã nguồn mở khi dung lượng bộ nhớ KV-cache trở thành nút thắt.

### Phạm vi ứng dụng còn lại của RNN cổ điển

Nhận diện từ khóa trên thiết bị biên (on-device keyword spotting), một số mô hình dự báo chuỗi thời gian cơ sở và các bài toán nghiên cứu quy mô nhỏ vẫn sử dụng `nn.RNN` hoặc `nn.GRU`. Trong khi đó, các hệ thống nhận dạng giọng nói (ASR) triển khai thực tế đã chuyển sang Conformer hoặc Whisper (Chương 08 và 19) thay vì dùng mạng Elman RNN gốc.

## 3. Thư viện và mã nguồn phổ biến

- [state-spaces/mamba](https://github.com/state-spaces/mamba) — mã nguồn và kernel chính thức của Mamba.
- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — mã nguồn huấn luyện và suy luận cho RWKV.
- PyTorch `nn.RNN` — API tiêu chuẩn cho các bài tập cốt lõi trong chương này.

## 4. Trích dẫn (2022–2026)

- [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752).
- [RWKV (Peng et al., 2023)](https://arxiv.org/abs/2305.13048).
- [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396) — kiến trúc tiền thân của dòng SSM.

## 5. Vai trò bổ trợ cho nội dung cốt lõi

Bạn nên hoàn thành phần cài đặt RNN gốc và bài về các bẫy thường gặp trước. Bài đọc thêm này chỉ nhằm đối chiếu **hồi quy tuyến tính hiện đại** với Transformer, không thay thế phần phân tích trải mạng (unfolding) và thuật toán BPTT.

## Video tham khảo

- **Xem trước:** [MAMBA and State Space Models explained | SSM explained](https://www.youtube.com/watch?v=vrF3MtGwD0Y) — AI Coffee Break with Letitia (tiếng Anh). Giải thích mô hình không gian trạng thái — lý do cơ chế hồi quy quay trở lại trong giai đoạn 2023–2024.
- [Recurrent Neural Networks (RNNs), Clearly Explained!!!](https://www.youtube.com/watch?v=AsNTP8Kwu80) — StatQuest with Josh Starmer (tiếng Anh). Ôn lại khái niệm trạng thái ẩn mà các kiến trúc hồi quy mới vẫn kế thừa.
- [MIT 6.S191 (2024): Recurrent Neural Networks, Transformers, and Attention](https://www.youtube.com/watch?v=dqoEU9Ac3ek) — Alexander Amini (tiếng Anh). Cho thấy cách các khóa học hiện đại đặt RNN cạnh attention trong cùng một khung phân tích.
