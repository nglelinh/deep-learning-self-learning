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
