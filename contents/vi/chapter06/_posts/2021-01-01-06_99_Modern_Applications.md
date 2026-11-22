---
layout: post
title: 06-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '06'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter06
lesson_type: optional
---

# Tùy chọn: Hồi quy có cổng sau LSTM — xLSTM và các ứng dụng hiện đại

> Bài này mang tính **tùy chọn** và **không** thay thế phần lý thuyết về cổng LSTM/GRU. Nội dung tóm lược hướng phát triển mới của RNN có cổng năm 2024 (xLSTM) cùng những bài toán thực tế nơi LSTM/GRU vẫn là lựa chọn phù hợp.

Phương trình trạng thái tế bào của LSTM:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

vẫn là một trong những biểu diễn trực quan nhất cho khái niệm **bộ nhớ có thể ghi và xóa**. Dù Transformer chiếm ưu thế trong các mô hình ngôn ngữ lớn, các kiến trúc hồi quy có cổng vẫn giữ vai trò riêng trong nhiều hệ thống thực tế.

## 1. xLSTM (2024): Cổng hàm mũ và bộ nhớ dạng ma trận

[Beck et al., 2024](https://arxiv.org/abs/2405.04517) mở rộng kiến trúc LSTM với hai biến thể chính:

- **sLSTM** — sử dụng bộ nhớ vô hướng với cổng vào/quên dạng hàm mũ kèm cơ chế chuẩn hóa để tăng tính ổn định so với ô LSTM cổ điển.
- **mLSTM** — sử dụng bộ nhớ dạng *ma trận* cập nhật theo quy tắc tương tự ma trận hiệp phương sai, cho phép huấn luyện song song như khối Transformer.

Kết quả trong bài báo cho thấy xLSTM tiệm cận hiệu năng của Transformer trên bài toán mô hình hóa ngôn ngữ ở quy mô vừa. Điều này chứng minh cơ chế **cổng kết hợp bộ nhớ** vẫn tiếp tục được nghiên cứu và cải tiến, dù không đồng nghĩa với việc thay thế hoàn toàn attention.

## 2. Ứng dụng thực tế hiện nay

### Chuỗi thời gian và luồng sự kiện dạng bảng

Nhiều hệ thống dự báo thực tế (nhu cầu bán lẻ, chỉ số sinh hiệu ICU, dữ liệu cảm biến công nghiệp) vẫn dùng **bộ mã hóa LSTM/GRU**, thường kết hợp trong kiến trúc Temporal Fusion Transformer hoặc mô hình seq2seq gọn nhẹ. Cơ chế cổng giúp các mô hình này xử lý ổn định dữ liệu có nhiễu và tần suất lấy mẫu không đều.

### Xử lý giọng nói và dữ liệu trực tuyến (streaming)

Các hệ thống phát hiện từ đánh thức (wake-word) trên thiết bị biên và một số bộ mã hóa nhận dạng giọng nói trực tuyến vẫn ưu tiên GRU vì trạng thái bộ nhớ chỉ chiếm vài kilobyte. Các mô hình ASR quy mô lớn như Whisper (Chương 19) dùng Transformer; ranh giới lựa chọn nằm ở **độ trễ và dung lượng bộ nhớ** hơn là riêng độ chính xác trên tập chuẩn như LibriSpeech.

### Kiến trúc lai

Một mẫu thiết kế phổ biến giai đoạn 2024–2026 là dùng Transformer (hoặc Mamba) làm mô hình nền, kết hợp **bộ giải mã LSTM** hoặc **bộ điều khiển GRU** ở đầu ra hành động trong lĩnh vực robot và hệ thống gợi ý theo phiên.

## 3. Thư viện và mã nguồn phổ biến

- PyTorch `nn.LSTM` / `nn.GRU` — các cài đặt tối ưu dùng trong sản xuất.
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm) — mã nguồn chính thức của xLSTM.
- [sktime](https://github.com/sktime/sktime) / [GluonTS](https://github.com/awslabs/gluonts) — các thư viện dự báo chuỗi thời gian hỗ trợ mô hình hồi quy.

## 4. Trích dẫn (2022–2026)

- [xLSTM (Beck et al., 2024)](https://arxiv.org/abs/2405.04517).
- [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) — đại diện nổi bật của nhóm mô hình không gian trạng thái hồi quy tuyến tính (đã giới thiệu ở Chương 05), thường được đặt cạnh xLSTM khi đối chiếu các kiến trúc hồi quy hiện đại.
- [Temporal Fusion Transformers (Lim et al., 2021)](https://arxiv.org/abs/1912.09363) — kiến trúc kết hợp bộ mã hóa LSTM và attention được ứng dụng rộng rãi giai đoạn 2022–2026.

## 5. Vai trò bổ trợ cho nội dung cốt lõi

Bạn nên nắm vững phần suy diễn cổng quên, cổng vào và cổng ra ở bài lý thuyết trước. Bài đọc thêm này chỉ cập nhật thêm **xLSTM** và các bài toán công nghiệp còn sử dụng LSTM/GRU, không làm thay đổi hệ phương trình cơ bản của chương.

## Video tham khảo

- **Xem trước:** [XLSTM - Extended LSTMs with sLSTM and mLSTM (paper explained)](https://www.youtube.com/watch?v=0aWGTNS03PU) — AI Bites (tiếng Anh). Giải thích bài báo xLSTM (2024) với hai biến thể sLSTM và mLSTM được đề cập trong bài.
- [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — StatQuest with Josh Starmer (tiếng Anh). Ôn lại ô LSTM gốc năm 1997 làm mốc đối chiếu cho các cải tiến mới.
- [Illustrated Guide to LSTM's and GRU's: A step by step explanation](https://www.youtube.com/watch?v=8HyCNIVRbSU) — The AI Hacker (tiếng Anh). Hệ thống sơ đồ cổng giúp dễ nhận ra những điểm thay đổi ở sLSTM và mLSTM.
