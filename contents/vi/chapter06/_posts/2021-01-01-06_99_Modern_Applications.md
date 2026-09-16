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

# Tùy chọn: hồi quy có cổng sau LSTM — xLSTM và khi nào cổng vẫn thắng

> Bài này là **tùy chọn**. Nó **không** viết lại lý thuyết cổng LSTM/GRU. Nó phủ các hồi sinh RNN có cổng năm 2024 và các ngách production nơi LSTM/GRU vẫn là công cụ đúng.

Trạng thái ô LSTM

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

vẫn là hình ảnh sạch nhất của **bộ nhớ ghi được**. Transformer lấy NLP quy mô lớn; chúng không xóa mọi mô hình hồi quy có cổng.

## 1. xLSTM (2024): cổng mũ + bộ nhớ ma trận

[Beck et al., 2024](https://arxiv.org/abs/2405.04517) xem lại LSTM với:

- **sLSTM** — bộ nhớ vô hướng với cổng input/forget dạng mũ và bộ chuẩn hóa (bản sửa ổn định của ô cổ điển),
- **mLSTM** — bộ nhớ *ma trận* cập nhật theo quy tắc giống hiệp phương sai, huấn luyện song song như khối Transformer.

Bài báo báo kết quả language modeling khít hơn với Transformer ở quy mô vừa. Hãy coi xLSTM là bằng chứng **cổng + bộ nhớ** không chỉ là ý tưởng 1997 — không phải lý do bỏ attention.

## 2. Ứng dụng cụ thể

### Chuỗi thời gian và luồng sự kiện dạng bảng

Các stack dự báo (nhu cầu bán lẻ, sinh hiệu ICU, cảm biến công nghiệp) vẫn giao **encoder LSTM/GRU**, thường dưới Temporal Fusion Transformer hoặc đầu seq2seq đơn giản. Phân tích cổng của chương này chính là lý do chúng sống sót với lấy mẫu ồn, không đều.

### Tiếng nói và streaming

Wake-word trên thiết bị và một số encoder ASR streaming giữ GRU vì trạng thái chỉ vài kilobyte. ASR lớn (Whisper, Chương 19) dựa Transformer; ranh giới là **latency / bộ nhớ**, không phải độ chính xác trên LibriSpeech.

### Stack lai

Mẫu 2024–2026 phổ biến: Transformer (hoặc Mamba) làm xương sống, **decoder LSTM** hoặc **bộ điều khiển GRU** cho đầu hành động trong robot và mô hình phiên gợi ý.

## 3. Phần mềm phổ biến

- PyTorch `nn.LSTM` / `nn.GRU` — kernel production.
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm) — cài đặt xLSTM chính thức.
- [sktime](https://github.com/sktime/sktime) / [GluonTS](https://github.com/awslabs/gluonts) — thư viện dự báo vẫn mở RNN estimator.

## 4. Trích dẫn (2022–2026)

- [xLSTM (Beck et al., 2024)](https://arxiv.org/abs/2405.04517).
- [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) — dòng “recurrence trở lại” kia; so với xLSTM, đừng trùng Chương 05.
- [Temporal Fusion Transformers (Lim et al., 2021)](https://arxiv.org/abs/1912.09363) — triển khai rộng 2022–2026; encoder LSTM + attention.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Suy ra cổng forget/input/output từ bài lý thuyết trước. Bài này chỉ thêm **xLSTM** và các use case LSTM/GRU công nghiệp còn lại — không đổi phương trình cổng.
