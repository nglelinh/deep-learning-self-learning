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

# Tùy chọn: mô hình chuỗi hồi quy sau Transformer

> Bài này là **tùy chọn**. Nó **không** thay lý thuyết RNN gốc hay cài đặt language model ký tự. Nó giải thích chỗ recurrence vẫn xuất hiện — và mô hình 2023–2025 nào làm sống lại ý “scan theo thời gian.”

RNN gốc:

$$\mathbf{h}_t = \sigma(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b}).$$

Transformer bỏ vòng trái-sang-phải lúc *huấn luyện*. Sau 2022, mô hình **state-space và hồi quy tuyến tính** đưa vòng lặp trở lại, nhưng dưới dạng *parallel scan* với chi phí tuyến tính theo độ dài chuỗi.

## 1. Vì sao RNN không biến mất

Transformer tốn $$O(T^2)$$ bộ nhớ/thời gian theo độ dài $$T$$ (trước khi xấp xỉ). Speech streaming, điều khiển, và genomics rất dài vẫn muốn bước $$O(T)$$ và trạng thái gọn. Đó đúng động cơ của chương này — với tham số hóa tốt hơn.

## 2. Ứng dụng cụ thể

### SSM có cấu trúc và Mamba

[Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) (Mamba) dùng SSM có chọn: cập nhật trạng thái rời rạc là hồi quy

$$\mathbf{h}_t = \overline{A}_t \mathbf{h}_{t-1} + \overline{B}_t \mathbf{x}_t, \quad \mathbf{y}_t = C_t \mathbf{h}_t,$$

trong đó $$\overline{A}_t$$ phụ thuộc đầu vào (cơ chế “selection”). Huấn luyện dùng parallel scan, không phải vòng `for` Python. Đây là câu trả lời 2023–2025 cho “recurrence có cạnh tranh được attention trên ngôn ngữ không?”

### RWKV

[Peng et al., 2023](https://arxiv.org/abs/2305.13048) trộn time-mix kiểu linear attention với channel-mix, nên suy luận giống RNN (trạng thái hằng) trong khi huấn luyện song song. Dùng trong mô hình chat mở khi bộ nhớ KV-cache là nút thắt.

### Chỗ RNN cổ điển còn lại

Nhận từ khóa trên thiết bị, một số baseline dự báo, và phòng lab vẫn dùng `nn.RNN` / `nn.GRU`. ASR production chuyển sang Conformer / Whisper (Chương 08 và 19), không phải Elman RNN gốc.

## 3. Phần mềm phổ biến

- [state-spaces/mamba](https://github.com/state-spaces/mamba) — kernel Mamba chính thức.
- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — huấn luyện và suy luận RWKV.
- PyTorch `nn.RNN` — API đúng cho bài tập cốt lõi.

## 4. Trích dẫn (2022–2026)

- [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752).
- [RWKV (Peng et al., 2023)](https://arxiv.org/abs/2305.13048).
- [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396) — tiền thân SSM.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Cài RNN gốc và bài pitfalls trước. Bài này chỉ đối chiếu **hồi quy tuyến tính hiện đại** với Transformer — không thay suy diễn unfolding / BPTT.
