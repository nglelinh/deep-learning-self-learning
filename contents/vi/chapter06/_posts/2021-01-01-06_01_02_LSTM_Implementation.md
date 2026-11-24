---
layout: post
title: 06-01-02 Triển khai LSTM
chapter: '06'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter06
---

Khi bạn chuyển từ sơ đồ LSTM sang code chạy thực tế, một quyết định tưởng nhỏ — như khởi tạo bias cổng quên `bias_f = 1.0` hay chọn đúng tuple `(h_n, c_n)` trả về từ `nn.LSTM` — có thể quyết định mô hình hội tụ sau 20 epoch hay đứng im không học được gì. Mục này nối thẳng phương trình toán học vào cài đặt thực chiến và kiểm thử trên các tác vụ bộ nhớ dài hạn.

## Bản đồ kiến thức nền: Ba trụ cột triển khai LSTM/GRU hiệu quả

1. **Thiết kế bài kiểm tra bộ nhớ dài hạn (Adding / Memorization Problem)**:
   Muốn chứng minh LSTM thực sự hơn RNN, ta cần bài toán chuẩn mà thông tin quan trọng nằm ở đầu chuỗi, ở giữa là hàng chục bước nhiễu, và đầu ra chỉ yêu cầu ở bước cuối (như dự đoán tổng `first + last`).
2. **Quản lý bộ đôi trạng thái `(h, c)` và kiến trúc đa tầng / hai chiều (Bidirectional)**:
   Với `nn.LSTM(num_layers=L, bidirectional=True)`, cả `h_0` lẫn `c_0` đều có hình dạng `(2*L, batch, hidden_size)` — khác biệt quan trọng so với `nn.RNN`/`nn.GRU` vốn chỉ có một tensor `h_0`.
3. **Kỹ thuật khởi tạo cổng quên và chẩn đoán qua nhật ký cổng**:
   Theo khuyến nghị của Jozefowicz et al. (2015), đặt bias cổng quên dương ($$b_f = 1$$ hoặc $$2$$) và in phân phối $$f_t, i_t$$ giúp phát hiện sớm hiện tượng bão hòa cổng.

### Khung giải thích (nên đọc chậm): Tính nhanh kích thước tensor cho `nn.LSTM` đa tầng hai chiều

Giả sử bạn khai báo `lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)` và đưa vào một batch `x` có shape `(32, 50, 64)` (32 câu, mỗi câu dài 50 token):

| Biến trả về | Công thức shape | Shape thực tế | Ý nghĩa |
| :--- | :--- | :---: | :--- |
| `output` | `(B, T, D * H)` | `(32, 50, 256)` | Trạng thái ẩn tầng cuối tại mọi bước thời gian (128 xuôi + 128 ngược) |
| `h_n` | `(D * L, B, H)` | `(4, 32, 128)` | Trạng thái ẩn cuối của 2 tầng $$\times$$ 2 chiều |
| `c_n` | `(D * L, B, H)` | `(4, 32, 128)` | Trạng thái tế bào cuối của 2 tầng $$\times$$ 2 chiều |

Để lấy biểu diễn tổng hợp cho cả câu (dùng cho phân loại văn bản), ta nối hai chiều của **tầng cuối cùng**: `torch.cat([h_n[-2], h_n[-1]], dim=-1)` $$\to$$ shape `(32, 256)`.

Phần cài đặt LSTM gồm hai nội dung chính:

1. **Triển khai Cốt lõi** — xây dựng và huấn luyện LSTM bằng NumPy và PyTorch trên bài toán bộ nhớ tầm xa.
2. **Phân tích Cổng, GRU và Bẫy thường gặp** — phân tích hoạt động của các cổng, so sánh LSTM với GRU, các bài báo nền tảng và lưu ý thực hành.

## Video tham khảo

- **Xem trước:** [Illustrated Guide to LSTM's and GRU's: A step by step explanation](https://www.youtube.com/watch?v=8HyCNIVRbSU) — The AI Hacker (tiếng Anh). Theo dõi rõ từng cổng trước khi chuyển sang viết các phép toán tensor tương ứng.
- [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — StatQuest with Josh Starmer (tiếng Anh). Đối chiếu các phương trình toán học cần khớp với phần cài đặt.
- [Simple Explanation of GRU (Gated Recurrent Units) | Deep Learning Tutorial 37 (Tensorflow & Python)](https://www.youtube.com/watch?v=tOuXgORsXJ4) — codebasics (tiếng Anh). Tài liệu bổ trợ cho phần cài đặt và so sánh GRU.
