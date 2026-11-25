---
layout: post
title: 05-01-02 Triển khai RNN
chapter: '05'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter05
---

Giữa việc hiểu phương trình $$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$ trên bảng và việc huấn luyện thành công một mô hình ngôn ngữ biết tự sinh văn bản là cả một khoảng cách kỹ thuật: cách tổ chức kích thước tensor 3D `(batch, seq_len, input_size)`, cách ngắt đồ thị tính toán (`detach()`) khi chạy chuỗi dài, và cách lấy mẫu phân phối đầu ra bằng nhiệt độ (*temperature*).

## Kiến thức cần nhớ: Ba trụ cột thực hành khi lập trình RNN

1. **Quản lý hình dạng tensor 3D và cờ `batch_first`**:
   Trong PyTorch, `nn.RNN` mặc định kỳ vọng `(seq_len, batch, input_size)` trừ khi bạn truyền tường minh `batch_first=True` để đổi thành `(batch, seq_len, input_size)` — nhầm thứ tự hai trục đầu là lỗi phổ biến nhất.
2. **Vòng lặp BPTT và tách trạng thái ẩn (`h.detach()`)**:
   Khi huấn luyện trên văn bản dài hàng triệu ký tự, ta cắt thành các đoạn $$T$$ bước và gọi `h = h.detach()` giữa các đoạn để truyền bộ nhớ sang đoạn sau mà không lưu đồ thị đạo hàm phình to vô hạn.
3. **Suy luận tự hồi quy (Autoregressive Sampling) với Temperature**:
   Chia logit cho nhiệt độ $$\tau$$ trước khi `softmax`: $$\tau < 1$$ làm phân phối nhọn hơn (an toàn, lặp), còn $$\tau > 1$$ làm phân phối phẳng hơn (sáng tạo nhưng dễ sai chính tả).

### Giải thích: Tính tay số tham số của `nn.RNN(input_size=10, hidden_size=32, num_layers=1)`

Một tầng `nn.RNN` trong PyTorch chứa hai ma trận trọng số `weight_ih` (từ đầu vào $$x_t$$), `weight_hh` (từ trạng thái ẩn $$h_{t-1}$$) và hai vector bias tương ứng `bias_ih`, `bias_hh`:

| Thành phần tham số | Kích thước ma trận/vector | Số lượng phần tử |
| :--- | :---: | ---: |
| `weight_ih_l0` ($$W_{xh}$$) | $$32 \times 10$$ | 320 |
| `weight_hh_l0` ($$W_{hh}$$) | $$32 \times 32$$ | 1,024 |
| `bias_ih_l0` + `bias_hh_l0` | $$32 + 32$$ | 64 |
| **Tổng cộng tầng RNN** | | **1,408** |

Lưu ý chi tiết thực tế: công thức tổng quát của `nn.RNN` trong PyTorch có **hai** vector bias ($$2 \cdot d_h$$) để tương thích chuẩn cuDNN, nhiều hơn $$d_h$$ tham số so với công thức sách giáo khoa chỉ viết 1 bias $$b_h$$.

Phần cài đặt RNN gồm hai nội dung chính:

1. **Triển khai Cốt lõi** — cài đặt RNN từ đầu bằng NumPy, dự đoán chuỗi và đối chiếu với PyTorch.
2. **Mô hình ngôn ngữ cấp ký tự, Papers và Bẫy thường gặp** — xây dựng mô hình ngôn ngữ, các khái niệm liên quan, bài báo nền tảng và lưu ý thực hành.

## Video tham khảo

- **Xem trước:** [Lecture 10 | Recurrent Neural Networks](https://www.youtube.com/watch?v=6niqTuYFZLQ) — Stanford University School of Engineering (tiếng Anh). Trình bày cấu trúc cài đặt của một ô RNN cùng hàm mất mát trên chuỗi.
- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy (tiếng Anh). Cài đặt trực tiếp mô hình ngôn ngữ cấp ký tự bằng mã nguồn, đúng với mục tiêu thực hành của bài.
- [Recurrent Neural Networks (RNNs), Clearly Explained!!!](https://www.youtube.com/watch?v=AsNTP8Kwu80) — StatQuest with Josh Starmer (tiếng Anh). Tài liệu đối chiếu nhanh về kích thước các vectơ trạng thái ẩn h và đầu vào x khi đọc mã.
