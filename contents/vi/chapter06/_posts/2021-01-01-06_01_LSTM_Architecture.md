---
layout: post
title: 06-01 Mạng Long Short-Term Memory
chapter: '06'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter06
---

Nhìn vào sơ đồ nội bộ một ô LSTM lần đầu tiên, ai cũng dễ choáng ngợp trước mạng lưới mũi tên, dấu nhân chấm $$\odot$$, dấu cộng $$+$$, ba khối $$\sigma$$ và hai khối $$\tanh$$. Nhưng nếu đọc nó như một chiếc máy tính siêu nhỏ có **bộ nhớ dài hạn (RAM)** và **ba chiếc van điều khiển**, mọi phương trình sẽ lập tức trở nên sáng rõ và logic đến bất ngờ.

## Kiến thức cần nhớ: Ba tầng hiểu biết để làm chủ Kiến trúc LSTM

1. **Giải phẫu phương trình 4 nhánh ($$f_t, i_t, \tilde{c}_t, o_t$$)**:
   Nắm rõ vì sao các cổng dùng $$\sigma \in (0,1)$$ (vai trò tỷ lệ mở van) còn ứng viên nội dung $$\tilde{c}_t$$ và đầu ra tế bào dùng $$\tanh \in (-1,1)$$ (vai trò giá trị có dấu âm/dương).
2. **Tổ chức cài đặt tối ưu hóa ma trận gộp**:
   Thay vì gọi 4 phép nhân ma trận nhỏ riêng lẻ, các thư viện như PyTorch gộp cả 4 trọng số thành **một ma trận lớn duy nhất** kích thước $$4 d_h \times (d_x + d_h)$$ rồi dùng `chunk(4)` để tách cổng — tận dụng tối đa băng thông GPU.
3. **So sánh thực nghiệm LSTM vs. GRU và phân tích hành vi cổng**:
   Quan sát trực tiếp giá trị số của $$f_t, i_t$$ trên chuỗi có nhiễu để thấy cổng tự đóng lại trước tín hiệu rác và mở rộng khi gặp thông tin quan trọng.

### Giải thích: Tính tay kích thước bộ tham số của LSTM so với RNN và GRU

Với cùng kích thước đầu vào $$d_x = 16$$ và kích thước ẩn $$d_h = 32$$ (tính theo công thức chuẩn 1 bias cho mỗi cổng):

| Kiến trúc | Số nhánh biến đổi | Công thức tham số | Kết quả cụ thể ($$d_x=16, d_h=32$$) | Tỷ lệ so với RNN |
| :--- | :---: | :--- | ---: | :---: |
| **RNN thuần** | 1 | $$1 \times d_h (d_x + d_h + 1)$$ | $$32 \times 49 = \mathbf{1{,}568}$$ | $$1\times$$ |
| **GRU** | 3 ($$r, z, \tilde{h}$$) | $$3 \times d_h (d_x + d_h + 1)$$ | $$3 \times 1{,}568 = \mathbf{4{,}704}$$ | $$3\times$$ |
| **LSTM** | 4 ($$f, i, \tilde{c}, o$$) | $$4 \times d_h (d_x + d_h + 1)$$ | $$4 \times 1{,}568 = \mathbf{6{,}272}$$ | $$4\times$$ |

Cái giá để đánh đổi khả năng nhớ dài hạn rất rõ ràng: LSTM cần đúng **gấp 4 lần** số tham số của RNN, còn GRU tiết kiệm đúng **25%** so với LSTM (tỷ lệ $$3 : 4$$).

Bài học này trình bày về LSTM và các kiến trúc hồi quy có cổng liên quan qua hai phần:

1. **Lý thuyết LSTM và GRU** — cơ chế cổng, trạng thái tế bào, nền tảng toán học và trực giác qua ví dụ.
2. **Triển khai LSTM** — mã nguồn đầy đủ, phân tích cổng, so sánh GRU, bài báo nền tảng và các bẫy thường gặp.

## Video tham khảo

- **Xem trước:** [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — StatQuest with Josh Starmer (tiếng Anh). Tổng quan ngắn gọn trước khi đi vào phần lý thuyết và mã nguồn.
- [Illustrated Guide to LSTM's and GRU's: A step by step explanation](https://www.youtube.com/watch?v=8HyCNIVRbSU) — The AI Hacker (tiếng Anh). Minh họa từng cổng tương ứng với cấu trúc mã nguồn trong bài cài đặt.
- [Simple Explanation of GRU (Gated Recurrent Units) | Deep Learning Tutorial 37 (Tensorflow & Python)](https://www.youtube.com/watch?v=tOuXgORsXJ4) — codebasics (tiếng Anh). Giới thiệu biến thể GRU với cấu trúc cổng tinh giản hơn.
