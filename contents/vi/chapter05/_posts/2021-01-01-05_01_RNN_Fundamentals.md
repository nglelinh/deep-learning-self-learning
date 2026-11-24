---
layout: post
title: 05-01 Mạng Neuron Hồi quy - Cơ bản
chapter: '05'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter05
---

Để hiểu vì sao những mô hình ngôn ngữ khổng lồ ngày nay phải tốn hàng tỷ đô-la nghiên cứu kiến trúc chuỗi, bước đi đúng đắn nhất là bắt đầu từ chính phương trình hồi quy đơn giản nhất của RNN. Nắm vững cách trải phẳng (unrolling) theo thời gian và lan truyền ngược BPTT sẽ cho bạn nền tảng vững chắc để đọc hiểu mọi kiến trúc xử lý chuỗi về sau.

## Bản đồ kiến thức nền: Ba trụ cột học phần Cơ bản về RNN

1. **Lý thuyết và trực giác trải phẳng theo thời gian (Unrolling)**:
   Biến một vòng lặp hồi tiếp thành một mạng sâu $$T$$ tầng chia sẻ chung trọng số — nhờ đó quy tắc chuỗi (chain rule) quen thuộc áp dụng được ngay.
2. **Lan truyền ngược theo thời gian (BPTT) và phổ trị riêng của $$W_{hh}$$**:
   Đạo hàm truyền lùi $$k$$ bước chứa lũy thừa $$(W_{hh}^T)^k$$; nếu trị riêng lớn nhất $$|\lambda_{\max}| < 1$$ gradient tắt lịm, còn $$|\lambda_{\max}| > 1$$ gradient bùng nổ theo cấp số nhân.
3. **Triển khai thực chiến và kỹ thuật ổn định huấn luyện**:
   Tự lập trình RNN từ NumPy/PyTorch, xây dựng mô hình ngôn ngữ cấp ký tự, áp dụng gradient clipping và truncated BPTT để huấn luyện ổn định.

### Khung giải thích (nên đọc chậm): Tính tay tốc độ suy giảm và bùng nổ của lũy thừa trọng số sau 20 bước

Trong BPTT, đạo hàm từ bước $$T$$ truyền ngược về bước $$T - k$$ tỷ lệ với lũy thừa $$\lambda^k$$ (với $$\lambda$$ là độ lớn trọng số hồi tiếp hiệu dụng sau khi nhân đạo hàm $$\tanh' \le 1$$):

| Hệ số mỗi bước $$\lambda$$ | Sau $$k = 5$$ bước | Sau $$k = 10$$ bước | Sau $$k = 20$$ bước | Hiện tượng xảy ra |
| :---: | :---: | :---: | :---: | :--- |
| $$0{,}5$$ | $$0{,}03125$$ | $$9{,}77 \times 10^{-4}$$ | $$\mathbf{9{,}54 \times 10^{-7}}$$ | **Vanishing Gradient** — mạng "quên" hoàn toàn đầu câu |
| $$1{,}0$$ | $$1{,}000$$ | $$1{,}000$$ | $$\mathbf{1{,}000}$$ | Lý tưởng nhưng cực khó duy trì chính xác bằng tay |
| $$1{,}5$$ | $$7{,}594$$ | $$57{,}67$$ | $$\mathbf{3{,}325{,}26}$$ | **Exploding Gradient** — trọng số vọt lên `NaN` |

Chỉ cần nhìn cột $$k = 20$$ (một câu tiếng Việt bình thường đã dài 20 từ!), bạn thấy ngay lý do toán học trần trụi khiến RNN thuần túy thất bại trên phụ thuộc tầm xa.

Bài học này giới thiệu mạng neuron hồi quy cho dữ liệu tuần tự qua hai phần:

1. **Lý thuyết và Trực giác RNN** — tổng quan khái niệm, nền tảng toán học và trực giác qua ví dụ cụ thể.
2. **Triển khai RNN** — mã nguồn đầy đủ, các khái niệm liên quan, bài báo nền tảng, bẫy thường gặp và tóm tắt điểm chính.

## Video tham khảo

- **Xem trước:** [Recurrent Neural Networks (RNNs), Clearly Explained!!!](https://www.youtube.com/watch?v=AsNTP8Kwu80) — StatQuest with Josh Starmer (tiếng Anh). Nên xem trước khi đọc các bài lý thuyết và cài đặt.
- [Lecture 10 | Recurrent Neural Networks](https://www.youtube.com/watch?v=6niqTuYFZLQ) — Stanford University School of Engineering (tiếng Anh). Bài giảng hệ thống về phép hồi quy và thuật toán lan truyền ngược theo thời gian.
- [Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=L8HKweZIOmg) — StatQuest with Josh Starmer (tiếng Anh). Giới thiệu kiến trúc chuỗi-sang-chuỗi (seq2seq) được đề cập trong phần tổng quan và phần cài đặt.
