---
layout: post
title: 07-02 Cơ chế Chú ý Đi sâu
chapter: '07'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter07
---

Nếu bài học trước cho bạn thấy động lực ra đời của Attention trong bài toán dịch máy Seq2Seq, thì phần đi sâu này sẽ nâng Attention lên thành một **phép toán tensor tổng quát** có thể đứng độc lập mà không cần RNN. Đây chính là bước chuyển hóa biến Attention từ một "phụ kiện" gắn kèm LSTM trở thành trái tim của toàn bộ kiến trúc Transformer.

## Bản đồ kiến thức nền: Ba trụ cột khi đi sâu vào Cơ chế Chú ý

1. **Toán học của Scaled Dot-Product Attention và lập luận phương sai**:
   Hiểu chính xác tại sao phải chia tích vô hướng $$Q K^T$$ cho $$\sqrt{d_k}$$ trước khi đưa qua `softmax`, và cách viết toàn bộ phép tính gọn gàng thành một tích ma trận theo batch.
2. **Mở rộng sang Multi-Head Attention (MHA)**:
   Tách không gian $$d_{\text{model}}$$ thành $$H$$ đầu chú ý độc lập kích thước $$d_k = d_{\text{model}} / H$$ để mô hình đồng thời theo dõi nhiều loại quan hệ khác nhau (cú pháp, tham chiếu đại từ, ngữ nghĩa).
3. **Cài đặt chuẩn xác và tránh lỗi mặt nạ (Masking)**:
   Nắm vững kỹ thuật che bằng $$-\infty$$ (trước softmax, không phải nhân 0 sau softmax) cho cả **padding mask** lẫn **causal mask**, cùng kiểm chứng hình dạng tensor 4D `(B, H, L, d_k)`.

### Khung giải thích (nên đọc chậm): Tính tay sự bùng nổ phương sai khi KHÔNG chia cho $$\sqrt{d_k}$$

Giả sử các phần tử của $$\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$$ độc lập, có trung bình $$0$$ và phương sai $$1$$. Khi đó tích vô hướng $$s = \sum_{i=1}^{d_k} q_i k_i$$ có phương sai bằng đúng $$d_k$$ (độ lệch chuẩn $$\sigma = \sqrt{d_k}$$):

| Chiều $$d_k$$ | Độ lệch chuẩn $$\sigma$$ của $$s = \mathbf{q}\cdot\mathbf{k}$$ | Độ lệch chuẩn sau khi chia $$\sqrt{d_k}$$ | Tác động lên Softmax |
| :---: | :---: | :---: | :--- |
| $$d_k = 4$$ | $$2$$ | **$$1$$** | Ổn định |
| $$d_k = 64$$ | $$8$$ | **$$1$$** | Nếu không chia: điểm số trải từ $$-16$$ đến $$+16 \Rightarrow e^{16} \approx 8{,}9 \times 10^6$$, softmax bão hòa gần 1-hot! |
| $$d_k = 512$$ | $$22{,}6$$ | **$$1$$** | Nếu không chia: gradient của softmax co về gần $$0$$, mạng ngừng học |

Chia cho $$\sqrt{d_k}$$ đưa phương sai của điểm chú ý về lại đúng **$$1$$** ở mọi kích thước mô hình!

# 07-02 Cơ chế Chú ý Đi sâu

Phần này mở rộng sâu hơn so với bài cơ sở (07-01). Sau khi đã hiểu lý do tồn tại của attention và các công thức nền tảng, ta sẽ hệ thống hóa chặt chẽ phần toán học rồi cài đặt đầy đủ từng thành phần.

## Nội dung chính

1. **Toán học Attention**  
   Thống nhất các hàm chấm điểm, dạng ma trận QKV theo batch, self-attention như phép tương tác $$L\times L$$, đại số multi-head, lý thuyết padding/causal mask trước softmax, độ phức tạp tính toán và lập luận về độ dài đường truyền thông tin.

2. **Cài đặt Attention**  
   Cài đặt từ đầu attention Bahdanau và scaled dot-product attention, xây dựng module multi-head, minh họa masking, phác thảo mô hình seq2seq có attention, điểm qua các bài báo nền tảng và các lỗi thường gặp trong thực tế.

## Lộ trình đọc

| Thứ tự | Bài | Mục tiêu đạt được |
|--------|-----|------------------|
| 1 | **07-02-01 Toán học Attention** | Nắm chắc và tự viết lại được toàn bộ các công thức cốt lõi |
| 2 | **07-02-02 Cài đặt** (tổng quan) | Đối chiếu từ công thức toán sang cấu trúc các module mã nguồn |
| 3 | **07-02-02-01 Cài đặt cốt lõi** | Chạy và kiểm chứng được mã nguồn Bahdanau và self-attention |
| 4 | **07-02-02-02 Multi-Head, Papers, Cạm bẫy** | Nắm vững multi-head, causal mask, tài liệu nền tảng và quy trình gỡ lỗi |

Nếu một biểu thức trong bài **07-02-01** còn trừu tượng, bạn có thể xem trước đoạn mã minh họa tương ứng ở **07-02-02-01** rồi quay lại đối chiếu.

## Liên kết trong khóa học

- **Xem lại bài 07-01** nếu cần ôn lại trực giác và bối cảnh lịch sử của attention.
- **Tiếp nối sang Chương 08 (Transformer)** để kết hợp self-attention với mạng feed-forward, kết nối phần dư và mã hóa vị trí.

Khi đã sẵn sàng, hãy bắt đầu với bài **Toán học Attention**.

## Video tham khảo

- **Xem trước:** [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Trình bày chi tiết scaled dot-product attention, trọng tâm của phần đi sâu này.
- [Attention for Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=PSs6nxngL6k) — StatQuest with Josh Starmer (tiếng Anh). Minh họa dạng attention cho mô hình chuỗi để đối chiếu với self-attention.
- [A Dive Into Multihead Attention, Self-Attention and Cross-Attention](https://www.youtube.com/watch?v=mmzRYGCfTzc) — Machine Learning Studio (tiếng Anh). Phân tách rõ ba dạng multi-head, self-attention và cross-attention đúng theo cấu trúc phần này.
