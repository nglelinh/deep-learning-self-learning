---
layout: post
title: 11-01 Mô hình Sinh - Các Khái niệm Cốt lõi
chapter: '11'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter11
---

Trước khi bước vào từng kiến trúc chuyên sâu như Autoencoder, VAE hay GAN ở các chương sau, bạn cần một tấm la bàn chung để không bị lạc giữa rừng thuật ngữ: tại sao có mô hình cần tích phân, mô hình lại cần hai mạng đấu nhau, còn mô hình khác lại dự đoán từng pixel một? Mục tổng quan cốt lõi này đặt tất cả các họ mô hình sinh lên cùng một hệ quy chiếu xác suất.

## Bản đồ kiến thức nền: Hệ quy chiếu thống nhất các mô hình sinh cốt lõi

- **Bản chất học mật độ dữ liệu:** Mọi mô hình sinh đều hướng tới việc xấp xỉ $$p_{\text{data}}(\mathbf{x})$$, khác biệt nằm ở việc chúng xấp xỉ *trực tiếp hàm mật độ* hay *trực tiếp bộ lấy mẫu*.
- **Tam giác đánh đổi của mô hình sinh:** Ba thuộc tính khó đồng thời đạt tối ưu tuyệt đối — chất lượng mẫu cao, độ phủ phân phối (đa dạng) rộng, và tốc độ lấy mẫu nhanh.
- **Kết nối lý thuyết sang mã nguồn:** Liên kết bài **11-01-01 (Lý thuyết)** về MLE/KL/ELBO với bài **11-01-02 (Cài đặt)** để kiểm chứng trực tiếp sự đánh đổi trên dữ liệu thật.
- **Bước đệm tới mô hình hiện đại:** Đặt nền móng để đọc hiểu cách khuếch tán (Diffusion) và Flow Matching dung hòa các đánh đổi của thế hệ cổ điển.

### Khung giải thích (nên đọc chậm): Đối chiếu tốc độ sinh và khả năng tính xác suất trên một ảnh $$32 \times 32 \times 3$$

Xét một ảnh màu nhỏ kích thước $$32 \times 32 \times 3 = 3{,}072$$ giá trị kênh màu. Giả sử một lượt suy luận mạng mất $$1\text{ ms}$$ trên GPU:

1. **Mô hình tự hồi quy (kiểu PixelCNN):** Để sinh trọn 1 ảnh, phải chạy mạng tuần tự qua từng thành phần: $$3{,}072 \times 1\text{ ms} = 3{,}072\text{ ms} \approx 3{,}07\text{ giây/ảnh}$$, đổi lại ta tính được chính xác $$\log p(\mathbf{x})$$ chỉ trong 1 lượt song song khi huấn luyện.
2. **VAE / GAN:** Sinh toàn bộ $$3{,}072$$ giá trị trong đúng **1 lượt xuôi** từ vector ẩn $$\mathbf{z}$$: thời gian chỉ $$1\text{ ms/ảnh}$$ — nhanh hơn hơn **3.000 lần** lúc sinh! Đổi lại, VAE chỉ cho cận dưới ELBO, còn GAN không tính được $$\log p(\mathbf{x})$$.
3. **Bài học kiến trúc:** Chỉ từ một phép nhân đơn giản, ta thấy ngay lý do công nghiệp vừa cần tính toán song song (lúc huấn luyện) vừa cần giảm số bước lấy mẫu (lúc suy luận) — động lực trực tiếp thúc đẩy các kỹ thuật chưng cất và flow matching sau này.

Nội dung nền tảng về mô hình sinh gồm hai phần:

1. **Lý thuyết** — tổng quan khái niệm, cơ sở toán học và ví dụ trực quan.
2. **Cài đặt** — mã nguồn minh họa, bài báo nền tảng, các cạm bẫy thường gặp và điểm then chốt.

## Video tham khảo

- **Xem trước:** [MIT 6.S191 (2024): Deep Generative Modeling](https://www.youtube.com/watch?v=Dmm4UG-6jxA) — Alexander Amini, tiếng Anh. Làm rõ sự phân biệt giữa mô hình mật độ tường minh và mô hình lấy mẫu ẩn.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, tiếng Anh. Cách phân loại tương tự trong bài giảng CS231n mà phần học này bám theo.
- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy, tiếng Anh. Ví dụ cài đặt mô hình tự hồi quy, tương ứng với bài mã đầu tiên của phần này.
