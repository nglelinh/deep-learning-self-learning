---
layout: post
title: 00-02 Đại số tuyến tính cơ bản
chapter: '00'
order: 7
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Một bức ảnh màu độ phân giải Full HD không gì khác hơn là một mảng số gồm hơn 6 triệu tọa độ; một câu tiếng Việt trước khi đi vào Transformer cũng được hóa thành chuỗi các vector hàng nghìn chiều. Toàn bộ thế giới dữ liệu của Học sâu vận hành bên trong không gian đại số tuyến tính, nơi GPU có thể thực hiện hàng nghìn tỷ phép nhân ma trận mỗi giây. Bài tổng quan này phác họa ba mảnh ghép cốt lõi—vector, ma trận và giá trị riêng—để bạn nhìn đại số tuyến tính bằng con mắt hình học thay vì những bảng số vô hồn.

## Kiến thức cần nhớ: Ba trụ cột Đại số Tuyến tính cho Học sâu

1. **Vector, Tích vô hướng và Chuẩn (Vectors & Norms)**:
   Mã hóa từng mẫu dữ liệu hoặc embedding ngữ nghĩa thành một điểm/mũi tên trong $$\mathbb{R}^n$$; tích vô hướng đo độ tương đồng góc (cơ chế Attention), còn chuẩn $$L_1, L_2$$ đo độ lớn và kiểm soát regularization.

2. **Ma trận như Phép biến đổi Không gian (Linear Transformations)**:
   Mỗi lớp tuyến tính `nn.Linear` không chỉ là một bảng trọng số mà là phép xoay, kéo giãn, nén chiếu toàn bộ không gian đầu vào sang một không gian đặc trưng mới dễ phân tách hơn.

3. **Giá trị riêng và Vector riêng (Eigenvalues & Eigenvectors)**:
   Tiết lộ các "trục chính bất biến" của ma trận—chìa khóa để hiểu PCA giảm chiều dữ liệu, phân tích độ cong Hessian của hàm mất mát, và giải thích hiện tượng gradient bùng nổ/biến mất khi nhân lặp ma trận qua nhiều tầng sâu.

### Giải thích: Theo dõi một vector đặc trưng đi qua một lớp tuyến tính

Giả sử ta có vector đầu vào 2 chiều $$\mathbf{x} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$ đi qua lớp tuyến tính $$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$ với ma trận trọng số $$\mathbf{W} = \begin{bmatrix} 2 & -1 \\ 0 & 3 \end{bmatrix}$$ và độ lệch $$\mathbf{b} = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$:

1. **Phép biến đổi tuyến tính $$\mathbf{W}\mathbf{x}$$**:
   - Thành phần thứ nhất: $$2(1) + (-1)(2) = 0$$.
   - Thành phần thứ hai: $$0(1) + 3(2) = 6$$.
   Ta thu được vector $$\begin{bmatrix} 0 \\ 6 \end{bmatrix}$$: không gian đã bị kéo giãn mạnh theo trục dọc (gấp 3) và trượt nghiêng theo trục ngang.
2. **Phép tịnh tiến cộng bias $$\mathbf{b}$$**:
   $$\mathbf{y} = \begin{bmatrix} 0 \\ 6 \end{bmatrix} + \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 1 \\ 4 \end{bmatrix}.$$
3. **Đo chuẩn $$L_2$$ trước và sau lớp**: $$\|\mathbf{x}\|_2 = \sqrt{1^2 + 2^2} = \sqrt{5} \approx 2.236$$, còn $$\|\mathbf{y}\|_2 = \sqrt{1^2 + 4^2} = \sqrt{17} \approx 4.123$$. Chỉ một ma trận nhỏ đã khuếch đại độ lớn vector gần gấp đôi—nếu xếp 50 lớp như vậy liên tiếp mà không khởi tạo trọng số chuẩn, tín hiệu sẽ tăng theo cấp số nhân!

Phần này trình bày các khái niệm đại số tuyến tính cốt lõi cho tối ưu hóa, được chia thành ba bài học chính giúp bạn tiếp cận một cách có hệ thống.

## Video tham khảo

- **Xem trước:** [Vectors | Chapter 1, Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs) — 3Blue1Brown, tiếng Anh. Giới thiệu trực quan về vector trong $$\mathbb{R}^n$$ — đối tượng trung tâm của toàn bộ phần đại số tuyến tính.
- [Linear transformations and matrices | Chapter 3, Essence of linear algebra](https://www.youtube.com/watch?v=kYB8IZa5AuE) — 3Blue1Brown, tiếng Anh. Trình bày cách nhìn ma trận như phép biến đổi tuyến tính, tương ứng với bài thứ hai.
- [Eigenvectors and eigenvalues | Chapter 14, Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g) — 3Blue1Brown, tiếng Anh. Minh họa trực giác hình học về giá trị riêng và vector riêng trước khi đi vào các thuật toán tính toán.
