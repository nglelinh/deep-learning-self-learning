---
layout: post
title: 02-03 Hàm Kích hoạt Chi tiết
chapter: '02'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

Thử tưởng tượng bạn xây một tòa tháp gồm 100 tầng mạng nơ-ron, đầu tư cả dàn GPU đắt tiền, nhưng quên mất một chi tiết nhỏ: **không đặt hàm kích hoạt phi tuyến** giữa các tầng. Kết quả? Về mặt toán học, cả 100 tầng đó lập tức sụp đổ thành đúng **một phép nhân ma trận duy nhất**—không mạnh hơn một mô hình hồi quy tuyến tính 1 tầng! Hàm kích hoạt chính là nguồn gốc của tính phi tuyến và khả năng học các ranh giới phức tạp. Mục tổng quan này mở đường cho hai bài chuyên sâu đi từ Sigmoid/Tanh cổ điển đến ReLU, GELU và cuộc chiến chống gradient biến mất.

## Kiến thức cần nhớ: Ba vai trò của Hàm Kích hoạt

1. **Phá vỡ Tính Tuyến tính (Non-linearity)**:
   Nhờ phép hợp $$\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x})$$ không thể gộp thành một ma trận duy nhất, mỗi tầng mới thực sự cộng thêm năng lực uốn cong không gian biểu diễn.

2. **Kiểm soát Dòng chảy Gradient (Gradient Flow)**:
   Đạo hàm $$\sigma'(z)$$ được nhân liên tiếp qua từng tầng trong lan truyền ngược. Nếu $$\sigma'(z) \ll 1$$ (như Sigmoid bão hòa), gradient tắt ngấm; nếu $$\sigma'(z) \approx 1$$ (như ReLU trên miền dương), mạng sâu học mượt mà.

3. **Định hình Miền Đầu ra theo Tác vụ (Output Semantics)**:
   Lớp ẩn ưu tiên tốc độ và dòng gradient (ReLU, GELU, SwiGLU), còn lớp đầu ra phải khớp miền xác suất của bài toán: Linear cho hồi quy, Sigmoid cho nhị phân/đa nhãn, và Softmax cho phân loại đa lớp loại trừ.

### Giải thích: Chứng minh bằng tay sự sụp đổ của mạng 2 tầng không có hàm kích hoạt

Xét mạng 2 tầng *tuyến tính thuần túy* với $$\mathbf{W}_1 = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix}$$, $$\mathbf{W}_2 = \begin{bmatrix} 3 & 1 \end{bmatrix}$$ (bỏ qua bias cho gọn):

1. **Tính qua 2 tầng lần lượt** cho $$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$:
   - Tầng 1: $$\mathbf{h} = \mathbf{W}_1\mathbf{x} = \begin{bmatrix} x_1 + 2x_2 \\ -x_2 \end{bmatrix}$$.
   - Tầng 2: $$y = \mathbf{W}_2\mathbf{h} = 3(x_1 + 2x_2) + 1(-x_2) = 3x_1 + 5x_2$$.
2. **Gộp trực tiếp thành 1 tầng duy nhất**:
   $$\mathbf{W}_{\text{gộp}} = \mathbf{W}_2 \mathbf{W}_1 = \begin{bmatrix} 3 & 1 \end{bmatrix}\begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix} = \begin{bmatrix} 3 & 5 \end{bmatrix} \implies y = \mathbf{W}_{\text{gộp}}\mathbf{x} = 3x_1 + 5x_2.$$
Dù bạn có chồng thêm 1{,}000 ma trận nữa, tích $$\mathbf{W}_{1000}\cdots\mathbf{W}_1$$ vẫn chỉ là **một ma trận tuyến tính đơn độc**. Chỉ khi chèn hàm phi tuyến $$\sigma(\cdot)$$ vào giữa, phép kết hợp mới bị phá vỡ và mạng sâu mới thực sự có ý nghĩa!

Phần này trình bày chi tiết về các hàm kích hoạt (*activation functions*):

1. **Hàm kích hoạt cổ điển** — vai trò của hàm kích hoạt, các hàm phổ biến và bảng so sánh
2. **Hàm kích hoạt hiện đại và gradient biến mất** — cách lựa chọn hàm kích hoạt, cài đặt và hiện tượng gradient biến mất (*vanishing gradients*)

## Video tham khảo

- **Xem trước:** [Why Non-linear Activation Functions (C1W3L07)](https://www.youtube.com/watch?v=NkOv_k7r6no) — DeepLearningAI, tiếng Anh. Giải thích vì sao thiếu tính phi tuyến sẽ khiến toàn bộ mạng nhiều lớp thu gọn về một phép biến đổi tuyến tính duy nhất.
- [Neural Networks Pt. 3: ReLU In Action!!!](https://www.youtube.com/watch?v=68BZ5f7P94E) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trực quan ReLU — hàm kích hoạt mặc định được phân tích ở phần đầu.
- [The GELU, SiLU and SwiGLU activation functions, clearly explained!!!](https://www.youtube.com/watch?v=2FaI2Fen1mQ) — StatQuest with Josh Starmer, tiếng Anh. Trình bày các hàm kích hoạt hiện đại được giới thiệu ở phần sau để so sánh với ReLU.
