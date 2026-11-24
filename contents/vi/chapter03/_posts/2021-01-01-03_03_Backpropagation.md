---
layout: post
title: 03-03 Thuật toán Lan truyền Ngược
chapter: '03'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

Nếu phải chọn ra một thuật toán duy nhất làm nên cả đế chế Trí tuệ Nhân tạo hiện đại, đó chắc chắn là **Lan truyền Ngược (*Backpropagation*)**—được phổ biến rộng rãi bởi Rumelhart, Hinton và Williams năm 1986. Trước khi có Backpropagation hiệu quả, việc tính đạo hàm cho một mạng 1 triệu tham số bằng sai phân hữu hạn đòi hỏi chạy lan truyền xuôi tới **1 triệu lần** cho mỗi bước cập nhật! Backpropagation rút gọn chi phí khổng lồ đó xuống chỉ còn **đúng 1 lượt đi ngược**. Mục tổng quan này phác họa trực giác trước khi bạn tự tay giải toán và tự code một động cơ Backprop hoàn chỉnh.

## Bản đồ kiến thức nền: Ba trụ cột của Thuật toán Lan truyền Ngược

1. **Đồ thị Tính toán và Quy tắc Dây chuyền (Computational Graph & Chain Rule)**:
   Mọi mạng nơ-ron phức tạp đều phân rã được thành đồ thị các phép toán sơ cấp (cộng, nhân, mũ, ReLU). Mỗi nút chỉ cần biết đạo hàm cục bộ của riêng mình rồi nhân với tín hiệu lỗi truyền từ phía sau về.

2. **Tái sử dụng Kết quả Trung gian bằng Quy hoạch Động (Reverse-Mode Autodiff)**:
   Thay vì tính lại đạo hàm từ đầu cho từng trọng số, Backpropagation tính tín hiệu lỗi $$\boldsymbol{\delta}^{(\ell)} = \frac{\partial L}{\partial \mathbf{Z}^{(\ell)}}$$ từ tầng cuối $$L$$ ngược về tầng $$1$$, tái sử dụng kết quả của tầng sau cho toàn bộ các tầng trước.

3. **Kiểm tra Gradient bằng Số học (Numerical Gradient Checking)**:
   So sánh gradient giải tích từ Backpropagation với sai phân trung tâm $$\frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}$$—vũ khí kiểm chứng chuẩn xác tuyệt đối mỗi khi bạn tự viết lớp tùy chỉnh.

### Khung giải thích (nên đọc chậm): So sánh chi phí tính Gradient giữa Sai phân Hữu hạn và Backpropagation

Giả sử mạng nơ-ron của bạn có $$P = 1{,}000{,}000$$ tham số và 1 lần chạy lan truyền xuôi mất $$1\text{ ms}$$:

1. **Cách ngây thơ (Sai phân hữu hạn — nhích từng tham số $$\theta_i \pm \varepsilon$$)**:
   Để tính gradient cho cả $$1{,}000{,}000$$ tham số, cần chạy $$2 \times 1{,}000{,}000 = 2{,}000{,}000$$ lượt forward $$\implies$$ tốn **2{,}000 giây (hơn 33 phút)** chỉ cho **1 bước cập nhật duy nhất**!
2. **Cách của Backpropagation (Reverse-Mode Autodiff)**:
   1 lượt forward ($$1\text{ ms}$$) + 1 lượt backward ($$\approx 2\text{ ms}$$) = **3 mili-giây** cho toàn bộ 1 triệu đạo hàm cùng lúc—nhanh hơn gần **700{,}000 lần**!

Chính sự chênh lệch tốc độ hàng trăm nghìn lần này biến việc huấn luyện mạng hàng trăm tỷ tham số từ điều bất khả thi thành hiện thực hàng ngày.

Phần này trình bày thuật toán lan truyền ngược (*backpropagation*) — cơ chế tính gradient cốt lõi của học sâu:

1. **Lý thuyết và ví dụ tính toán** — khái niệm, nền tảng toán học và ví dụ số từng bước
2. **Triển khai** — cài đặt từ đầu bằng NumPy và bằng PyTorch, các bài báo nền tảng, lỗi thường gặp và điểm then chốt

## Video tham khảo

- **Xem trước:** [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown, tiếng Anh. Diễn giải trực quan cách mỗi mẫu dữ liệu điều chỉnh từng trọng số trong mạng.
- [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown, tiếng Anh. Phép tính quy tắc dây chuyền đứng sau quá trình cập nhật trọng số.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, tiếng Anh. Cài đặt hoàn chỉnh thuật toán lan truyền ngược bằng mã nguồn, tương ứng với bài thứ hai trong phần này.
