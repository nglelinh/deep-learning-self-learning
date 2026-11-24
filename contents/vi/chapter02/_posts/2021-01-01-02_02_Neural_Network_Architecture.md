---
layout: post
title: 02-02 Kiến trúc Mạng Neuron
chapter: '02'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter02
---

Một viên gạch riêng lẻ không thể che nắng mưa, nhưng biết cách xếp gạch thành vòm, cột và mái sẽ tạo nên một tòa nhà vững chãi. Mạng nơ-ron cũng vậy: từ cùng một loại nơ-ron đơn giản, cách bạn tổ chức số tầng (*depth*), số nơ-ron mỗi tầng (*width*), kiểu kết nối (*dense*, *residual*, *bottleneck*) sẽ quyết định liệu mạng có thể học được bài toán phức tạp hay bị kẹt cứng khi huấn luyện. Mục tổng quan này mở đầu hai bài chuyên sâu về lý thuyết kiến trúc và nghệ thuật cân bằng độ sâu—độ rộng.

## Bản đồ kiến thức nền: Ba trụ cột của Thiết kế Kiến trúc Mạng Nơ-ron

1. **Lớp vào, Lớp ẩn và Lớp ra (Input, Hidden, Output Layers)**:
   Lớp vào giữ biểu diễn thô, các lớp ẩn đóng vai trò "xưởng chế biến" liên tục uốn cong không gian dữ liệu để các lớp bị trộn lẫn trở nên tách rời tuyến tính, còn lớp ra chiếu về đúng định dạng bài toán.

2. **Định lý Xấp xỉ Phổ quát (Universal Approximation Theorem)**:
   Khẳng định một mạng chỉ có 1 lớp ẩn phi tuyến đủ rộng có thể xấp xỉ *mọi* hàm liên tục trên tập compact với sai số nhỏ tùy ý—nhưng cái giá về số nơ-ron có thể phình to theo hàm mũ.

3. **Đánh đổi Độ sâu vs. Độ rộng và Mẫu Thiết kế Thực chiến**:
   Mạng sâu hiệu quả tham số hơn nhờ tái sử dụng đặc trưng phân cấp, nhưng khó huấn luyện hơn nếu thiếu khởi tạo chuẩn, chuẩn hóa tầng (Norm) và kết nối tắt (Skip/Residual Connections).

### Khung giải thích (nên đọc chậm): So sánh tay số tham số của mạng Nông-Rộng vs. mạng Sâu-Hẹp

Xét một bài toán nhận đầu vào $$d_{\text{in}} = 100$$ chiều và dự đoán $$d_{\text{out}} = 10$$ lớp:

1. **Mạng Nông–Rộng** (`100 -> 512 -> 10`):
   - Lớp 1: $$100 \times 512 + 512 = 51{,}712$$ tham số.
   - Lớp 2: $$512 \times 10 + 10 = 5{,}130$$ tham số.
   - **Tổng**: $$56{,}842$$ tham số (chỉ 1 lần uốn phi tuyến).
2. **Mạng Sâu–Hẹp** (`100 -> 128 -> 128 -> 128 -> 10`):
   - Lớp 1: $$100 \times 128 + 128 = 12{,}928$$.
   - Lớp 2 & 3: $$2 \times (128 \times 128 + 128) = 33{,}024$$.
   - Lớp 4: $$128 \times 10 + 10 = 1{,}290$$.
   - **Tổng**: $$47{,}242$$ tham số—**ít hơn 17% tham số** nhưng có tới **3 tầng biến đổi phi tuyến liên tiếp**, giúp biểu diễn các ranh giới quyết định phức tạp mượt mà hơn nhiều!

Phần này trình bày kiến trúc mạng neuron, từ một neuron đơn lẻ đến các hệ thống nhiều lớp:

1. **Lý thuyết kiến trúc** — tổng quan khái niệm, nền tảng toán học và trực giác
2. **Độ sâu, độ rộng và mẫu thiết kế** — các lựa chọn thiết kế, cách biểu diễn, cài đặt và những lưu ý thực tế

## Video tham khảo

- **Xem trước:** [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Trình bày độ sâu dưới góc nhìn chuỗi các hàm hợp được học từ dữ liệu.
- [The Essential Main Ideas of Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trực quan vai trò của độ rộng: thêm neuron ẩn giúp đường cong xấp xỉ linh hoạt hơn.
- [Lecture 4 | Introduction to Neural Networks](https://www.youtube.com/watch?v=d14TUNcbn1k) — Stanford University School of Engineering, tiếng Anh. Giới thiệu tổng quan kiến trúc mạng liên kết đầy đủ trước khi các chương sau chuyển sang các kiểu kết nối chuyên biệt hơn.
