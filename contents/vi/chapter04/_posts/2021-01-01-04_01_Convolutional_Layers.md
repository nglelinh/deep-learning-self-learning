---
layout: post
title: 04-01 Tầng Tích chập
chapter: '04'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

Mọi hệ thống nhận diện khuôn mặt, xe tự lái hay chẩn đoán ảnh X-quang đều bắt đầu từ một phép toán tưởng chừng đơn giản: trượt một ma trận số nhỏ qua bức ảnh và tính tổng tích từng phần tử. Thế nhưng đằng sau thao tác trượt ấy là cả một hệ thống toán học chặt chẽ về kích thước tensor, đa kênh, bước nhảy và độ đệm — nơi chỉ cần sai một đơn vị là toàn bộ đường ống kiến trúc bị lệch chiều.

## Bản đồ kiến thức nền: Bốn nhánh kiến thức của Tầng Tích chập

1. **Đại số tensor và hình học kích thước**:
   Hiểu chính xác mối quan hệ giữa kích thước đầu vào $$(H, W, C_{\text{in}})$$, kích thước nhân $$k$$, đệm $$p$$, bước nhảy $$s$$ và giãn cách $$d$$ để dự đoán ngay đầu ra mà không phải "chạy thử rồi sửa lỗi shape".
2. **Trực giác xử lý tín hiệu và trường tiếp nhận**:
   Nhìn thấy một bộ lọc $$3 \times 3$$ thực chất là một bộ phát hiện mẫu cục bộ, và khi xếp chồng nhiều tầng, vùng ảnh gốc mà một neuron sâu "nhìn thấy" (receptive field) mở rộng nhanh chóng.
3. **Từ công thức toán đến cài đặt NumPy và PyTorch**:
   Tự viết vòng lặp tích chập bằng NumPy giúp hiểu từng phép nhân cộng bên trong, trước khi chuyển sang `nn.Conv2d` tối ưu hóa trên GPU bằng thuật toán `im2col`/Winograd.
4. **Biến thể hiện đại và cạm bẫy kỹ thuật**:
   Phân biệt tích chập chuẩn với tích chập $$1\times 1$$, depthwise separable và dilated convolution, đồng thời tránh các lỗi kinh điển như nhầm thứ tự chiều NCHW/NHWC hay quên bias khi dùng BatchNorm.

### Khung giải thích (nên đọc chậm): Tính nhanh một tầng tích chập 4 tham số trên giấy

Cho ảnh đầu vào kích thước $$H = 32, W = 32, C_{\text{in}} = 3$$ đi qua tầng `Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)`:

1. **Bước 1 — Tính chiều cao và rộng đầu ra**:
   $$H_{\text{out}} = \left\lfloor \frac{32 + 2(1) - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{31}{2} \right\rfloor + 1 = 15 + 1 = 16$$
   Tương tự $$W_{\text{out}} = 16$$. Do đó tensor đầu ra có kích thước $$16 \times 16 \times 16$$ (giảm một nửa mỗi chiều không gian).
2. **Bước 2 — Tính số tham số học được**:
   Mỗi bộ lọc có kích thước $$3 \times 3 \times 3 = 27$$ trọng số cộng $$1$$ bias = $$28$$ tham số. Với 16 bộ lọc: $$16 \times 28 = 448$$ tham số.
3. **Bước 3 — Ước lượng số phép nhân-cộng (MACs)**:
   Tại mỗi vị trí trong $$16 \times 16 = 256$$ điểm đầu ra, mỗi bộ lọc thực hiện $$27$$ phép nhân-cộng $$\Rightarrow 256 \times 16 \times 27 = 110{,}592$$ phép tính MACs.

Phần này trình bày chi tiết về các tầng tích chập (*convolutional layers*) — thành phần cốt lõi của CNN — theo bốn mục chính:

1. **Toán học Tích chập và Kích thước** — khái niệm nền tảng, tích chập đa kênh, công thức tính kích thước đầu ra và số lượng tham số
2. **Trực giác Tích chập và Ví dụ** — phát hiện cạnh, hệ thống đặc trưng phân cấp và sự mở rộng của trường tiếp nhận (*receptive field*)
3. **Cài đặt Tích chập** — cài đặt từ đầu bằng NumPy, sử dụng API PyTorch và xây dựng khối CNN hoàn chỉnh
4. **Khái niệm Liên quan, Cạm bẫy và Bài báo** — mối liên hệ với các tầng khác, lỗi thường gặp, mẹo thực hành và các công trình nền tảng

Bạn nên đọc tuần tự từng mục trước khi chuyển sang phần tầng gộp (*pooling*) và các kiến trúc CNN kinh điển.

## Video tham khảo

- **Xem trước:** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, tiếng Anh. Giới thiệu trực giác chung về phép tích chập trước khi đi sâu vào công thức và mã nguồn.
- [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, tiếng Anh. Mở rộng phép tích chập thành tầng nhiều kênh trong mạng CNN.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trọn vẹn cách một tầng tích chập xử lý ảnh cụ thể.
