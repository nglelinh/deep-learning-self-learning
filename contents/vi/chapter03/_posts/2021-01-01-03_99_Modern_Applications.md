---
layout: post
title: 03-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '03'
order: 14
owner: Deep Learning Course
lang: vi
categories:
- chapter03
lesson_type: optional
---

# Tùy chọn: Lan truyền ngược dưới góc nhìn trình biên dịch

> Bài này là **tùy chọn** và **không** thay thế lý thuyết về hàm mất mát, gradient descent hay lan truyền ngược. Mục đích của bài là kết nối các thuật toán nền tảng đó với trình biên dịch và nhân tính toán (*kernel*) dùng trong thực tế huấn luyện từ sau năm 2022.

Về bản chất toán học, lan truyền ngược vẫn là vi phân tự động chế độ ngược cho hàm mất mát vô hướng $$\mathcal{L}$$. Điểm thay đổi lớn giai đoạn 2022–2026 nằm ở cách thực thi: **đồ thị tính toán được biên dịch trước** — nhiều toán tử được hợp nhất (*fused*), các giá trị kích hoạt có thể được tính lại thay vì lưu toàn bộ, và huấn luyện độ chính xác hỗn hợp (*mixed precision*) trở thành thiết lập mặc định.

## 1. Cùng quy tắc dây chuyền, khác cách lập lịch thực thi

Nếu một lớp tính $$\mathbf{y} = f(\mathbf{x}; \theta)$$, lượt ngược vẫn cần tính $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$$ và $$\frac{\partial \mathcal{L}}{\partial \theta}$$. Các trình biên dịch như `torch.compile`, XLA hay Triton sẽ tự động quyết định:

- những giá trị trung gian nào nên **lưu lại** và những giá trị nào nên **tính lại** (*activation checkpointing*),
- những phép toán theo từng phần tử nào nên **hợp nhất** (*fuse*) thành một nhân tính toán GPU duy nhất,
- có thực thi phép tính ở định dạng BF16/FP8 và tích lũy ở FP32 hay không.

Toán học của Chương 03 giữ nguyên; điều thay đổi là thời gian chạy thực tế (*wall-clock time*) và lượng bộ nhớ tiêu thụ.

## 2. Các ứng dụng và kỹ thuật tiêu biểu

### `torch.compile` và JAX `jit`

[Ansel et al., 2024](https://arxiv.org/abs/2404.14294) mô tả kiến trúc Dynamo + AOTAutograd + TorchInductor của PyTorch 2: bytecode Python được ghi nhận thành đồ thị FX rồi biên dịch xuống mã máy tối ưu cho phần cứng. JAX áp dụng tư duy tương tự thông qua `jit` và `grad`. Cả hai là lời giải trong thực tế sản xuất cho câu hỏi: "Sau khi hiểu lan truyền ngược bằng NumPy, hệ thống lớn triển khai nó như thế nào?"

### Huấn luyện độ chính xác hỗn hợp và tỷ lệ hóa mất mát (*loss scaling*)

Một bước cập nhật tiêu chuẩn có dạng:

$$\theta \leftarrow \theta - \eta \cdot \mathrm{Cast}_{fp32}(\widehat{\nabla \mathcal{L}}),$$

trong đó $$\widehat{\nabla \mathcal{L}}$$ được tính bằng BF16 hoặc FP16. Để tránh hiện tượng tràn số dưới (*underflow*) ở FP16, giá trị hàm mất mát thường được nhân thêm một hệ số trước khi chạy lượt ngược (cơ chế AMP trong PyTorch). Bản chất cập nhật SGD/Adam không đổi, chỉ bổ sung chính sách quản lý định dạng dấu phẩy động.

### Lượt ngược mức kernel cho attention

[Dao et al., 2022](https://arxiv.org/abs/2205.14135) (FlashAttention) thiết kế **lượt ngược tối ưu truy cập bộ nhớ (IO-aware)** mà không cần lưu tường minh ma trận attention kích thước $$N\times N$$. Đạo hàm của softmax-attention vẫn giống hệt công thức lý thuyết ở Chương 07, nhưng phần cài đặt là một nhân tính toán lượt ngược đã hợp nhất xử lý theo từng khối nhỏ (*tiling*). Đây là một trong những cải tiến kỹ thuật quan trọng nhất về lan truyền ngược trong giai đoạn gần đây.

## 3. Thư viện và công cụ phổ biến

- [PyTorch 2](https://pytorch.org/) — `torch.compile`, `torch.func`, `torch.utils.checkpoint`.
- [JAX](https://github.com/google/jax) — `grad`, `value_and_grad`, trình biên dịch XLA.
- [Triton](https://github.com/triton-lang/triton) — ngôn ngữ viết nhân GPU tùy chỉnh cho lượt xuôi/ngược khi trình biên dịch mặc định chưa đủ tối ưu.

## 4. Trích dẫn (2022–2026)

- [PyTorch 2 (Ansel et al., 2024)](https://arxiv.org/abs/2404.14294) — ghi nhận đồ thị động kết hợp autograd biên dịch.
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — tính chính xác gradient attention với lượt ngược chia khối tiết kiệm băng thông bộ nhớ.
- [Online softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867) — thuật toán softmax trực tuyến dùng bên trong các kernel này và vẫn là nền tảng số học phổ biến giai đoạn 2024–2026.

## 5. Mối liên hệ với các bài học cốt lõi

Nên hoàn thành phần lý thuyết lan truyền ngược và cài đặt bằng NumPy trước. Bài này chỉ bổ sung lớp hạ tầng kỹ thuật mà ngành công nghiệp đặt xung quanh thuật toán cốt lõi — trình biên dịch, AMP và kernel lượt ngược hợp nhất — chứ không thay đổi bản chất toán học của quá trình học.

## Video tham khảo

- **Xem trước:** [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, tiếng Anh. Ôn lại quy tắc dây chuyền cốt lõi — phần không đổi dù lớp biên dịch và tối ưu thực thi bên dưới có thay đổi.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, tiếng Anh. Giới thiệu Adam — bộ tối ưu mặc định thường dùng song song với các kỹ thuật huấn luyện hiện đại.
- [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, tiếng Anh. So sánh tổng quan Momentum, RMSprop, AdaGrad và Adam trong cùng một khung nhìn.
