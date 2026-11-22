---
layout: post
title: 00-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '00'
order: 25
owner: Deep Learning Course
lang: vi
categories:
- chapter00
lesson_type: optional
---

# Tùy chọn: Giải tích, đại số tuyến tính và xác suất trong hệ sinh thái 2022–2026

> Bài này là phần **tùy chọn**, **không** thay thế các bài học cốt lõi về giải tích, đại số tuyến tính hay xác suất. Mục đích của bài là minh họa các khái niệm toán học đó đang được dùng như thế nào trong phần mềm khoa học và hệ thống thực tế hiện nay.

Ý tưởng xuyên suốt của chương là gradient, ma trận và phân phối xác suất tạo thành ngôn ngữ chung của quá trình học máy. Trong giai đoạn 2022–2026, ngôn ngữ này được thực thi trực tiếp ở cấp trình biên dịch: đồ thị toán học được biên dịch thành kernel GPU, còn mã tính toán khoa học dùng chung cơ chế vi phân tự động (*autodiff*) với học sâu.

## 1. Vì sao kiến thức toán nền tảng vẫn cần thiết

Vi phân tự động (*automatic differentiation*) là phiên bản được chuẩn hóa ở quy mô công nghiệp của quy tắc dây chuyền trong giải tích. Nếu $$f = g \circ h$$ thì:

$$Df(\mathbf{x}) = Dg(h(\mathbf{x})) \cdot Dh(\mathbf{x}).$$

Chế độ ngược (*reverse-mode*, được dùng trong PyTorch và JAX) nhân chuỗi đạo hàm này từ đầu ra ngược về đầu vào — chính là thuật toán lan truyền ngược cho hàm mất mát vô hướng. Điểm mới của các hệ thống hiện đại không nằm ở công thức đạo hàm, mà ở *cách* tích đạo hàm được lập lịch, gộp phép toán (*fusion*) và kiểm soát độ ổn định số.

## 2. Một số ứng dụng cụ thể

### Học máy khoa học (Scientific ML) và PINN

Mạng nơ-ron tích hợp vật lý (PINN — *Physics-Informed Neural Networks*) cực tiểu hóa hàm mục tiêu gồm sai số khớp dữ liệu và phần dư của phương trình đạo hàm riêng (PDE). Phần dư ứng với toán tử vi phân $$\mathcal{N}$$ có dạng:

$$\mathcal{L}_{\text{PDE}} = \mathbb{E}_{\mathbf{x}}\big\|\mathcal{N}[\hat{u}](\mathbf{x})\big\|^2,$$

trong đó đạo hàm của mạng $$\hat{u}$$ được tính bằng vi phân tự động thay vì sai phân hữu hạn. Các thư viện như [Modulus](https://docs.nvidia.com/deeplearning/modulus/index.html) của NVIDIA cùng hệ sinh thái JAX ([JAX-CFD](https://github.com/google/jax-cfd), [JAX-MD](https://github.com/jax-md/jax-md)) đã đưa cách tiếp cận này trở thành công cụ phổ biến từ sau năm 2022.

### Lập trình khả vi ngoài mạng nơ-ron

Các phép biến đổi `grad`, `vmap`, `jit` trong JAX xem mỗi bước mô phỏng như một hàm có thể ghép nối Jacobian. Bản chất của chúng vẫn là đại số tuyến tính (tích Jacobian–vector và vector–Jacobian) kết hợp với khai triển Taylor: một vòng lặp Newton–Schulz hay thuật toán gradient liên hợp (*conjugate gradient*) đều có thể trở thành một khối tính toán khả vi.

### Xác suất trong huấn luyện mô hình nền tảng

Các nghiên cứu về quy luật mở rộng (*scaling law*) xem quá trình huấn luyện như một thí nghiệm thống kê, mô tả hàm mất mát $$L$$ theo năng lực tính toán, số token và số tham số. Nghiên cứu Chinchilla ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) khớp dữ liệu thực nghiệm bằng mô hình luật lũy thừa:

$$L(N, D) \approx E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

vận dụng cùng ngôn ngữ kỳ vọng và hàm hợp lý trong các bài xác suất để phân bổ ngân sách huấn luyện cho mô hình ngôn ngữ lớn (LLM).

## 3. Công cụ phần mềm phổ biến

- [JAX](https://github.com/google/jax) — hỗ trợ kết hợp linh hoạt `grad`, `vmap` và `jit`.
- [PyTorch 2](https://pytorch.org/get-started/pytorch-2.0/) `torch.compile` — biên dịch đồ thị autograd để tối ưu hiệu năng ([Ansel et al., 2024](https://arxiv.org/abs/2404.14294)).
- [tinygrad](https://github.com/tinygrad/tinygrad) và [MLX](https://github.com/ml-explore/mlx) — các bộ máy vi phân tự động gọn nhẹ, thể hiện rõ bản chất giải tích bên dưới.

## 4. Tài liệu tham khảo (2022–2026)

- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — quy luật Chinchilla về việc cân đối đồng thời số lượng token và số tham số.
- [PyTorch 2 (Ansel et al., 2024)](https://arxiv.org/abs/2404.14294) — trình biên dịch cho đồ thị autograd.
- [JAX](https://github.com/google/jax) — nền tảng nghiên cứu phổ biến cho vi phân tự động và xử lý song song theo lô (*SIMD batching*).

## 5. Vai trò của bài này đối với phần cốt lõi

Bạn vẫn cần dùng các bài giải tích và đại số tuyến tính cốt lõi để tự tay tính $$\nabla f$$ và phân tích ma trận. Bài tùy chọn này chỉ bổ sung bức tranh về cách các đối tượng toán học đó vận hành trong trình biên dịch và học máy khoa học, không thay thế phần lý thuyết nền tảng.

## Video tham khảo

- **Xem trước:** [Backpropagation calculus | Deep Learning Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown, tiếng Anh. Giải thích giải tích đằng sau lan truyền ngược — chính là quy tắc dây chuyền chế độ ngược mà PyTorch và JAX cài đặt.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, tiếng Anh. Tự xây dựng bộ máy vi phân tự động chế độ ngược từ đầu bằng mã nguồn, minh họa trực tiếp kiến thức giải tích của chương.
- [Visualizing the chain rule and product rule | Chapter 4, Essence of calculus](https://www.youtube.com/watch?v=YG15m2VwSjA) — 3Blue1Brown, tiếng Anh. Minh họa quy tắc dây chuyền một biến, cơ sở của đẳng thức $$Df = Dg \cdot Dh$$ trong bài.
