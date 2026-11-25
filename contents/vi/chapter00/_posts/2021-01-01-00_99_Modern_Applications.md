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

Một câu hỏi rất thực tế mà nhiều kỹ sư trẻ đặt ra trong giai đoạn 2022–2026 là: *"Khi đã có PyTorch, JAX, FlashAttention và các mô hình nền tảng hàng trăm tỷ tham số chạy sẵn qua API, liệu tôi còn cần học giải tích đa biến, giá trị riêng hay xác suất làm gì nữa?"* Thực tế tại các phòng thí nghiệm AI hàng đầu cho thấy điều hoàn toàn ngược lại: chính ở biên giới hiện đại—từ tinh chỉnh mô hình hạng thấp (LoRA), mạng nơ-ron tích hợp vật lý (PINN), cho tới mô hình sinh Diffusion và căn chỉnh RLHF—toán học nền tảng lại là thứ phân biệt người chỉ biết gọi thư viện với kỹ sư có thể chẩn đoán, tối ưu và sáng tạo kiến trúc mới.

## Kiến thức cần nhớ: Bốn cầu nối từ Toán nền tảng Chương 00 đến Deep Learning 2022–2026

1. **Đại số tuyến tính hạng thấp $$\to$$ Tinh chỉnh hiệu quả tham số (LoRA / QLoRA)**:
   Thay vì cập nhật toàn bộ ma trận trọng số $$\mathbf{W} \in \mathbb{R}^{d \times d}$$ của LLM, LoRA phân rã độ lệch $$\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$$ với hạng $$r \ll d$$, cắt giảm tới >99% số tham số huấn luyện mà vẫn giữ nguyên hiệu năng.

2. **Giải tích vi phân tự động bậc cao $$\to$$ Học máy Khoa học (PINN & Neural ODE)**:
   Đưa trực tiếp các phương trình đạo hàm riêng (PDE) như Navier–Stokes hay phương trình truyền nhiệt vào hàm mất mát để mạng nơ-ron học nghiệm tuân thủ định luật vật lý.

3. **Phân phối Gaussian & Quá trình ngẫu nhiên $$\to$$ Mô hình Diffusion & Flow Matching**:
   Sinh ảnh/video độ phân giải cao bằng cách học đảo ngược quá trình cộng nhiễu Gaussian từng bước thông qua dự đoán hàm điểm (*score function* $$\nabla_{\mathbf{x}} \log p(\mathbf{x})$$).

4. **KL-Divergence & Tối ưu có ràng buộc $$\to$$ Căn chỉnh LLM (RLHF / DPO)**:
   Giữ cho mô hình ngôn ngữ sau tinh chỉnh không trôi quá xa phân phối gốc bằng số hạng phạt $$\beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$.

### Giải thích: Tính tay tỷ lệ tiết kiệm tham số của LoRA bằng phép nhân ma trận hạng thấp

Giả sử một lớp chiếu (*projection layer*) trong mô hình ngôn ngữ lớn có ma trận trọng số vuông $$\mathbf{W} \in \mathbb{R}^{4096 \times 4096}$$:

1. **Tinh chỉnh toàn phần (Full Fine-Tuning)**:
   Số tham số cần cập nhật và lưu trạng thái optimizer cho riêng ma trận này là:
   $$4096 \times 4096 = 16{,}777{,}216 \text{ tham số } (\approx 16.78\text{ triệu}).$$
2. **Tinh chỉnh bằng LoRA với hạng $$r = 8$$**:
   Ta đóng băng $$\mathbf{W}$$ và chỉ học hai ma trận mỏng $$\mathbf{B} \in \mathbb{R}^{4096 \times 8}$$ và $$\mathbf{A} \in \mathbb{R}^{8 \times 4096}$$ sao cho $$\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$$:
   - Số tham số của $$\mathbf{B}$$: $$4096 \times 8 = 32{,}768$$.
   - Số tham số của $$\mathbf{A}$$: $$8 \times 4096 = 32{,}768$$.
   - Tổng số tham số huấn luyện: $$65{,}536$$ tham số.
3. **Tỷ lệ cắt giảm**:
   $$\frac{65{,}536}{16{,}777{,}216} = \frac{1}{256} \approx 0.39\%!$$
Chỉ từ một ý tưởng đại số tuyến tính cổ điển (phân rã hạng thấp), LoRA giúp bạn tinh chỉnh mô hình hàng tỷ tham số ngay trên một GPU cá nhân.

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
