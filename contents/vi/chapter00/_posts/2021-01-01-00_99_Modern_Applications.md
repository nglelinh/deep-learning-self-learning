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

# Tùy chọn: giải tích, đại số tuyến tính và xác suất trong các stack 2022–2026

> Bài này là **tùy chọn**. Nó **không** thay các ghi chú cốt lõi về giải tích, đại số tuyến tính hay xác suất. Nó chỉ cho thấy các đối tượng toán ấy xuất hiện thế nào trong phần mềm khoa học và production hiện nay.

Ý tưởng trung tâm của chương: gradient, ma trận và phân phối là ngôn ngữ của học. Từ khoảng 2022–2026 ngôn ngữ đó trở thành *chạy được*: compiler biến đồ thị toán thành kernel GPU, và mã khoa học dùng chung autodiff với deep learning.

## 1. Vì sao chương toán vẫn cần

Automatic differentiation là dạng công nghiệp của quy tắc chuỗi trong bài giải tích. Nếu $$f = g \circ h$$ thì

$$Df(\mathbf{x}) = Dg(h(\mathbf{x})) \cdot Dh(\mathbf{x}).$$

Reverse-mode (PyTorch, JAX) nhân tích này từ đầu ra về — đúng là lan truyền ngược cho loss vô hướng. Cái mới không phải đạo hàm khác, mà là *cách* tích được lập lịch, fuse và kiểm tra số.

## 2. Ứng dụng cụ thể

### Scientific ML và PINN

Mạng thông tin vật lý vẫn cực tiểu hóa phần dư gồm loss dữ liệu và phần dư PDE. Phần dư của toán tử vi phân $$\mathcal{N}$$ là

$$\mathcal{L}_{\text{PDE}} = \mathbb{E}_{\mathbf{x}}\big\|\mathcal{N}[\hat{u}](\mathbf{x})\big\|^2,$$

trong đó đạo hàm của mạng $$\hat{u}$$ đến từ autodiff, không phải sai phân hữu hạn. [Modulus](https://docs.nvidia.com/deeplearning/modulus/index.html) của NVIDIA và hệ sinh thái JAX ([JAX-CFD](https://github.com/google/jax-cfd), [JAX-MD](https://github.com/jax-md/jax-md)) biến mẫu này thành công cụ thường ngày sau 2022.

### Lập trình khả vi ngoài mạng neuron

`grad`, `vmap`, `jit` của JAX coi một bước mô phỏng như hàm có Jacobian ghép được. Đó là đại số tuyến tính (tích Jacobian–vector / vector–Jacobian) cộng góc nhìn Taylor: một vòng Newton–Schulz hay conjugate gradient cũng chỉ là một khối khả vi.

### Xác suất trong huấn luyện foundation model

Các bài scaling law coi training như thí nghiệm thống kê: loss $$L$$ theo compute, token và số tham số. [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) (Chinchilla) khớp các luật lũy thừa

$$L(N, D) \approx E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

cùng ngôn ngữ kỳ vọng / hợp lý như các bài xác suất, áp vào ngân sách huấn luyện LLM.

## 3. Phần mềm phổ biến

- [JAX](https://github.com/google/jax) — `grad` / `vmap` / `jit` kết hợp được.
- [PyTorch 2](https://pytorch.org/get-started/pytorch-2.0/) `torch.compile` — bắt đồ thị autograd ([Ansel et al., 2024](https://arxiv.org/abs/2404.14294)).
- [tinygrad](https://github.com/tinygrad/tinygrad) và [MLX](https://github.com/ml-explore/mlx) — engine autodiff nhỏ, lộ rõ giải tích.

## 4. Trích dẫn (2022–2026)

- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — luật Chinchilla: phân bổ token và tham số cùng lúc.
- [PyTorch 2 (Ansel et al., 2024)](https://arxiv.org/abs/2404.14294) — compiler cho đồ thị autograd.
- [JAX](https://github.com/google/jax) — stack nghiên cứu chuẩn cho autodiff + batch hóa SIMD.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Vẫn dùng các bài giải tích và đại số để tính $$\nabla f$$ và phân tích ma trận bằng tay. Bài tùy chọn chỉ chỉ chỗ các đối tượng ấy sống trong compiler và scientific ML — không thay các chứng minh.
