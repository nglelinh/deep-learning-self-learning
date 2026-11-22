---
layout: post
title: 02-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '02'
order: 10
owner: Deep Learning Course
lang: vi
categories:
- chapter02
lesson_type: optional
---

# Tùy chọn: Khối MLP hiện đại

> Bài này là **tùy chọn** và **không** trình bày lại lý thuyết về perceptron, kiến trúc, hàm kích hoạt hay lan truyền xuôi. Mục đích của bài là cho thấy các thành phần cơ bản đó được tổ chức ra sao trong các mô hình nền tảng (*foundation model*) giai đoạn 2022–2026.

Ở mức đơn vị cơ bản, neuron vẫn thực hiện phép tính $$z = \mathbf{w}^\top \mathbf{x} + b$$ theo sau bởi một hàm phi tuyến. Điểm thay đổi lớn nằm ở **cấu trúc khối**: đường nối tắt residual, khối MLP có cổng (*gated MLP*) và chuẩn hóa RMSNorm đã trở thành cấu phần mặc định khi xếp chồng các lớp.

## 1. Từ một neuron tới khối MLP của Transformer

Phần lớn các mô hình ngôn ngữ lớn (LLM) hiện nay dùng khối feed-forward có cổng (SwiGLU hoặc GeGLU) thay cho cấu trúc truyền thống `Linear → ReLU → Linear`. Với đầu vào $$\mathbf{x} \in \mathbb{R}^{d}$$:

$$\mathrm{SwiGLU}(\mathbf{x}) = \big(\mathrm{SiLU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)\big) W_3,$$

trong đó $$\mathrm{SiLU}(z) = z\,\sigma(z)$$. Về bản chất, đây vẫn là đại số lan truyền xuôi của Chương 02 nhưng bổ sung thêm một nhánh cổng thông qua tích Hadamard (nhân theo từng phần tử). [Shazeer, 2020](https://arxiv.org/abs/2002.05202) giới thiệu các biến thể GLU này, và chúng đã trở thành lựa chọn chuẩn trong PaLM, LLaMA cũng như nhiều mô hình mã nguồn mở về sau.

## 2. Các ứng dụng và biến thể tiêu biểu

### Pre-norm kết hợp RMSNorm thay cho BatchNorm

Các mô hình LLM hầu như không sử dụng BatchNorm mà chuyển sang **RMSNorm**:

$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \odot \boldsymbol{\gamma}$$

và đặt lớp chuẩn hóa này trước nhánh residual (*pre-norm*). Đây là giải pháp thực tiễn giúp kiểm soát thang đo kích hoạt và hạn chế gradient biến mất ở quy mô từ 7B đến 400B tham số.

### MLP-Mixer và hướng tiếp cận "toàn MLP" trong thị giác

[Tolstikhin et al., 2021](https://arxiv.org/abs/2105.01601) (MLP-Mixer) chứng minh rằng chỉ dùng các lớp MLP trộn theo token và trộn theo kênh vẫn có thể phân loại tốt trên ImageNet mà không cần phép tích chập. Giai đoạn 2022–2024, Mixer thường được dùng như một mô hình cơ sở gọn nhẹ bên cạnh ConvNeXt và ViT chứ không thay thế hoàn toàn CNN.

### Kolmogorov–Arnold Networks (KAN)

[Liu et al., 2024](https://arxiv.org/abs/2404.19756) đề xuất thay các hàm kích hoạt cố định đặt trên nút bằng các hàm một biến học được đặt trên **cạnh**. Nên xem đây là một hướng nghiên cứu thử nghiệm xoay quanh câu hỏi "cấu trúc neuron có thể khác đi thế nào?", chưa phải lựa chọn mặc định cho các hệ thống thị giác hay NLP thực tế.

## 3. Thư viện và phần mềm phổ biến

- `torch.nn.SiLU`, `RMSNorm` và khối SwiGLU trong [LLaMA](https://github.com/meta-llama/llama) và [Hugging Face transformers](https://github.com/huggingface/transformers).
- [timm](https://github.com/huggingface/pytorch-image-models) — thư viện cung cấp các khối MLP và ConvNeXt hiện đại cho thị giác máy tính.

## 4. Trích dẫn (2022–2026)

- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) — nền tảng cho khối MLP có cổng trong các mô hình ngôn ngữ giai đoạn 2024–2026.
- [RMSNorm (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) — lớp chuẩn hóa mặc định trong họ mô hình LLaMA.
- [KAN (Liu et al., 2024)](https://arxiv.org/abs/2404.19756) — kích hoạt học được trên cạnh; một hướng nghiên cứu bổ sung chứ không thay thế MLP.

## 5. Mối liên hệ với các bài học cốt lõi

Các bài về perceptron và hàm kích hoạt giúp bạn hiểu rõ $$z$$ và $$\sigma$$. Bài này chỉ bổ sung **cách đóng gói hiện đại** (SwiGLU, RMSNorm, residual) để bạn dễ đọc hiểu khi mở một checkpoint LLM hoặc tài liệu mô hình trên `timm`. Bài **26-07** sẽ đặt khối này vào toàn bộ kiến trúc decoder cùng RoPE, GQA và phần tính toán chi phí đi kèm.

## Video tham khảo

- **Xem trước:** [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, tiếng Anh. Minh họa khối MLP bên trong Transformer — dạng lớp liên kết đầy đủ hiện đại được đề cập trong bài.
- [The GELU, SiLU and SwiGLU activation functions, clearly explained!!!](https://www.youtube.com/watch?v=2FaI2Fen1mQ) — StatQuest with Josh Starmer, tiếng Anh. Giải thích chi tiết SwiGLU — hàm kích hoạt được sử dụng phổ biến trong các khối MLP hiện đại.
- [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Ôn lại cấu trúc mạng neuron xếp chồng cơ bản để thấy rõ sự phát triển từ một neuron tới MLP của Transformer.
