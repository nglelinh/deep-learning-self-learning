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

Nhiều người nghĩ mạng truyền thẳng (MLP) học ở Chương 02 chỉ là món đồ chơi giáo khoa đã lỗi thời trước thời đại Transformer. Sự thật sẽ khiến bạn bất ngờ: trong các mô hình ngôn ngữ lớn hàng trăm tỷ tham số như GPT, Llama hay Gemma, **hơn 2/3 tổng số tham số của toàn bộ mô hình lại nằm chính trong các khối MLP** (thường gọi là khối FFN — Feed-Forward Network)! Tuy nhiên, khối MLP năm 2022–2026 đã tiến hóa mạnh mẽ so với MLP cổ điển với bộ ba vũ khí mới: **Pre-Norm + RMSNorm**, **cơ chế cổng nhân SwiGLU**, và kiến trúc thưa **Mixture-of-Experts (MoE)**.

## Kiến thức cần nhớ: Bốn nâng cấp biến MLP cổ điển thành Trái tim của Transformer Hiện đại

1. **Pre-Norm kết hợp RMSNorm (thay cho Post-Norm & BatchNorm)**:
   Đưa chuẩn hóa ra *trước* khối MLP và chỉ chia cho căn trung bình bình phương (RMS, bỏ bước trừ trung bình) giúp giảm chi phí tính toán và ổn định huấn luyện mạng hàng trăm tầng mà không cần bước warmup quá mong manh.

2. **Hàm kích hoạt Cổng nhân SwiGLU**:
   Thay vì $$\mathbf{W}_2\,\text{ReLU}(\mathbf{W}_1\mathbf{x})$$, SwiGLU dùng hai nhánh song song: $$\mathbf{W}_{\text{down}}\big(\text{SiLU}(\mathbf{W}_{\text{gate}}\mathbf{x}) \odot (\mathbf{W}_{\text{up}}\mathbf{x})\big)$$, giúp mạng điều tiết dòng thông tin mềm dẻo hơn hẳn.

3. **Liên kết Residual ($$\mathbf{x} + \text{MLP}(\text{Norm}(\mathbf{x}))$$)**:
   Bảo toàn luồng thông tin gốc và cho phép gradient đi thẳng từ tầng cuối về tầng đầu mà không bị bóp nghẹt.

4. **Mixture-of-Experts (MoE — MLP Thưa)**:
   Thay 1 khối MLP khổng lồ bằng $$E$$ khối MLP "chuyên gia" song song kèm một bộ định tuyến (*router*) chỉ kích hoạt top-$$k$$ chuyên gia (ví dụ $$k=2$$ trong $$E=8$$) cho mỗi token—tăng dung lượng tri thức lên gấp nhiều lần mà giữ nguyên FLOPs suy luận!

### Giải thích: Tính tay tỷ lệ mở rộng chiều ẩn của SwiGLU để giữ nguyên ngân sách tham số

Trong Transformer cổ điển (dùng ReLU/GELU), khối FFN mở rộng chiều ẩn gấp $$4$$ lần ($$d_{\text{ff}} = 4d$$) với **2 ma trận** $$\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$$ và $$\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$$:
- Tổng số tham số FFN cổ điển: $$2 \times (d \times 4d) = 8d^2$$.

Sang **SwiGLU**, khối FFN có tới **3 ma trận** ($$\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}}, \mathbf{W}_{\text{down}}$$) với chiều ẩn $$d_{\text{ff}}'$$:
- Tổng số tham số SwiGLU: $$3 \times (d \times d_{\text{ff}}') = 3\,d\,d_{\text{ff}}'$$.
- Để giữ đúng ngân sách $$8d^2$$ tham số ngang bằng FFN cổ điển, ta giải:
  $$3\,d\,d_{\text{ff}}' = 8d^2 \implies d_{\text{ff}}' = \frac{8}{3}d \approx 2.67\,d.$$

Thử với $$d = 4096$$ (cỡ Llama-7B): $$\frac{8}{3} \times 4096 \approx 10{,}923$$ (làm tròn lên bội của 256 thành $$11{,}008$$). Đó chính xác là lý do bạn luôn thấy con số tưởng chừng kỳ lạ $$\frac{8}{3}d$$ hoặc $$11{,}008$$ trong bảng thông số kỹ thuật của Llama và Mistral!

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
