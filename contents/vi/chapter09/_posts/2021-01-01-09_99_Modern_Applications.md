---
layout: post
title: 09-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '09'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter09
lesson_type: optional
---

# Tùy chọn: Chính quy hóa hiện đại — SAM, DropPath và công thức huấn luyện LLM

> Bài này mang tính **tùy chọn** và **không** thay thế phần lý thuyết về dropout hay BatchNorm. Nội dung tổng hợp các kỹ thuật chính quy hóa thường xuất hiện trong quy trình huấn luyện giai đoạn 2022–2026.

Công thức inverted dropout:

$$\tilde{\mathbf{h}} = \frac{\mathbf{m} \odot \mathbf{h}}{1-p}, \quad m_i \sim \mathrm{Bernoulli}(1-p)$$

vẫn là mô hình tư duy nền tảng. Tuy nhiên, trong các kiến trúc Transformer và mạng thị giác lớn, thực hành hiện đại thường ưu tiên bỏ ngẫu nhiên **nhánh phần dư** (*stochastic depth* / DropPath) thay vì từng neuron riêng lẻ, đồng thời dựa nhiều hơn vào **weight decay kết hợp quy mô dữ liệu**.

## 1. Sharpness-Aware Minimization (SAM)

[Foret et al., 2021](https://arxiv.org/abs/2010.01412) đề xuất SAM (được ứng dụng rộng rãi trong thị giác máy tính từ sau năm 2022) nhằm tìm bộ tham số nằm trong vùng cực tiểu phẳng, vẫn đạt mất mát thấp trên cả một lân cận:

$$\min_\theta \max_{\|\boldsymbol{\varepsilon}\|_2 \le \rho} \mathcal{L}(\theta + \boldsymbol{\varepsilon}).$$

Trong thực tế, mỗi bước cập nhật cần hai lượt truyền xuôi–ngược: trước hết dịch tới điểm xấu nhất trong lân cận $$\theta + \rho \frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|}$$, sau đó dùng gradient tại điểm đó để cập nhật tham số gốc. Đây là cách chính quy hóa trực tiếp **quỹ đạo tối ưu**, bổ trợ cho nhiễu kích hoạt kiểu dropout.

## 2. Ứng dụng thực tế tiêu biểu

### Stochastic Depth và DropPath

Các cấu hình huấn luyện tiêu biểu của ConvNeXt và ViT (như mặc định trong thư viện `timm` giai đoạn 2022–2026) thường ngẫu nhiên bỏ qua toàn bộ một khối phần dư trong quá trình huấn luyện. Thay vì vô hiệu hóa từng đơn vị ẩn, kỹ thuật này hoạt động như dropout theo **đường dẫn**, tạo hiệu ứng tổ hợp ẩn của các mạng có độ sâu khác nhau.

### Cách các mô hình ngôn ngữ lớn (LLM) chính quy hóa

Quy trình tiền huấn luyện kiểu LLaMA chủ yếu dựa vào **weight decay** (với AdamW), hầu như không dùng dropout ở cấp độ token hoặc chỉ dùng tỷ lệ rất nhỏ trên các mô hình cỡ nhỏ. Ở quy mô từ 7 tỷ tham số trở lên, yếu tố kiểm soát tổng quát hóa quan trọng nhất là **tổ hợp dữ liệu huấn luyện, đóng gói chuỗi (packing) và weight decay**, theo sau bởi giai đoạn căn chỉnh RLHF/DPO (Chương 21) — khác hẳn mức dropout $$p=0.5$$ quen thuộc ở các bài tập MLP cổ điển.

### Các biến thể chuẩn hóa

LayerNorm và RMSNorm thay thế BatchNorm trong hầu hết mô hình chuỗi hiện đại. Kết hợp giữa kết nối phần dư và chuẩn hóa lớp giúp giữ ổn định thang kích hoạt để mô hình có thể huấn luyện sâu và dài mà không bị phân kỳ.

## 3. Thư viện và mã nguồn phổ biến

- `DropPath` trong thư viện `timm`, cùng các cài đặt SAM trong [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer) và [sam](https://github.com/davda54/sam).
- `torch.optim.AdamW` với tham số `weight_decay` trong PyTorch (xem thêm Chương 10).
- Các công thức huấn luyện mặc định của Hugging Face (`Trainer`, `trl`) thường ưu tiên weight decay và dùng rất ít hidden dropout.

## 4. Trích dẫn (2022–2026)

- [SAM (Foret et al., 2021)](https://arxiv.org/abs/2010.01412) — mục tiêu tối ưu vùng phẳng được áp dụng rộng rãi trong thị giác máy tính.
- [ConvNeXt (Liu et al., 2022)](https://arxiv.org/abs/2201.03545) — sử dụng stochastic depth như thành phần tiêu chuẩn trong cấu hình huấn luyện.
- [AdamW (Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) — tách riêng weight decay khỏi bước cập nhật thích nghi, vẫn là bộ tối ưu mặc định cho LLM.

## 5. Vai trò bổ trợ cho nội dung cốt lõi

Bạn nên nắm vững cơ chế inverted dropout và vị trí đặt BatchNorm trong phần cốt lõi trước. Bài đọc thêm này cập nhật thêm **SAM** và **DropPath**, đồng thời làm rõ sự dịch chuyển của các mô hình nền tảng rời xa dropout nặng trên từng đơn vị ẩn.

## Video tham khảo

- **Xem trước:** [Sharpness-Aware Minimization (SAM) in 7 minutes](https://www.youtube.com/watch?v=k6-jJ58MFKU) — Yuxiang "Shawn" Wang (tiếng Anh). Giới thiệu ngắn gọn phương pháp SAM giúp tìm cực tiểu phẳng và cải thiện khả năng tổng quát hóa.
- [Overfitting in a Neural Network explained](https://www.youtube.com/watch?v=DEMmkFC6IGM) — deeplizard (tiếng Anh). Ôn lại vấn đề quá khớp mà SAM và DropPath tiếp tục giải quyết.
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — StatQuest with Josh Starmer (tiếng Anh). Kết nối góc nhìn độ sắc của cực tiểu với khung lý thuyết bias–variance quen thuộc.
