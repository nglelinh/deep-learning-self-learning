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

Nếu thập niên 2010 là sân khấu của Dropout và BatchNorm, thì giai đoạn 2022–2026 chứng kiến sự dịch chuyển ngoạn mục trong cách chính quy hóa các mô hình khổng lồ: Vision Transformer thay Dropout bằng **Stochastic Depth (DropPath)**, các mô hình LLM nghìn tỷ token gần như **tắt hẳn Dropout** để tận dụng dữ liệu khổng lồ kết hợp **RMSNorm + Weight Decay tách rời**, còn giới tối ưu hóa đón nhận **SAM (Sharpness-Aware Minimization)** để chủ động săn lùng các vùng cực tiểu phẳng!

## Bản đồ kiến thức nền: Bốn xu hướng Chính quy hóa Hiện đại (2022–2026)

1. **Sharpness-Aware Minimization (SAM)**:
   Thay vì chỉ tìm điểm $$w$$ có mất mát thấp, SAM tìm điểm $$w$$ sao cho **cả vùng lân cận bán kính $$\rho$$** quanh nó đều có mất mát thấp: $$\min_w \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}(w + \epsilon)$$. Cực tiểu phẳng (*flat minima*) tổng quát hóa tốt hơn hẳn cực tiểu nhọn (*sharp minima*).
2. **Stochastic Depth / DropPath**:
   Thay vì tắt từng neuron rời rạc như Dropout, DropPath tắt ngẫu nhiên **cả một nhánh khối Residual** ($$y = x + \text{DropPath}(F(x))$$) theo tỷ lệ tăng dần theo chiều sâu — chuẩn mực bắt buộc khi huấn luyện ViT, Swin Transformer và ConvNeXt.
3. **Công thức chính quy hóa của LLM hiện đại (Pre-training vs. Fine-tuning)**:
   Khi tiền huấn luyện trên hàng nghìn tỷ token (mỗi văn bản chỉ đi qua 1 lần — 1 epoch), bản thân dữ liệu khổng lồ đã ngăn overfitting nên hầu hết LLM lớn (Llama 3, Qwen, DeepSeek) đặt `dropout = 0.0` và dựa vào **decoupled weight decay (AdamW)**, gradient clipping và z-loss; chỉ bật lại Dropout nhẹ khi tinh chỉnh (LoRA/SFT) trên tập nhỏ.
4. **Chuẩn hóa ổn định ở quy mô lớn (RMSNorm & QK-Norm)**:
   Bên cạnh RMSNorm thay LayerNorm, nhiều mô hình đa phương thức và LLM mới áp dụng thêm **QK-Norm** (chuẩn hóa riêng Query và Key trước khi nhân $$Q K^T$$) để dập tắt hoàn toàn các đột biến *loss spike* khi huấn luyện FP16/BF16 sâu hàng trăm tầng.

### Khung giải thích (nên đọc chậm): Tính tay 2 bước cập nhật của thuật toán SAM (Sharpness-Aware Minimization)

Xét hàm mất mát 1 chiều $$\mathcal{L}(w) = (w - 1)^2$$ tại điểm hiện tại $$w = 2{,}0$$, bán kính nhiễu SAM $$\rho = 0{,}5$$ và tốc độ học $$\eta = 0{,}2$$:

1. **Bước 1 — Leo lên đỉnh xấu nhất trong bán kính $$\rho$$ (Ascent Step)**:
   - Gradient tại $$w = 2{,}0$$: $$g = \mathcal{L}'(2{,}0) = 2(2{,}0 - 1) = \mathbf{2{,}0}$$.
   - Vector nhiễu hướng xấu nhất: $$\hat{\epsilon} = \rho \frac{g}{|g|} = 0{,}5 \times \frac{2{,}0}{2{,}0} = \mathbf{+0{,}5}$$.
   - Điểm bị nhiễu: $$w_{\text{adv}} = w + \hat{\epsilon} = 2{,}0 + 0{,}5 = \mathbf{2{,}5}$$.
2. **Bước 2 — Tính gradient tại điểm xấu nhất $$w_{\text{adv}}$$, rồi cập nhật cho điểm gốc $$w$$ (Descent Step)**:
   - Gradient tại $$w_{\text{adv}} = 2{,}5$$: $$g_{\text{adv}} = \mathcal{L}'(2{,}5) = 2(2{,}5 - 1) = \mathbf{3{,}0}$$.
   - Cập nhật $$w$$: $$w_{\text{new}} = w - \eta g_{\text{adv}} = 2{,}0 - 0{,}2(3{,}0) = \mathbf{1{,}40}$$.

So với SGD thường ($$w_{\text{new}} = 2{,}0 - 0{,}2(2{,}0) = 1{,}60$$), SAM "nhìn trước" độ dốc ở lân cận $$w + \hat{\epsilon}$$ nên đẩy mạnh hơn khỏi các thung lũng hẹp có độ cong lớn — đánh đổi là tốn **gấp 2 lần** lượt forward/backward mỗi bước!

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
