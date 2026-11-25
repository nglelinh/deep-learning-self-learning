---
layout: post
title: 24-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '24'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter24
lesson_type: optional
---

Khi cố nhìn trực tiếp vào một nơ-ron đơn lẻ bên trong một Mô hình Ngôn ngữ Lớn, các nhà nghiên cứu phát hiện một điều gây choáng: *cùng một nơ-ron* vừa sáng lên khi thấy mã nguồn Python, vừa sáng lên khi thấy trích dẫn Kinh Thánh, lại vừa sáng lên khi thấy công thức hóa học! Hiện tượng **Chồng chập (Superposition)** này khiến việc đọc từng nơ-ron thất bại — và thúc đẩy cuộc cách mạng **Diễn giải Cơ chế (Mechanistic Interpretability)** giai đoạn 2023–2026 với **Sparse Autoencoder (SAE)** và **Circuit Analysis**.

## Kiến thức cần nhớ: Ba đột phá của Diễn giải Mô hình hiện đại (2022–2026)

- **Giả thuyết chồng chập (Superposition) và nơ-ron đa nghĩa (Polysemanticity):** Mạng nơ-ron nhồi nhét $$M \gg d$$ khái niệm thưa vào một không gian kích hoạt $$d$$ chiều dưới dạng các hướng gần trực giao.
- **Sparse Autoencoder (SAE) tách đặc trưng đơn nghĩa:** Mở rộng vector kích hoạt $$\mathbf{h} \in \mathbb{R}^d$$ sang không gian từ điển rộng hơn nhiều $$\mathbf{f} \in \mathbb{R}^M$$ ($$M = 8d\text{–}64d$$) với phạt thưa $$L_1$$ hoặc TopK để tách riêng từng khái niệm sạch (ví dụ đặc trưng "Cầu Cổng Vàng", "lỗi bảo mật SQL").
- **Phân tích mạch (Circuit Analysis) và Vá kích hoạt (Activation Patching):** Thay thế kích hoạt ở từng đầu chú ý/tầng MLP giữa câu sạch và câu nhiễu để chứng minh *quan hệ nhân quả* của các mạch như Induction Heads.
- **Ứng dụng điều hướng hành vi (Feature Steering):** Tăng hoặc khóa trực tiếp cường độ của một đặc trưng SAE đã phát hiện để kiểm soát phong cách hoặc giảm hành vi không an toàn của LLM ngay lúc suy luận.

### Giải thích: Hiểu định lượng về Chồng chập (Superposition) và không gian từ điển của Sparse Autoencoder

Trong một tầng Transformer có chiều ẩn $$d = 4{,}096$$, giả sử mô hình cần biểu diễn $$M = 65{,}536$$ khái niệm thực tế (gấp $$16\times$$ số chiều sẵn có!), nhưng tại mỗi token trung bình chỉ có $$k = 32$$ khái niệm thực sự xuất hiện cùng lúc.

1. **Tỷ lệ mở rộng (Expansion Factor) của SAE:** $$M / d = 65{,}536 / 4{,}096 = 16\times$$.
2. **Mật độ kích hoạt (Sparsity $$L_0$$):** Chỉ $$32 / 65{,}536 \approx 0{,}049\%$$ số đặc trưng SAE khác 0 trên mỗi token.
3. **Hàm mục tiêu của SAE trên một vector kích hoạt $$\mathbf{h}$$:**
   $$\mathcal{L}_{\text{SAE}} = \|\mathbf{h} - \hat{\mathbf{h}}\|_2^2 + \lambda \|\mathbf{f}\|_1.$$
   Ví dụ nếu $$\|\mathbf{h} - \hat{\mathbf{h}}\|_2^2 = 0{,}08$$, tổng độ lớn đặc trưng $$\|\mathbf{f}\|_1 = 24{,}0$$ và $$\lambda = 0{,}005$$, thì $$\mathcal{L}_{\text{SAE}} = 0{,}08 + 0{,}12 = 0{,}20$$. Chính sự giằng co giữa sai số tái tạo và phạt $$L_1$$ buộc SAE chọn ra bộ từ điển các khái niệm **đơn nghĩa** nhất thay vì tái tạo bằng các tổ hợp pha trộn!

# Tùy chọn: Diễn giải sau LIME/SHAP — Mạch Transformer và Autoencoder thưa

> Bài này là **tùy chọn**, **không** thay thế các phần lý thuyết và cài đặt về *saliency*, LIME hay SHAP. Nội dung giới thiệu nhánh **diễn giải cơ chế** (*mechanistic interpretability*) cho Transformer — hướng nghiên cứu phát triển mạnh cho LLM trong giai đoạn 2022–2026.

SHAP vẫn trả lời tốt câu hỏi “đặc trưng đầu vào nào làm thay đổi điểm dự đoán?”. Tuy nhiên, với các mô hình ngôn ngữ lớn cỡ 70B tham số, câu hỏi trọng tâm thường chuyển sang: “**đặc trưng nội tại** nào bên trong mạng đảm nhiệm hành vi từ chối, cú pháp mã nguồn hay một sự kiện tri thức cụ thể?”.

## 1. Autoencoder thưa (Sparse Autoencoder) trên giá trị kích hoạt

[Cunningham et al., 2023](https://arxiv.org/abs/2309.08600) và [Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Anthropic, *Scaling Monosemanticity*) huấn luyện *autoencoder* thưa (SAE) trên các vector kích hoạt của luồng phần dư (*residual stream*) $$x$$:

$$\hat{x} = W_{\mathrm{dec}}\,\mathrm{ReLU}(W_{\mathrm{enc}}x + b) + b', \quad \mathcal{L}=\|x-\hat{x}\|_2^2 + \lambda\|f\|_1.$$

Mã thưa $$f$$ giúp phân tách chồng chập thành các **đặc trưng đơn nghĩa** (*monosemantic features*) dễ diễn giải hơn. Từ đó, nhà nghiên cứu có thể thử điều hướng (*steering*) hoặc triệt tiêu từng đặc trưng riêng lẻ, dù tiêu chuẩn đánh giá định lượng vẫn đang hoàn thiện. Ở đây, *autoencoder* (Chương 12) đóng vai trò công cụ giải phẫu mô hình thay vì mô hình sinh dữ liệu.

## 2. Ứng dụng cụ thể

### Các công cụ quy kết vẫn giữ vai trò trong triển khai thực tế

Integrated Gradients trong [Captum](https://captum.ai/) và *attention rollout* vẫn được dùng rộng rãi trong các ngành chịu sự quản lý để lập hồ sơ mô hình (*model card*). Với dữ liệu dạng bảng (như chấm điểm tín dụng), SHAP và LIME trong bài lý thuyết vẫn là lựa chọn phù hợp hơn là áp dụng SAE.

### Phân tích mạch (Circuit Analysis)

Loạt nghiên cứu [Transformer Circuits (Elhage et al.)](https://transformer-circuits.pub/) (2022–2024) phân tích ngược các cấu trúc như *induction head* và những mạch tính toán con bên trong Transformer. Đây là tài liệu mở rộng, không thay thế bài tập thực hành Grad-CAM cơ bản.

### Đánh giá an toàn mô hình

Diễn giải cơ chế ngày càng được dùng làm đầu vào cho **quản trị và an toàn AI**, chẳng hạn tìm hiểu liệu có thể nhận diện các đặc trưng liên quan đến hành vi sai lệch hay không. Các kết quả hiện vẫn mang tính nghiên cứu ban đầu; mục tiêu của bài là giới thiệu bộ công cụ chứ chưa khẳng định mô hình lớn đã được giải thích hoàn toàn.

## 3. Phần mềm phổ biến

- [captum](https://github.com/pytorch/captum) — thư viện quy kết cho PyTorch trong sản xuất.
- [openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder) và [SAELens](https://github.com/decoderesearch/SAELens).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — công cụ phân tích mạch Transformer thông qua cơ chế *hook*.

## 4. Trích dẫn (2022–2026)

- [Sparse Autoencoders Find Highly Interpretable Features (Cunningham et al., 2023)](https://arxiv.org/abs/2309.08600).
- [Scaling Monosemanticity (Templeton et al., 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- [A Mathematical Framework for Transformer Circuits (Elhage et al.)](https://transformer-circuits.pub/2021/framework/index.html).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Trước hết cần thực hành thành thạo LIME, SHAP và bản đồ *saliency* trong các bài cốt lõi. Bài này bổ sung nhóm **công cụ diễn giải cơ chế cho LLM** (SAE và phân tích mạch) xuất hiện sau hệ thống phương pháp XAI cổ điển.

## Video tham khảo

- **Xem trước** [A Window  Into LLMs | Sparse Autoencoders Explained](https://www.youtube.com/watch?v=TKozVZoXAYs) — Papers Are Wonderful, tiếng Anh. Giải thích cách *autoencoder* thưa trích xuất đặc trưng diễn giải được từ giá trị kích hoạt của LLM.
- [The Dark Matter of AI \[Mechanistic Interpretability\]](https://www.youtube.com/watch?v=UGO_Ehywuxc) — Welch Labs, tiếng Anh. Tổng quan về chương trình nghiên cứu diễn giải cơ chế, nơi các *autoencoder* thưa được ứng dụng.
- [Lecture 12 | Visualizing and Understanding](https://www.youtube.com/watch?v=6wcs6szJWMY) — Stanford University School of Engineering, tiếng Anh. Nền tảng trực quan hóa mạng nơ-ron trước đó mà hướng diễn giải cơ chế tiếp nối.
