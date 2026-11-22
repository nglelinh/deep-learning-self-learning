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
