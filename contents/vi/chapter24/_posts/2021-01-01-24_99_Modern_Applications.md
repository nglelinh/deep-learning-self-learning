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

# Tùy chọn: diễn giải sau LIME/SHAP — mạch và autoencoder thưa

> Bài này là **tùy chọn**. Nó **không** thay saliency, LIME, SHAP, hay ghi chú cài đặt. Nó phủ **diễn giải cơ chế** Transformer — dòng nghiên cứu/sản phẩm 2022–2026 cho LLM.

SHAP vẫn trả lời “đặc trưng đầu vào nào làm dịch điểm?” Với mô hình chat 70B câu hỏi thú vị thường là “**đặc trưng nội** nào thực hiện từ chối / cú pháp mã / một sự kiện ẩn?”

## 1. Autoencoder thưa trên activation

[Cunningham et al., 2023](https://arxiv.org/abs/2309.08600) và [Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Anthropic, *Scaling Monosemanticity*) huấn luyện AE thưa trên activation dòng residual $$x$$:

$$\hat{x} = W_{\mathrm{dec}}\,\mathrm{ReLU}(W_{\mathrm{enc}}x + b) + b', \quad \mathcal{L}=\|x-\hat{x}\|_2^2 + \lambda\|f\|_1.$$

Mã thưa $$f$$ được coi như **đặc trưng đơn nghĩa**. Đội rồi lái hoặc cắt các đặc trưng ấy (đánh giá chuẩn vàng vẫn lộn xộn). Đây là autoencoder (Chương 12) dùng như *dụng cụ diễn giải*, không phải mô hình sinh.

## 2. Ứng dụng cụ thể

### Attribution vẫn được giao

Integrated gradients [Captum](https://captum.ai/) và attention-rollout vẫn là thứ ngành được quản lý đưa vào model card. Dùng SHAP/LIME của bài lý thuyết cho DS dạng bảng; đừng thay chúng bằng SAE trên mô hình tín dụng.

### Phân tích kiểu mạch

[Transformer Circuits (Elhage et al.)](https://transformer-circuits.pub/) (2022–2024) dịch ngược induction head và các motif khác. Đọc bổ sung, không thay homework Grad-CAM.

### Đánh giá an toàn

Diễn giải giờ là đầu vào **quản trị**: ta có định vị được đặc trưng “lừa dối”? Paper còn sơ bộ; bài học của khóa là *công cụ*, không phải tuyên bố mô hình đã được giải thích hết.

## 3. Phần mềm phổ biến

- [captum](https://github.com/pytorch/captum) — attribution production.
- [openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder) và [SAELens](https://github.com/decoderesearch/SAELens).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — làm mạch bằng hook.

## 4. Trích dẫn (2022–2026)

- [Sparse Autoencoders Find Highly Interpretable Features (Cunningham et al., 2023)](https://arxiv.org/abs/2309.08600).
- [Scaling Monosemanticity (Templeton et al., 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- [A Mathematical Framework for Transformer Circuits (Elhage et al.)](https://transformer-circuits.pub/2021/framework/index.html).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Chạy LIME/SHAP và saliency của chương trước. Bài này chỉ thêm **công cụ cơ chế LLM** (SAE, mạch) xuất hiện sau catalog XAI cổ điển.
