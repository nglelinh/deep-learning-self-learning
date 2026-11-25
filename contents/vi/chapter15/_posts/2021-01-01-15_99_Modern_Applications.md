---
layout: post
title: 15-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '15'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter15
lesson_type: optional
---

Khi mô hình nền lớn lên tới 7 tỷ, 70 tỷ hay hàng trăm tỷ tham số, việc "mở khóa toàn bộ mạng để tinh chỉnh" theo kiểu cổ điển trở nên xa xỉ tới mức chỉ vài phòng thí nghiệm lớn mới đủ GPU. Bước ngoặt mang tên **PEFT (Parameter-Efficient Fine-Tuning)** — với hai ngôi sao **LoRA** và **QLoRA** — đã dân chủ hóa hoàn toàn học chuyển giao, giúp bạn tinh chỉnh một mô hình khổng lồ ngay trên một chiếc GPU tiêu dùng mà chất lượng vẫn tiệm cận tinh chỉnh toàn phần.

## Kiến thức cần nhớ: Bốn trụ cột của Học Chuyển giao hiện đại (PEFT 2022–2026)

- **Giả thuyết hạng thấp nội tại (Intrinsic Low-Rank) của LoRA:** Thay vì cập nhật ma trận đầy đủ $$\Delta W \in \mathbb{R}^{d \times k}$$, phân tích $$\Delta W = \frac{\alpha}{r} B A$$ với $$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$ ($$r \ll \min(d, k)$$) và khởi tạo $$B = 0$$ để lúc đầu không làm xáo trộn mô hình gốc.
- **QLoRA (Quantized LoRA):** Đóng băng xương sống ở định dạng 4-bit NormalFloat (NF4) kết hợp lượng tử hóa kép (double quantization) và chỉ giữ trạng thái bộ tối ưu Adam cho các ma trận LoRA nhỏ bé ở BF16.
- **Triển khai không tăng độ trễ (Weight Merging):** Sau khi huấn luyện xong, cộng gộp trực tiếp $$W_{\text{mới}} = W_0 + \frac{\alpha}{r} B A$$ để suy luận nhanh y hệt mô hình gốc, hoặc hoán đổi nóng nhiều adapter LoRA theo từng người dùng.
- **Mở rộng sang Thị giác & Đa phương thức:** Áp dụng LoRA cho lớp chiếu Attention của CLIP, Stable Diffusion/Flux (học phong cách/nhân vật chỉ từ vài chục ảnh) và mô hình ngôn ngữ thị giác (VLM).

### Giải thích: Tính tay số tham số và độ giảm VRAM của LoRA & QLoRA trên một ma trận $$4096 \times 4096$$

Xét một ma trận trọng số chiếu tuyến tính $$W_0 \in \mathbb{R}^{4096 \times 4096}$$ trong Transformer ($$16{,}777{,}216$$ tham số).

1. **So sánh số tham số huấn luyện khi gắn LoRA hạng $$r = 16$$:**
   - Ma trận $$A \in \mathbb{R}^{16 \times 4096}$$: $$65{,}536$$ tham số; ma trận $$B \in \mathbb{R}^{4096 \times 16}$$: $$65{,}536$$ tham số.
   - Tổng tham số LoRA: $$131{,}072$$ tham số — chỉ bằng $$\frac{131{,}072}{16{,}777{,}216} = \frac{1}{128} \approx 0{,}78\%$$ so với tinh chỉnh đầy đủ!
2. **So sánh bộ nhớ lưu trọng số + trạng thái Adam (mô hình 7 tỷ tham số):**
   - **Full Fine-Tuning (FP16 trọng số + FP32 bản sao + Adam $$m, v$$):** $$7\text{B} \times (2 + 4 + 4 + 4)\text{ byte} \approx 98\text{ GB VRAM}$$ (chưa kể kích hoạt).
   - **QLoRA (4-bit NF4 cho xương sống + Adam chỉ cho $$0{,}8\%$$ tham số LoRA):** Xương sống tốn $$7\text{B} \times 0{,}5\text{ byte} = 3{,}5\text{ GB}$$, toàn bộ trọng số & trạng thái Adam của LoRA tốn chưa tới $$0{,}8\text{ GB}$$ — tổng cộng dưới $$5\text{ GB}$$ cho phần trọng số/optimizer, vừa vặn trên một GPU 16–24 GB!

# Tùy chọn: Học chuyển giao theo hướng PEFT — LoRA, QLoRA và Adapter

> Bài này mang tính **tùy chọn** và **không** thay thế phần lý thuyết so sánh trích xuất đặc trưng với tinh chỉnh toàn phần. Các bài chính xuất phát từ ResNet và BERT; bài này trình bày cách tiếp cận mặc định giai đoạn 2022–2026 cho các mô hình **hàng tỷ tham số**.

Tinh chỉnh toàn phần cập nhật mọi tham số $$\theta$$. Tinh chỉnh tiết kiệm tham số (PEFT) đóng băng $$\theta$$ và chỉ học một phần bổ sung $$\Delta$$ có kích thước nhỏ:

$$W' = W + \frac{\alpha}{r} BA, \quad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r\ll \min(d,k).$$

Đây chính là công thức của [LoRA (Hu et al., 2021/2022)](https://arxiv.org/abs/2106.09685). Về bản chất vẫn là *học chuyển giao*: đặc trưng tiền huấn luyện được thích nghi cho tác vụ đích, nhưng bị ràng buộc trong một không gian con hạng thấp.

## 1. QLoRA và chuyển giao trên mô hình lượng tử hóa

[Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) giữ nguyên bộ trọng số nền ở định dạng lượng tử hóa 4-bit (NF4) và chỉ huấn luyện các ma trận LoRA ở độ chính xác 16-bit, giúp tinh chỉnh mô hình 65B trên một GPU 48 GB. Sự kết hợp giữa [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) và [Hugging Face PEFT](https://github.com/huggingface/peft) đã biến QLoRA thành lựa chọn phổ biến trong khoa học dữ liệu ứng dụng.

## 2. Một số ứng dụng tiêu biểu

### Adapter theo miền và theo chỉ dẫn

Thay vì lưu một bản sao mô hình hoàn chỉnh cho từng khách hàng hay lĩnh vực, hệ thống chỉ cần triển khai **một mô hình nền kết hợp nhiều adapter LoRA** (pháp lý, y tế, SQL). Các nền tảng phục vụ suy luận như vLLM hay SGLang hỗ trợ hoán đổi adapter linh hoạt theo từng yêu cầu.

### PEFT trong thị giác và đa phương thức

Áp dụng LoRA lên các tầng attention của U-Net hoặc DiT (như DreamBooth-LoRA, SDXL LoRA) cho phép học thêm một phong cách hoặc đối tượng mới chỉ từ khoảng 10 ảnh — sử dụng cùng công thức hạng thấp như LoRA ngôn ngữ nhưng đặt trên các tensor thị giác.

### Khi nào *không* nên dùng LoRA

Với mạng CNN cỡ nhỏ, mô hình dữ liệu bảng huấn luyện từ đầu hoặc trường hợp miền đích lệch mạnh nhưng có nhiều dữ liệu gán nhãn, tinh chỉnh toàn phần hoặc thay mới nhiều tầng vẫn phù hợp hơn — tương đồng với nguyên tắc chọn giữa trích xuất đặc trưng và tinh chỉnh trong bài lý thuyết.

## 3. Thư viện và mã nguồn phổ biến

- [huggingface/peft](https://github.com/huggingface/peft) — hỗ trợ LoRA, AdaLoRA, IA3 và prefix-tuning.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) kết hợp [QLoRA](https://github.com/artidoro/qlora).
- [unsloth](https://github.com/unslothai/unsloth) — thư viện tăng tốc huấn luyện LoRA phổ biến trong giai đoạn 2024–2026.

## 4. Tài liệu tham khảo (2022–2026)

- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685).
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314).
- [Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691) — một hướng tiếp cận khác trong nhóm PEFT.

## 5. Phần bổ sung so với nội dung cốt lõi

Bạn nên hoàn thành phần thực hành tinh chỉnh ResNet/BERT trước. Bài này bổ sung cách tiếp cận chuyển giao **hạng thấp và lượng tử hóa**, thay thế cho cách "mở băng vài tầng cuối" truyền thống khi làm việc với LLM.

## Video tham khảo

- **Xem trước:** [QLoRA: Efficient Finetuning of Quantized LLMs | Tim Dettmers](https://www.youtube.com/watch?v=fQirE9N5q_Y) — AMLD Intelligence Summit, tiếng Anh. Bài thuyết trình của chính tác giả Tim Dettmers về QLoRA, phương pháp tinh chỉnh hạng thấp trên mô hình lượng tử hóa được trình bày trong bài.
- [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, tiếng Anh. Giải thích nền tảng số học độ chính xác thấp mà QLoRA dựa vào.
- [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, tiếng Anh. Ôn lại tinh chỉnh truyền thống để thấy QLoRA thay đổi phương thức tính toán chứ không thay đổi mục tiêu chuyển giao.
