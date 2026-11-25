---
layout: post
title: 16-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '16'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter16
lesson_type: optional
---

Nếu năm 2020 SimCLR vẫn còn là câu chuyện nghiên cứu cần batch size 4.096 và phải tinh chỉnh lại cho từng tác vụ, thì giai đoạn 2023–2026 đã chứng kiến bước trưởng thành vượt bậc: **DINOv2** cho ra đặc trưng thị giác đóng băng dùng thẳng cho ước lượng độ sâu lẫn phân đoạn, **SigLIP** thay thế Softmax toàn cục bằng hàm sigmoid song song cực gọn, còn **I-JEPA** của Yann LeCun dự đoán trực tiếp trong không gian biểu diễn thay vì vẽ lại từng điểm ảnh.

## Kiến thức cần nhớ: Ba hướng đột phá của Học Tự giám sát hiện đại (2022–2026)

- **DINOv2 và xương sống thị giác phổ dụng:** Kết hợp chưng cất tự thân cấp ảnh + cấp patch (iBOT) cùng quy trình lọc dữ liệu tự động LVD-142M để tạo ra bộ đặc trưng đóng băng xuất sắc cho cả phân loại, phân đoạn lẫn ước lượng độ sâu 3D.
- **SigLIP / SigLIP 2 (Sigmoid Contrastive Loss):** Thay bài toán Softmax phụ thuộc toàn batch của CLIP bằng phân loại nhị phân độc lập trên từng cặp ảnh–văn bản, loại bỏ nút cổ chai đồng bộ hóa GPU và mở rộng dễ dàng ở mọi batch size.
- **I-JEPA / V-JEPA (Joint-Embedding Predictive Architecture):** Dự đoán biểu diễn ẩn của vùng bị che thay vì tái tạo pixel thô, loại bỏ chi tiết nhiễu không liên quan ngữ nghĩa (như vị trí từng chiếc lá rung trong gió).
- **Hội tụ với hệ sinh thái đa phương thức:** Các bộ mã hóa SSL hiện đại (SigLIP, DINOv2) chính là "đôi mắt" mặc định được cắm trực tiếp vào các mô hình ngôn ngữ thị giác (VLM) thế hệ mới.

### Giải thích: So sánh tính toán của Softmax CLIP vs. Sigmoid SigLIP trên một batch $$B = 4$$ cặp ảnh–văn bản

Xét ma trận độ tương đồng $$4 \times 4$$ giữa 4 ảnh và 4 chú thích văn bản (đường chéo chính là 4 cặp dương $$y_{ij} = +1$$, 12 ô ngoài đường chéo là cặp âm $$y_{ij} = -1$$). Giả sử ô dương $$(1,1)$$ có điểm $$z_{11} = +3{,}0$$ (sau hệ số thang $$t$$ và bias $$b$$) và một ô âm $$(1,2)$$ có $$z_{12} = -2{,}0$$.

1. **Với CLIP (Softmax Cross-Entropy):** Muốn tính xác suất cho ô $$(1,1)$$, GPU phải tập hợp (all-gather) điểm tương đồng với **toàn bộ** $$B$$ văn bản trên mọi GPU để tính mẫu số $$\sum_{j=1}^B e^{z_{1j}}$$.
2. **Với SigLIP (Pairwise Sigmoid Loss):** Mỗi ô $$(i, j)$$ tính độc lập bằng công thức $$\ell_{ij} = \ln(1 + e^{-y_{ij} z_{ij}})$$:
   - Ô dương $$(1,1)$$ ($$y = +1, z = +3{,}0$$): $$\ell_{11} = \ln(1 + e^{-3{,}0}) \approx \ln(1{,}0498) \approx 0{,}0486$$.
   - Ô âm $$(1,2)$$ ($$y = -1, z = -2{,}0$$): $$\ell_{12} = \ln(1 + e^{-(-1)(-2{,}0)}) = \ln(1 + e^{-2{,}0}) \approx \ln(1{,}1353) \approx 0{,}1269$$.
3. **Ưu thế hệ thống:** Vì không cần mẫu số chuẩn hóa toàn cục, SigLIP cho phép tính chia khối (chunked) giữa các GPU với bộ nhớ cực nhỏ và hoạt động ổn định ngay cả ở batch size nhỏ lẫn cực lớn (hàng chục nghìn cặp).

# Tùy chọn: Học tự giám sát sau SimCLR — DINOv2, SigLIP và JEPA

> Bài này là **tùy chọn**, **không** thay thế phần lý thuyết học đối chiếu / MLM hay bản phác thảo SimCLR trong chương. Nội dung cập nhật các phương pháp học tự giám sát (SSL) cho **thị giác và đa phương thức** được dùng rộng rãi từ sau năm 2022.

Các bài trước trong chương đã giới thiệu SimCLR, CLIP và MAE. Trong giai đoạn 2023–2025, các hệ thống thực tế chuyển dần sang **mô hình thầy–trò mạnh hơn**, **hàm mất mát sigmoid cho cặp ảnh–văn bản**, và mục tiêu **dự đoán trong không gian biểu diễn ẩn** (JEPA).

## 1. DINOv2: Mạng xương sống đóng băng dùng trực tiếp

[Oquab et al., 2023](https://arxiv.org/abs/2304.07193) kết hợp cơ chế thầy–trò (*student–teacher*) kiểu DINO/iBOT với quy trình chọn lọc dữ liệu quy mô lớn. Đặc trưng từ ViT-g có thể dùng trực tiếp cho truy xuất ảnh, ước lượng độ sâu và phân đoạn mà **không** cần tinh chỉnh theo từng tác vụ trong nhiều bài toán thử nghiệm. Cách tiếp cận này xem học tự giám sát như phương pháp xây dựng **bộ mã hóa nền tảng**, thay vì chỉ dừng ở một tác vụ tiền huấn luyện đơn lẻ.

## 2. Ứng dụng cụ thể

### SigLIP và SigLIP 2

[Zhai et al., 2023](https://arxiv.org/abs/2303.15343) thay hàm mất mát softmax của CLIP (vốn đòi hỏi batch mẫu âm lớn) bằng hàm mất mát **sigmoid** tính trên từng cặp:

$$\mathcal{L} = -\frac{1}{B}\sum_{i,j}\log\sigma\big(y_{ij}\, t\, x_i^\top y_j\big),$$

trong đó $$y_{ij}=\pm 1$$ cho biết cặp $$(i, j)$$ có khớp nhau hay không. Nhiều mô hình thị giác–ngôn ngữ mở (như PaliGemma và các VLM giai đoạn 2024–2025) sử dụng nhánh mã hóa thị giác SigLIP làm thành phần khởi đầu.

### I-JEPA và dự đoán trong không gian biểu diễn

[Assran et al., 2023](https://arxiv.org/abs/2301.08243) (I-JEPA) dự đoán **vector nhúng** (*embedding*) của các mảng ảnh (*patch*) bị che thay vì tái tạo trực tiếp giá trị pixel như MAE (đã nhắc ở bài tùy chọn Chương 12). Hướng nghiên cứu JEPA của Yann LeCun đặt giả thuyết rằng học tự giám sát nên mô hình hóa cấu trúc và biến đổi trong không gian ẩn thay vì giải mã chi tiết từng pixel RGB.

### Những nền tảng vẫn giữ nguyên giá trị

MLM và cơ chế ghép cặp ảnh–văn bản kiểu CLIP vẫn là nền tảng trong NLP và mô hình đa phương thức, còn SimCLR vẫn là bước dẫn nhập cơ bản để hiểu cơ chế học đối chiếu.

## 3. Phần mềm phổ biến

- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
- Trọng số SigLIP trong OpenCLIP trên Hugging Face Hub.
- [facebookresearch/jepa](https://github.com/facebookresearch/jepa).

## 4. Trích dẫn (2022–2026)

- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193).
- [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343).
- [I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Phần suy diễn NT-Xent và công thức MLM của BERT trong các bài cốt lõi vẫn giữ nguyên giá trị. Bài này bổ sung các **mô hình tiền huấn luyện giai đoạn 2023–2025** (DINOv2, SigLIP, JEPA) thường được tải về sử dụng trực tiếp thay vì huấn luyện SimCLR từ đầu.

## Video tham khảo

- **Xem trước** [DINOv2 from Meta AI - Finally a Foundational Model in Computer Vision?](https://www.youtube.com/watch?v=csEgtSh7jV4) — AI Papers Academy, tiếng Anh. Giới thiệu DINOv2, mạng xương sống thị giác đóng băng có thể dùng cho nhiều tác vụ mà không cần tinh chỉnh toàn bộ.
- [DINO: Emerging Properties in Self-Supervised Vision Transformers (Facebook AI Research Explained)](https://www.youtube.com/watch?v=h3ij3F3cPIk) — Yannic Kilcher, tiếng Anh. Trình bày DINO, nền tảng tiền thân mà DINOv2 mở rộng.
- [Contrastive Learning with SimCLR | Deep Learning Animated](https://www.youtube.com/watch?v=UqJauYELn6c) — Deepia, tiếng Anh. Minh họa mục tiêu học đối chiếu, tiền đề của nhiều mô hình thị giác tự giám sát hiện đại.
