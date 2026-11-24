---
layout: post
title: 23-01 Nén Mô hình và Hiệu quả
chapter: '23'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter23
---

Một phát hiện gây chấn động từ bài báo *The Lottery Ticket Hypothesis* cho thấy bên trong một mạng nơ-ron dày đặc hàng triệu tham số thường ẩn giấu một "mạng con trúng số" chỉ chiếm **10–20% số trọng số** nhưng đạt độ chính xác ngang ngửa mạng đầy đủ! Bài học này đi sâu vào cơ sở toán học của ba trụ cột nén mô hình kinh điển: **Cắt tỉa (Pruning)**, **Lượng tử hóa đồng đều (Affine/Symmetric Quantization)** và **Chưng cất tri thức với nhiệt độ $$T$$**.

## Bản đồ kiến thức nền: Cơ sở toán học của ba kỹ thuật nén mô hình cốt lõi

- **Cắt tỉa theo độ lớn và cấu trúc (Magnitude & Structured Pruning):** Xếp hạng trọng số theo $$|w|$$ hoặc chuẩn $$L_1/L_2$$ của bộ lọc, cắt bỏ tỷ lệ $$p$$ rồi tinh chỉnh lại; hiểu sự đánh đổi giữa độ thưa và hiệu suất thực trên nhân CUDA.
- **Lượng tử hóa tuyến tính (Symmetric & Asymmetric Quantization):** Ánh xạ số thực $$r \in [r_{\min}, r_{\max}]$$ sang số nguyên $$q = \text{clip}(\text{round}(r/s) + z,\; q_{\min}, q_{\max})$$ cùng quá trình giải lượng tử hóa $$\hat{r} = s(q - z)$$, phân biệt PTQ và QAT (dùng Straight-Through Estimator).
- **Chưng cất tri thức (Hinton Knowledge Distillation):** Làm mềm phân phối Softmax bằng nhiệt độ $$T > 1$$ để lộ ra "tri thức ẩn" (*dark knowledge*) về độ tương đồng giữa các lớp sai, kết hợp mất mát KL (nhân $$T^2$$) với Cross-Entropy nhãn cứng.
- **Tích chập tách biệt chiều sâu (Depthwise Separable Convolution):** Phân rã phép tích chập $$K \times K \times C_{\text{in}} \times C_{\text{out}}$$ thành bước lọc từng kênh rồi trộn $$1 \times 1$$ để giảm $$8\text{–}9\times$$ số phép tính FLOPs.

### Khung giải thích (nên đọc chậm): Tính tay một bước lượng tử hóa đối xứng INT8 và sai số lượng tử hóa

Xét một vector trọng số thực $$\mathbf{w} = [-0{,}82,\; 0{,}15,\; 1{,}27,\; -0{,}04]$$ cần lượng tử hóa đối xứng sang số nguyên có dấu INT8 trong đoạn $$[-127, 127]$$.

1. **Tính hệ số tỷ lệ (Scale Factor $$s$$):**
   $$s = \frac{\max_i |w_i|}{127} = \frac{1{,}27}{127} = 0{,}01.$$
2. **Lượng tử hóa sang số nguyên ($$q_i = \text{round}(w_i / s)$$):**
   - $$-0{,}82 / 0{,}01 = -82$$ → $$q_1 = -82$$
   - $$0{,}15 / 0{,}01 = 15$$ → $$q_2 = 15$$
   - $$1{,}27 / 0{,}01 = 127$$ → $$q_3 = 127$$
   - $$-0{,}04 / 0{,}01 = -4$$ → $$q_4 = -4$$
3. **Giải lượng tử hóa và đo sai số:** Nếu phần tử thứ nhất là $$-0{,}824$$ thì $$\text{round}(-82{,}4) = -82$$, giải lượng tử hóa ra $$-82 \times 0{,}01 = -0{,}820$$ — sai số tối đa luôn bị chặn bởi $$s/2 = 0{,}005$$! Tuy nhiên, nếu xuất hiện một **giá trị ngoại lai (outlier)** bằng $$12{,}7$$ trong cùng nhóm, $$s$$ nhảy vọt lên $$0{,}10$$ làm toàn bộ trọng số nhỏ bị làm tròn thô gấp 10 lần — đó chính là lý do các thuật toán lượng tử hóa LLM hiện đại phải xử lý riêng các kênh ngoại lai!

# Học sâu Hiệu quả: Nén và Tăng tốc Mô hình

## 1. Tổng quan khái niệm

Khi các mô hình học sâu mở rộng đến hàng tỷ tham số, việc triển khai chúng trên thiết bị hạn chế tài nguyên (điện thoại di động, hệ thống nhúng, thiết bị biên) hoặc phục vụ hàng triệu truy vấn cùng lúc trở nên khó khăn về chi phí và độ trễ. Các kỹ thuật nén mô hình (*model compression*) giúp giảm kích thước lưu trữ, mức tiêu thụ bộ nhớ và khối lượng tính toán mà vẫn giữ độ chính xác ở mức chấp nhận được. Bốn nhóm kỹ thuật chính gồm: cắt tỉa (*pruning*), lượng tử hóa (*quantization*), chưng cất tri thức (*knowledge distillation*) và thiết kế kiến trúc hiệu quả.

Cắt tỉa loại bỏ các tham số hoặc kết nối ít đóng góp, dựa trên nhận xét rằng mạng nơ-ron thường có mức độ dư thừa lớn và vẫn hoạt động tốt sau khi lược bỏ 50–90% trọng số. Cắt tỉa có cấu trúc (*structured pruning*) loại bỏ nguyên cả bộ lọc hoặc tầng, dễ mang lại tốc độ thực thi nhanh hơn trên phần cứng thông thường. Cắt tỉa không cấu trúc (*unstructured pruning*) xóa từng trọng số riêng lẻ, đạt tỷ lệ nén cao hơn nhưng thường cần thư viện hoặc phần cứng hỗ trợ ma trận thưa để tăng tốc thực tế.

Lượng tử hóa giảm độ chính xác biểu diễn số học của trọng số và giá trị kích hoạt (chẳng hạn từ số thực 32-bit sang số nguyên 8-bit), giúp giảm 4 lần dung lượng bộ nhớ và tận dụng phép tính số nguyên nhanh hơn trên nhiều bộ xử lý. Lượng tử hóa sau huấn luyện (*post-training quantization*, PTQ) áp dụng trực tiếp lên mô hình đã huấn luyện mà không cần huấn luyện lại. Huấn luyện nhận biết lượng tử hóa (*quantization-aware training*, QAT) mô phỏng quá trình lượng tử hóa ngay trong vòng lặp huấn luyện để mô hình thích nghi với sai số làm tròn.

Chưng cất tri thức chuyển giao năng lực từ một mạng “giáo viên” lớn sang một mạng “học sinh” nhỏ hơn bằng cách huấn luyện học sinh khớp với phân bố dự đoán mềm (*soft targets*) của giáo viên bên cạnh nhãn gốc. Nhờ học thêm thông tin về độ bất định và sự tương đồng giữa các lớp từ giáo viên, mạng học sinh thường đạt kết quả tốt hơn so với khi chỉ huấn luyện trực tiếp trên nhãn cứng. Với **chưng cất trong kỷ nguyên LLM** (hạn chế khi khớp logit toàn từ vựng, chưng cất qua dữ liệu tổng hợp và phạm vi sử dụng hợp lệ), xem **Chương 26**, bài **26-05**; bài này tập trung vào thiết lập cổ điển với không gian đầu ra cố định.

Song song với việc nén mô hình có sẵn, các kiến trúc hiệu quả như MobileNet và EfficientNet được thiết kế tối ưu ngay từ đầu nhờ tích chập tách theo chiều sâu (*depthwise separable convolution*), tìm kiếm kiến trúc nơ-ron (NAS) và quy tắc mở rộng kích thước cân bằng, đạt độ chính xác cạnh tranh với chi phí tính toán thấp hơn nhiều so với mạng chuẩn.

## 2. Nền tảng toán học

### Cắt tỉa (Pruning)

Mức độ quan trọng của tham số $$w$$ thường được ước lượng theo một trong hai cách:

$$I(w) = |w|$$ (cắt tỉa theo độ lớn trọng số)

hoặc

$$I(w) = \left|w \cdot \frac{\partial \mathcal{L}}{\partial w}\right|$$ (xấp xỉ khai triển Taylor bậc nhất)

Các trọng số có $$I(w) < \tau$$ (với $$\tau$$ là ngưỡng chọn trước) bị loại bỏ, sau đó mô hình được tinh chỉnh lại trên các trọng số còn giữ.

### Lượng tử hóa (Quantization)

Lượng tử hóa ánh xạ giá trị liên tục sang tập mức rời rạc. Với lượng tử hóa tuyến tính 8-bit không dấu:

$$w_{\text{quant}} = \text{round}\left(\frac{w - w_{\min}}{w_{\max} - w_{\min}} \cdot 255\right)$$

Khi cần khôi phục giá trị xấp xỉ (giải lượng tử hóa) để tính toán:

$$w \approx w_{\min} + w_{\text{quant}} \cdot \frac{w_{\max} - w_{\min}}{255}$$

### Chưng cất tri thức (Knowledge Distillation)

Mạng học sinh tối ưu tổng có trọng số giữa hàm mất mát trên nhãn cứng và hàm mất mát trên phân bố mềm từ giáo viên:

$$\mathcal{L} = \alpha \mathcal{L}_{\text{hard}}(y, \hat{y}_{\text{student}}) + (1-\alpha) \mathcal{L}_{\text{soft}}(\hat{y}_{\text{teacher}}, \hat{y}_{\text{student}})$$

Phân bố mềm sử dụng tham số nhiệt độ $$T$$:

$$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Giá trị $$T$$ lớn hơn làm phẳng phân bố xác suất, giúp bộc lộ rõ hơn quan hệ tương đồng giữa các lớp mà giáo viên học được.

## 3. Ví dụ và trực giác

Xét ví dụ nén mô hình ResNet-50 (khoảng 25 triệu tham số, tương đương khoảng 100 MB ở định dạng FP32 — 4 byte mỗi tham số). Giả sử bước cắt tỉa loại bỏ 60% trọng số có độ lớn nhỏ hơn 0.001 do đóng góp không đáng kể, đưa mô hình xuống còn 10 triệu tham số (40 MB ở FP32) với mức suy giảm độ chính xác rất nhỏ.

Tiếp theo, lượng tử hóa từ FP32 sang INT8 (1 byte mỗi tham số) giảm bộ nhớ thêm 4 lần: 10 triệu tham số chỉ còn khoảng 10 MB thay vì 40 MB. Kết hợp cắt tỉa và lượng tử hóa giúp giảm dung lượng lưu trữ trọng số từ 100 MB xuống 10 MB — tỷ lệ nén tổng cộng 10 lần.

Cuối cùng, chưng cất tri thức sang một kiến trúc gọn hơn như MobileNet (4 triệu tham số) giúp mạng học sinh học từ dự đoán mềm của ResNet, đạt 75% độ chính xác thay vì 70% khi chỉ huấn luyện trên nhãn cứng — phần chênh lệch 5% đến từ thông tin phân bố của giáo viên trên các mẫu khó.

Tổng kết lại, mô hình học sinh 4 triệu tham số ở định dạng INT8 chỉ chiếm khoảng 4 MB (nhỏ hơn 25 lần so với 100 MB FP32 ban đầu), suy luận nhanh hơn nhiều lần và chỉ giảm 3% độ chính xác so với mức 78% của ResNet-50 gốc, đủ gọn để triển khai trực tiếp trên thiết bị di động.

## 4. Mã minh họa

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pruning example
def magnitude_prune(model, sparsity=0.5):
    """Remove smallest magnitude weights"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight.data.abs()
            threshold = weight.quantile(sparsity)
            mask = weight > threshold
            module.weight.data *= mask.float()

# Quantization
def quantize_tensor(tensor, num_bits=8):
    """Quantize tensor to num_bits"""
    qmin = 0
    qmax = 2**num_bits - 1
    
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    
    q = torch.round((tensor - min_val) / scale).clamp(qmin, qmax)
    
    return q, scale, min_val

# Knowledge distillation
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    """Combine hard and soft targets"""
    hard_loss = F.cross_entropy(student_logits, labels)
    
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * T * T
    
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

## 5. Khái niệm liên quan

Nén mô hình gắn liền với tìm kiếm kiến trúc nơ-ron (*neural architecture search*, NAS), phương pháp tự động tìm kiếm kiến trúc cân bằng tốt giữa độ chính xác và chi phí tính toán.

Ngoài ra, cắt tỉa còn liên hệ với *giả thuyết vé số* (*lottery ticket hypothesis*): một mạng khởi tạo ngẫu nhiên thường chứa sẵn những mạng con mà nếu được huấn luyện lại từ đúng trọng số khởi tạo ban đầu sẽ đạt hiệu năng tương đương mạng đầy đủ.

## 6. Các bài báo nền tảng

**["Learning both Weights and Connections for Efficient Neural Networks" (2015)](https://arxiv.org/abs/1506.02626)**  
*Tác giả*: Song Han, Jeff Pool, John Tran, William Dally  
Chứng minh cắt tỉa theo độ lớn trọng số có thể nén AlexNet và VGGNet từ 9 đến 13 lần mà hầu như không giảm độ chính xác.

**["Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531)**  
*Tác giả*: Geoffrey Hinton, Oriol Vinyals, Jeff Dean  
Đặt nền tảng cho chưng cất tri thức, cho thấy mạng học sinh học hiệu quả hơn khi dùng phân bố mềm của giáo viên thay vì chỉ dùng nhãn cứng.

**["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)](https://arxiv.org/abs/1704.04861)**  
*Tác giả*: Andrew Howard et al.  
Sử dụng tích chập tách theo chiều sâu để giảm khối lượng tính toán 8–9 lần so với tích chập tiêu chuẩn trong khi vẫn duy trì độ chính xác tốt cho các ứng dụng thị giác trên thiết bị di động.

## Điểm chính cần nhớ

Nén mô hình giúp triển khai học sâu hiệu quả trên thiết bị hạn chế tài nguyên thông qua bốn hướng chính: cắt tỉa (loại bỏ trọng số dư thừa), lượng tử hóa (giảm số bit biểu diễn), chưng cất tri thức (chuyển giao năng lực từ mô hình lớn sang mô hình nhỏ) và thiết kế kiến trúc gọn nhẹ ngay từ đầu. Kết hợp hợp lý các kỹ thuật này có thể giảm mạnh dung lượng bộ nhớ và chi phí suy luận với mức suy giảm độ chính xác nhỏ, hỗ trợ suy luận thời gian thực trên thiết bị biên và tiết kiệm năng lượng tính toán.

## Video tham khảo

- **Xem trước** [Lecture 15 | Efficient Methods and Hardware for Deep Learning](https://www.youtube.com/watch?v=eZdOkDtYMoo) — Stanford University School of Engineering, tiếng Anh. Trình bày tổng hợp cắt tỉa, lượng tử hóa và các ràng buộc phần cứng.
- [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, tiếng Anh. Giải thích cơ chế biểu diễn trọng số ở độ chính xác thấp, bổ sung trực quan cho phần lượng tử hóa.
- [2.3 Knowledge Distillation | Geoffery Hinton | Soft Labels | Softmax with Temperature](https://www.youtube.com/watch?v=g1r4Esohw-g) — AI_Snippets, tiếng Anh. Minh họa nhãn mềm và tham số nhiệt độ của Hinton — nội dung chưng cất cổ điển trước khi chuyển sang Chương 26.
