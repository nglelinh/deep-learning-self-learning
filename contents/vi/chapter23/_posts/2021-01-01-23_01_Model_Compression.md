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

# Học sâu Hiệu quả: Nén và Tăng tốc

## 1. Tổng quan khái niệm
Khi mô hình học sâu phát triển đến hàng tỷ tham số, triển khai chúng trên thiết bị hạn chế tài nguyên (điện thoại di động, hệ thống nhúng, thiết bị biên) hoặc phục vụ ở quy mô lớn (hàng triệu truy vấn) trở nên thách thức. Các kỹ thuật nén mô hình (*model compression*) giảm kích thước mô hình, dấu chân bộ nhớ và yêu cầu tính toán trong khi duy trì độ chính xác, cho phép triển khai trong các kịch bản nơi mô hình đầy đủ không thực tế. Các kỹ thuật này — cắt tỉa (*pruning*), lượng tử hóa (*quantization*), chưng cất tri thức (*knowledge distillation*) và kiến trúc hiệu quả — đại diện cho các đổi mới kỹ thuật then chốt giúp học sâu tiếp cận được ngoài máy chủ đám mây với GPU mạnh.

Cắt tỉa loại bỏ tham số hoặc kết nối không cần thiết, khai thác quan sát rằng nhiều trọng số mạng neuron đóng góp tối thiểu vào đầu ra. Nghiên cứu cho thấy mạng vẫn hiệu quả sau khi loại bỏ 50–90% trọng số, gợi ý sự dư thừa đáng kể. Cắt tỉa có cấu trúc (*structured pruning*) loại bỏ toàn bộ bộ lọc hoặc tầng, mang lại tăng tốc thân thiện phần cứng. Cắt tỉa không cấu trúc (*unstructured pruning*) loại bỏ từng trọng số riêng lẻ, đạt nén cao hơn nhưng cần phần cứng chuyên biệt để tăng tốc.

Lượng tử hóa giảm độ chính xác số, biểu diễn trọng số và kích hoạt bằng ít bit hơn (số nguyên 8-bit thay vì float 32-bit), giảm bộ nhớ 4× và cho phép số học nguyên nhanh hơn trên nhiều bộ xử lý. Lượng tử hóa sau huấn luyện (*post-training quantization*) áp dụng cho mô hình đã huấn luyện mà không cần huấn luyện lại. Huấn luyện nhận biết lượng tử hóa (*quantization-aware training*) đưa lượng tử hóa vào vòng lặp huấn luyện, cho phép mạng thích ứng với độ chính xác giảm.

Chưng cất tri thức chuyển tri thức từ mạng “giáo viên” lớn sang mạng “học sinh” nhỏ bằng cách huấn luyện học sinh khớp dự đoán của giáo viên (mục tiêu mềm) chứ không chỉ nhãn cứng. Học sinh học từ sự bất định của giáo viên và độ tương tự giữa các lớp, thường đạt hiệu năng tốt hơn huấn luyện chỉ trên nhãn dù nhỏ hơn nhiều.

Các kiến trúc hiệu quả như MobileNet và EfficientNet được thiết kế vì hiệu quả ngay từ đầu qua tích chập tách theo chiều sâu (*depthwise separable convolution*), tìm kiếm kiến trúc neuron, và mở rộng quy mô cẩn thận. Chúng đạt độ chính xác cạnh tranh với một phần tính toán/tham số so với kiến trúc chuẩn.

## 2. Nền tảng toán học
### Cắt tỉa (Pruning)

Định nghĩa điểm quan trọng cho tham số $$w$$:

$$I(w) = |\frac{\partial \mathcal{L}}{\partial w}|$$ (cắt tỉa theo độ lớn)

hoặc

$$I(w) = |w \cdot \frac{\partial \mathcal{L}}{\partial w}|$$ (xấp xỉ khai triển Taylor)

Loại bỏ trọng số có $$I(w) < \tau$$ (ngưỡng). Sau cắt tỉa, tinh chỉnh các trọng số còn lại.

### Lượng tử hóa (Quantization)

Ánh xạ giá trị liên tục sang mức rời rạc. Với lượng tử hóa 8-bit:

$$w_{\text{quant}} = \text{round}\left(\frac{w - w_{\min}}{w_{\max} - w_{\min}} \cdot 255\right)$$

Giải lượng tử hóa để tính toán:

$$w \approx w_{\min} + w_{\text{quant}} \cdot \frac{w_{\max} - w_{\min}}{255}$$

### Chưng cất Tri thức (Knowledge Distillation)

Mạng học sinh được huấn luyện trên mục tiêu mềm từ giáo viên:

$$\mathcal{L} = \alpha \mathcal{L}_{\text{hard}}(y, \hat{y}_{\text{student}}) + (1-\alpha) \mathcal{L}_{\text{soft}}(\hat{y}_{\text{teacher}}, \hat{y}_{\text{student}})$$

Mục tiêu mềm dùng nhiệt độ $$T$$:

$$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

$$T$$ cao hơn tạo phân bố mềm hơn, tiết lộ sự bất định của giáo viên.

## 3. Ví dụ / Trực giác

Hãy tưởng tượng nén ResNet-50 (25M tham số, 4GB bộ nhớ). Cắt tỉa có thể phát hiện 60% trọng số có độ lớn < 0.001 và đóng góp không đáng kể. Loại bỏ chúng cho 10M tham số, 1.6GB — giảm 60% với mất độ chính xác tối thiểu.

Lượng tử hóa sang INT8 (số nguyên 8-bit) từ FP32 (float 32-bit) giảm bộ nhớ 4×: 10M tham số giờ chỉ 10MB so với 40MB. Kết hợp với cắt tỉa: 2.5GB → 1GB → 250MB — nén tổng 10×.

Chưng cất tri thức chuyển tri thức mô hình nén này sang MobileNet (4M tham số). MobileNet học từ dự đoán mềm của ResNet, đạt 75% độ chính xác so với 70% khi huấn luyện chỉ trên nhãn — 5% thêm đến từ học những gì ResNet nghĩ về các mẫu khó.

Kết quả: mô hình 4M tham số, 40MB INT8 đạt 75% độ chính xác so với 25M tham số, 4GB FP32 gốc đạt 78%. Nhỏ hơn 100×, suy luận nhanh hơn 10×, chỉ mất 3% độ chính xác — cho phép triển khai di động!

## 4. Mã minh họa
```python
import torch
import torch.nn as nn

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
Nén mô hình gắn với tìm kiếm kiến trúc neuron (*neural architecture search*, NAS), tự động khám phá kiến trúc hiệu quả. NAS khám phá không gian kiến trúc, đánh giá ứng viên theo đánh đổi độ chính xác–hiệu quả.

Nén liên quan đến giả thuyết vé số (*lottery ticket hypothesis*): mạng ngẫu nhiên chứa mạng con mà khi được huấn luyện khớp hiệu năng mạng đầy đủ. Tìm các “vé thắng” này cung cấp lựa chọn thay thế cho cắt tỉa sau huấn luyện.

## 6. Các Bài báo Nền tảng

**["Learning both Weights and Connections for Efficient Neural Networks" (2015)](https://arxiv.org/abs/1506.02626)**  
*Tác giả*: Song Han, Jeff Pool, John Tran, William Dally  
Giới thiệu cắt tỉa theo độ lớn đạt nén 9–13× trên AlexNet và VGGNet với mất độ chính xác tối thiểu.

**["Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531)**  
*Tác giả*: Geoffrey Hinton, Oriol Vinyals, Jeff Dean  
Bài báo chưng cất tri thức cho thấy mạng học sinh học tốt hơn từ dự đoán giáo viên so với chỉ từ nhãn.

**["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)](https://arxiv.org/abs/1704.04861)**  
*Tác giả*: Andrew Howard et al.  
MobileNets dùng tích chập tách theo chiều sâu, giảm tính toán 8–9× so với tích chập chuẩn trong khi duy trì độ chính xác.

## Điểm Chính Cần Nhớ

Nén mô hình làm cho học sâu thực tiễn trên thiết bị hạn chế tài nguyên qua cắt tỉa (loại bỏ tham số không cần thiết), lượng tử hóa (giảm độ chính xác số), chưng cất tri thức (chuyển tri thức mô hình lớn sang mô hình nhỏ), và kiến trúc hiệu quả (thiết kế vì hiệu quả ngay từ đầu). Các kỹ thuật này cho phép nén 10–100× với mất độ chính xác tối thiểu, dân chủ hóa triển khai học sâu ngoài máy chủ đám mây sang thiết bị biên, cho phép suy luận thời gian thực, và giảm tác động môi trường qua yêu cầu tính toán thấp hơn.
