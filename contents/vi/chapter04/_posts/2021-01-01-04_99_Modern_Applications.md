---
layout: post
title: 04-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '04'
order: 14
owner: Deep Learning Course
lang: vi
categories:
- chapter04
lesson_type: optional
---

Khi Vision Transformer (ViT) xuất hiện năm 2020, từng có dự đoán rằng mạng tích chập (CNN) sẽ sớm trở thành "đồ cổ" trong thị giác máy tính. Thế nhưng thực tế giai đoạn 2022–2026 lại kể một câu chuyện thú vị hơn nhiều: khi được cập nhật đúng quy trình huấn luyện hiện đại và thiết kế khối tối ưu như **ConvNeXt**, CNN không những bắt kịp ViT về độ chính xác mà còn tiếp tục thống trị thị giác thời gian thực, ảnh y khoa và thiết bị biên (edge AI).

## Kiến thức cần nhớ: Bốn xu hướng hiện đại của CNN (2022–2026)

1. **ConvNeXt — Hiện đại hóa ResNet bằng công thức của Transformer**:
   Áp dụng nhân depthwise lớn $$7 \times 7$$, khối cổ chai đảo ngược (inverted bottleneck), LayerNorm thay cho BatchNorm, GELU thay cho ReLU và công thức huấn luyện AdamW + Mixup + Stochastic Depth.
2. **Kiến trúc lai (Hybrid CNN–Transformer)**:
   Kết hợp tầng tích chập ở các giai đoạn đầu (khai thác thiên kiến cục bộ và hạ độ phân giải nhanh) với tầng Self-Attention ở giai đoạn sâu (mô hình hóa ngữ cảnh toàn cục), như trong CoAtNet hay FasterViT.
3. **Vai trò cốt lõi trong Mô hình Nền tảng và Phát hiện Vật thể**:
   Dù SAM hay DINOv2 dùng encoder ViT, các bộ phát hiện thời gian thực (họ YOLO v8–v11) và decoder phân đoạn ảnh y khoa (U-Net hiện đại) vẫn đặt nền móng trên tích chập.
4. **Đánh đổi Độ phức tạp: $$O(N)$$ của CNN vs. $$O(N^2)$$ của Attention**:
   Với ảnh độ phân giải cực cao (ảnh vệ tinh, ảnh giải phẫu bệnh học), chi phí bậc hai theo số patch của Self-Attention tăng vọt, khiến CNN tuyến tính theo số pixel vẫn giữ ưu thế áp đảo về độ trễ và năng lượng.

### Giải thích: Tính tay sự chênh lệch chi phí giữa Depthwise Conv $$7 \times 7$$ và Self-Attention trên ảnh độ phân giải cao

Xét một bản đồ đặc trưng kích thước $$56 \times 56$$ (tổng cộng $$N = 56 \times 56 = 3{,}136$$ vị trí không gian) với số kênh $$C = 96$$:

1. **Self-Attention toàn cục** (chi phí tương tác giữa mọi cặp vị trí $$N \times N \times C$$):
   Số phép nhân-cộng riêng cho tích ma trận $$Q K^T$$ và $$A V$$ là:
   $$2 \times N^2 \times C = 2 \times (3{,}136)^2 \times 96 \approx \mathbf{1{,}888 \text{ triệu MACs}}$$
2. **Depthwise Convolution $$7 \times 7$$ trong ConvNeXt** (mỗi vị trí chỉ nhân với $$7 \times 7 = 49$$ láng giềng trên từng kênh):
   $$N \times k^2 \times C = 3{,}136 \times 49 \times 96 \approx \mathbf{14{,}75 \text{ triệu MACs}}$$

Ở độ phân giải $$56 \times 56$$, phép lọc không gian của ConvNeXt rẻ hơn Self-Attention toàn cục tới **128 lần** — đó là lý do cốt lõi khiến CNN hiện đại vẫn là lựa chọn hàng đầu cho suy luận độ trễ thấp trên điện thoại, camera và NPU biên.

# Tùy chọn: ConvNet sau ConvNeXt và Mô hình Nền tảng Thị giác

> Bài này là **tùy chọn** và **không** trình bày lại toán tích chập, tầng gộp hay lịch sử LeNet–ResNet. Nội dung tập trung cập nhật những kiến trúc và cách triển khai phổ biến trong thị giác máy tính từ sau năm 2022.

Phép tính cốt lõi của tích chập $$(\mathbf{x} * \mathbf{k})_{i} = \sum_{u} x_{i+u}\,k_{u}$$ không thay đổi. Điều thay đổi nằm ở **thiết kế kiến trúc tổng thể** (như ConvNeXt) và **cách sử dụng** (dùng các mạng xương sống tích chập hoặc lai, thường được đóng băng trọng số, trong các mô hình nền tảng thị giác).

## 1. ConvNeXt: Hiện đại hóa ConvNet bằng Quy trình Huấn luyện Transformer

[Liu et al., 2022](https://arxiv.org/abs/2201.03545) (*"A ConvNet for the 2020s"*) cho thấy một kiến trúc thuần tích chập xuất phát từ ResNet, khi được hiện đại hóa thiết kế khối và huấn luyện bằng quy trình của ViT (AdamW, Mixup, stochastic depth, kích thước nhân lớn hơn), có thể đạt hiệu năng ngang bằng các Transformer phân cấp trên ImageNet. Tiếp đó, [ConvNeXt V2](https://arxiv.org/abs/2301.00808) (2023) bổ sung lớp chuẩn hóa phản hồi toàn cục (*Global Response Normalization* — GRN) cùng giai đoạn tiền huấn luyện autoencoder che khuất (*masked autoencoder*) thuần tích chập.

Thông điệp chính cho chương này: **tích chập tách theo chiều sâu (*depthwise convolution*) kết hợp cổ chai đảo ngược (*inverted bottleneck*)** vẫn là một thiên kiến quy nạp hiệu quả; khoảng cách từng thấy giữa CNN và ViT đầu thập niên 2020 chủ yếu đến từ công thức huấn luyện và quy mô dữ liệu, chứ không có nghĩa tích chập đã lỗi thời.

## 2. Ứng dụng Thực tế

### Mô hình nền tảng thị giác vẫn giữ thành phần tích chập

- **DINOv2** ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)) — học đặc trưng tự giám sát với ViT làm xương sống cho truy xuất ảnh, ước lượng độ sâu và phân đoạn; các biến thể ConvNeXt cũng thường được so sánh trực tiếp trên cùng bộ chuẩn.
- **SAM** ([Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)) và **SAM 2** ([Ravi et al., 2024](https://arxiv.org/abs/2408.00714)) — mô hình phân đoạn theo gợi ý (*prompt*) dùng encoder ảnh kiểu ViT, nhưng các hệ thống triển khai thực tế vẫn thường ghép kèm bộ phát hiện đối tượng tích chập và phần cổ (*neck*) CNN gọn nhẹ.
- Các bài toán ảnh y khoa và viễn thám vẫn ưu tiên **decoder tích chập** kiểu U-Net đặt trên các encoder hiện đại.

### Khi nào CNN vẫn là lựa chọn phù hợp

Các bài toán dự đoán dày đặc ở độ phân giải cao, yêu cầu độ trễ thấp trên NPU thiết bị di động hoặc tập dữ liệu quy mô nhỏ thường tiếp tục sử dụng EfficientNetV2, ConvNeXt-Tiny hay phần cổ (*neck*) của họ YOLO. Trong khi đó, Transformer phát huy thế mạnh rõ rệt khi có thể tiền huấn luyện ở quy mô dữ liệu rất lớn.

## 3. Thư viện và Phần mềm Phổ biến

- [timm](https://github.com/huggingface/pytorch-image-models) — cung cấp ConvNeXt, EfficientNetV2 cùng nhiều mô hình tiền huấn luyện.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — bộ phát hiện đối tượng dựa trên tích chập (YOLOv8/v11) phổ biến trong công nghiệp.
- [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) và [facebookresearch/sam2](https://github.com/facebookresearch/sam2).

## 4. Trích dẫn (2022–2026)

- [A ConvNet for the 2020s (Liu et al., 2022)](https://arxiv.org/abs/2201.03545) — giới thiệu ConvNeXt.
- [ConvNeXt V2 (Woo et al., 2023)](https://arxiv.org/abs/2301.00808) — bổ sung GRN và tiền huấn luyện FCMAE.
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) — đặc trưng thị giác tự giám sát quy mô lớn.
- [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643) — mô hình nền tảng cho bài toán phân đoạn theo gợi ý.

## 5. Mối liên hệ với các Bài học Cốt lõi

Các bài tập tính kích thước và phép tính tích chập ở phần cốt lõi vẫn giữ nguyên giá trị. Bài này chỉ bổ sung góc nhìn thực tiễn về các kiến trúc thay thế VGG/ResNet sau năm 2022 — gồm ConvNet hiện đại và mô hình nền tảng thị giác — chứ không thay đổi định nghĩa của phép lọc tích chập.

## Video tham khảo

- **Xem trước:** [ConvNeXt: A ConvNet for the 2020s – Paper Explained (with animations)](https://www.youtube.com/watch?v=QqejV0LNDHA) — AI Coffee Break with Letitia, tiếng Anh. Giải thích chi tiết cách ConvNeXt kế thừa công thức huấn luyện và thiết kế từ Transformer.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, tiếng Anh. Tổng quan kiến trúc CNN hiện đại làm nền cho các cải tiến trong bài.
- [Residual Networks (ResNet) \[Physics Informed Machine Learning\]](https://www.youtube.com/watch?v=w1UsKanMatM) — Steve Brunton, tiếng Anh. Ôn lại cấu trúc khối dư mà ConvNeXt vẫn tiếp tục giữ lại.
