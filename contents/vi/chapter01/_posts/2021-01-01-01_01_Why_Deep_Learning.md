---
layout: post
title: 01-01 Vì sao học sâu?
chapter: '01'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter01
---

Nếu các thuật toán kinh điển như Random Forest, XGBoost hay SVM đã hoạt động xuất sắc suốt nhiều thập kỷ, vì sao thế giới lại đổ hàng trăm tỷ đô-la vào GPU và Học sâu? Câu trả lời không nằm ở sự cường điệu truyền thông, mà nằm ở **nút thắt đặc trưng** (*feature bottleneck*) trên dữ liệu phi cấu trúc: ảnh, video, âm thanh và ngôn ngữ tự nhiên. Bài tổng quan này định vị chính xác nơi Học sâu tạo ra bước nhảy vọt về chất, đồng thời giữ cho bạn cái đầu lạnh để chọn đúng công cụ cho đúng bài toán.

## Bản đồ kiến thức nền: Ba trụ cột lý giải "Vì sao Học sâu?"

1. **Giới hạn của Đặc trưng Thủ công (Hand-crafted Features)**:
   Các bộ mô tả cổ điển như SIFT, HOG (ảnh) hay TF-IDF (văn bản) tốn công sức chuyên gia thiết kế, dễ gãy khi đổi ánh sáng, góc chụp hay ngữ cảnh ngôn ngữ, và không thể tinh chỉnh theo hàm mất mát cuối cùng.

2. **Định lý Biểu diễn Sâu vs. Nông (Depth Efficiency)**:
   Nhiều hàm phức tạp đòi hỏi số nơ-ron tăng theo hàm mũ nếu chỉ dùng 1 tầng ẩn, nhưng chỉ cần số nơ-ron tuyến tính khi xếp nhiều tầng sâu tái sử dụng đặc trưng trung gian.

3. **Tiêu chí Chọn Công cụ Thực tế (DL vs. Classical ML)**:
   Dữ liệu bảng (tabular) cỡ nhỏ–trung bình thường là sân nhà của Gradient Boosted Trees (XGBoost, LightGBM); còn dữ liệu phi cấu trúc quy mô lớn, đa phương thức (multimodal) là lãnh địa thống trị tuyệt đối của Học sâu.

### Khung giải thích (nên đọc chậm): So sánh định lượng hai bài toán thực tế để quyết định có dùng Deep Learning hay không

| Tiêu chí đánh giá | Bài toán A: Dự đoán khách hàng rời bỏ (Bảng 20 cột, 8{,}000 dòng) | Bài toán B: Phân loại lỗi sản phẩm từ ảnh camera nhà máy (200{,}000 ảnh $$224\times 224$$) |
| :--- | :--- | :--- |
| **Số chiều đầu vào thô** | $$20$$ đặc trưng đã có nghĩa rõ ràng | $$224 \times 224 \times 3 = 150{,}528$$ điểm ảnh thô |
| **Khả năng kỹ sư tự tạo đặc trưng** | Cao (tỷ lệ dùng dịch vụ, số lần gọi CSKH) | Rất thấp (vết xước, vết nứt đổi góc sáng liên tục) |
| **Lựa chọn tối ưu** | **XGBoost / LightGBM** (nhanh, chính xác, dễ giải thích SHAP) | **Deep Learning (CNN / ViT pretrained)** |

Kỹ sư giỏi không phải người đem mạng nơ-ron 100 triệu tham số đi giải mọi bài toán, mà là người biết nhìn vào cấu trúc dữ liệu và số chiều để chọn vũ khí sắc nhất.

Mục này giải thích lý do học sâu đạt hiệu quả cao và khi nào nên lựa chọn phương pháp này:

1. **Hạn chế của học máy truyền thống và điều làm học sâu khác biệt** — phân tích giới hạn của đặc trưng thủ công, biểu diễn cố định và lợi thế học biểu diễn phân cấp.
2. **Khi nào dùng học sâu và hệ sinh thái** — các bài toán tiêu biểu, vai trò của dữ liệu và tính toán, quy trình triển khai, công cụ, tài liệu và xu hướng hiện nay.

## Video tham khảo

- **Xem trước:** [Machine Learning vs Deep Learning](https://www.youtube.com/watch?v=q6kJ71tEYqM) — IBM Technology, tiếng Anh. So sánh những điểm học sâu bổ sung so với học máy cổ điển, trả lời trực tiếp câu hỏi trọng tâm của phần này.
- [MIT Introduction to Deep Learning (2024) | 6.S191](https://www.youtube.com/watch?v=ErnWZxJovaM) — Alexander Amini, tiếng Anh. Phần mở đầu trình bày các yếu tố về quy mô dữ liệu và tính toán giúp học sâu phát triển, làm rõ bối cảnh của bài học.
- [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, tiếng Anh. Minh họa cách biểu diễn đặc trưng phân cấp giúp mạng sâu xử lý ảnh hiệu quả.
