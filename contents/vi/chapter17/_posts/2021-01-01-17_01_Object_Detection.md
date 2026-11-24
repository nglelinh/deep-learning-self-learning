---
layout: post
title: 17-01 Cơ sở Phát hiện Đối tượng
chapter: '17'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter17
---

Chọn YOLO tốc độ 120 khung hình/giây hay Faster R-CNN độ chính xác cao cho vật thể nhỏ? Câu trả lời không bao giờ là "cái nào mới hơn thì tốt hơn", mà phụ thuộc vào ngân sách độ trễ, kích thước vật thể và phần cứng triển khai của bạn. Mục tổng quan này kết nối trực tiếp lý thuyết hình học (**17-01-01**) với cài đặt PyTorch (**17-01-02**) để bạn xây dựng trực giác thiết kế hệ thống phát hiện đối tượng.

## Bản đồ kiến thức nền: Hệ quy chiếu chọn kiến trúc Phát hiện Đối tượng

- **Hai giai đoạn (Faster R-CNN + FPN):** Mạng đề xuất vùng (RPN) lọc ứng viên trước rồi trích xuất đặc trưng `RoIAlign` trên từng hộp — mạnh về vật thể nhỏ, cảnh dày đặc, nhưng độ trễ cao hơn.
- **Một giai đoạn (YOLO / RetinaNet / FCOS):** Dự đoán trực tiếp hộp bao và xác suất lớp trên các lưới đặc trưng đa tỷ lệ — tối ưu cho camera thời gian thực và thiết bị biên.
- **Kim tự tháp đặc trưng (Feature Pyramid Network — FPN):** Trộn đặc trưng sâu (giàu ngữ nghĩa) với đặc trưng nông (độ phân giải cao) giúp phát hiện đồng thời cả biển báo nhỏ xíu lẫn tòa nhà lớn.
- **Mắt xích lý thuyết – thực hành:** Nắm chắc IoU/NMS/mAP trước khi tinh chỉnh Faster R-CNN trên tập dữ liệu tùy chỉnh.

### Khung giải thích (nên đọc chậm): Tính nhanh tổng số hộp neo (Anchor Boxes) trên kim tự tháp đặc trưng 3 mức

Giả sử ảnh đầu vào có kích thước $$640 \times 640$$ đi qua xương sống FPN tạo ra 3 bản đồ đặc trưng với bước nhảy (stride) $$8, 16, 32$$, và mỗi vị trí lưới đặt $$3$$ hộp neo với các tỷ lệ khung hình khác nhau:

1. **Mức P3 (stride 8 — bắt vật thể nhỏ):** Lưới $$80 \times 80 = 6{,}400$$ ô → $$6{,}400 \times 3 = 19{,}200$$ hộp neo.
2. **Mức P4 (stride 16 — bắt vật thể trung bình):** Lưới $$40 \times 40 = 1{,}600$$ ô → $$1{,}600 \times 3 = 4{,}800$$ hộp neo.
3. **Mức P5 (stride 32 — bắt vật thể lớn):** Lưới $$20 \times 20 = 400$$ ô → $$400 \times 3 = 1{,}200$$ hộp neo.
4. **Tổng cộng:** $$19{,}200 + 4{,}800 + 1{,}200 = 25{,}200$$ hộp ứng viên trên mỗi khung hình — trong đó hơn $$99\%$$ là nền! Con số này giải thích ngay vì sao **Focal Loss** và **NMS** lại tối quan trọng để kiểm soát mất cân bằng lớp và lọc hộp trùng.

Chủ đề phát hiện đối tượng (*object detection*) được chia thành hai phần:

1. **Lý thuyết** — IoU, hồi quy hộp bao (*bounding box*), RPN và trực giác hoạt động.
2. **Cài đặt** — mã nguồn minh họa, các bài báo nền tảng và những lỗi thường gặp.

## Video tham khảo

- **Xem trước** [How YOLO Object Detection Works](https://www.youtube.com/watch?v=svn9-xV7wjk) — DeepBean, tiếng Anh. Trình bày cách chia lưới, dự đoán hộp bao và điểm tin cậy đối tượng (*objectness*).
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — Stanford University School of Engineering, tiếng Anh. Đặt YOLO trong bức tranh chung cùng các bộ phát hiện hai giai đoạn được so sánh trong bài.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, tiếng Anh. Làm rõ loại đặc trưng mà mạng xương sống tích chập cần trích xuất trước khi gắn thêm đầu phát hiện.
