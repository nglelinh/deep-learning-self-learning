---
layout: post
title: 15-98 Luyện phỏng vấn (tách đặc trưng và chuyển giao)
chapter: '15'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter15
lesson_type: optional
---

# Tùy chọn: luyện phỏng vấn — tách đặc trưng và chuyển giao

> Bài này là **tùy chọn**. Nó **không** thay thế lý thuyết tách đặc trưng so với fine-tune. Sau các ghi chú đó, dùng bài luyện này, rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để làm các câu đã giải về tách đặc trưng CNN. Hình học kiến trúc ở **04-98**.

**Chủ đề sách cho chương này (đọc trong PDF, không phải ở đây):** dùng CNN đã huấn luyện như bộ tách đặc trưng; thân đóng băng so với thân huấn luyện được.

Hub: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu luyện (gốc)

### Q1. Linear probe so với fine-tune khối cuối

Bạn có 800 ảnh y khoa có nhãn và ResNet tiền huấn luyện trên ImageNet. So (a) đóng băng thân, huấn luyện đầu tuyến tính và (b) mở khóa giai đoạn residual cuối cùng cộng đầu. (b) thêm rủi ro gì mà (a) không có?

**Gợi ý.** Số tham số huấn luyện được so với mức gần của thống kê nguồn và đích.

**Thảo luận.** (a) là linear probe: rẻ, khó phá bộ lọc đã học, thường là baseline mạnh. (b) có thể thích nghi bộ lọc trung/cao với tổn thương hoặc kết cấu ImageNet chưa thấy, nhưng 800 ảnh có thể ghi đè các bộ lọc đó (fine-tune thảm họa). Thương lượng thực tế là learning rate nhỏ trên thân và lớn trên đầu, hoặc adapter (ghi chú hiện đại tùy chọn Chương 15) — nói như phần mở rộng, không phải câu trả lời lõi.

### Q2. Khi đặc trưng đóng băng là sai công cụ

Đưa một miền đích cụ thể mà kích hoạt ImageNet là biểu diễn yếu dù nguồn và đích đều là “ảnh,” và nêu chỗ lệch.

**Gợi ý.** Nghĩ cảm biến, góc nhìn, hoặc độ mịn nhãn — không phải “tập nhỏ.”

**Thảo luận.** Ví dụ: mô bệnh học xám 40×, kênh vệ tinh ngoài RGB, hoặc khuyết tật công nghiệp cận cảnh. Chỗ lệch là *thống kê thấp và ngữ nghĩa*, không chỉ cỡ mẫu. Fine-tune nhiều phần thân hơn, hoặc tiền huấn luyện trong miền (tự giám sát, Chương 16), thẳng thắn hơn là chồng MLP sâu lên đặc trưng vật-thể RGB đóng băng.

### Q3. Bạn thực sự tách cái gì

Một đồng đội “tách đặc trưng” bằng cách lấy bản đồ $$7\times7\times2048$$ của ResNet-50 rồi làm phẳng để huấn luyện SVM. Đưa một lý do nên global-average-pool trước, và một tác vụ bạn sẽ *giữ* bản đồ không gian.

**Gợi ý.** Bất biến so với định vị.

**Thảo luận.** GAP thành $$2048$$ chiều cho mô tả chịu tịnh tiến và số chiều SVM vừa phải. Detection, segmentation, hoặc tác vụ cần biết vật *ở đâu* nên giữ tensor không gian (hoặc dùng kim tự tháp đặc trưng). “Tách đặc trưng” không phải một shape vector; đó là chọn bất biến bạn muốn.

### Q4. Lệch miền trong một bức tranh

Ảnh huấn luyện là ảnh sản phẩm studio; ảnh production là ảnh điện thoại trong kho. Bạn chỉ fine-tune đầu và accuracy val rất đẹp (val cũng là studio). Bạn đã đo gì, và thí nghiệm nào cần thêm trước khi giao hàng?

**Gợi ý.** Val i.i.d. so với phân phối triển khai.

**Thảo luận.** Bạn đo chất lượng đầu trong miền, không phải chuyển giao. Thêm một lát có nhãn trong kho (hoặc proxy khớp thị giác) và báo cáo số đó. Augmentation màu / crop bắt chước ánh sáng kho là bước trung gian rẻ; không thay được val đích thật. Người phỏng vấn muốn sự khiêm tốn này hơn một tên kiến trúc thần kỳ.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Tải PDF từ arXiv để xem toàn bộ Q&A đã giải. Trang này là bài luyện gốc của khóa, không phải bản in lại.
