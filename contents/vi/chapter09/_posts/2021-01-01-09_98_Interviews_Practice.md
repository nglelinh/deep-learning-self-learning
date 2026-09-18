---
layout: post
title: 09-98 Luyện phỏng vấn (overfitting và chính quy)
chapter: '09'
order: 9
owner: Deep Learning Course
lang: vi
categories:
- chapter09
lesson_type: optional
---

# Tùy chọn: luyện phỏng vấn — overfitting và chính quy

> Bài này là **tùy chọn**. Nó **không** thay thế dropout, BatchNorm, hay các ghi chú chính quy khác. Sau những bài đó, dùng bài luyện này, rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để làm các câu đã giải về overfitting, kiểm định, và kiểm soát dung lượng.

**Chủ đề sách cho chương này (đọc trong PDF, không phải ở đây):** overfitting / tổng quát hóa, kiểm định, và các bộ chính quy trong vùng chủ đề học sâu mở rộng.

Hub: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu luyện (gốc)

### Q1. Loss train giảm, loss val tăng

Cross-entropy huấn luyện vẫn giảm sau epoch 40; cross-entropy kiểm định tăng từ epoch 25. Nêu hai can thiệp *khác nhau* (không phải “thêm GPU”) và mỗi cách giả định gì về khoảng cách đó.

**Gợi ý.** Một cách đổi lớp giả thuyết hoặc mục tiêu hiệu dụng; một cách đổi *lúc* bạn dừng.

**Thảo luận.** (1) Chính quy mạnh hơn (weight decay cao hơn, thêm dropout, thêm augmentation) giả định mô hình vẫn khớp được tập train nhưng quá linh hoạt. (2) Dừng sớm / chọn mô hình trên split validation thật giả định bạn đã qua điểm dung lượng hữu ích. Thu thập thêm nhãn tấn công cùng khoảng cách từ phía cỡ mẫu. Huấn luyện lại mạng nhỏ hơn là đổi dung lượng, không phải chính quy lúc train.

### Q2. L2 và L1 trên đầu tuyến tính

Với một tầng tuyến tính, so việc thêm $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$ với $$\lambda\|\mathbf{w}\|_1$$ vào loss. Cái nào đẩy từng trọng số về *đúng* không dễ hơn, và vì sao đó chỉ là một phần câu chuyện trong mạng ReLU?

**Gợi ý.** Subgradient L1 chứa một khoảng tại 0; gradient L2 tuyến tính, co nhưng hiếm khi chạm không trong SGD.

**Thảo luận.** L1 (kiểu lasso) có thể tạo số không đúng nghĩa; L2 kéo trọng số về gốc mà không khuyến khích thưa. Trong mạng ReLU sâu, “đặc trưng” bị tắt là một *đường kích hoạt*, không phải một tọa độ của $$\mathbf{w}$$, nên L1 trên trọng số thô là công cụ thô. Weight decay (L2) vẫn mặc định vì thân thiện với quay trong trường hợp tuyến tính và rẻ.

### Q3. Dropout lúc train và lúc eval

Một đơn vị bị drop với xác suất $$p=0.5$$ lúc huấn luyện. Bạn phải làm gì lúc đánh giá để kỳ vọng tiền kích hoạt khớp lúc train? Quên thì sao?

**Gợi ý.** Inverted dropout so với scale kiểu cổ điển.

**Thảo luận.** Inverted dropout (mặc định framework) nhân đơn vị sống sót với $$1/(1-p)$$ *lúc train* và dùng đủ mạng lúc eval. Dropout cổ điển để train không scale và nhân trọng số với $$1-p$$ lúc eval. Nếu drop lúc train rồi không làm gì lúc eval, mọi tầng lớn hơn có hệ thống so với mạng bạn đã tối ưu — cả hiệu chỉnh xác suất lẫn accuracy đều kém. Nói “dropout là ensemble $$2^n$$ mạng” là khẩu hiệu; điểm vận hành là hệ số scale.

### Q4. BatchNorm lúc train và lúc test

Vì sao BatchNorm lưu trung bình chạy, và điều gì hỏng nếu bạn đánh giá checkpoint mới nạp ở chế độ train trên batch kích thước 1?

**Gợi ý.** Train dùng thống kê batch; eval dùng trung bình trượt.

**Thảo luận.** Moment lưu lại là ước lượng lúc test của $$\mathbb{E}[\mathbf{h}]$$ và $$\mathrm{Var}(\mathbf{h})$$. Batch kích thước 1 làm phương sai batch không xác định hoặc rất ồn, nên BN chế độ train không phải eval thay thế. Huấn luyện batch nhỏ có cùng vấn đề — đó là lý do LayerNorm / GroupNorm xuất hiện ở công thức chương khác, nhưng câu trả lời phỏng vấn *của chương này* là: biết thống kê nào đang được dùng.

### Q5. Augmentation như một bộ chính quy

Bạn không tăng được $$\lambda$$ vì loss train đã cao. Đưa một bộ chính quy phía dữ liệu không thêm hạng phạt vào $$J(\mathbf{w})$$, và một chế độ hỏng.

**Gợi ý.** Phân phối huấn luyện trở thành phiên bản làm mượt của phân phối gốc.

**Thảo luận.** Crop ngẫu nhiên, lật, hoặc nhiễu màu bơm bất biến bạn quan tâm mà không đổi hạng phạt tham số. Hỏng: augmentation phá nhãn (ví dụ crop cắt mất vật) thêm *nhiễu*, không phải bất biến hữu ích, và có thể tăng cả lỗi train lẫn val. Dừng sớm là chính quy không phạt khác; nó không thay thế chính sách augmentation kém.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Tải PDF từ arXiv để xem toàn bộ Q&A đã giải. Trang này là bài luyện gốc của khóa, không phải bản in lại.
