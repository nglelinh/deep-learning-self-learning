---
layout: post
title: 10-98 Luyện phỏng vấn (bộ tối ưu)
chapter: '10'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter10
lesson_type: optional
---

# Tùy chọn: luyện phỏng vấn — bộ tối ưu

> Bài này là **tùy chọn**. Nó **không** thay thế lý thuyết momentum, RMSprop, hay Adam. Sau các ghi chú đó, dùng bài luyện này, rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để làm các câu đã giải về phương pháp thích nghi, siêu tham số, và động lực huấn luyện.

**Chủ đề sách cho chương này (đọc trong PDF, không phải ở đây):** thuật toán tối ưu (kể cả cập nhật họ Adam), chọn siêu tham số, và phần huấn luyện của vùng chủ đề học sâu mở rộng.

Hub: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu luyện (gốc)

### Q1. Momentum trên thung lũng hẹp

SGD thuần dao động ngang thung lũng hẹp và bò dọc trục dài. Mỗi hướng một câu: bộ đệm vận tốc $$ \mathbf{v}\leftarrow \beta\mathbf{v}+\nabla J $$ đổi gì?

**Gợi ý.** Đảo dấu tần số cao bị trung bình hóa; gradient cùng chiều thì cộng dồn.

**Thảo luận.** Ngang thung lũng dấu gradient đảo, trung bình momentum nhỏ và dao động tắt. Dọc thung lũng gradient giữ dấu, $$\mathbf{v}$$ lớn dần và bước bò nhanh hơn. Đó là hình ảnh phỏng vấn; không cần khai triển $$\beta$$ cụ thể trừ khi được hỏi.

### Q2. Vì sao Adam có hiệu chỉnh bias

Moment bậc nhất của Adam là $$\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t$$ với $$\mathbf{m}_0=\mathbf{0}$$. Vì sao $$\mathbf{m}_t$$ quá nhỏ tại $$t=1$$, và chia cho $$1-\beta_1^t$$ làm gì?

**Gợi ý.** Khai một bước: $$\mathbf{m}_1=(1-\beta_1)\mathbf{g}_1$$.

**Thảo luận.** Tại $$t=1$$ bạn chỉ có một phần $$(1-\beta_1)$$ của gradient đầu (thường $$0.1$$ nếu $$\beta_1=0.9$$). Hệ số $$1-\beta_1^t$$ là tổng trọng số hình học; chia sẽ gỡ co khởi động lạnh. Cùng câu chuyện với moment bậc hai và $$\beta_2$$. Sau vài trăm bước $$\beta^t\approx 0$$ và hiệu chỉnh gần như tùy chọn trên thực tế — nhưng người phỏng vấn vẫn muốn phép tính $$t=1$$.

### Q3. Adam và AdamW

Bạn muốn phạt L2 hiệu dụng $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$. Vì sao “cộng $$\lambda\mathbf{w}$$ vào gradient, rồi để Adam rescale” không giống suy giảm $$\mathbf{w}$$ một lượng $$\lambda\eta$$ *ngoài* mẫu số thích nghi?

**Gợi ý.** Adam chia gradient (và mọi thứ bạn cộng vào) cho $$\sqrt{\hat{\mathbf{v}}}+\epsilon$$.

**Thảo luận.** Weight decay gắn kèm bị méo: tọa độ có moment bậc hai lớn bị *ít* suy giảm hơn. AdamW áp decay trực tiếp lên trọng số, nên mọi tọa độ co khoảng $$\lambda\eta$$ (trừ lịch learning rate). Đó là mặc định trong công thức CNN / Transformer hiện đại và là phân biệt Chương 10 đã nêu; câu này chỉ kiểm tra bạn nói được *vì sao*.

### Q4. Khi SGD+momentum có thể thắng Adam

Một ConvNet trên tập ảnh lớn tổng quát kém hơn với Adam chỉnh kỹ so với SGD+momentum chạy dài hơn. Đưa một giả thuyết nghiêng tối ưu và một giả thuyết nghiêng chính quy ẩn — không phải “Adam lỗi.”

**Gợi ý.** Phương pháp thích nghi viết lại preconditioner mỗi bước; SGD giữ một thang toàn cục.

**Thảo luận.** Tối ưu: Adam có thể vào vùng sắc của $$J$$ nhanh rồi ở lại vì thang từng tọa độ vẫn lớn trên tọa độ gradient nhỏ. Thiên kiến ẩn: nhiễu SGD + learning rate chung có phân phối dừng trên các cực tiểu khác nhiễu đã precondition của Adam. Không giả thuyết nào là định lý phải chứng trên bảng; bạn *nên* nói sẽ kiểm lại bằng SGD dài hơn và val set đúng, không phải một seed.

### Q5. Warmup trong một dòng

Bạn khởi Transformer với AdamW ở learning rate đỉnh và loss nhảy NaN. Vì sao tăng tuyến tính $$\eta$$ vài nghìn bước có thể sửa dù $$\eta$$ *cuối* không đổi?

**Gợi ý.** Lúc khởi tạo, kích hoạt và logit attention chưa ra thang; bước lớn được đi với $$\mathbf{m},\mathbf{v}$$ vô nghĩa.

**Thảo luận.** Gradient sớm rất lớn hoặc hỗn loạn; ước lượng moment bậc hai của Adam cũng chưa đáng tin (Q2). $$\eta$$ nhỏ cho moment và residual stream ổn rồi mới bước cỡ ImageNet / LLM. Warmup không phải phép màu — nếu $$\eta$$ đỉnh vốn quá lớn, bạn vẫn phân kỳ sau khi warmup kết thúc.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Tải PDF từ arXiv để xem toàn bộ Q&A đã giải. Trang này là bài luyện gốc của khóa, không phải bản in lại.
