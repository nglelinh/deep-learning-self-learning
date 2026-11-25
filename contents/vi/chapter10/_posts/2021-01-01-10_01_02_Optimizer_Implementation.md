---
layout: post
title: 10-01-02 Cài đặt Bộ tối ưu
chapter: '10'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter10
---

Đằng sau dòng lệnh quen thuộc `optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)` chỉ là một vòng lặp rất ngắn cập nhật từng tensor tham số dựa trên `.grad` và từ điển trạng thái `state`. Khi tự viết lại các bộ tối ưu từ đầu và quan sát chúng chạy đua trên cùng một bề mặt hàm mất mát 2D, bạn sẽ phát triển được "cảm giác trực quan" để chẩn đoán ngay lập tức khi quá trình huấn luyện bị dao động hoặc đứng im.

## Kiến thức cần nhớ: Ba trụ cột khi triển khai và đối chiếu Bộ tối ưu

1. **Thiết kế giao diện chuẩn hóa (`step()` và `zero_grad()`)**:
   Mỗi bộ tối ưu lưu từ điển trạng thái (vận tốc `v`, bình phương trượt `s`, bộ đếm bước `t`) gắn liền với từng tensor tham số, và tuyệt đối phải xóa gradient cũ trước mỗi bước backward (trừ khi cố ý dùng gradient accumulation).
2. **Trực quan hóa quỹ đạo 2D trên hàm thách thức (Rosenbrock / Thung lũng hẹp)**:
   So sánh trực diện SGD, Momentum, RMSprop và Adam trên cùng một xuất phát điểm để thấy cách từng thuật toán xử lý hướng dốc và hướng phẳng.
3. **Phân biệt L2 Regularization và Decoupled Weight Decay trong mã nguồn**:
   Chỉ khác nhau đúng 1 dòng code — cộng `wd * param` vào `grad` (Adam + L2) hay trừ trực tiếp `lr * wd * param` vào `param` (AdamW) — nhưng tạo ra sự khác biệt lớn về độ tổng quát hóa!

### Giải thích: Tính tay sự khác biệt giữa Adam + L2 và AdamW trên 1 tham số

Giả sử $$w = 2{,}0$$, gradient dữ liệu $$g = 0{,}0$$ (chỉ còn hiệu ứng phân rã trọng số), $$\eta = 0{,}1$$, hệ số weight decay $$\lambda = 0{,}5$$, và ước lượng moment bậc 2 hiện tại là $$\sqrt{\hat{v}_t} = 4{,}0$$ (một tham số có lịch sử gradient lớn):

| Cách cài đặt | Bước xử lý trọng số phạt | Độ giảm thực tế của $$w$$ | Nhận xét |
| :--- | :--- | :---: | :--- |
| **Adam + L2 truyền thống** | Cộng $$\lambda w = 1{,}0$$ vào gradient, rồi **chia cho $$\sqrt{\hat{v}_t} = 4{,}0$$** | $$\eta \frac{\lambda w}{\sqrt{\hat{v}_t}} = 0{,}1 \times \frac{1{,}0}{4{,}0} = \mathbf{0{,}025}$$ | Tham số có gradient lớn bị phạt **yếu đi 4 lần** (mất tác dụng phạt!) |
| **AdamW (Tách rời)** | Trừ trực tiếp $$\eta \lambda w$$ **không chia cho $$\sqrt{\hat{v}_t}$$** | $$\eta \lambda w = 0{,}1 \times 0{,}5 \times 2{,}0 = \mathbf{0{,}100}$$ | Mọi tham số đều co đều đúng tỷ lệ $$\eta \lambda = 5\%$$ |

Chính sự khác biệt ở mẫu số $$\sqrt{\hat{v}_t}$$ này giải thích vì sao Loshchilov & Hutter (2019) chỉ ra rằng **Adam + L2 không hề tương đương với Weight Decay**, và AdamW lập tức trở thành tiêu chuẩn mới!

Nội dung phần cài đặt bộ tối ưu gồm hai phần:

1. **Cài đặt Cốt lõi** — cài đặt từ đầu các bộ tối ưu từ Momentum đến AdamW.
2. **So sánh, Bài báo và Cạm bẫy** — minh họa quỹ đạo 2D, so sánh trên bài toán phân loại, tổng hợp bài báo nền tảng và các lưu ý thực hành.

## Video tham khảo

- **Xem trước:** [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, tiếng Anh. Trình bày quy tắc cập nhật Adam sẽ được cài đặt trong bài mã.
- [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, tiếng Anh. Đối chiếu với RMSprop và AdaGrad giúp dễ phát hiện sai sót khi cài đặt từng bộ đệm.
- [PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) — Patrick Loeber, tiếng Anh. Cách gọi bộ tối ưu trong vòng lặp huấn luyện PyTorch sau khi đã hiểu rõ phép cập nhật.
