---
layout: post
title: 15-01 Nền tảng Học Chuyển giao
chapter: '15'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter15
---

Trong thực tế công nghiệp, hơn 90% dự án học sâu thành công không bắt đầu bằng việc phát minh một kiến trúc mạng mới, mà bắt đầu bằng việc chọn đúng một mô hình nền (foundation/pretrained model) và biết cách tinh chỉnh nó thật khéo. Mục tổng quan nền tảng này hệ thống hóa ma trận quyết định kinh điển của Học Chuyển giao trước khi đi sâu vào lý thuyết (**15-01-01**) và mã nguồn PyTorch (**15-01-02**).

## Kiến thức cần nhớ: Ma trận quyết định 4 tình huống trong Học Chuyển giao

- **Dữ liệu đích ít + Miền tương đồng nguồn:** Đóng băng toàn bộ xương sống, chỉ huấn luyện bộ phân loại tuyến tính (Linear Probing) hoặc dùng đặc trưng trích xuất cho SVM/Logistic Regression.
- **Dữ liệu đích nhiều + Miền tương đồng nguồn:** Tinh chỉnh toàn bộ mạng (hoặc phần lớn các khối trên) với tốc độ học nhỏ để đạt độ chính xác tối đa.
- **Dữ liệu đích ít + Miền khác biệt nguồn:** Trích xuất đặc trưng từ các *tầng nông/giữa* (nơi biểu diễn còn tổng quát) thay vì tầng cuối quá chuyên biệt cho nhãn nguồn.
- **Dữ liệu đích nhiều + Miền khác biệt nguồn:** Khởi tạo từ trọng số tiền huấn luyện (vẫn hội tụ nhanh hơn khởi tạo ngẫu nhiên) và mở khóa huấn luyện lại toàn bộ mạng.

### Giải thích: Áp dụng ma trận quyết định và ước lượng tốc độ học phân tầng cho 2 bài toán thực tế

Xét mô hình xương sống 4 giai đoạn (`stage1` → `stage4` + `head`) với tốc độ học cơ sở cho đầu phân loại $$\eta_0 = 10^{-3}$$ và hệ số suy giảm theo tầng $$\gamma = 0{,}3$$:

1. **Tình huống A — Phân loại 12 giống chó địa phương (600 ảnh, rất giống ImageNet):**
   - Đóng băng `stage1`–`stage4` trong 5 epoch đầu, chỉ train `head` với $$\eta_0 = 10^{-3}$$.
   - Sau đó mở khóa `stage4` với $$\eta_4 = \eta_0 \cdot \gamma = 3 \times 10^{-4}$$ trong 5 epoch nữa.
2. **Tình huống B — Phân loại vết nứt bề mặt kim loại (25.000 ảnh, khác ImageNet):**
   - Mở khóa toàn bộ mạng với tốc độ học phân tầng: `head` $$= 10^{-3}$$, `stage4` $$= 3 \times 10^{-4}$$, `stage3` $$= 9 \times 10^{-5}$$, `stage2` $$= 2{,}7 \times 10^{-5}$$, `stage1` $$= 8{,}1 \times 10^{-6}$$.
   Cách chia bậc số học này giúp các bộ lọc cạnh ở `stage1` dịch chuyển cực chậm và ổn định trong khi các tầng cao nhanh chóng thích nghi với kết cấu kim loại.

Nội dung nền tảng về học chuyển giao gồm hai phần:

1. **Lý thuyết** — khái niệm cốt lõi, các chiến lược chuyển giao, lý thuyết thích ứng miền và ví dụ minh họa.
2. **Cài đặt** — so sánh trích xuất đặc trưng và tinh chỉnh trong PyTorch, bài báo nền tảng và các cạm bẫy thường gặp.

## Video tham khảo

- **Xem trước:** [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, tiếng Anh. Phân biệt rõ chiến lược trích xuất đặc trưng và tinh chỉnh.
- [Pytorch Transfer Learning and Fine Tuning Tutorial](https://www.youtube.com/watch?v=qaDe0qQZ5AQ) — Aladdin Persson, tiếng Anh. Minh họa cả hai chiến lược bằng mã PyTorch.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Giải thích các đặc trưng thị giác tầng sớm được tái sử dụng trong cả hai chiến lược.
