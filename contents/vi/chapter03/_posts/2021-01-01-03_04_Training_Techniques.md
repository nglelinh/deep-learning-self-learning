---
layout: post
title: 03-04 Kỹ thuật Huấn luyện Thực tế
chapter: '03'
order: 11
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

Hai kỹ sư cùng dùng một kiến trúc mạng y hệt nhau và cùng một tập dữ liệu, nhưng người thứ nhất đạt độ chính xác 96% chỉ sau 15 epoch còn người thứ hai mãi dậm chân ở 50% hoặc vọt lỗi `NaN`. Sự khác biệt không nằm ở công thức mới lạ nào, mà nằm ở **kỹ thuật huấn luyện thực tế**: cách khởi tạo trọng số ban đầu, cách chuẩn hóa dữ liệu tránh rò rỉ (*data leakage*), cách chọn batch size và cách đọc biểu đồ loss để dừng đúng lúc. Mục tổng quan này mở đầu hai bài thực chiến giúp biến kiến thức lý thuyết thành mô hình chạy ổn định ngoài đời thực.

## Kiến thức cần nhớ: Ba trụ cột của Kỹ thuật Huấn luyện Thực tế

1. **Khởi tạo Trọng số & Tiền xử lý Sạch (Initialization & Preprocessing)**:
   Dùng Xavier cho Sigmoid/Tanh và He/Kaiming cho ReLU để giữ phương sai tín hiệu ổn định; luôn tính tham số chuẩn hóa ($$\mu, \sigma$$) **chỉ trên tập Train** rồi mới áp dụng lên Validation và Test.

2. **Chiến lược Chia tập & Chọn Batch Size (Train / Val / Test Split)**:
   Tách bạch rạch ròi vai trò của 3 tập: Train để cập nhật trọng số, Validation để chọn siêu tham số và kiến trúc, và Test giữ kín tuyệt đối cho lần đánh giá cuối cùng.

3. **Giám sát Đường cong Học & Dừng sớm (Monitoring & Early Stopping)**:
   Đọc bệnh của mô hình qua khoảng cách giữa `train_loss` và `val_loss`, lưu lại *checkpoint* tốt nhất và dừng huấn luyện ngay khi `val_loss` ngừng cải thiện trong `patience` epoch liên tiếp.

### Giải thích: Bảng chẩn đoán nhanh 4 triệu chứng đường cong Loss khi huấn luyện

| Triệu chứng quan sát được | Nguyên nhân gốc rễ khả năng cao nhất | Hành động xử lý ưu tiên |
| :--- | :--- | :--- |
| **Cả Train Loss và Val Loss đều cao & đi ngang** | Underfitting, learning rate quá nhỏ, hoặc bug code | Kiểm tra lại phép thử overfit 1 batch, tăng dung lượng mạng hoặc tăng learning rate |
| **Train Loss giảm sâu nhưng Val Loss quay đầu tăng** | **Overfitting** (mô hình học thuộc nhiễu tập train) | Bật Early Stopping, thêm Regularization (Dropout, Weight Decay), tăng dữ liệu / Data Augmentation |
| **Loss dao động dữ dội hoặc bùng thành `NaN`** | Learning rate quá lớn, chưa chuẩn hóa dữ liệu, hoặc chia 0 | Hạ learning rate 5–10 lần, bật Gradient Clipping, kiểm tra chuẩn hóa đầu vào |
| **Val Loss thấp hơn Train Loss một chút ở đầu** | Bình thường khi đang bật Dropout mạnh lúc train (tắt lúc val) | Tiếp tục theo dõi; không phải lỗi! |

Phần này tổng hợp các kỹ thuật thực tế giúp huấn luyện mạng neuron ổn định và hiệu quả:

1. **Khởi tạo, tiền xử lý, batch và chia tập** — khởi tạo trọng số, chuẩn hóa đặc trưng, tạo mini-batch và chia train/validation/test
2. **Giám sát, dừng sớm và vòng lặp huấn luyện** — theo dõi chỉ số, dừng sớm (*early stopping*), cắt ngưỡng gradient và tinh chỉnh siêu tham số

## Video tham khảo

- **Xem trước:** [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering, tiếng Anh. Trình bày các bước chuẩn bị huấn luyện: khởi tạo trọng số, tiền xử lý và chia batch.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering, tiếng Anh. Phần tiếp nối bàn về tối ưu hóa, chính quy hóa và theo dõi quá trình huấn luyện.
- [Train, Test, & Validation Sets explained](https://www.youtube.com/watch?v=Zi-0rlM4RDs) — deeplizard, tiếng Anh. Giải thích vai trò của ba tập train, validation và test trong quy trình chuẩn bị dữ liệu.
