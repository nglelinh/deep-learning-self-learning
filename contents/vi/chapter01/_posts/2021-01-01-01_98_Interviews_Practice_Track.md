---
layout: post
title: 01-98 Lộ trình luyện Deep Learning Interviews
chapter: '01'
order: 9
owner: Deep Learning Course
lang: vi
categories:
- chapter01
lesson_type: optional
---

# Tùy chọn: Lộ trình luyện Deep Learning Interviews

> Bài này là **tùy chọn**. Nó **không** thay thế bất kỳ chương lý thuyết nào. Đây là bản đồ và kế hoạch học để dùng một cuốn sách phỏng vấn miễn phí như bài luyện thêm — sau khi bạn đã xong ghi chú khóa học tương ứng.

## Sách là gì

**Deep Learning Interviews** (ấn bản thứ hai) của **Shlomo Kashani** và **Amir Ivry** là tập bài toán phỏng vấn / thi cử đã giải đầy đủ, phủ các chủ đề cốt lõi của AI. Sách viết cho học viên cao học và ứng viên cần trình bày rõ dưới áp lực thời gian; cũng hữu ích như danh sách kiểm tra nền tảng nghiên cứu.

Khóa học này **không** in lại sách. Hãy mở nguồn chính thức:

- PDF / tóm tắt: [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)
- Kho kèm theo: [github.com/BoltzmannEntropy/interviews.ai](https://github.com/BoltzmannEntropy/interviews.ai)
- Trang tác giả: [interviews.ai](http://www.interviews.ai/)

Tải PDF từ arXiv (hoặc liên kết của tác giả). Tôn trọng điều khoản trên các trang đó. Đừng dán đoạn trích dài vào ghi chú, issue, hay đáp án dùng chung.

## Cách dùng cùng khóa học

Chọn một chế độ và giữ kỷ luật thời gian:

1. **Luyện phỏng vấn.** Sau mỗi chương, trả lời *thành tiếng* các câu gốc trong bài tùy chọn của chương đó, rồi mở sách để làm bộ khó hơn, đã giải, cùng *chủ đề*.
2. **Quiz trên lớp.** Giảng viên có thể giao 2–3 câu companion như bài kiểm tra ngắn. Dùng sách làm đáp án giáo viên / bài về nhà thêm, không sao chép thành phiếu bài tập.
3. **Nền tảng nghiên cứu.** Trước khi đọc bài báo dựa trên lý thuyết thông tin, autodiff, hoặc bất định, lướt *vùng chủ đề* tương ứng trong sách (không dump cả chương) và bài lý thuyết của khóa.

Làm đóng sách trước. Rồi xem gợi ý khóa học. Chỉ sau đó mới mở PDF để đọc thêm Q&A đã giải.

## Kế hoạch học: chủ đề sách → chương 00–25

Sách sắp theo chủ đề phỏng vấn, không theo số chương khóa học. Dùng bảng như tờ chỉ đường. Tên chủ đề dưới đây chỉ là **nhãn mức cao** — không phải mục lục sao chép.

| Chương khóa | Chủ đề khóa | Chủ đề sách nên đọc (trong PDF) | Bài tùy chọn |
| --- | --- | --- | --- |
| 00 | Toán tiên quyết | Lý thuyết thông tin (entropy, KL, thông tin tương hỗ); giải tích / trực giác autodiff | **00-98** |
| 01 | Giới thiệu và hướng dẫn | Cách dùng sách; hub này | bài này |
| 02 | Nền tảng mạng neuron | Perceptron, hàm kích hoạt, phần kiến trúc “deep learning mở rộng” | **02-98** |
| 03 | Huấn luyện, loss, lan truyền ngược | Vi phân thuật toán; trực giác Hessian; logistic / phân loại | **03-98** |
| 04 | CNN | Tích chập, kiến trúc CNN, ý tưởng tách đặc trưng ban đầu | **04-98** |
| 05–08 | RNN, LSTM, attention, Transformer | Volume I không có nhà — bám ghi chú khóa; Volume II dự kiến NLP / chuỗi | chỉ trên hub |
| 09 | Chính quy hóa | Overfitting, kiểm định, kiểm soát dung lượng | **09-98** |
| 10 | Bộ tối ưu | Phương pháp thích nghi (Adam và họ hàng), siêu tham số, động lực huấn luyện | **10-98** |
| 11–14 | Mô hình sinh (kể cả VAE, GAN) | Volume I không có nhà — kế hoạch Volume II nhắc GAN / VAE; dùng chương khóa | chỉ trên hub |
| 15 | Học chuyển giao | Tách đặc trưng CNN và backbone đóng băng so với tinh chỉnh | **15-98** |
| 16–23 | SSL, CV, NLP, tiếng nói, RL, GNN, hiệu năng | Volume I không có nhà. Volume II dự kiến detection, segmentation, NLP, RL — dùng chương khóa khớp | chỉ trên hub |
| 24 | Giải thích mô hình | Chỉ bàn liên quan; *bất định* Bayesian nằm ở 25 | chỉ trên hub |
| 25 | Chủ đề nâng cao | Học sâu Bayesian và tư duy xác suất | **25-98** |

### Chủ đề không có chương riêng

Giữ các mục này trên hub. **Không** tạo chương bắt buộc mới.

**Ensemble (bagging / boosting / stacking).** Sách coi ensemble mạng neuron là chủ đề phỏng vấn. Khóa không có chương ensemble riêng. Nếu bị hỏi:

- Bagging: trung bình nhiều mô hình huấn luyện độc lập để giảm phương sai.
- Boosting: thêm mô hình sửa lỗi phần dư (giảm bias nhiều hơn bagging).
- Stacking: một mô hình tầng hai học cách tổ hợp các dự đoán gốc.

**Thảo luận (gốc, không lấy từ sách).** Khi nào bạn nên bagging mười CNN độc lập hơn là huấn luyện một mô hình lớn gấp mười? Điều gì hỏng nếu các mô hình gốc tương quan mạnh?

**Volume II (dự kiến).** Tác giả phác thảo tập sau về CNN nâng cao, detection, segmentation, NLP, GAN, VAE, và học tăng cường. Cho đến khi tập đó là giáo trình được giao, hãy dùng Chương 11–21 của khóa này cho các chủ đề ấy và coi PDF Volume I là nguồn luyện đã giải cho bảng trên.

## Khởi động gốc (chỉ trên hub)

Ba câu sau do khóa soạn. Không lấy từ sách.

### H1. Người phỏng vấn đang kiểm tra gì?

Bạn được hỏi “vì sao dùng cross-entropy chứ không phải MSE cho phân loại 10 lớp?” trong hai phút. Nêu lý do *mô hình hóa* và lý do *tối ưu*, rồi dừng.

**Gợi ý.** Nghĩ hợp lý của phân phối categorical, và dạng gradient khi dự đoán sai nhưng rất chắc.

**Thảo luận.** Mô hình hóa: softmax là phân phối rời rạc; cross-entropy là âm log-hợp lý của mô hình đó. Tối ưu: MSE trên xác suất thường cho gradient yếu khi lớp dự đoán đã nhọn ở nhãn sai; NLL vẫn đẩy logit của lớp đúng lên. Sau đó mở PDF để xem thêm câu loss / phân loại — đừng chép vào đây.

### H2. Đóng sách và mở sách

Bạn được dùng PDF trong bài take-home nhưng không được dùng trong buổi onsite 45 phút. Việc luyện hàng tuần nên đổi thế nào?

**Gợi ý.** Phỏng vấn onsite thưởng các biến đổi ngắn viết được trên bảng (đạo hàm sigmoid, kích thước đầu ra conv, một dòng cập nhật Adam).

**Thảo luận.** Luyện câu companion theo đồng hồ. Dùng sách để đào sâu và kiểm tra cách diễn đạt. Nếu chỉ giải được khi mở PDF, chủ đề đó chưa sẵn sàng cho phỏng vấn.

### H3. Ensemble khi khóa không có chương ensemble

Một vòng tuyển dụng yêu cầu giảm phương sai của ResNet đã chỉnh tốt mà không đổi kiến trúc. Đưa hai cách *không phải* “thu thập thêm nhãn.”

**Gợi ý.** Một cách đổi cách dùng mô hình sẵn có; một cách đổi cách dùng dữ liệu sẵn có.

**Thảo luận.** Snapshot / fine-tune bagging, tăng cường lúc suy luận, và dropout-như-ensemble lúc test đều giảm phương sai. Mô hình lớn hơn là đổi dung lượng, không phải ensemble giảm phương sai. Các bài ensemble đã giải đầy đủ thuộc về sách, không thuộc repo này.

## Mục lục companion

Sau bài lý thuyết khớp, mở:

- **00-98** — lý thuyết thông tin và giải tích / autodiff
- **02-98** — neuron, hàm kích hoạt, cấu trúc MLP
- **03-98** — autodiff, trực giác Hessian, phân loại logistic
- **04-98** — hình học và kiến trúc CNN
- **09-98** — overfitting và bộ chính quy
- **10-98** — bộ tối ưu họ Adam
- **15-98** — tách đặc trưng và chuyển giao
- **25-98** — chủ đề Bayesian / bất định

Mỗi companion có 3–6 câu **gốc** kèm gợi ý ngắn. Chúng không viết lại lý thuyết.

## Ghi công và nhắc giấy phép

Kashani, S., và Ivry, A. *Deep Learning Interviews*. arXiv:2201.00650. GitHub: [BoltzmannEntropy/interviews.ai](https://github.com/BoltzmannEntropy/interviews.ai).

Ngân hàng câu hỏi đã giải là tác phẩm của các tác giả. Khóa này chỉ trích tiêu đề, tác giả, mã arXiv, và tên chủ đề, rồi thêm bài luyện gốc của riêng mình. **Người học tải sách từ arXiv** (hoặc GitHub / interviews.ai chính thức). Đừng đưa bản PDF vào kho này và đừng dán nguyên văn đáng kể của sách vào pull request.
