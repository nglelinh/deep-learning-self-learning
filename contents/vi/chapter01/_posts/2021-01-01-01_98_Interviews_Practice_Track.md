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

Chuẩn bị phỏng vấn Deep Learning mà chỉ học tủ danh sách câu hỏi rời rạc rất dễ rơi vào cảnh "gặp câu hỏi biến tấu một chút là đứng hình". Lý do đơn giản là mọi câu hỏi phỏng vấn sắc bén đều bắt nguồn từ một hệ thống kiến thức liên thông: từ xác suất và giải tích (Chương 00), qua kiến trúc mạng và lan truyền ngược (Chương 02–03), đến tối ưu hóa, CNN, Transformer và thiết kế hệ thống ML. Bài học này thiết kế lộ trình đối chiếu từng chương của khóa học với các chủ đề trọng tâm của sách *Deep Learning Interviews*, giúp bạn tích lũy năng lực phỏng vấn song hành với tiến độ học hàng ngày.

## Kiến thức cần nhớ: Bốn nhóm câu hỏi cốt lõi trong Phỏng vấn Deep Learning

1. **Nền tảng Toán & Thống kê (Ch 00–03)**:
   Tự tay đạo hàm Softmax + Cross-Entropy, giải thích liên hệ MLE–MSE–Cross-Entropy, phân biệt L1/L2 Regularization, và phân tích nguyên nhân gradient biến mất/bùng nổ.

2. **Kiến trúc & Cơ chế Hiện đại (Ch 04–16)**:
   Tính kích thước đầu ra tầng tích chập (Conv2D), so sánh BatchNorm vs. LayerNorm, giải thích cơ chế Scaled Dot-Product Attention $$\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}$$ và lý do chia cho $$\sqrt{d_k}$$.

3. **Gỡ lỗi Huấn luyện & Thực chiến (Training & Debugging)**:
   Chẩn đoán các kịch bản quen thuộc: train loss giảm nhưng val loss tăng (overfitting), cả hai cùng cao (underfitting/bug), loss bất ngờ vọt lên `NaN` (learning rate cao hoặc chia 0).

4. **Thiết kế Hệ thống ML (ML System Design)**:
   Biết chọn metric kinh doanh và metric mô hình, xử lý mất cân bằng dữ liệu, trôi phân phối (*data drift*) và tối ưu độ trễ suy luận (*latency/throughput*).

### Giải thích: Áp dụng khung trả lời 4 bước cho một câu hỏi phỏng vấn kinh điển

**Đề bài mẫu**: *"Tại sao trong Scaled Dot-Product Attention ta phải chia tích vô hướng $$\mathbf{q} \cdot \mathbf{k}$$ cho $$\sqrt{d_k}$$ trước khi đưa vào Softmax?"*

1. **Giả định thống kê**: Giả sử các thành phần $$q_i, k_i$$ độc lập, kỳ vọng $$0$$, phương sai $$1$$.
2. **Tính tay phương sai của tích vô hướng**:
   $$\text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i)\text{Var}(k_i) = d_k.$$
   Với $$d_k = 64$$, độ lệch chuẩn là $$\sqrt{64} = 8$$; với $$d_k = 512$$, độ lệch chuẩn vọt lên $$\approx 22.6$$!
3. **Hệ quả nếu KHÔNG chia**: Các logit lớn tới $$\pm 20$$ đẩy Softmax vào vùng bão hòa (một phần tử $$\approx 1$$, mọi phần tử khác $$\approx 0$$), làm đạo hàm của Softmax gần bằng $$0$$ (vanishing gradient).
4. **Hiệu quả khi chia cho $$\sqrt{d_k}$$**:
   $$\text{Var}\!\left(\frac{\mathbf{q}\cdot\mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{(\sqrt{d_k})^2} = 1,$$
   giữ phương sai đầu vào Softmax luôn ổn định bằng $$1$$ bất kể $$d_k$$ lớn đến đâu! Trả lời đủ 4 bước định lượng này, bạn sẽ luôn thuyết phục được hội đồng phỏng vấn.

# Tùy chọn: Lộ trình luyện Deep Learning Interviews

> Bài này là phần **tùy chọn**, **không** thay thế bất kỳ chương lý thuyết nào. Đây là bảng đối chiếu và kế hoạch học giúp bạn kết hợp một cuốn sách luyện phỏng vấn miễn phí như nguồn bài tập bổ sung sau khi đã học xong phần lý thuyết tương ứng trong khóa.

## Giới thiệu cuốn sách

**Deep Learning Interviews** (ấn bản thứ hai) của **Shlomo Kashani** và **Amir Ivry** là tuyển tập các bài toán phỏng vấn và thi cử có lời giải chi tiết, bao phủ nhiều chủ đề cốt lõi của trí tuệ nhân tạo. Sách phù hợp cho học viên cao học và ứng viên cần trình bày rõ ràng dưới áp lực thời gian, đồng thời có thể dùng như danh mục kiểm tra kiến thức nền tảng cho nghiên cứu.

Khóa học này **không** sao chép lại nội dung sách. Bạn hãy truy cập trực tiếp các nguồn chính thức:

- PDF và tóm tắt: [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)
- Kho mã nguồn kèm theo: [github.com/BoltzmannEntropy/interviews.ai](https://github.com/BoltzmannEntropy/interviews.ai)
- Trang của tác giả: [interviews.ai](http://www.interviews.ai/)

Hãy tải tệp PDF từ arXiv hoặc liên kết chính thức của tác giả, tuân thủ điều khoản sử dụng trên các trang đó và không dán các đoạn trích dài vào ghi chú, issue hay lời giải công khai.

## Cách kết hợp sách với khóa học

Hãy chọn một mục tiêu học phù hợp và giữ kỷ luật về thời gian:

1. **Luyện phỏng vấn:** Sau khi hoàn thành mỗi chương, hãy tự trình bày *thành tiếng* câu trả lời cho các câu hỏi gốc trong bài luyện tập bổ trợ của chương đó, rồi mở sách để làm thêm các bài khó hơn có lời giải cùng chủ đề.
2. **Kiểm tra ngắn trên lớp:** Giảng viên có thể chọn 2–3 câu trong bài bổ trợ làm bài kiểm tra nhanh, dùng sách làm nguồn tham khảo lời giải hoặc bài tập về nhà mở rộng mà không sao chép trực tiếp thành phiếu bài tập.
3. **Củng cố nền tảng nghiên cứu:** Trước khi đọc các bài báo dựa trên lý thuyết thông tin, vi phân tự động (*autodiff*) hoặc ước lượng bất định, hãy rà soát phần chủ đề tương ứng trong sách (tránh đọc dàn trải cả chương) kết hợp với bài lý thuyết của khóa.

Nguyên tắc chung là tự giải khi **đóng sách** trước, sau đó xem phần gợi ý của khóa học, và chỉ mở tệp PDF sau cùng để đối chiếu thêm các câu hỏi–đáp có lời giải.

## Kế hoạch học: đối chiếu chủ đề sách với Chương 00–26

Cuốn sách được sắp xếp theo nhóm chủ đề phỏng vấn chứ không theo thứ tự chương của khóa học. Bảng dưới đây giúp bạn tra cứu tương ứng; tên chủ đề chỉ là **nhãn khái quát ở mức cao**, không phải mục lục sao chép từ sách.

| Chương khóa | Chủ đề khóa | Chủ đề nên đọc trong sách (PDF) | Bài tùy chọn |
| --- | --- | --- | --- |
| 00 | Toán tiên quyết | Lý thuyết thông tin (entropy, phân kỳ KL, thông tin tương hỗ); giải tích và trực giác autodiff | **00-98** |
| 01 | Giới thiệu và hướng dẫn | Cách sử dụng sách; trang tổng hợp này | bài này |
| 02 | Nền tảng mạng nơ-ron | Perceptron, hàm kích hoạt, phần kiến trúc mở rộng của học sâu | **02-98** |
| 03 | Huấn luyện, hàm mất mát, lan truyền ngược | Vi phân thuật toán; trực giác ma trận Hessian; hồi quy logistic và phân loại | **03-98** |
| 04 | CNN | Phép tích chập, kiến trúc CNN, trích xuất đặc trưng ban đầu | **04-98** |
| 05–08 | RNN, LSTM, attention, Transformer | Tập I chưa có mục riêng — bám sát bài giảng khóa học; Tập II dự kiến đề cập NLP và chuỗi | chỉ trên trang này |
| 09 | Chính quy hóa | Quá khớp (*overfitting*), kiểm định, kiểm soát dung lượng mô hình | **09-98** |
| 10 | Bộ tối ưu | Phương pháp thích nghi (Adam và các biến thể), siêu tham số, động lực học huấn luyện | **10-98** |
| 11–14 | Mô hình sinh (gồm VAE, GAN) | Tập I chưa có mục riêng — kế hoạch Tập II nhắc tới GAN và VAE; dùng bài giảng khóa học | chỉ trên trang này |
| 15 | Học chuyển giao | Trích xuất đặc trưng CNN, so sánh đóng băng *backbone* và tinh chỉnh (*fine-tuning*) | **15-98** |
| 16–23 | SSL, CV, NLP, tiếng nói, RL, GNN, hiệu năng | Tập I chưa có mục riêng; Tập II dự kiến gồm phát hiện đối tượng, phân đoạn, NLP, RL — dùng các chương tương ứng của khóa | chỉ trên trang này |
| 24 | Giải thích mô hình | Chỉ đề cập ở mức liên quan; phần *bất định* Bayesian nằm ở Chương 25 | chỉ trên trang này |
| 25 | Chủ đề nâng cao | Học sâu Bayesian và tư duy xác suất | **25-98** |
| 26 | Mô hình ngôn ngữ lớn | Tập I chưa có mục riêng — dùng mục tổng hợp LLM của khóa (toán tiền huấn luyện, prompting, căn chỉnh, phục vụ, chưng cất, cơ chế nội tại) | **26-06** |

### Các chủ đề không có chương riêng

Những nội dung này được tóm tắt ngay tại đây thay vì tạo thêm chương bắt buộc mới.

**Tổ hợp mô hình (*ensemble*: bagging / boosting / stacking):** Cuốn sách coi tổ hợp mạng nơ-ron là một chủ đề phỏng vấn quan trọng, dù khóa học không có chương riêng. Nếu được hỏi, bạn cần phân biệt rõ ba hướng chính:

- **Bagging:** lấy trung bình nhiều mô hình huấn luyện độc lập để giảm phương sai.
- **Boosting:** huấn luyện tuần tự các mô hình nhằm sửa phần dư của mô hình trước, giúp giảm độ chệch (*bias*) mạnh hơn *bagging*.
- **Stacking:** dùng một mô hình tầng hai học cách kết hợp dự đoán của các mô hình cơ sở.

**Câu hỏi thảo luận (do khóa biên soạn, không lấy từ sách):** Khi nào nên áp dụng *bagging* với mười mạng CNN độc lập thay vì huấn luyện một mô hình lớn gấp mười lần? Điều gì sẽ xảy ra nếu các mô hình cơ sở có tương quan cao với nhau?

**Tập II (dự kiến):** Các tác giả đã phác thảo tập tiếp theo về CNN nâng cao, phát hiện đối tượng, phân đoạn ảnh, NLP, GAN, VAE và học tăng cường. Khi tập này chưa được đưa vào giáo trình, bạn hãy học các chủ đề đó qua Chương 11–21 của khóa học và dùng tệp PDF Tập I làm nguồn bài tập có lời giải theo bảng đối chiếu trên.

## Ba câu khởi động gốc (chỉ có tại trang này)

Ba câu hỏi dưới đây do khóa học tự soạn, không trích từ sách.

### H1. Người phỏng vấn đang kiểm tra điều gì?

Bạn được yêu cầu giải thích trong hai phút: "Vì sao bài toán phân loại 10 lớp thường dùng *cross-entropy* thay vì MSE?" Hãy nêu rõ một lý do về *mô hình hóa* và một lý do về *tối ưu hóa*, rồi dừng lại.

**Gợi ý:** Liên hệ với hàm hợp lý (*likelihood*) của phân phối phân loại (*categorical*) và độ lớn của gradient khi mô hình dự đoán sai nhưng lại có độ tự tin rất cao.

**Thảo luận:** Về mô hình hóa, *softmax* biểu diễn một phân phối xác suất rời rạc, và *cross-entropy* chính là âm log-hợp lý (*negative log-likelihood*) của phân phối đó. Về tối ưu hóa, MSE tính trên xác suất thường cho gradient nhỏ khi xác suất tập trung sai vào một lớp, trong khi *negative log-likelihood* vẫn duy trì tín hiệu đẩy *logit* của lớp đúng lên. Sau khi tự trả lời, bạn có thể mở tệp PDF để luyện thêm các câu về hàm mất mát và phân loại (không sao chép lại vào đây).

### H2. Luyện tập đóng sách và mở sách

Nếu bạn được phép tra cứu PDF trong bài làm tại nhà (*take-home*) nhưng không được dùng tài liệu trong buổi phỏng vấn trực tiếp 45 phút, kế hoạch luyện tập hằng tuần nên thay đổi như thế nào?

**Gợi ý:** Phỏng vấn trực tiếp thường đánh giá khả năng biến đổi ngắn gọn ngay trên bảng (như đạo hàm hàm sigmoid, công thức kích thước đầu ra tầng tích chập, hay một bước cập nhật của Adam).

**Thảo luận:** Hãy bấm giờ khi làm các câu trong bài luyện tập bổ trợ, còn cuốn sách dùng để đào sâu và chuẩn hóa cách diễn đạt. Nếu một chủ đề chỉ giải được khi mở PDF, nghĩa là bạn chưa đủ sẵn sàng cho vòng phỏng vấn trực tiếp.

### H3. Tổ hợp mô hình khi khóa không có chương riêng

Trong một vòng tuyển dụng, bạn được yêu cầu giảm phương sai của một mạng ResNet đã tinh chỉnh tốt mà không thay đổi kiến trúc. Hãy nêu hai cách *không phải* là "thu thập thêm dữ liệu có nhãn".

**Gợi ý:** Một cách thay đổi cách sử dụng mô hình sẵn có; cách còn lại thay đổi cách sử dụng dữ liệu sẵn có.

**Thảo luận:** Có thể áp dụng *snapshot ensemble* hoặc *bagging* nhiều bản tinh chỉnh, tăng cường dữ liệu lúc suy luận (*test-time augmentation*), hoặc dùng *dropout* nhiều lần lúc kiểm thử như một dạng tổ hợp. Việc tăng kích thước mô hình làm thay đổi dung lượng chứ không phải tổ hợp giảm phương sai. Các bài toán tổ hợp mô hình có lời giải chi tiết hơn nằm trong sách, không lưu trong kho này.

## Danh mục các bài luyện tập bổ trợ

Sau khi hoàn thành bài lý thuyết tương ứng, bạn có thể mở các bài sau:

- **00-98** — lý thuyết thông tin, giải tích và vi phân tự động (*autodiff*).
- **02-98** — nơ-ron, hàm kích hoạt và cấu trúc MLP.
- **03-98** — vi phân tự động, trực giác Hessian và phân loại logistic.
- **04-98** — hình học phép tích chập và kiến trúc CNN.
- **09-98** — quá khớp (*overfitting*) và các kỹ thuật chính quy hóa.
- **10-98** — họ bộ tối ưu Adam.
- **15-98** — trích xuất đặc trưng và học chuyển giao.
- **25-98** — học sâu Bayesian và bất định.
- **26-06** — mục tổng hợp LLM (toán tiền huấn luyện, phục vụ, cơ chế nội tại, chưng cất; chỉ gồm câu hỏi gốc).

Mỗi bài bổ trợ gồm 3–6 câu hỏi **gốc** kèm gợi ý ngắn, không viết lại phần lý thuyết đã học.

## Ghi công và lưu ý bản quyền

Kashani, S., và Ivry, A. *Deep Learning Interviews*. arXiv:2201.00650. GitHub: [BoltzmannEntropy/interviews.ai](https://github.com/BoltzmannEntropy/interviews.ai).

Ngân hàng câu hỏi có lời giải thuộc quyền tác giả của cuốn sách. Khóa học này chỉ dẫn tiêu đề, tên tác giả, mã arXiv và tên chủ đề khái quát, đồng thời bổ sung bộ câu hỏi luyện tập riêng. **Người học tự tải sách từ arXiv** (hoặc trang GitHub / interviews.ai chính thức); không đưa tệp PDF vào kho lưu trữ này và không dán các đoạn trích nguyên văn lớn từ sách vào *pull request*.

## Video tham khảo

- **Xem trước:** [MIT Introduction to Deep Learning (2024) | 6.S191](https://www.youtube.com/watch?v=ErnWZxJovaM) — Alexander Amini, tiếng Anh. Cung cấp khung đề cương tổng thể tương ứng với bản đồ các chủ đề phỏng vấn trong lộ trình này.
- [\[1hr Talk\] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Andrej Karpathy, tiếng Anh. Bổ sung mảng kiến thức hiện đại về mô hình ngôn ngữ lớn — phần chưa có mặt khi các tài liệu phỏng vấn trước đây được biên soạn.
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy, tiếng Anh. Hướng dẫn tự triển khai chi tiết thuật toán lan truyền ngược, hỗ trợ trực tiếp cho kỹ năng tự suy diễn công thức khi phỏng vấn.
