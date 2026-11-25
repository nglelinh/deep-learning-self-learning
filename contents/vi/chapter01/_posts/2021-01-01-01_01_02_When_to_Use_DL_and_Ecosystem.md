---
layout: post
title: 01-01-02 Khi nào dùng học sâu và hệ sinh thái
chapter: '01'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter01
---

Có một câu nói nổi tiếng trong giới kỹ sư dữ liệu: *"Đừng dùng pháo đại bác để bắn chim sẻ."* Dù Học sâu đã chinh phục cờ vây, gập protein và ngôn ngữ tự nhiên, hàng nghìn hệ thống sản xuất thực tế trong ngân hàng, logistics và thương mại điện tử vẫn chạy nhanh hơn, rẻ hơn và chính xác hơn với hồi quy tuyến tính hoặc Gradient Boosted Trees. Bài học này cung cấp bộ khung ra quyết định thực dụng giữa Học sâu và Học máy truyền thống, đồng thời giới thiệu hệ sinh thái phần mềm—phần cứng (PyTorch, JAX, TensorFlow, CUDA, Hugging Face) đang vận hành ngành AI hiện nay.

## Kiến thức cần nhớ: Bốn trục đánh giá khi Chọn Mô hình và Hệ sinh thái

1. **Bản chất Dữ liệu (Phi cấu trúc vs. Bảng biểu)**:
   Ảnh, âm thanh, văn bản, đồ thị phức tạp $$\to$$ ưu tiên Deep Learning; bảng dữ liệu có cấu trúc với đặc trưng hỗn hợp (số + phân loại) cỡ nhỏ/trung bình $$\to$$ ưu tiên XGBoost / LightGBM / CatBoost.

2. **Ràng buộc Độ trễ, Chi phí Phần cứng và Khả năng Giải thích**:
   Triển khai trên vi điều khiển (Edge) hay ngành tài chính/y tế đòi hỏi giải trình từng quyết định cần cân nhắc mô hình gọn nhẹ, giải thích được, hoặc áp dụng chưng cất (*distillation*) và lượng tử hóa (*quantization*).

3. **Hạ tầng Tăng tốc Phần cứng (CPU vs. GPU / TPU)**:
   CPU có ít nhân mạnh tối ưu cho xử lý tuần tự độ trễ thấp; GPU/TPU có hàng nghìn nhân tính toán song song thiết kế riêng cho phép nhân ma trận dày đặc (GEMM) và Tensor Cores.

4. **Hệ sinh thái Phần mềm Hiện đại**:
   PyTorch thống trị nghiên cứu và phần lớn sản xuất nhờ đồ thị động trực quan; JAX mạnh về tính toán khoa học và biên dịch XLA; Hugging Face và ONNX/TensorRT kết nối mượt mà từ mã nguồn tới triển khai thực tế.

### Giải thích: Tính tay tốc độ tính toán (FLOPs) và lý do GPU bỏ xa CPU trong Deep Learning

Xét một bước nhân ma trận đơn giản trong một lớp tuyến tính với mini-batch $$B = 64$$, chiều vào $$d_{\text{in}} = 1024$$, chiều ra $$d_{\text{out}} = 1024$$:

1. **Đếm số phép toán dấu phẩy động (FLOPs)**:
   Mỗi phần tử đầu ra cần $$1024$$ phép nhân và $$1024$$ phép cộng ($$\approx 2 \times 1024$$ FLOPs).
   Với $$64 \times 1024$$ phần tử đầu ra, tổng số phép toán là:
   $$\text{FLOPs} = 2 \times 64 \times 1024 \times 1024 = 134{,}217{,}728 \approx 0.134\text{ GFLOPs}.$$
2. **So sánh thời gian lý thuyết giữa CPU và GPU**:
   - Một CPU văn phòng (~50 GFLOPs thực tế cho `fp32`): mất $$\frac{0.134}{50} \approx 2.68\text{ ms}$$ cho riêng 1 lớp.
   - Một GPU hiện đại có Tensor Cores (~100{,}000 GFLOPs = 100 TFLOPs ở `bf16`/`fp16`): chỉ mất $$\frac{0.134}{100{,}000} \approx 0.00134\text{ ms}$$—nhanh hơn về mặt tính toán tới **2{,}000 lần**!
3. **Ý nghĩa thực tế**: Khi mạng có 100 lớp và huấn luyện trên 1 triệu mẫu, chênh lệch đó chính là khác biệt giữa **chạy 1 buổi chiều** và **chạy suốt 1 năm**.

## Khi nào nên dùng học sâu

### Học sâu phát huy thế mạnh khi:

- ✅ Có **lượng dữ liệu lớn**
- ✅ Cần học các **mẫu phức tạp**
- ✅ Dữ liệu đầu vào có **số chiều cao** (ảnh, văn bản, âm thanh)
- ✅ Muốn **học từ đầu đến cuối (*end-to-end*)**
- ✅ Có **đủ tài nguyên tính toán**
- ✅ Dữ liệu chứa các **quan hệ phi tuyến**

### Học máy truyền thống thường phù hợp hơn khi:

- ⚠️ Tập dữ liệu nhỏ (< 1.000 mẫu)
- ⚠️ Yêu cầu cao về khả năng diễn giải (chẳng hạn một số quyết định hỗ trợ chẩn đoán y khoa)
- ⚠️ Tài nguyên tính toán hạn chế
- ⚠️ Bài toán đơn giản và đã hiểu rõ cấu trúc
- ⚠️ Cần thời gian huấn luyện nhanh
- ⚠️ Mô hình tuyến tính hoặc đặc trưng đơn giản đã đáp ứng đủ

## Một số dấu mốc ứng dụng tiêu biểu

### Thị giác máy tính: ImageNet (2012)

**AlexNet** đạt tỷ lệ lỗi 15,3%, giảm mạnh so với mức khoảng 26% của các phương pháp truyền thống:
- Đánh dấu bước ngoặt của học sâu trong bài toán thị giác máy tính.
- Thúc đẩy làn sóng nghiên cứu và ứng dụng mạng nơ-ron sâu.
- Gồm kiến trúc CNN 8 tầng với khoảng 60 triệu tham số.

### Xử lý ngôn ngữ tự nhiên: dịch máy

**Google Neural Machine Translation (2016):**
- Giảm sai sót dịch thuật khoảng 60% so với hệ thống trước đó.
- Vượt trội hơn các hệ thống dịch máy thống kê dựa trên cụm từ.
- Nâng chất lượng bản dịch tiến gần hơn tới bản dịch của con người.

### Trò chơi: AlphaGo (2016)

- Giành chiến thắng 4–1 trước kỳ thủ cờ vây hàng đầu thế giới Lee Sedol.
- Kết hợp học sâu với thuật toán tìm kiếm cây Monte Carlo (*Monte Carlo Tree Search*).
- Chinh phục cờ vây — trò chơi có không gian trạng thái lớn hơn rất nhiều so với cờ vua.
- Đưa ra nhiều nước đi mới mẻ ngoài các khuôn mẫu truyền thống.

### Y tế: phân tích ảnh y khoa

**Phát hiện ung thư da (2017):**
- Mô hình học sâu đạt hiệu năng tương đương bác sĩ da liễu trên tập đánh giá.
- Phân tích trực tiếp ảnh soi da (*dermoscopy*).
- Mở ra hướng hỗ trợ sàng lọc chuyên sâu ở quy mô rộng hơn.

### Tiếng nói: trợ lý ảo

- Độ chính xác nhận dạng tiếng nói tiệm cận con người trong nhiều điều kiện.
- Là thành phần cốt lõi trong Siri, Alexa và Google Assistant.
- Hỗ trợ nhiều giọng đọc và ngôn ngữ khác nhau.

### Hệ thống gợi ý: luồng video ngắn (kiểu TikTok)

![Thuật toán gợi ý và mạng neuron](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_tiktok_title_nn.jpg)
*Hình: Hệ thống gợi ý nội dung quy mô lớn dựa trên học máy và học sâu. (Minh họa từ video giải thích hệ thống gợi ý)*

Đây là một ứng dụng quen thuộc hằng ngày của học máy và học sâu: tự động chọn video hoặc sản phẩm phù hợp cho từng người dùng. Một quy trình gợi ý điển hình thường gồm các thành phần sau:

1. **Lọc cộng tác (*collaborative filtering*):** Nếu nhiều người xem mục A cũng thích mục B, hệ thống sẽ gợi ý B cho người xem A mà không cần phân tích sâu nội dung của từng mục.

![Lọc cộng tác: xem A gợi ý B](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_collaborative_filtering.jpg)
*Hình: Khi người dùng xem mục A, hệ thống gợi ý các mục liên quan (B, …). (Minh họa từ video giải thích hệ thống gợi ý)*

2. **Gợi ý theo nội dung (*content-based*):** So sánh đặc trưng của các mục (thẻ gắn, phong cách, mức giá, …) để tìm những mục có thuộc tính tương đồng.

![Content-based: item A vs B](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_content_based_similar.jpg)
*Hình: Tìm các mục tương tự dựa trên thuộc tính nội dung. (Minh họa từ video giải thích hệ thống gợi ý)*

![Gán tag đặc trưng item](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_item_feature_tags.jpg)
*Hình: Mỗi mục được biểu diễn thông qua tập thẻ và đặc trưng. (Minh họa từ video giải thích hệ thống gợi ý)*

3. **Khởi động lạnh (*cold start*):** Khi người dùng hoặc nội dung mới chưa có lịch sử tương tác, lọc cộng tác kém hiệu quả nên hệ thống cần dựa vào đặc trưng nội dung, mức độ phổ biến chung hoặc khảo sát ngắn ban đầu.

![Cold start: chưa có dữ liệu](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_cold_start.jpg)
*Hình: Trạng thái chưa có lịch sử tương tác — bài toán khởi động lạnh (*cold start*). (Minh họa từ video giải thích hệ thống gợi ý)*

4. **Xếp hạng và tín hiệu tương tác:** Hệ thống sử dụng các tín hiệu như lượt thích, thời lượng xem hoặc hành động bỏ qua để chọn ra ứng viên phù hợp nhất trong thời gian rất ngắn.

![Ứng viên và tín hiệu engagement](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_engagement_candidates.jpg)
*Hình: Từ nhiều video ứng viên, hệ thống xếp hạng và chọn nội dung phù hợp theo hành vi xem. (Minh họa từ video giải thích hệ thống gợi ý)*

Đây là ví dụ rõ nét về học sâu ở quy mô hàng tỷ người dùng: dữ liệu rất lớn, phản hồi theo thời gian thực và phải cân bằng đồng thời nhiều mục tiêu (mức độ gắn kết, tính đa dạng và an toàn nội dung). Chi tiết về *embedding* và kiến trúc *two-tower* được trình bày ở Chương 18; cơ chế khám phá như ε-greedy nằm ở Chương 20.

## Lợi thế về dữ liệu

### Vì sao thêm dữ liệu giúp cải thiện mô hình

Học máy truyền thống thường bão hòa sau một ngưỡng dữ liệu:
```
Performance
    |     _______________
    |    /
    |   /
    |  /
    |_/________________
         Amount of Data
```

Học sâu có khả năng tận dụng lượng dữ liệu ngày càng lớn:
```
Performance
    |              /
    |            /
    |          /
    |        /
    |      /
    |    /
    |__/________________
         Amount of Data
```

### Quy luật mở rộng (*scaling law*)

Quan sát thực nghiệm cho thấy sai số của mô hình học sâu thường giảm theo quy luật lũy thừa của kích thước dữ liệu:

$$\text{Error} \propto \frac{1}{(\text{Data size})^\alpha}$$

trong đó $$\alpha \approx 0.5$$ trên nhiều tác vụ.

<div class="textbook-theorem" markdown="1">
**Hệ quả:**
- Tăng dữ liệu gấp 4 lần → giảm sai số khoảng 2 lần.
- Tăng dữ liệu gấp 100 lần → giảm sai số khoảng 10 lần.
</div>

## Yêu cầu về tính toán

### Vai trò của GPU

Việc huấn luyện mạng nơ-ron sâu trở nên khả thi trong thực tế phần lớn nhờ khả năng tính toán song song của GPU:

| Phép toán | Thời gian CPU | Thời gian GPU | Mức tăng tốc |
|-----------|---------------|---------------|--------------|
| Nhân ma trận (1000×1000) | 100ms | 5ms | 20× |
| Tầng Conv2D | 1000ms | 10ms | 100× |
| Huấn luyện mô hình đầy đủ | Ngày/Tuần | Giờ/Ngày | 10–100× |

### Hạ tầng tính toán hiện đại

- **Điện toán đám mây:** AWS, Google Cloud, Azure.
- **Phần cứng chuyên dụng:** TPU và các bộ xử lý mạng nơ-ron (*Neural Processing Units*).
- **Huấn luyện phân tán:** chạy song song trên nhiều GPU và nhiều máy chủ.
- **Độ chính xác hỗn hợp (*mixed precision*):** sử dụng định dạng FP16 để tăng tốc độ huấn luyện và tiết kiệm bộ nhớ.

## Quy trình xây dựng hệ thống học sâu

Một dự án học sâu điển hình trải qua 7 bước lặp lại:

1. **Thu thập dữ liệu:** xây dựng tập dữ liệu đủ lớn, đảm bảo chất lượng và độ đa dạng.
2. **Chuẩn bị dữ liệu:** làm sạch, tiền xử lý, chia tập huấn luyện/kiểm định/kiểm thử (*train/val/test*) và tăng cường dữ liệu (*data augmentation*) khi cần.
3. **Thiết kế mô hình:** lựa chọn kiến trúc, định nghĩa các tầng, kết nối và thiết lập siêu tham số.
4. **Huấn luyện:** khởi tạo tham số, chạy chu trình lan truyền xuôi → tính mất mát → lan truyền ngược → cập nhật trọng số và theo dõi chỉ số đánh giá.
5. **Đánh giá:** đo hiệu năng trên tập giữ lại, phân tích lỗi sai và trực quan hóa dự đoán.
6. **Lặp và cải tiến:** cải thiện chất lượng dữ liệu, điều chỉnh kiến trúc và tinh chỉnh siêu tham số.
7. **Triển khai:** tối ưu tốc độ suy luận, đưa vào môi trường vận hành (*production*) và giám sát hiệu năng liên tục.

## Công cụ và framework

### Các framework học sâu phổ biến

- **PyTorch:** hỗ trợ đồ thị tính toán động, giao diện gần gũi với Python và rất phổ biến trong nghiên cứu lẫn thực tế.
- **TensorFlow/Keras:** hệ sinh thái phong phú, sẵn sàng cho triển khai sản xuất (*production*) với nhiều công cụ hỗ trợ.
- **JAX:** tiếp cận theo phong cách lập trình hàm, linh hoạt, hiệu năng cao và ngày càng được dùng rộng rãi trong nghiên cứu.

### Thư viện hỗ trợ

- **NumPy:** tính toán mảng số học.
- **Pandas:** xử lý và phân tích dữ liệu dạng bảng.
- **Matplotlib/Seaborn:** trực quan hóa dữ liệu và kết quả.
- **Scikit-learn:** tiền xử lý dữ liệu, mô hình học máy cổ điển và các chỉ số đánh giá.

## Sách khuyến nghị cho khóa học

Để củng cố kiến thức, bạn có thể tham khảo các tài liệu sau:

### 1. **Deep Learning** — Ian Goodfellow, Yoshua Bengio và Aaron Courville
- Sách giáo khoa chuẩn mực, bao quát toàn diện cả nền tảng toán học lẫn lý thuyết học sâu.
- Có bản đọc miễn phí trực tuyến tại [deeplearningbook.org](http://www.deeplearningbook.org/).
- **Phù hợp cho:** củng cố nền tảng toán học và chiều sâu lý thuyết.

### 2. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** — Aurélien Géron
- Hướng dẫn thực hành từng bước với nhiều ví dụ mã nguồn và dự án hoàn chỉnh sử dụng Scikit-Learn, TensorFlow và Keras.
- **Phù hợp cho:** kỹ năng cài đặt và triển khai dự án thực tế.

### 3. **Understanding Deep Learning** — Simon J.D. Prince
- Tài liệu nhập môn hiện đại, giải thích rõ ràng bằng trực giác và hình minh họa chất lượng, bao gồm cả các kiến trúc mới như Transformer.
- **Phù hợp cho:** xây dựng trực giác và hiểu khái niệm một cách trực quan.

### 4. **MIT Deep Learning Book**
- Cách tiếp cận học thuật chặt chẽ, nhấn mạnh vào chứng minh, suy diễn toán học và góc nhìn nghiên cứu.
- **Phù hợp cho:** đào sâu lý thuyết và chuẩn bị cho công việc nghiên cứu.

### 5. **Deep Learning Interviews** — Shlomo Kashani và Amir Ivry
- Tuyển tập bài toán phỏng vấn và thi cử kèm lời giải miễn phí.
- PDF: [arXiv:2201.00650](https://arxiv.org/abs/2201.00650) — GitHub: [interviews.ai](https://github.com/BoltzmannEntropy/interviews.ai).
- Khóa học này ánh xạ các *chủ đề* trong sách (không sao chép lại câu hỏi) sang các bài `*-98` tùy chọn.
- **Phù hợp cho:** luyện phỏng vấn sau khi học xong phần lý thuyết tương ứng; bắt đầu từ bài **01-98**.

### Gợi ý lộ trình đọc sách

**Người mới bắt đầu:**
1. Đọc *Understanding Deep Learning* để xây dựng trực giác.
2. Học theo cấu trúc của khóa học này.
3. Tra cứu *Hands-On ML* khi thực hành cài đặt.
4. Đọc sâu *Deep Learning* để củng cố lý thuyết.

**Trình độ trung cấp:**
1. Dùng khóa học này làm khung chính.
2. Tham khảo *Hands-On ML* để nắm kinh nghiệm triển khai.
3. Đọc các chương tương ứng trong *Deep Learning* để đào sâu.
4. Dùng *Understanding DL* khi cần làm rõ trực giác.

**Trình độ nâng cao:**
1. Dùng khóa học này để rà soát hệ thống kiến thức.
2. Nghiên cứu *Deep Learning* cho phần lý thuyết chặt chẽ.
3. Áp dụng kỹ thuật thực hành từ *Hands-On ML*.
4. Tham khảo thêm sách MIT cho các chi tiết hướng nghiên cứu.

## Xu hướng hiện tại (2024–2025)

### Mô hình ngôn ngữ lớn (LLM)
- Các họ mô hình như GPT-4, Claude, Gemini.
- Quy mô từ hàng tỷ đến hàng nghìn tỷ tham số.
- Xuất hiện nhiều năng lực nổi (*emergent capabilities*) khi tăng quy mô.

### Mô hình đa phương thức (*multimodal*)
- CLIP kết hợp thị giác và ngôn ngữ.
- GPT-4V xử lý đồng thời văn bản và hình ảnh.
- Hướng tới biểu diễn thống nhất qua nhiều dạng dữ liệu (*modality*).

### AI hiệu quả
- Nén mô hình và lượng tử hóa (*quantization*).
- Tìm kiếm kiến trúc mạng (*neural architecture search*).
- Triển khai trên thiết bị biên (*edge devices*).

### Mô hình nền tảng (*foundation models*)
- Tiền huấn luyện trên tập dữ liệu rất lớn.
- Tinh chỉnh (*fine-tune*) cho từng tác vụ cụ thể.
- Ứng dụng học chuyển giao ở quy mô rộng.

## Thách thức phía trước

### Thách thức kỹ thuật
- **Hiệu quả mẫu:** học tốt từ ít dữ liệu hơn.
- **Độ bền vững:** duy trì hiệu năng khi phân phối dữ liệu thay đổi (*domain shift*).
- **Khả năng diễn giải:** hiểu rõ cơ sở đằng sau quyết định của mô hình.
- **Chi phí tính toán:** giảm chi phí cho cả huấn luyện lẫn suy luận.

### Thách thức xã hội
- Thiên lệch thuật toán và tính công bằng.
- Bảo vệ quyền riêng tư dữ liệu.
- Tác động môi trường từ tiêu thụ năng lượng.
- Ảnh hưởng đến thị trường lao động.
- Rủi ro thông tin sai lệch (như *deepfake*).

## Tóm tắt

Học sâu đạt hiệu quả cao nhờ:
- ✅ Học đặc trưng tự động.
- ✅ Mở rộng tốt theo quy mô dữ liệu và tính toán.
- ✅ Xử lý hiệu quả dữ liệu nhiều chiều.
- ✅ Đạt kết quả hàng đầu trên nhiều bài toán phức tạp.
- ✅ Hỗ trợ học từ đầu đến cuối (*end-to-end*).

Nên cân nhắc dùng học sâu khi:
- Có tập dữ liệu lớn.
- Bài toán chứa các mẫu phức tạp.
- Có đủ tài nguyên tính toán.
- Yêu cầu cao về độ chính xác.

Lĩnh vực này tiếp tục phát triển nhanh với nhiều kiến trúc, kỹ thuật và ứng dụng mới.

**Tiếp theo:** Chương sau sẽ đi sâu vào nền tảng mạng nơ-ron để hiểu cơ chế hoạt động bên trong của chúng.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ video của kênh *Machine Learning Thực Chiến*. Nguồn gốc để tra cứu:

- [Thuật toán gợi ý (TikTok-style recommendation)](https://www.facebook.com/reel/1444915507374636)

## Video tham khảo

- **Xem trước:** [AI, Machine Learning, Deep Learning and Generative AI Explained](https://www.youtube.com/watch?v=qYNweeDHiyU) — IBM Technology, tiếng Anh. Giúp xác định một bài toán thực sự cần học sâu hay chỉ cần học máy cổ điển.
- [MIT Introduction to Deep Learning (2024) | 6.S191](https://www.youtube.com/watch?v=ErnWZxJovaM) — Alexander Amini, tiếng Anh. Giới thiệu các framework và yêu cầu quy mô dữ liệu gắn với phần hệ sinh thái trong bài.
- [Machine Learning vs Deep Learning](https://www.youtube.com/watch?v=q6kJ71tEYqM) — IBM Technology, tiếng Anh. Cung cấp tiêu chí đối chiếu nhanh trước khi quyết định dùng mô hình sâu thay cho mô hình đơn giản hơn.
