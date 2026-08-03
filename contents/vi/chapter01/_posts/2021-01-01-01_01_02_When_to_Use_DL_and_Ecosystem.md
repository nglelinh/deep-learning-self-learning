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

## Khi nào nên dùng học sâu

### Học sâu mạnh khi:

✅ Có **lượng dữ liệu lớn**  
✅ Cần học **mẫu phức tạp**  
✅ Đầu vào **chiều cao** (ảnh, văn bản, âm thanh)  
✅ Muốn **học đầu-cuối**  
✅ Có **đủ tài nguyên tính toán**  
✅ Dữ liệu chứa **quan hệ phi tuyến**

### Học máy truyền thống có thể tốt hơn khi:

⚠️ Tập dữ liệu nhỏ (< 1000 mẫu)  
⚠️ Cần khả năng diễn giải (quyết định chẩn đoán y khoa)  
⚠️ Tài nguyên tính toán hạn chế  
⚠️ Bài toán đơn giản, đã hiểu rõ  
⚠️ Cần huấn luyện nhanh  
⚠️ Quan hệ tuyến tính là đủ

## Câu chuyện thành công

### Thị giác máy tính: ImageNet (2012)

**AlexNet** đạt tỷ lệ lỗi 15,3% (so với 26% của phương pháp truyền thống)
- Chiến thắng học sâu đầu tiên trong thị giác máy tính
- Khơi mào cuộc cách mạng học sâu
- CNN 8 tầng với 60 triệu tham số

### Ngôn ngữ tự nhiên: dịch máy

**Google Neural Machine Translation (2016)**
- Giảm lỗi dịch khoảng 60%
- Học dịch tốt hơn hệ thống dựa trên cụm từ
- Cho phép dịch gần chất lượng con người

### Game: AlphaGo (2016)

- Thắng kỳ thủ số một thế giới Lee Sedol 4–1
- Kết hợp học sâu với tìm kiếm cây Monte Carlo
- Làm chủ cờ vây — được coi là khó hơn cờ vua nhiều
- Thể hiện lối chơi sáng tạo, trực giác

### Y tế: ảnh y khoa

**Phát hiện ung thư da (2017)**
- Học sâu ngang hiệu năng bác sĩ da liễu
- Phân tích ảnh dermoscopy
- Tiềm năng dân chủ hóa chẩn đoán mức chuyên gia

### Tiếng nói: trợ lý ảo

- Độ chính xác nhận dạng tiếng nói gần con người
- Vận hành Siri, Alexa, Google Assistant
- Hoạt động qua nhiều giọng và ngôn ngữ

### Hệ thống gợi ý: feed video ngắn (TikTok-style)

![Thuật toán gợi ý và mạng neuron](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_tiktok_title_nn.jpg)
*Hình: Gợi ý nội dung quy mô lớn dựa trên học máy / học sâu. (Minh họa từ video giải thích recommendation system)*

Ứng dụng “hàng ngày” của ML/DL: feed tự động chọn video/sản phẩm phù hợp từng người. Pipeline điển hình gồm vài tầng:

1. **Lọc cộng tác (*collaborative filtering*)** — người xem item A thường cũng thích item B → gợi ý B (không cần hiểu “nội dung” là gì).

![Lọc cộng tác: xem A gợi ý B](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_collaborative_filtering.jpg)
*Hình: Khi có ai xem A, hệ thống gợi ý các item liên quan (B, …). (Minh họa từ video giải thích recommendation system)*

2. **Gợi ý theo nội dung (*content-based*)** — so đặc trưng item (tag, phong cách, giá, …) để tìm “giống nhau”.

![Content-based: item A vs B](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_content_based_similar.jpg)
*Hình: Tìm item tương tự theo thuộc tính (minh họa bằng icon sản phẩm). (Minh họa từ video giải thích recommendation system)*

![Gán tag đặc trưng item](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_item_feature_tags.jpg)
*Hình: Item được biểu diễn bằng tập tag/feature. (Minh họa từ video giải thích recommendation system)*

3. **Cold start** — người dùng hoặc item mới chưa có lịch sử → CF yếu, cần content-based, phổ biến toàn cục, hoặc survey ngắn.

![Cold start: chưa có dữ liệu](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_cold_start.jpg)
*Hình: Trang trống khi chưa có lịch sử — bài toán cold start. (Minh họa từ video giải thích recommendation system)*

4. **Xếp hạng & tín hiệu tương tác** — like, thời gian xem, skip… chọn ứng viên “ăn khớp” trong thời gian rất ngắn.

![Ứng viên và tín hiệu engagement](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_engagement_candidates.jpg)
*Hình: Từ nhiều video ứng viên, hệ thống chọn cái phù hợp theo hành vi xem. (Minh họa từ video giải thích recommendation system)*

Đây là ví dụ học sâu “đụng” hàng tỷ người dùng: dữ liệu lớn, phản hồi realtime, nhiều mục tiêu (giữ chân, đa dạng, an toàn). Chi tiết embedding / two-tower ở Chương 18; ε-greedy / khám phá ở Chương 20.

## Lợi thế dữ liệu

### Vì sao nhiều dữ liệu hơn giúp ích

Học máy truyền thống:
```
Performance
    |     _______________
    |    /
    |   /
    |  /
    |_/________________
         Amount of Data
```

Học sâu:
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

### Định luật scaling

Quan sát thực nghiệm: hiệu năng học sâu thường tuân theo:

$$\text{Error} \propto \frac{1}{(\text{Data size})^\alpha}$$

trong đó $$\alpha \approx 0.5$$ với nhiều tác vụ.

**Hệ quả**:
- Dữ liệu ×4 → giảm lỗi khoảng ×2
- Dữ liệu ×100 → giảm lỗi khoảng ×10

## Yêu cầu tính toán

### Cuộc cách mạng GPU

Học sâu trở nên thực tế nhờ GPU:

| Phép toán | Thời gian CPU | Thời gian GPU | Tăng tốc |
|-----------|---------------|---------------|----------|
| Nhân ma trận (1000×1000) | 100ms | 5ms | 20× |
| Tầng Conv2D | 1000ms | 10ms | 100× |
| Huấn luyện mô hình đầy đủ | Ngày/Tuần | Giờ/Ngày | 10–100× |

### Hạ tầng hiện đại

- **Điện toán đám mây**: AWS, Google Cloud, Azure
- **Phần cứng chuyên biệt**: TPU, Neural Processing Units
- **Huấn luyện phân tán**: multi-GPU, multi-machine
- **Mixed precision**: huấn luyện FP16 để tăng tốc

## Quy trình học sâu

### Quy trình điển hình

1. **Thu thập dữ liệu**
   - Thu thập tập dữ liệu lớn
   - Đảm bảo chất lượng và đa dạng

2. **Chuẩn bị dữ liệu**
   - Làm sạch và tiền xử lý
   - Chia train/val/test
   - Tăng cường dữ liệu nếu cần

3. **Thiết kế mô hình**
   - Chọn kiến trúc
   - Định nghĩa tầng và kết nối
   - Thiết lập siêu tham số

4. **Huấn luyện**
   - Khởi tạo tham số
   - Lan truyền xuôi → mất mát → lan truyền ngược
   - Cập nhật trọng số
   - Theo dõi metric

5. **Đánh giá**
   - Kiểm tra trên dữ liệu giữ lại
   - Phân tích lỗi
   - Trực quan hóa dự đoán

6. **Lặp cải tiến**
   - Cải thiện dữ liệu
   - Tinh chỉnh kiến trúc
   - Điều chỉnh siêu tham số

7. **Triển khai**
   - Tối ưu cho suy luận
   - Đưa vào production
   - Giám sát hiệu năng

## Công cụ và framework

### Framework học sâu

**PyTorch**
- Đồ thị tính toán động
- Giao diện Pythonic
- Phổ biến trong nghiên cứu

**TensorFlow/Keras**
- Sẵn sàng cho production
- Hệ sinh thái phong phú
- Triển khai dễ dàng

**JAX**
- Tiếp cận hàm
- Nhanh và linh hoạt
- Đang được áp dụng rộng hơn

### Thư viện hỗ trợ

- **NumPy**: tính toán số
- **Pandas**: thao tác dữ liệu
- **Matplotlib/Seaborn**: trực quan hóa
- **Scikit-learn**: tiền xử lý, metric

## Sách khuyến nghị cho khóa học

Để đào sâu hiểu biết, các sách học sâu thiết yếu sau được khuyến nghị:

### 1. **Deep Learning** của Ian Goodfellow, Yoshua Bengio và Aaron Courville
- **Sách giáo khoa học sâu mang tính chuẩn mực**
- Bao quát toàn diện lý thuyết và toán học
- Viết bởi các nhà tiên phong trong lĩnh vực
- Miễn phí trực tuyến: [deeplearningbook.org](http://www.deeplearningbook.org/)
- **Dùng cho**: nền tảng toán, chiều sâu lý thuyết

### 2. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** của Aurélien Géron
- **Hướng dẫn triển khai thực hành tốt nhất**
- Ví dụ mã từng bước
- Dự án ML đầu-cuối
- Bao gồm Scikit-Learn, TensorFlow và Keras
- **Dùng cho**: triển khai, dự án thực tế

### 3. **Understanding Deep Learning** của Simon J.D. Prince
- **Giới thiệu hiện đại, dễ tiếp cận**
- Giải thích rõ với trực quan hóa xuất sắc
- Bao gồm kiến trúc gần đây (Transformer, v.v.)
- Tiếp cận trực giác với khái niệm phức tạp
- **Dùng cho**: xây dựng trực giác, hiểu trực quan

### 4. **MIT Deep Learning Book**
- **Xử lý học thuật nghiêm ngặt**
- Nền tảng lý thuyết vững
- Góc nhìn hướng nghiên cứu
- Chứng minh và suy diễn toán học
- **Dùng cho**: chiều sâu học thuật, chuẩn bị nghiên cứu

### Cách dùng các sách này

**Lộ trình người mới**:
1. Bắt đầu với "Understanding Deep Learning" để có trực giác
2. Theo khóa học này để học có cấu trúc
3. Tham chiếu "Hands-On ML" cho triển khai
4. Đọc sâu "Deep Learning" cho lý thuyết

**Lộ trình trung cấp**:
1. Dùng khóa học này làm hướng dẫn chính
2. Tham chiếu "Hands-On ML" cho mẹo thực hành
3. Đọc các chương "Deep Learning" để đào sâu
4. Tham khảo "Understanding DL" để làm rõ

**Lộ trình nâng cao**:
1. Dùng khóa học này cho bao quát toàn diện
2. Nghiên cứu "Deep Learning" cho lý thuyết nghiêm ngặt
3. Triển khai theo kỹ thuật "Hands-On ML"
4. Tham chiếu sách MIT cho chi tiết nghiên cứu

## Xu hướng hiện tại (2024–2025)

### Mô hình ngôn ngữ lớn (LLM)
- GPT-4, Claude, Gemini
- Hàng tỷ đến hàng nghìn tỷ tham số
- Năng lực nổi (*emergent capabilities*) ở quy mô lớn

### Mô hình đa phương thức
- CLIP: thị giác + ngôn ngữ
- GPT-4V: văn bản + ảnh
- Hiểu thống nhất qua các modality

### AI hiệu quả
- Nén mô hình
- Lượng tử hóa
- Neural architecture search
- Triển khai edge

### Mô hình nền tảng (*foundation models*)
- Tiền huấn luyện trên dữ liệu khổng lồ
- Fine-tune cho tác vụ cụ thể
- Chuyển giao học ở quy mô lớn

## Thách thức phía trước

### Thách thức kỹ thuật
- Hiệu quả mẫu (học từ ít dữ liệu hơn)
- Độ bền vững (xử lý domain shift)
- Khả năng diễn giải (hiểu quyết định)
- Chi phí tính toán (huấn luyện và suy luận)

### Thách thức xã hội
- Thiên lệch và công bằng
- Quyền riêng tư
- Tác động môi trường
- Thay thế việc làm
- Thông tin sai lệch (deepfake)

## Tóm tắt

Học sâu thành công vì:
- ✅ Học đặc trưng tự động
- ✅ Mở rộng theo dữ liệu và tính toán
- ✅ Xử lý đầu vào chiều cao
- ✅ Đạt kết quả state-of-the-art
- ✅ Cho phép học đầu-cuối

Nên dùng khi:
- Có tập dữ liệu lớn
- Tồn tại mẫu phức tạp
- Đủ tài nguyên tính toán
- Yêu cầu hiệu năng cao

Lĩnh vực tiếp tục tiến hóa nhanh với kiến trúc, kỹ thuật và ứng dụng mới liên tục xuất hiện.

**Tiếp theo**: ta sẽ đi sâu vào nền tảng mạng nơ-ron và hiểu cách chúng thực sự hoạt động.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Thuật toán gợi ý (TikTok-style recommendation)](https://www.facebook.com/reel/1444915507374636)
