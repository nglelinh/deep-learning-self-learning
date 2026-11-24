---
layout: post
title: 17-01-01 Lý thuyết Phát hiện Đối tượng
chapter: '17'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter17
---

Phát hiện đối tượng không đơn thuần là ghép bài toán phân loại với bài toán hồi quy 4 tọa độ: khi một bức ảnh có thể chứa 0, 3 hoặc 47 vật thể với đủ kích thước từ chú kiến nhỏ xíu đến chiếc xe buýt choán gần hết khung hình, mạng nơ-ron sẽ phải dự đoán bao nhiêu hộp bao? Bài lý thuyết này giải mã trọn bộ công cụ hình học giúp máy tính trả lời câu hỏi đó: **IoU**, **Anchor Boxes**, **hồi quy độ lệch log-scale**, **NMS** và thước đo chuẩn vàng **mAP**.

## Bản đồ kiến thức nền: Cơ sở toán học của Phát hiện Đối tượng

- **Intersection over Union (IoU) và các biến thể GIoU/CIoU:** Thước đo bất biến tỷ lệ giữa hộp dự đoán $$B_p$$ và hộp thật $$B_{gt}$$, làm nền tảng cho cả gán nhãn huấn luyện lẫn hàm mất mát hộp bao.
- **Tham số hóa hộp bao theo Anchor (Bounding Box Regression):** Dự đoán độ lệch tương đối $$(t_x, t_y, t_w, t_h)$$ chuẩn hóa theo kích thước hộp neo và lấy logarit cho chiều rộng/cao để bất biến tỷ lệ và giữ kích thước luôn dương.
- **Hàm mất mát đa tác vụ và Focal Loss:** Cân bằng giữa loss phân loại (xử lý mất cân bằng cực độ nền/vật thể bằng Focal Loss) và Smooth L1 / IoU loss cho tọa độ.
- **Non-Maximum Suppression (NMS) và Mean Average Precision (mAP):** Hậu xử lý lọc hộp trùng lặp và tính diện tích dưới đường cong Precision–Recall ở các ngưỡng IoU (mAP@0.5, mAP@0.5:0.95).

### Khung giải thích (nên đọc chậm): Tính tay IoU của hai hộp bao và một bước lọc Non-Maximum Suppression (NMS)

Xét hộp nhãn thật $$B_{gt} = [x_1, y_1, x_2, y_2] = [10, 10, 50, 50]$$ và hai hộp dự đoán cho lớp `ô tô`: hộp A $$= [20, 20, 60, 60]$$ (điểm tin cậy $$0{,}92$$) và hộp B $$= [18, 18, 56, 56]$$ (điểm tin cậy $$0{,}75$$).

1. **Tính IoU giữa hộp A và hộp thật $$B_{gt}$$:**
   - Diện tích $$B_{gt}$$: $$(50-10)\times(50-10) = 1{,}600$$; diện tích A: $$(60-20)\times(60-20) = 1{,}600$$.
   - Vùng giao: $$[\max(10,20),\max(10,20),\min(50,60),\min(50,60)] = [20, 20, 50, 50]$$, diện tích giao $$= 30 \times 30 = 900$$.
   - Diện tích hợp: $$1{,}600 + 1{,}600 - 900 = 2{,}300$$.
   - $$\text{IoU}(A, B_{gt}) = \frac{900}{2{,}300} \approx 0{,}3913$$ (chưa đạt ngưỡng $$0{,}50$$ nên bị tính là False Positive ở mAP@0.5!).
2. **Chạy NMS với ngưỡng $$\tau_{\text{NMS}} = 0{,}5$$ giữa A và B:**
   - Chọn hộp có điểm cao nhất trước là A ($$0{,}92$$). Tính vùng giao giữa A và B: $$[20, 20, 56, 56]$$ có diện tích $$36 \times 36 = 1{,}296$$.
   - Diện tích B là $$38 \times 38 = 1{,}444$$, hợp $$A \cup B = 1{,}600 + 1{,}444 - 1{,}296 = 1{,}748$$.
   - $$\text{IoU}(A, B) = \frac{1{,}296}{1{,}748} \approx 0{,}7414 > 0{,}5$$ → **loại bỏ hộp B** vì trùng lặp với A!

# Phát hiện Đối tượng: Định vị và Nhận dạng

![Object Detection Example](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg/800px-Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg)
*Hình ảnh: Phát hiện đối tượng với YOLO — nhận dạng và định vị đồng thời nhiều vật thể. Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm

Phát hiện đối tượng (*object detection*) mở rộng bài toán phân loại ảnh từ câu hỏi “trong ảnh có đối tượng gì?” sang “có những đối tượng nào và chúng nằm ở đâu?”. Sự mở rộng tưởng như nhỏ này thực chất đòi hỏi giải đồng thời nhiều bài toán con liên kết với nhau: đề xuất các vùng có khả năng chứa đối tượng (*region proposal*), phân loại nội dung trong từng vùng (*recognition*), tinh chỉnh biên giới hạn (*localization*), và xử lý nhiều đối tượng thuộc nhiều lớp ở các tỷ lệ kích thước khác nhau (*multi-scale, multi-class detection*). Việc phối hợp các thành phần này trong khi vẫn đảm bảo tốc độ thời gian thực khiến phát hiện đối tượng trở thành một bài toán nhiều thách thức trong thị giác máy tính.

Quá trình phát triển của các phương pháp phát hiện đối tượng phản ánh bước chuyển từ thị giác máy tính cổ điển sang học sâu. Các phương pháp cổ điển dùng đặc trưng thiết kế thủ công (SIFT, HOG) kết hợp cửa sổ trượt (*sliding window*) quét qua mọi vị trí và tỷ lệ có thể, rồi phân loại từng cửa sổ bằng bộ phân loại như SVM. Cách này tiêu tốn nhiều tài nguyên tính toán (phải đánh giá hàng triệu cửa sổ trên mỗi ảnh) và bị giới hạn bởi chất lượng đặc trưng thủ công. Học sâu khắc phục hạn chế đó bằng đặc trưng học trực tiếp từ dữ liệu và các kiến trúc huấn luyện đầu–cuối, cải thiện đáng kể cả độ chính xác lẫn tốc độ.

Phát hiện đối tượng hiện đại gồm hai hướng kiến trúc chính. Bộ phát hiện hai giai đoạn (*two-stage detector*) như R-CNN, Fast R-CNN và Faster R-CNN trước tiên đề xuất các vùng có khả năng chứa đối tượng, sau đó mới phân loại và tinh chỉnh hộp bao cho các vùng đó. Việc tách riêng bước đề xuất vùng và bước nhận dạng giúp tập trung năng lực tính toán vào các vùng tiềm năng và đạt độ chính xác cao. Bộ phát hiện một giai đoạn (*single-stage detector*) như YOLO và SSD dự đoán trực tiếp hộp bao cùng xác suất lớp tại các vị trí trên lưới đều, bỏ qua giai đoạn đề xuất để đạt tốc độ thời gian thực, đánh đổi bằng độ chính xác thấp hơn một chút đối với các đối tượng nhỏ.

Để hiểu rõ cơ chế hoạt động của các hệ thống phát hiện đối tượng, cần nắm bốn thành phần kỹ thuật cốt lõi. Mạng đề xuất vùng (*Region Proposal Network*, RPN) học cách sinh đề xuất thay cho thuật toán thủ công, giúp toàn bộ quy trình khả vi. Hộp neo (*anchor box*) xử lý các đối tượng có kích thước và tỷ lệ khung hình khác nhau thông qua các hộp tham chiếu định sẵn. Ức chế không cực đại (*non-maximum suppression*, NMS) loại bỏ các phát hiện trùng lặp khi mô hình sinh ra nhiều hộp chồng lấn cho cùng một đối tượng. Mạng kim tự tháp đặc trưng (*Feature Pyramid Network*, FPN) hỗ trợ phát hiện đa tỷ lệ bằng cách xây dựng kim tự tháp đặc trưng giàu thông tin ngữ nghĩa ở mọi độ phân giải. Khi kết hợp lại, các thành phần này cho phép phát hiện và định vị hàng chục đối tượng thuộc nhiều lớp chỉ trong vài mili giây, ứng dụng trong xe tự lái, phân tích ảnh y khoa và thực tế tăng cường.

## 2. Nền tảng toán học

Phát hiện đối tượng cần mô tả chặt chẽ đầu ra cần dự đoán và thước đo đánh giá. Mỗi đối tượng được biểu diễn bằng bộ $$(\text{class}, x, y, w, h)$$, gồm nhãn lớp và hộp bao (tọa độ tâm $$x, y$$ cùng chiều rộng $$w$$ và chiều cao $$h$$). Với ảnh có $$N$$ đối tượng, nhãn thực tế (*ground truth*) là tập hợp $$\{(\text{class}_i, x_i, y_i, w_i, h_i)\}_{i=1}^N$$. Bộ phát hiện phải dự đoán tập hợp này trong điều kiện số lượng $$N$$ thay đổi tùy theo từng ảnh.

### Giao trên Hợp (Intersection over Union, IoU)

Chất lượng định vị được đo bằng chỉ số Giao trên Hợp (IoU) giữa hộp dự đoán và hộp nhãn thực tế:

$$\text{IoU}(\text{box}_{\text{pred}}, \text{box}_{\text{gt}}) = \frac{\text{Area}(\text{box}_{\text{pred}} \cap \text{box}_{\text{gt}})}{\text{Area}(\text{box}_{\text{pred}} \cup \text{box}_{\text{gt}})}$$

IoU nhận giá trị từ 0 (không chồng lấn) đến 1 (trùng khít hoàn toàn). Thông thường, một phát hiện được xem là đúng nếu $$\text{IoU} \geq 0.5$$ và nhãn lớp dự đoán khớp với nhãn thực tế. Ngưỡng này cân bằng giữa yêu cầu định vị chính xác và độ dung sai hợp lý cho vị trí hộp bao.

### Hồi quy hộp bao (Bounding Box Regression)

Thay vì dự đoán trực tiếp tọa độ tuyệt đối, các bộ phát hiện hiện đại dự đoán độ lệch so với hộp neo (hộp tham chiếu định sẵn). Cho hộp neo $$(\hat{x}, \hat{y}, \hat{w}, \hat{h})$$ và hộp nhãn thực tế $$(\bar{x}, \bar{y}, \bar{w}, \bar{h})$$, mục tiêu hồi quy được tham số hóa như sau:

$$t_x = \frac{\bar{x} - \hat{x}}{\hat{w}}, \quad t_y = \frac{\bar{y} - \hat{y}}{\hat{h}}$$

$$t_w = \log\frac{\bar{w}}{\hat{w}}, \quad t_h = \log\frac{\bar{h}}{\hat{h}}$$

Mạng dự đoán bộ giá trị $$(t_x, t_y, t_w, t_h)$$, sau đó giải mã ngược về tọa độ tuyệt đối:

$$x = \hat{x} + \hat{w} \cdot t_x, \quad y = \hat{y} + \hat{h} \cdot t_y$$

$$w = \hat{w} \cdot \exp(t_w), \quad h = \hat{h} \cdot \exp(t_h)$$

Cách tham số hóa này dễ học hơn dự đoán tọa độ trực tiếp vì các độ lệch thường có giá trị nhỏ và cùng thang đo, trong khi tọa độ tuyệt đối biến thiên trên toàn ảnh với thang đo rất khác nhau giữa đối tượng nhỏ và lớn.

### Hàm mất mát đa tác vụ (Multi-Task Loss)

Bộ phát hiện tối ưu đồng thời hàm mất mát phân loại và hàm mất mát định vị:

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{box}}$$

trong đó $$\mathcal{L}_{\text{cls}}$$ là hàm mất mát phân loại (cross-entropy) và $$\mathcal{L}_{\text{box}}$$ là hàm mất mát hồi quy hộp bao (smooth L1 hoặc hàm mất mát dựa trên IoU). Hệ số $$\lambda$$ cân bằng hai mục tiêu: nếu $$\lambda$$ quá lớn, mô hình ưu tiên định vị mà giảm chất lượng phân loại; nếu quá nhỏ, mô hình phân loại tốt nhưng hộp bao kém chính xác.

Trong Faster R-CNN, hàm mất mát phân loại dùng cross-entropy trên tất cả các lớp đối tượng cộng thêm lớp nền:

$$\mathcal{L}_{\text{cls}} = -\log p_{\text{class}}$$

trong đó $$\text{class}$$ là nhãn lớp thực tế (hoặc lớp nền nếu IoU < 0.5 với mọi hộp nhãn thực tế).

Hàm mất mát hồi quy hộp bao là smooth L1:

$$\mathcal{L}_{\text{box}} = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L1}(t_i - \hat{t}_i)$$

$$\text{smooth}_{L1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

Smooth L1 ít nhạy với các điểm ngoại lai hơn L2 (chuyển từ bậc hai sang tuyến tính khi sai số lớn) đồng thời khả vi tại mọi điểm (khác với L1 thuần túy).

### Mạng đề xuất vùng (Region Proposal Network, RPN)

Faster R-CNN đưa ra RPN — một mạng tích chập hoàn toàn dùng để dự đoán các vùng đề xuất chứa đối tượng. Tại mỗi vị trí trên bản đồ đặc trưng, RPN dự đoán:
- Điểm *objectness*: $$k$$ hộp neo × 2 giá trị (có đối tượng hay là nền).
- Độ tinh chỉnh hộp bao: $$k$$ hộp neo × 4 tọa độ.

Với bản đồ đặc trưng kích thước $$H \times W$$ và $$k=9$$ hộp neo tại mỗi vị trí, đầu ra của RPN gồm:
- Điểm *objectness*: $$H \times W \times 9 \times 2$$ giá trị.
- Độ lệch hộp bao: $$H \times W \times 9 \times 4$$ giá trị.

Tổng cộng có $$HW \times 9$$ đề xuất. Thuật toán NMS lọc lại còn khoảng 2000 đề xuất có điểm *objectness* cao nhất để đưa vào đầu phát hiện.

Hàm mất mát của RPN kết hợp cả phân loại (*objectness*) và hồi quy hộp bao:

$$\mathcal{L}_{\text{RPN}} = \frac{1}{N_{\text{cls}}}\sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \frac{\lambda}{N_{\text{box}}}\sum_i p_i^* \mathcal{L}_{\text{box}}(t_i, t_i^*)$$

trong đó $$p_i^* = 1$$ nếu hộp neo $$i$$ chồng lấn với hộp nhãn thực tế với IoU > 0.7 (mẫu dương), $$p_i^* = 0$$ nếu IoU < 0.3 (mẫu âm), và các trường hợp nằm giữa hai ngưỡng này bị bỏ qua để tránh gây nhiễu cho quá trình huấn luyện.

## 3. Ví dụ và trực giác

Giả sử bạn cần tìm và nhận diện tất cả mọi người trong một bức ảnh đông đúc. Cách làm tự nhiên thường gồm ba bước:

1. **Quét nhanh** các vùng có khả năng chứa người (dựa trên hình dáng đầu, thân).
2. **Xem xét kỹ** từng vùng tiềm năng (đây thực sự là người hay là tượng? là ai?).
3. **Tinh chỉnh** biên giới hạn (hộp bao của từng người bắt đầu và kết thúc chính xác ở đâu?).

Quy trình này phản ánh trực tiếp cơ chế của bộ phát hiện hai giai đoạn. RPN thực hiện bước quét nhanh, đề xuất khoảng 2000 vùng có khả năng chứa đối tượng (người, xe, chó, v.v.). Đầu phát hiện kiểm tra kỹ từng đề xuất để phân loại nội dung và tinh chỉnh hộp bao, sau đó NMS loại bỏ các hộp trùng lặp cho cùng một đối tượng.

Xét ví dụ phát hiện xe trên đường phố. Một khung hình có thể chứa:
- 3 chiếc xe ở các khoảng cách khác nhau (tương ứng với kích thước khác nhau).
- 2 người đi bộ.
- 1 biển báo giao thông.
- Nền phức tạp (nhà cửa, cây cối).

Bộ phát hiện một giai đoạn như YOLO chia ảnh thành một lưới (chẳng hạn 13×13). Mỗi ô lưới dự đoán:
- Nhiều hộp bao (ví dụ 3 hộp với các tỷ lệ khung hình khác nhau: cao, rộng, vuông).
- Xác suất lớp cho từng hộp.
- Điểm tin cậy (*confidence*) phản ánh khả năng có đối tượng tại ô đó.

Tại ô lưới (5, 8) nằm gần một chiếc xe, mô hình có thể dự đoán:
- Hộp 1: `class=car`, `confidence=0.95`, tọa độ lệch so với tâm ô.
- Hộp 2: `class=background`, `confidence=0.05`.
- Hộp 3: `class=background`, `confidence=0.02`.

Sau khi xử lý toàn bộ 13×13 ô, ta thu được 13×13×3 = 507 dự đoán, phần lớn là nền với điểm tin cậy gần 0. NMS chỉ giữ lại các hộp có độ tin cậy cao và không chồng lấn:
- Xe 1: `confidence=0.95`, `box=[120, 200, 60, 40]`
- Xe 2: `confidence=0.89`, `box=[300, 180, 80, 50]`
- Xe 3: `confidence=0.76`, `box=[450, 220, 40, 25]` (xe ở xa nên kích thước nhỏ hơn)
- Người 1: `confidence=0.92`, `box=[200, 150, 30, 80]`

Như vậy, bộ phát hiện đã đồng thời nhận dạng lớp và định vị vị trí từng đối tượng bằng hộp bao.

## Video tham khảo

- **Xem trước** [How YOLO Object Detection Works](https://www.youtube.com/watch?v=svn9-xV7wjk) — DeepBean, tiếng Anh. Minh họa cách một lượt lan truyền xuôi dự đoán trực tiếp các hộp bao, bám sát thiết kế lý thuyết trong bài.
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — Stanford University School of Engineering, tiếng Anh. Trình bày IoU và cách gán hộp dự đoán với nhãn thực tế — phần toán học chi tiết mà video ngắn chưa đi sâu.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Ôn lại các đặc trưng tích chập làm nền tảng cho cả hai video trên.
