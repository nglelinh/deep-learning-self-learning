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

# Phát hiện Đối tượng: Định vị và Nhận dạng

![Object Detection Example](https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg/800px-Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg)
*Hình ảnh: Phát hiện đối tượng với YOLO — phát hiện và định vị nhiều vật thể. Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm
Phát hiện đối tượng (*object detection*) mở rộng phân loại ảnh từ câu hỏi “trong ảnh có những đối tượng nào?” sang “có những đối tượng nào và chúng nằm ở đâu?”. Sự mở rộng tưởng chừng nhỏ này thực chất đòi hỏi giải đồng thời nhiều bài toán liên kết: đề xuất vùng có thể chứa đối tượng (*region proposal*), phân loại nội dung từng vùng (*recognition*), tinh chỉnh biên của các phát hiện (*localization*), và xử lý nhiều đối tượng thuộc nhiều lớp ở nhiều tỷ lệ khác nhau (*multi-scale, multi-class detection*). Việc phối hợp các thành phần này trong khi vẫn duy trì hiệu năng thời gian thực đã khiến phát hiện đối tượng trở thành một trong những lĩnh vực thách thức và được nghiên cứu sôi động nhất của thị giác máy tính.

Sự tiến hóa của các phương pháp phát hiện đối tượng cho thấy lộ trình thú vị từ thị giác máy tính cổ điển đến học sâu hiện đại. Các phương pháp cổ điển dùng đặc trưng thủ công (SIFT, HOG) với cửa sổ trượt (*sliding window*) quét mọi vị trí và tỷ lệ khả dĩ, rồi áp dụng bộ phân loại như SVM. Cách này tốn tính toán (đánh giá hàng triệu cửa sổ mỗi ảnh) và bị giới hạn bởi chất lượng đặc trưng. Cuộc cách mạng học sâu đã biến đổi phát hiện đối tượng thông qua đặc trưng học được và các hệ thống huấn luyện đầu–cuối, mang lại cải thiện lớn về cả độ chính xác lẫn tốc độ.

Phát hiện đối tượng hiện đại chia thành hai mô hình chính. Bộ phát hiện hai giai đoạn (*two-stage detector*) như R-CNN, Fast R-CNN và Faster R-CNN trước hết đề xuất các vùng có khả năng chứa đối tượng, rồi phân loại và tinh chỉnh các đề xuất đó. Việc tách rõ đề xuất vùng và nhận dạng cho phép đạt độ chính xác cao nhờ tập trung tính toán vào các vùng tiềm năng. Bộ phát hiện một giai đoạn (*single-stage detector*) như YOLO và SSD dự đoán trực tiếp hộp bao và xác suất lớp từ các vị trí lưới đều, đạt hiệu năng thời gian thực bằng cách bỏ qua giai đoạn đề xuất, với đánh đổi là độ chính xác hơi thấp hơn trên đối tượng nhỏ.

Để hiểu sâu phát hiện đối tượng, cần nắm các đổi mới kỹ thuật then chốt. Mạng đề xuất vùng (*Region Proposal Network*, RPN) học cách sinh đề xuất thay vì dùng quy tắc thủ công, khiến toàn bộ đường ống khả vi. Hộp neo (*anchor box*) xử lý đối tượng có tỷ lệ khung hình và kích thước khác nhau thông qua các mẫu hộp định sẵn. Ức chế không cực đại (*non-maximum suppression*, NMS) loại bỏ các phát hiện trùng lặp — vì bộ phát hiện tốt thường sinh nhiều hộp chồng lấn cho cùng một đối tượng. Mạng kim tự tháp đặc trưng (*feature pyramid network*, FPN) cho phép phát hiện đa tỷ lệ bằng cách xây kim tự tháp đặc trưng giàu ngữ nghĩa ở mọi mức. Các thành phần này, mỗi phần giải một bài toán con, kết hợp thành hệ thống có thể phát hiện và định vị hàng chục đối tượng thuộc nhiều lớp trong mili giây, phục vụ từ xe tự lái đến phân tích ảnh y khoa và thực tế tăng cường.

## 2. Nền tảng toán học
Phát hiện đối tượng đòi hỏi hình thức hóa cái ta dự đoán và cách đo thành công. Một phát hiện đối tượng là bộ $$(\text{class}, x, y, w, h)$$ chỉ định lớp đối tượng và hộp bao (tọa độ tâm $$x,y$$ và kích thước $$w,h$$). Với ảnh có $$N$$ đối tượng, nhãn gốc là tập các bộ như vậy: $$\{(\text{class}_i, x_i, y_i, w_i, h_i)\}_{i=1}^N$$. Bộ phát hiện phải dự đoán tập này — thách thức vì $$N$$ thay đổi giữa các ảnh.

### Giao trên Hợp (Intersection over Union, IoU)

Để đo chất lượng định vị, ta dùng Giao trên Hợp giữa hộp dự đoán và hộp gốc:

$$\text{IoU}(\text{box}_{\text{pred}}, \text{box}_{\text{gt}}) = \frac{\text{Area}(\text{box}_{\text{pred}} \cap \text{box}_{\text{gt}})}{\text{Area}(\text{box}_{\text{pred}} \cup \text{box}_{\text{gt}})}$$

IoU nằm trong khoảng từ 0 (không chồng lấn) đến 1 (chồng lấn hoàn hảo). Thông thường, một phát hiện được coi là đúng nếu IoU $$\geq 0.5$$ và lớp dự đoán khớp nhãn gốc. Ngưỡng này cân bằng giữa yêu cầu định vị chính xác và cho phép biến thiên hợp lý của hộp bao.

### Hồi quy Hộp bao (Bounding Box Regression)

Thay vì dự đoán trực tiếp tọa độ hộp, các bộ phát hiện hiện đại dự đoán độ lệch so với hộp neo (hộp tham chiếu định sẵn). Cho hộp neo $$(\hat{x}, \hat{y}, \hat{w}, \hat{h})$$ và nhãn gốc $$(\bar{x}, \bar{y}, \bar{w}, \bar{h})$$, ta tham số hóa mục tiêu:

$$t_x = \frac{\bar{x} - \hat{x}}{\hat{w}}, \quad t_y = \frac{\bar{y} - \hat{y}}{\hat{h}}$$

$$t_w = \log\frac{\bar{w}}{\hat{w}}, \quad t_h = \log\frac{\bar{h}}{\hat{h}}$$

Mạng dự đoán $$(t_x, t_y, t_w, t_h)$$, rồi giải mã ra tọa độ tuyệt đối:

$$x = \hat{x} + \hat{w} \cdot t_x, \quad y = \hat{y} + \hat{h} \cdot t_y$$

$$w = \hat{w} \cdot \exp(t_w), \quad h = \hat{h} \cdot \exp(t_h)$$

Tham số hóa này dễ học hơn dự đoán tọa độ trực tiếp vì các độ lệch thường là số nhỏ có cùng thang đo, trong khi tọa độ tuyệt đối trải trên toàn ảnh với thang đo rất khác giữa đối tượng nhỏ và lớn.

### Hàm mất Đa tác vụ (Multi-Task Loss)

Bộ phát hiện tối ưu tổ hợp mất mát phân loại và định vị:

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{box}}$$

trong đó $$\mathcal{L}_{\text{cls}}$$ là mất mát phân loại (cross-entropy) và $$\mathcal{L}_{\text{box}}$$ là mất mát hồi quy hộp (smooth L1 hoặc mất IoU). Trọng số $$\lambda$$ cân bằng hai mục tiêu — quá cao thì bộ phát hiện tập trung định vị chính xác nhưng kém phân loại; quá thấp thì phân loại tốt nhưng hộp định vị kém.

Với Faster R-CNN, mất phân loại dùng cross-entropy trên các lớp cộng nền:

$$\mathcal{L}_{\text{cls}} = -\log p_{\text{class}}$$

trong đó $$\text{class}$$ là lớp gốc (hoặc nền nếu IoU < 0.5 với mọi hộp gốc).

Mất hộp là smooth L1:

$$\mathcal{L}_{\text{box}} = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L1}(t_i - \hat{t}_i)$$

$$\text{smooth}_{L1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

Smooth L1 ít nhạy với ngoại lai hơn L2 (phần bậc hai trở thành tuyến tính khi sai số lớn) đồng thời khả vi mọi nơi (khác L1 thuần).

### Mạng Đề xuất Vùng (Region Proposal Network, RPN)

Faster R-CNN giới thiệu RPN — mạng tích chập đầy đủ dự đoán các đề xuất đối tượng. Tại mỗi vị trí trên bản đồ đặc trưng, RPN dự đoán:
- Điểm objectness: $$k$$ neo × 2 giá trị (đối tượng vs nền)
- Tinh chỉnh hộp: $$k$$ neo × 4 tọa độ

Với bản đồ đặc trưng $$H \times W$$ và $$k=9$$ neo mỗi vị trí, RPN xuất ra:
- Objectness: $$H \times W \times 9 \times 2$$ điểm
- Delta hộp: $$H \times W \times 9 \times 4$$ giá trị

Tổng: $$HW \times 9$$ đề xuất. NMS lọc còn top $$\sim$$2000 theo điểm objectness, rồi đưa vào đầu phát hiện.

Mất RPN kết hợp phân loại (objectness) và hồi quy:

$$\mathcal{L}_{\text{RPN}} = \frac{1}{N_{\text{cls}}}\sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \frac{\lambda}{N_{\text{box}}}\sum_i p_i^* \mathcal{L}_{\text{box}}(t_i, t_i^*)$$

trong đó $$p_i^* = 1$$ nếu neo $$i$$ chồng lấn nhãn gốc với IoU > 0.7 (dương), $$p_i^* = 0$$ nếu IoU < 0.3 (âm), và bỏ qua nếu nằm giữa (xử lý trường hợp mơ hồ).

## 3. Ví dụ / Trực giác

Hãy tưởng tượng bạn cần tìm và nhận diện mọi người trong một bức ảnh đông đúc. Chiến lược có thể là:

1. **Quét nhanh** các vùng có khả năng chứa người (dáng đầu, đường viền thân)
2. **Xem xét kỹ** các vùng tiềm năng (đây có thực sự là người hay là bức tượng? là ai?)
3. **Tinh chỉnh** biên (hộp bao của người này bắt đầu/kết thúc chính xác ở đâu?)

Quy trình ba giai đoạn này phản ánh phát hiện đối tượng hai giai đoạn. RPN thực hiện quét nhanh, đề xuất ~2000 vùng có thể chứa đối tượng (người, xe, chó, bất kỳ thứ gì). Đầu phát hiện xem xét từng đề xuất, phân loại nội dung và tinh chỉnh hộp bao. NMS loại bỏ trùng lặp (nhiều hộp chồng lấn cho cùng một người).

Xét phát hiện xe trên đường phố. Ảnh có thể chứa:
- 3 xe ở các khoảng cách khác nhau (kích thước khác nhau)
- 2 người đi bộ
- 1 biển báo giao thông
- Nền phức tạp (nhà, cây)

Bộ phát hiện một giai đoạn như YOLO chia ảnh thành lưới (ví dụ 13×13). Mỗi ô lưới dự đoán:
- Nhiều hộp bao (ví dụ 3, với tỷ lệ khung hình khác: cao, rộng, vuông)
- Xác suất lớp cho mỗi hộp
- Điểm tin cậy (có đối tượng tại đây không?)

Với ô lưới tại vị trí (5, 8) gần một chiếc xe, nó có thể dự đoán:
- Hộp 1: class=car, confidence=0.95, tọa độ lệch so với tâm ô
- Hộp 2: class=background, confidence=0.05
- Hộp 3: class=background, confidence=0.02

Sau khi xử lý tất cả 13×13 ô, ta có 13×13×3 = 507 dự đoán. Hầu hết là nền (tin cậy gần 0). NMS chỉ giữ các hộp tin cậy cao, không chồng lấn:
- Xe 1: confidence=0.95, box=[120, 200, 60, 40]
- Xe 2: confidence=0.89, box=[300, 180, 80, 50]
- Xe 3: confidence=0.76, box=[450, 220, 40, 25] (xe xa, nhỏ hơn)
- Người 1: confidence=0.92, box=[200, 150, 30, 80]

Bộ phát hiện đã nhận diện mọi đối tượng, phân loại chúng và định vị bằng hộp bao — đúng những gì phát hiện đối tượng yêu cầu.
