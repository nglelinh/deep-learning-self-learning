---
layout: post
title: 15-98 Luyện phỏng vấn (tách đặc trưng và chuyển giao)
chapter: '15'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter15
lesson_type: optional
---

Trong các buổi phỏng vấn kỹ sư Machine Learning, câu hỏi về Transfer Learning hiếm khi dừng ở "hãy nêu định nghĩa fine-tuning" — người phỏng vấn muốn thấy bạn phản xạ thế nào khi dữ liệu bị lệch miền, khi chỉ có 200 mẫu gắn nhãn, hoặc khi cần quyết định giữa trích xuất đặc trưng đóng băng với tinh chỉnh toàn mạng. Bài luyện phỏng vấn này trang bị cho bạn khung trả lời chặt chẽ, có số liệu và lập luận kỹ thuật sắc bén.

## Kiến thức cần nhớ: Bốn năng lực cốt lõi được kiểm tra khi phỏng vấn về Transfer Learning

- **Đánh đổi Bias–Variance giữa Linear Probe và Fine-Tuning:** Biết cách lập luận định lượng về số tham số tự do, nguy cơ quá khớp và khả năng thích nghi biểu diễn.
- **Chẩn đoán giới hạn của đặc trưng đóng băng:** Nhận diện rõ những bài toán miền xa (ảnh vệ tinh đa phổ, siêu âm y khoa, chữ viết cổ) hoặc tác vụ cần định vị tinh (segmentation) nơi tầng cuối ImageNet đã đánh mất thông tin vị trí.
- **Bảo toàn cấu trúc không gian khi trích xuất đặc trưng:** Phân biệt khi nào dùng vector sau Global Average Pooling (phân loại) và khi nào phải giữ feature map 2D/kim tự tháp FPN (phát hiện vật thể, phân đoạn).
- **Tư duy chi phí tính toán và lưu trữ:** So sánh bộ nhớ lưu nhiều bản sao checkpoint đầy đủ với việc chỉ lưu các đầu tuyến tính hoặc adapter LoRA siêu nhẹ.

### Giải thích: Tính toán chi phí lưu trữ khi phục vụ 50 tác vụ đích bằng Full Fine-Tuning vs. Linear Probe / Adapter

Giả sử doanh nghiệp của bạn triển khai cùng một xương sống thị giác `ViT-Large` ($$304\text{ triệu}$$ tham số, lưu FP16 tốn $$2\text{ byte/tham số}$$) cho $$50$$ khách hàng với $$50$$ bài toán phân loại $$10$$ lớp khác nhau (chiều đặc trưng $$d = 1024$$).

1. **Full Fine-Tuning cho từng khách hàng:**
   - Mỗi bản sao mô hình: $$304 \times 10^6 \times 2\text{ byte} \approx 608\text{ MB}$$.
   - Tổng lưu trữ cho 50 khách hàng: $$50 \times 608\text{ MB} = 30{,}4\text{ GB}$$ (và rất tốn VRAM nếu nạp đồng thời).
2. **Đóng băng xương sống + 50 đầu Linear Probe:**
   - 1 xương sống dùng chung: $$608\text{ MB}$$.
   - Mỗi đầu phân loại `Linear(1024, 10)`: $$(1024 \times 10 + 10) \times 2\text{ byte} \approx 20{,}5\text{ KB}$$.
   - Tổng 50 đầu: $$50 \times 20{,}5\text{ KB} \approx 1{,}025\text{ MB}$$. Tổng cộng chỉ $$\sim 609\text{ MB}$$ — **tiết kiệm gần 50 lần bộ nhớ** và có thể gộp chung batch cho cả 50 khách hàng trên một GPU!

# Tùy chọn: Luyện phỏng vấn — Trích xuất đặc trưng và Học chuyển giao

> Bài này mang tính **tùy chọn** và **không** thay thế phần lý thuyết so sánh trích xuất đặc trưng với tinh chỉnh. Sau khi học xong các bài chính, bạn có thể luyện với các câu hỏi dưới đây rồi mở *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để xem thêm các câu hỏi có lời giải về trích xuất đặc trưng CNN. Phần hình học kiến trúc CNN nằm ở **04-98**.

**Chủ đề tương ứng trong sách (đọc trong bản PDF, không tóm tắt lại ở đây):** sử dụng CNN đã huấn luyện làm bộ trích xuất đặc trưng; so sánh thân mạng đóng băng với thân mạng cho phép tinh chỉnh.

Trang tổng hợp: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu hỏi luyện tập (tự soạn)

### Q1. Linear probe so với tinh chỉnh khối cuối

Bạn có 800 ảnh y khoa có nhãn và một mạng ResNet tiền huấn luyện trên ImageNet. Hãy so sánh hai phương án: (a) đóng băng toàn bộ thân mạng, chỉ huấn luyện một tầng đầu ra tuyến tính và (b) mở băng khối residual cuối cùng cùng tầng đầu ra. Phương án (b) mang lại thêm rủi ro gì mà (a) không có?

**Gợi ý.** So sánh số tham số cần huấn luyện với mức độ gần nhau về phân phối thống kê giữa miền nguồn và miền đích.

**Thảo luận.** Phương án (a) (*linear probe*) tốn ít chi phí, không làm hỏng các bộ lọc đã học và thường là mốc cơ sở vững chắc. Phương án (b) cho phép điều chỉnh các bộ lọc mức trung và cao cho phù hợp với tổn thương hoặc kết cấu chưa có trong ImageNet, nhưng với chỉ 800 ảnh rất dễ quá khớp và ghi đè các bộ lọc hữu ích (tinh chỉnh quá mức). Giải pháp cân bằng trong thực tế là đặt learning rate nhỏ trên thân mạng và lớn hơn trên tầng đầu ra, hoặc dùng adapter (xem bài tùy chọn hiện đại Chương 15) — nên nhắc đến như phần mở rộng sau khi đã trả lời đúng trọng tâm.

### Q2. Khi nào đặc trưng đóng băng không phù hợp

Hãy nêu một miền đích cụ thể mà đặc trưng đóng băng từ ImageNet hoạt động kém dù cả miền nguồn lẫn miền đích đều là ảnh, và chỉ rõ nguồn gốc của sự sai lệch.

**Gợi ý.** Xét đến loại cảm biến, góc chụp hoặc bản chất nhãn — không chỉ lý do "tập dữ liệu nhỏ".

**Thảo luận.** Ví dụ điển hình gồm ảnh mô bệnh học độ phóng đại 40×, ảnh vệ tinh có kênh ngoài dải RGB, hoặc ảnh kiểm tra khuyết tật công nghiệp cận cảnh. Sự sai lệch nằm ở cả *thống kê mức thấp lẫn ngữ nghĩa*, không chỉ ở kích thước mẫu. Trong các trường hợp này, tinh chỉnh sâu hơn vào thân mạng hoặc tiền huấn luyện trong miền bằng học tự giám sát (Chương 16) sẽ hiệu quả hơn việc chỉ gắn thêm mạng MLP sâu lên trên đặc trưng RGB đóng băng.

### Q3. Nên giữ cấu trúc nào khi trích xuất đặc trưng

Một đồng nghiệp "trích xuất đặc trưng" bằng cách lấy bản đồ đặc trưng $$7\times7\times2048$$ của ResNet-50 rồi trải phẳng toàn bộ để huấn luyện SVM. Hãy nêu một lý do nên áp dụng *global average pooling* (GAP) trước, và một loại tác vụ mà ta cần *giữ nguyên* bản đồ không gian.

**Gợi ý.** Cân nhắc giữa tính bất biến tịnh tiến và yêu cầu định vị không gian.

**Thảo luận.** Áp dụng GAP rút gọn đặc trưng xuống vector $$2048$$ chiều có tính bất biến tịnh tiến và kích thước phù hợp để huấn luyện SVM (tránh bùng nổ số chiều thành 100.352). Ngược lại, các bài toán phát hiện đối tượng (*detection*), phân đoạn (*segmentation*) hoặc định vị vị trí vật thể cần giữ nguyên tensor không gian (hoặc dùng kim tự tháp đặc trưng). "Trích xuất đặc trưng" không chỉ là chọn kích thước vector mà là chọn tính bất biến phù hợp với bài toán.

### Q4. Lệch miền trong thực tế triển khai

Tập huấn luyện gồm ảnh sản phẩm chụp trong studio, còn ảnh thực tế khi triển khai là ảnh chụp bằng điện thoại trong kho hàng. Bạn chỉ tinh chỉnh tầng đầu ra và đạt độ chính xác rất cao trên tập validation (vốn cũng là ảnh studio). Bạn thực sự mới đo được điều gì, và cần bổ sung thí nghiệm nào trước khi đưa vào sử dụng?

**Gợi ý.** Phân biệt đánh giá trên tập validation cùng phân phối (*i.i.d.*) với phân phối triển khai thực tế.

**Thảo luận.** Kết quả đó mới chỉ đo chất lượng tầng đầu ra trong cùng miền studio chứ chưa kiểm chứng khả năng chuyển giao sang môi trường kho. Trước khi triển khai, cần thu thập và gán nhãn một tập kiểm tra nhỏ chụp trực tiếp trong kho để đo hiệu năng thực tế. Các phép tăng cường dữ liệu như biến đổi ánh sáng, màu sắc hay cắt khung có thể hỗ trợ thu hẹp khoảng cách nhưng không thể thay thế tập validation thuộc miền đích.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Bạn có thể tải bản PDF từ arXiv để xem toàn bộ bộ câu hỏi và lời giải. Các câu hỏi trên trang này được biên soạn riêng cho khóa học, không phải bản sao chép lại từ sách.

## Video tham khảo

- **Xem trước:** [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, tiếng Anh. Ôn lại các câu hỏi cốt lõi về trích xuất đặc trưng và tinh chỉnh.
- [Pytorch Transfer Learning and Fine Tuning Tutorial](https://www.youtube.com/watch?v=qaDe0qQZ5AQ) — Aladdin Persson, tiếng Anh. Nắm rõ nhóm tham số nào nhận gradient trong từng chiến lược.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Giải thích vì sao các bộ lọc ở tầng sớm thường nên được giữ lại.
