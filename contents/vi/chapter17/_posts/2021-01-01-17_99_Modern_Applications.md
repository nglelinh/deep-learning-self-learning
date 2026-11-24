---
layout: post
title: 17-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '17'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter17
lesson_type: optional
---

Suốt gần một thập kỷ, mọi bộ phát hiện đối tượng đều bị trói chặt vào hai thứ: danh sách lớp cố định (ví dụ đúng 80 lớp của COCO) và thuật toán hậu xử lý thủ công NMS. Từ năm 2022 đến 2026, cả hai rào cản đó đồng loạt bị phá vỡ: **RT-DETR** loại bỏ hoàn toàn NMS nhờ khớp hai phía Hungarian, **SAM / SAM 2** cắt tách bất kỳ vật thể nào trong ảnh/video chỉ bằng một cú nhấp chuột, còn **Grounding DINO** tìm vật thể bằng câu mô tả tiếng người tự do!

## Bản đồ kiến thức nền: Ba bước ngoặt của Thị giác Máy tính hiện đại (2022–2026)

- **Phân đoạn theo Prompt (SAM & SAM 2):** Tách biệt bộ mã hóa ảnh nặng (chạy 1 lần) với bộ giải mã mặt nạ siêu nhẹ nhận prompt điểm/hộp/văn bản; SAM 2 mở rộng thêm bộ nhớ thời gian để theo dõi đối tượng xuyên suốt video.
- **Bộ phát hiện Transformer thời gian thực (RT-DETR / RF-DETR):** Dùng phép khớp một-một Hungarian lúc huấn luyện nên mỗi truy vấn đối tượng (object query) tạo đúng một dự đoán, bỏ hẳn Anchor Boxes và NMS.
- **Phát hiện từ vựng mở (Open-Vocabulary Detection — Grounding DINO, YOLO-World):** Trộn chéo đặc trưng ảnh với vector văn bản để phát hiện cả những vật thể chưa từng có trong tập nhãn cố định.
- **Chuỗi kết hợp Grounding DINO + SAM:** Quy trình tiêu chuẩn công nghiệp cho gắn nhãn tự động (auto-labeling): nhập cụm từ văn bản → Grounding DINO khoanh hộp → SAM cắt chính xác mặt nạ pixel.

### Khung giải thích (nên đọc chậm): Tính tay bài toán khớp một-một Hungarian trong DETR (loại bỏ NMS)

Giả sử ảnh có 2 đối tượng thật ($$y_1$$: `mèo`, $$y_2$$: `chó`) và bộ giải mã DETR tạo ra 3 dự đoán $$(\hat{y}_1, \hat{y}_2, \hat{y}_3)$$. Ma trận chi phí khớp (càng nhỏ càng tốt, gộp sai số lớp + sai số hộp GIoU) với đối tượng 3 là lớp rỗng $$\varnothing$$ (no-object) cho bởi:
- $$\hat{y}_1$$: chi phí với `mèo` $$= 0{,}2$$, với `chó` $$= 2{,}5$$, với $$\varnothing = 1{,}0$$
- $$\hat{y}_2$$: chi phí với `mèo` $$= 0{,}3$$, với `chó` $$= 0{,}4$$, với $$\varnothing = 1{,}0$$
- $$\hat{y}_3$$: chi phí với `mèo` $$= 3{,}0$$, với `chó` $$= 2{,}8$$, với $$\varnothing = 0{,}1$$

1. **So sánh hai phương án gán song ánh:**
   - Phương án A ($$\hat{y}_1 \to \text{mèo},\; \hat{y}_2 \to \text{chó},\; \hat{y}_3 \to \varnothing$$): tổng chi phí $$= 0{,}2 + 0{,}4 + 0{,}1 = 0{,}7$$.
   - Phương án B ($$\hat{y}_2 \to \text{mèo},\; \hat{y}_1 \to \text{chó},\; \hat{y}_3 \to \varnothing$$): tổng chi phí $$= 0{,}3 + 2{,}5 + 0{,}1 = 2{,}9$$.
2. **Hệ quả kiến trúc:** Thuật toán Hungarian chọn phương án A ($$0{,}7$$). Vì **chỉ duy nhất một truy vấn** được khớp vào `mèo` (còn mọi truy vấn dư đều bị phạt hướng về $$\varnothing$$), mạng tự học cách không bao giờ đẻ ra hai hộp chồng lên nhau — xóa sổ hoàn toàn nhu cầu dùng NMS!

# Tùy chọn: Phát hiện và phân đoạn sau YOLO / Mask R-CNN

> Bài này là **tùy chọn**, **không** thay thế phần lý thuyết về bộ phát hiện một/hai giai đoạn hay các ghi chú về YOLO và Faster R-CNN. Nội dung bổ sung các mô hình được sử dụng rộng rãi trong thực tế giai đoạn 2023–2025: **SAM/SAM 2**, **RT-DETR**, **Grounding DINO** và các phiên bản YOLO mới.

Các khái niệm cốt lõi của chương như IoU, NMS và cơ chế ghép cặp dự đoán vẫn là tiêu chuẩn quyết định một hộp bao có chính xác hay không:

$$\mathrm{IoU}(A,B) = \frac{|A\cap B|}{|A\cup B|}.$$

Tuy nhiên, các kiến trúc xoay quanh những chỉ số này đã thay đổi đáng kể.

## 1. Phân đoạn theo prompt: SAM và SAM 2

[Kirillov et al., 2023](https://arxiv.org/abs/2304.02643) (Segment Anything) mô hình hóa bài toán phân đoạn như một tác vụ nền tảng **theo prompt**: đầu vào là điểm, hộp bao hoặc mặt nạ sơ bộ, đầu ra là mặt nạ của từng thể hiện đối tượng. [Ravi et al., 2024](https://arxiv.org/abs/2408.00714) (SAM 2) bổ sung bộ nhớ theo luồng thời gian để xử lý **video**. Trong quy trình khoa học dữ liệu thực tế, các mô hình này thường dùng để hỗ trợ gán nhãn nhanh (chấm vài điểm, xuất mặt nạ rồi huấn luyện một mô hình chuyên biệt nhỏ hơn), chứ không thay thế U-Net khi đã có tập dữ liệu y tế được gán nhãn chi tiết ở mức pixel.

## 2. Ứng dụng cụ thể

### Bộ phát hiện DETR thời gian thực

[Zhao et al., 2024](https://arxiv.org/abs/2304.08069) (RT-DETR) cải thiện tốc độ cho dòng bộ phát hiện dựa trên Transformer đến mức có thể thay thế YOLO trong một số hệ thống thực tế, trong khi vẫn đánh giá bằng chỉ số mAP như phần lý thuyết đã trình bày.

### Phát hiện từ vựng mở: Grounding DINO + SAM

[Liu et al., 2023](https://arxiv.org/abs/2303.05499) (Grounding DINO) nhận truy vấn **văn bản** (ví dụ “mũ bảo hiểm đỏ”) và trả về hộp bao tương ứng. Kết hợp Grounding DINO với SAM (“Grounded SAM”) là mẫu quy trình phổ biến trong gán nhãn dữ liệu và tri giác robot giai đoạn 2024–2026. Dù vậy, với các bài toán có tập nhãn cố định như kiểm đếm mã sản phẩm (SKU) trong nhà máy, bộ phát hiện tập đóng kiểu COCO vẫn là lựa chọn phù hợp.

### YOLO trong triển khai thực tế

Các phiên bản Ultralytics **YOLOv8 / YOLO11** vẫn giữ thiết kế tích chập một giai đoạn, phù hợp khi yêu cầu về độ trễ thấp và khả năng xuất sang các định dạng triển khai (ONNX, TensorRT, CoreML) quan trọng hơn khả năng nhận dạng từ vựng mở.

## 3. Phần mềm phổ biến

- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) và Segment Anything.
- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) và [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR).

## 4. Trích dẫn (2022–2026)

- [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643).
- [SAM 2 (Ravi et al., 2024)](https://arxiv.org/abs/2408.00714).
- [Grounding DINO (Liu et al., 2023)](https://arxiv.org/abs/2303.05499).
- [RT-DETR (Zhao et al., 2024)](https://arxiv.org/abs/2304.08069).

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Trước hết cần nắm vững cơ chế ghép cặp dự đoán và quy trình R-CNN/YOLO trong các bài cốt lõi. Bài này bổ sung các đầu phát hiện và phân đoạn **theo prompt hoặc từ vựng mở** được dùng phổ biến trong các hệ thống giai đoạn 2023–2026.

## Video tham khảo

- **Xem trước** [Explaining the Segment Anything Model - Network architecture, Dataset, Training](https://www.youtube.com/watch?v=OhxJkqD1vuE) — Neural Breakdown with AVB, tiếng Anh. Giải thích kiến trúc, dữ liệu và cách huấn luyện của Segment Anything — mô hình phân đoạn theo prompt trong bài.
- [Lecture 11 | Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — Stanford University School of Engineering, tiếng Anh. Ôn lại bài toán phát hiện và phân đoạn cổ điển mà SAM tổng quát hóa.
- [MIT 6.S191 (2024): Convolutional Neural Networks](https://www.youtube.com/watch?v=2xqkSUhmmXU) — Alexander Amini, tiếng Anh. Nền tảng về mạng xương sống thị giác bên dưới các mô hình phân đoạn theo prompt.
