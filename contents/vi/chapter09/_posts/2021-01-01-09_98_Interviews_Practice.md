---
layout: post
title: 09-98 Luyện phỏng vấn (overfitting và chính quy)
chapter: '09'
order: 9
owner: Deep Learning Course
lang: vi
categories:
- chapter09
lesson_type: optional
---

# Tùy chọn: Luyện phỏng vấn — Quá khớp và chính quy hóa

> Bài này mang tính **tùy chọn** và **không** thay thế các bài lý thuyết về dropout, BatchNorm hay các kỹ thuật chính quy hóa khác. Sau khi học xong phần cốt lõi, hãy dùng các câu hỏi dưới đây để tự kiểm tra, đồng thời tham khảo thêm bộ câu hỏi có lời giải trong *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)).

**Chủ đề tương ứng trong sách (đọc trong bản PDF):** quá khớp và tổng quát hóa, đánh giá kiểm định (validation) và các kỹ thuật kiểm soát dung lượng mô hình.

Trang tổng hợp lộ trình: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Bộ câu hỏi tự luyện

### Q1. Mất mát huấn luyện giảm nhưng mất mát kiểm định tăng

Giá trị cross-entropy trên tập huấn luyện vẫn tiếp tục giảm sau epoch 40, trong khi trên tập kiểm định lại bắt đầu tăng từ epoch 25. Hãy nêu hai biện pháp can thiệp *khác nhau* (không phải "tăng thêm GPU") và phân tích giả định đằng sau mỗi cách.

**Gợi ý:** Một hướng thay đổi không gian giả thuyết hoặc hàm mục tiêu hiệu dụng; hướng còn lại điều chỉnh *thời điểm* dừng huấn luyện.

**Thảo luận:** (1) Tăng cường chính quy hóa (tăng weight decay, bổ sung dropout hoặc data augmentation) dựa trên giả định mô hình có đủ khả năng khớp tập huấn luyện nhưng đang quá linh hoạt so với dữ liệu. (2) Áp dụng dừng sớm (*early stopping*) và chọn checkpoint tốt nhất trên tập kiểm định hợp lệ dựa trên giả định mô hình đã đi qua điểm tổng quát hóa tối ưu. Ngoài ra, thu thập thêm dữ liệu gán nhãn giúp thu hẹp khoảng cách từ phía cỡ mẫu, còn thu nhỏ kiến trúc mạng là thay đổi trực tiếp dung lượng thay vì chính quy hóa trong quá trình huấn luyện.

### Q2. So sánh chính quy hóa L2 và L1 trên tầng tuyến tính

Với một tầng tuyến tính, so sánh việc cộng thêm $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$ và $$\lambda\|\mathbf{w}\|_1$$ vào hàm mất mát. Dạng nào dễ đẩy trọng số về *đúng* bằng 0 hơn, và vì sao điều đó mới chỉ là một phần câu chuyện trong mạng ReLU sâu?

**Gợi ý:** Đạo hàm dưới (*subgradient*) của L1 chứa một đoạn tại 0, còn gradient của L2 tỷ lệ tuyến tính với $$\mathbf{w}$$ nên thu nhỏ dần nhưng hiếm khi chạm đúng 0 trong SGD.

**Thảo luận:** Chính quy hóa L1 (tương tự Lasso) tạo nghiệm thưa với nhiều trọng số bằng đúng 0, trong khi L2 kéo trọng số về gần gốc tọa độ mà không ép về 0 tuyệt đối. Trong mạng ReLU nhiều tầng, một "đặc trưng" bị tắt tương ứng với cả một đường dẫn kích hoạt chứ không chỉ một tọa độ riêng lẻ của $$\mathbf{w}$$, nên áp L1 trực tiếp lên trọng số thô có hiệu quả hạn chế. Do đó, weight decay (L2) vẫn là lựa chọn mặc định nhờ chi phí tính toán thấp và tính bất biến theo phép quay (*rotation-invariant*) trong trường hợp tuyến tính.

### Q3. Dropout giữa hai chế độ train và eval

Nếu một đơn vị bị loại bỏ với xác suất $$p=0.5$$ khi huấn luyện, cần xử lý thế nào lúc đánh giá để giá trị kỳ vọng trước kích hoạt khớp với lúc huấn luyện? Hậu quả sẽ ra sao nếu quên bước này?

**Gợi ý:** So sánh giữa *inverted dropout* và cách chia tỉ lệ cổ điển.

**Thảo luận:** Với *inverted dropout* (mặc định trong các framework hiện đại), các đơn vị được giữ lại sẽ nhân với $$1/(1-p)$$ ngay trong lúc huấn luyện, còn lúc đánh giá dùng nguyên toàn mạng. Với dropout cổ điển, lúc huấn luyện không chia tỉ lệ và khi đánh giá phải nhân trọng số với $$(1-p)$$. Nếu tắt ngẫu nhiên lúc huấn luyện mà bỏ qua bước cân chỉnh tỉ lệ, giá trị kích hoạt lúc suy luận sẽ lớn hơn có hệ thống so với lúc huấn luyện, làm suy giảm cả độ chính xác lẫn độ hiệu chuẩn xác suất. Bên cạnh cách diễn giải "ensemble $$2^n$$ mạng con", điểm vận hành thiết yếu nhất trong cài đặt chính là hệ số chia tỉ lệ này.

### Q4. BatchNorm giữa hai chế độ train và test

Vì sao BatchNorm phải duy trì trung bình và phương sai trượt, và điều gì xảy ra nếu đánh giá một checkpoint vừa nạp ở chế độ `train` với kích thước batch bằng 1?

**Gợi ý:** Phân biệt nguồn thống kê giữa chế độ `train` và `eval`.

**Thảo luận:** Các giá trị thống kê chạy lưu lại ước lượng quần thể của $$\mathbb{E}[\mathbf{h}]$$ và $$\mathrm{Var}(\mathbf{h})$$ để dùng ổn định khi suy luận. Nếu chạy ở chế độ `train` với batch size bằng 1, phương sai theo batch bằng 0 (hoặc rất nhiễu với batch quá nhỏ), làm hỏng hoàn toàn phép chuẩn hóa. Hạn chế với batch nhỏ cũng là lý do LayerNorm và GroupNorm được dùng thay thế trong nhiều kiến trúc khác.

### Q5. Tăng cường dữ liệu (Data Augmentation) như một dạng chính quy hóa

Giả sử bạn không thể tăng hệ số phạt $$\lambda$$ vì mất mát trên tập huấn luyện vốn đã cao. Hãy đề xuất một phương pháp chính quy hóa từ phía dữ liệu không cộng thêm số hạng phạt vào $$J(\mathbf{w})$$, kèm theo một tình huống thất bại của nó.

**Gợi ý:** Tăng cường dữ liệu làm mịn phân phối đầu vào theo các phép biến đổi bảo toàn nhãn.

**Thảo luận:** Các phép biến đổi như cắt ảnh ngẫu nhiên (random crop), lật ảnh hoặc thay đổi màu sắc đưa tính bất biến mong muốn trực tiếp vào dữ liệu huấn luyện mà không thêm hạng phạt lên trọng số. Tuy nhiên, nếu phép tăng cường làm biến đổi cả nhãn gốc (ví dụ cắt mất đối tượng cần nhận diện hoặc lật ngang chữ số), nó sẽ đưa thêm *nhiễu nhãn* thay vì tính bất biến hữu ích, khiến sai số tăng trên cả tập huấn luyện lẫn tập kiểm định. Dừng sớm cũng là một dạng chính quy hóa không dùng hạng phạt, nhưng không thể bù đắp cho chiến lược tăng cường dữ liệu sai lệch.

## Ghi công

Kashani, S., & Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Bạn có thể tải bản PDF từ arXiv để xem đầy đủ ngân hàng câu hỏi kèm lời giải; nội dung trên đây là bộ câu hỏi tự luyện do khóa học biên soạn.

## Video tham khảo

- **Xem trước:** [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — StatQuest with Josh Starmer (tiếng Anh). Hệ thống hóa khái niệm nền tảng cho các câu hỏi về quá khớp.
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — StatQuest with Josh Starmer (tiếng Anh). Ôn tập trực quan về chính quy hóa L2 thường xuất hiện trong phỏng vấn.
- [Dropout Regularization (C2W1L06)](https://www.youtube.com/watch?v=D8PJAL-MZv8) — DeepLearningAI (tiếng Anh). Củng cố cơ chế dropout để đối chiếu với L2 và các kỹ thuật khác.
