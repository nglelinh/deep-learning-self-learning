---
layout: post
title: 10-98 Luyện phỏng vấn (bộ tối ưu)
chapter: '10'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter10
lesson_type: optional
---

Trong phỏng vấn Machine Learning nâng cao, câu hỏi về bộ tối ưu hiếm khi dừng ở mức "Adam viết tắt của từ gì?". Người phỏng vấn muốn kiểm tra xem bạn có giải thích được bằng trực giác toán học **tại sao Adam cần bias correction**, **điểm khác biệt chí mạng giữa Adam và AdamW**, và **vì sao Transformer gần như không thể huấn luyện thiếu Learning Rate Warmup** hay không.

## Bản đồ kiến thức nền: Bốn chủ đề phỏng vấn trọng tâm về Bộ tối ưu

1. **Giải thích hình học của Momentum trên thung lũng hẹp**:
   Gradient trên trục dốc đổi dấu liên tục ($$+, -, +, -$$) nên khi cộng dồn vào vận tốc $$v_t$$ chúng **tự triệt tiêu nhau**; trong khi gradient dọc đáy thung lũng luôn cùng dấu nên **cộng hưởng tăng tốc** lên tới hệ số $$\frac{1}{1 - \beta}$$ (với $$\beta = 0{,}9$$ là gấp 10 lần!).
2. **Tại sao Adam cần Hiệu chỉnh độ lệch (Bias Correction)**:
   Vì $$m_0 = v_0 = 0$$, ước lượng EMA bị kéo lệch mạnh về 0 ở những bước đầu khi $$\beta_1 = 0{,}9, \; \beta_2 = 0{,}999$$; chia cho $$(1 - \beta^t)$$ chuẩn hóa lại tổng trọng số của trung bình trượt về đúng 1.
3. **Adam vs. AdamW (Decoupled Weight Decay)**:
   Chỉ ra đúng lỗi của L2 gộp vào gradient bị chia cho $$\sqrt{\hat{v}_t}$$, làm mất tính đồng đều của trọng số phân rã.
4. **Vai trò của Learning Rate Warmup**:
   Ở vài trăm–vài nghìn bước đầu, phương sai ước lượng $$\hat{v}_t$$ của Adam rất nhiễu và các lớp chuẩn hóa chưa ổn định; giữ $$\eta$$ nhỏ rồi tăng dần giúp các thống kê moment ổn định trước khi bước những bước lớn.

### Khung giải thích (nên đọc chậm): Tính tay hệ số khuếch đại vận tốc cực đại của Momentum khi gradient không đổi

Giả sử dọc theo đáy thung lũng phẳng, gradient luôn giữ nguyên một hướng với giá trị hằng số $$g_t = g$$, và hệ số quán tính là $$\beta = 0{,}9$$. Ta tính vận tốc $$v_t = \beta v_{t-1} + g$$ (với $$v_0 = 0$$):

- $$v_1 = g$$
- $$v_2 = 0{,}9 g + g = 1{,}9 g$$
- $$v_3 = 0{,}9^2 g + 0{,}9 g + g = 2{,}71 g$$
- Tổng quát sau $$t$$ bước (tổng cấp số nhân):
  $$v_t = g \sum_{k=0}^{t-1} \beta^k = g \cdot \frac{1 - \beta^t}{1 - \beta} \xrightarrow{t \to \infty} \mathbf{\frac{g}{1 - \beta}}$$

Thay $$\beta = 0{,}9$$: vận tốc cuối cùng đạt $$\frac{1}{1 - 0{,}9} = \mathbf{10 \, g}$$ — tăng tốc đúng **10 lần** dọc theo đáy thung lũng mà không cần tăng learning rate (còn nếu $$\beta = 0{,}99$$ thì tăng tới **100 lần**)!

# Tùy chọn: Luyện phỏng vấn — Bộ tối ưu

> Bài này mang tính **tùy chọn** và **không** thay thế phần lý thuyết về momentum, RMSprop hay Adam. Sau khi học xong các bài chính, bạn có thể dùng bộ câu hỏi dưới đây để tự kiểm tra, sau đó tham khảo thêm cuốn *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) để luyện các câu hỏi có lời giải chi tiết về phương pháp thích nghi, siêu tham số và động lực huấn luyện.

**Các chủ đề tương ứng trong sách (đọc trong bản PDF, không tóm tắt lại ở đây):** thuật toán tối ưu (bao gồm các phép cập nhật họ Adam), lựa chọn siêu tham số và phần huấn luyện trong nhóm chủ đề học sâu mở rộng.

Trang tổng hợp: **01-98 Lộ trình luyện Deep Learning Interviews**.

## Câu hỏi luyện tập (tự soạn)

### Q1. Momentum trên thung lũng hẹp

SGD thuần túy thường dao động qua lại ngang thung lũng hẹp và tiến rất chậm dọc theo trục dài. Hãy giải thích ngắn gọn (mỗi hướng một câu) bộ đệm vận tốc $$ \mathbf{v}\leftarrow \beta\mathbf{v}+\nabla J $$ làm thay đổi hành vi đó như thế nào?

**Gợi ý.** Các thành phần đổi dấu liên tục sẽ tự triệt tiêu khi lấy trung bình, còn các gradient cùng chiều sẽ cộng dồn.

**Thảo luận.** Theo phương ngang thung lũng, dấu của gradient đảo liên tục nên trung bình vận tốc nhỏ đi và dao động tắt dần. Theo phương dọc thung lũng, gradient giữ nguyên dấu nên $$\mathbf{v}$$ tích lũy lớn dần và giúp bước đi nhanh hơn. Khi phỏng vấn, chỉ cần trình bày trực giác này; không cần khai triển công thức theo $$\beta$$ trừ khi được yêu cầu.

### Q2. Vì sao Adam cần hiệu chỉnh độ lệch (bias correction)

Moment bậc nhất trong Adam được tính theo công thức $$\mathbf{m}_t=\beta_1\mathbf{m}_{t-1}+(1-\beta_1)\mathbf{g}_t$$ với $$\mathbf{m}_0=\mathbf{0}$$. Vì sao $$\mathbf{m}_t$$ lại quá nhỏ tại $$t=1$$, và phép chia cho $$1-\beta_1^t$$ có tác dụng gì?

**Gợi ý.** Thay trực tiếp $$t=1$$ vào công thức: $$\mathbf{m}_1=(1-\beta_1)\mathbf{g}_1$$.

**Thảo luận.** Tại $$t=1$$, $$\mathbf{m}_1$$ chỉ giữ lại một phần tỷ lệ $$(1-\beta_1)$$ của gradient đầu tiên (chỉ bằng $$0.1$$ khi $$\beta_1=0.9$$). Hệ số $$1-\beta_1^t$$ chính là tổng trọng số của chuỗi hình học đã cộng tới bước $$t$$; chia cho hệ số này giúp khử hiện tượng ước lượng bị co nhỏ lúc khởi đầu. Lập luận tương tự áp dụng cho moment bậc hai với $$\beta_2$$. Sau vài trăm bước, $$\beta^t\approx 0$$ nên tác động của phép hiệu chỉnh gần như không đáng kể, nhưng trong phỏng vấn bạn vẫn cần giải thích rõ trường hợp $$t=1$$.

### Q3. Phân biệt Adam và AdamW

Giả sử bạn muốn áp dụng thành phần phạt L2 hiệu dụng $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$. Vì sao việc cộng $$\lambda\mathbf{w}$$ vào gradient rồi để Adam tự điều chỉnh thang lại không tương đương với việc giảm trực tiếp $$\mathbf{w}$$ một lượng $$\lambda\eta$$ *nằm ngoài* mẫu số thích nghi?

**Gợi ý.** Adam chia toàn bộ gradient (kể cả phần bạn cộng thêm) cho $$\sqrt{\hat{\mathbf{v}}}+\epsilon$$.

**Thảo luận.** Khi gắn weight decay vào gradient, cường độ phạt bị bóp méo theo từng tọa độ: các tham số có moment bậc hai lớn lại bị suy giảm *ít* hơn. Ngược lại, AdamW áp dụng weight decay trực tiếp lên trọng số, giúp mọi tọa độ giảm đều theo tỷ lệ $$\lambda\eta$$ (chưa tính lịch learning rate). Đây là lựa chọn mặc định trong các cấu hình huấn luyện CNN và Transformer hiện đại; câu hỏi này kiểm tra xem bạn có hiểu rõ *lý do* đằng sau sự tách rời đó hay không.

### Q4. Khi nào SGD+momentum có thể vượt Adam

Một mạng tích chập huấn luyện trên tập ảnh lớn đạt khả năng tổng quát hóa kém hơn khi dùng Adam (dù đã tinh chỉnh) so với SGD+momentum chạy lâu hơn. Hãy đưa ra một giả thuyết từ góc độ tối ưu hóa và một giả thuyết từ góc độ chính quy hóa ẩn — tránh trả lời chung chung rằng "Adam bị lỗi".

**Gợi ý.** Phương pháp thích nghi liên tục thay đổi hệ số điều chỉnh theo từng tọa độ, trong khi SGD giữ một thang bước học chung trên toàn không gian.

**Thảo luận.** Về mặt tối ưu hóa: Adam có thể nhanh chóng đi vào vùng cực tiểu nhọn của $$J$$ và dừng lại ở đó do bước học theo từng tọa độ vẫn duy trì lớn ở các chiều có gradient nhỏ. Về mặt thiên kiến ẩn (*implicit bias*): nhiễu mini-batch của SGD kết hợp với learning rate đồng nhất tạo ra phân phối dừng trên tập các cực tiểu khác với nhiễu đã qua tiền điều kiện của Adam. Cả hai đều là giả thuyết thực nghiệm chứ không phải định lý cần chứng minh trên bảng; trong thực tế, bạn nên nêu rõ cách kiểm chứng lại bằng cách chạy SGD đủ dài và đánh giá trên tập validation với nhiều hạt giống ngẫu nhiên (seed) khác nhau.

### Q5. Vai trò của Warmup trong một ý

Bạn khởi động huấn luyện Transformer bằng AdamW ngay tại learning rate đỉnh và hàm mất mát lập tức sinh ra NaN. Vì sao tăng tuyến tính $$\eta$$ trong vài nghìn bước đầu lại có thể khắc phục, dù giá trị $$\eta$$ *đỉnh* không hề thay đổi?

**Gợi ý.** Lúc mới khởi tạo, giá trị kích hoạt và logit của tầng attention chưa ổn định về thang đo; các bước cập nhật lớn lại diễn ra khi ước lượng $$\mathbf{m},\mathbf{v}$$ chưa đáng tin cậy.

**Thảo luận.** Ở giai đoạn đầu, gradient thường rất lớn và biến động mạnh, đồng thời ước lượng moment bậc hai của Adam vẫn còn ít mẫu nên thiếu ổn định (liên hệ Q2). Giữ $$\eta$$ nhỏ lúc đầu cho phép các ước lượng moment và luồng kết nối dư (*residual stream*) ổn định trước khi áp dụng bước học lớn như trong huấn luyện ImageNet hay LLM. Tuy nhiên, warmup không thể cứu một cấu hình nếu bản thân learning rate đỉnh vốn đã quá lớn — khi đó mô hình vẫn sẽ phân kỳ ngay sau khi kết thúc warmup.

## Ghi công

Kashani, S., và Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Bạn có thể tải bản PDF từ arXiv để xem toàn bộ bộ câu hỏi và lời giải. Các câu hỏi trên trang này được biên soạn riêng cho khóa học, không phải bản sao chép lại từ sách.

## Video tham khảo

- **Xem trước:** [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, tiếng Anh. Nên xem lại trước khi trả lời các câu phỏng vấn về bộ tối ưu.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, tiếng Anh. Ôn lại cách suy ra công thức cập nhật của Adam để viết lại chính xác khi phỏng vấn.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, tiếng Anh. Ôn lại bước cập nhật thuần túy khi bỏ các thành phần moment.
