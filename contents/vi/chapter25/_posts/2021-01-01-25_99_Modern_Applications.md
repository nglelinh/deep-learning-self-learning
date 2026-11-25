---
layout: post
title: 25-99 Ứng dụng và cập nhật hiện đại (2022–2026)
chapter: '25'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter25
lesson_type: optional
---

Nếu thập kỷ trước là cuộc đua xây những mô hình ngày càng lớn hơn lúc *huấn luyện*, thì giai đoạn 2024–2026 mở ra một trục mở rộng hoàn toàn mới: **cho mô hình thêm thời gian suy nghĩ và công cụ lúc suy luận**! Từ các mô hình suy luận tự kiểm tra từng bước (Inference-Time Scaling), các tác tử lập trình/nghiên cứu gọi công cụ qua vòng lặp phản hồi (Tool-Using Agents), đến các mô hình đa phương thức nguyên bản hiểu liền mạch văn bản, ảnh, âm thanh và video — bài cập nhật này tổng kết những hướng nghiên cứu đã chính thức bước vào sản phẩm hàng ngày.

## Kiến thức cần nhớ: Ba trụ cột đã bước vào thực tế (2024–2026)

- **Mở rộng tính toán lúc suy luận (Inference-Time / Test-Time Scaling):** Dùng chuỗi tư duy dài (Long CoT), lấy mẫu song song Best-of-$$N$$ kết hợp bộ chấm điểm (Verifier/PRM) để đổi thêm token suy nghĩ lấy bước nhảy vượt bậc về độ chính xác toán/code.
- **Tác tử sử dụng công cụ (Tool-Using Agents — ReAct & Function Calling):** Đóng gói LLM trong vòng lặp `Suy nghĩ → Gọi công cụ (tìm kiếm, chạy Python, đọc API) → Quan sát kết quả → Tự sửa lỗi`, chuyển AI từ "trả lời một lượt" sang "giải quyết tác vụ đa bước".
- **Mô hình đa phương thức nguyên bản (Native Multimodal Models):** Huấn luyện chung văn bản, ảnh, âm thanh và video ngay từ tiền huấn luyện trong cùng một không gian biểu diễn thay vì ghép nối rời rạc các bộ mã hóa.
- **Kỹ thuật đánh giá và kiểm soát rủi ro tác tử:** Thiết kế hộp cát (sandbox), giới hạn quyền hành động, và theo dõi độ tin cậy tích lũy qua nhiều bước gọi công cụ liên tiếp.

### Giải thích: Tính xác suất thành công của Best-of-$$N$$ (Test-Time Scaling) và tích lũy rủi ro của Tác tử đa bước

**Phần A — Sức mạnh của Best-of-$$N$$ với bộ kiểm chứng chính xác (ví dụ chạy bộ unit test lập trình):**
Giả sử trên một bài lập trình khó, mô hình chỉ giải đúng với xác suất $$p = 0{,}20$$ ở lần thử đầu tiên (`pass@1 = 20%`). Nếu ta sinh độc lập $$N = 8$$ lời giải rồi dùng trình biên dịch + bộ test chọn lời giải chạy qua:
$$P(\text{ít nhất 1 lời giải đúng}) = 1 - (1 - 0{,}20)^8 = 1 - (0{,}80)^8 \approx 1 - 0{,}1678 = 0{,}8322\;(83{,}2\%!).$$
Chỉ bằng cách tăng tính toán lúc suy luận gấp 8 lần mà **không cần huấn luyện lại mô hình**, tỷ lệ giải đúng đã vọt từ $$20\%$$ lên $$83{,}2\%$$!

**Phần B — Rủi ro tích lũy của Tác tử đa bước:**
Ngược lại, nếu một tác tử cần thực hiện chuỗi $$10$$ bước gọi công cụ phụ thuộc nhau, và xác suất thành công mỗi bước là $$95\%$$ ($$0{,}95$$), thì xác suất hoàn thành trọn vẹn cả 10 bước mà không có cơ chế tự sửa lỗi chỉ còn:
$$(0{,}95)^{10} \approx 0{,}5987\;(59{,}9\%).$$
Con số này cho thấy vì sao mọi tác tử thực chiến đều bắt buộc phải có **vòng lặp quan sát phản hồi (feedback loop) và tự phục hồi lỗi**!

# Tùy chọn: Những hướng đã đi vào thực tế — Tính toán lúc suy luận, Tác tử và Đa phương thức

> Bài này là **tùy chọn**, **không** viết lại phần tổng quan về định luật mở rộng, đa phương thức hay an toàn AI của chương. Nội dung cập nhật **các hệ thống cụ thể đã được triển khai trong giai đoạn 2024–2026**, giúp kết nối các xu hướng tương lai với những công cụ có thể sử dụng ngay.

Định luật lũy thừa trong bài cốt lõi

$$L \propto N^{-\alpha}$$

vẫn mô tả quá trình tiền huấn luyện. Trục mở rộng mới trong thực tế công nghiệp là **tính toán lúc suy luận** (*test-time compute*): chi thêm tài nguyên tính toán cho *mỗi truy vấn* thay vì chỉ mở rộng lúc huấn luyện.

## 1. Mở rộng tính toán lúc suy luận (Inference-Time Scaling)

[OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/) (2024) và các mô hình suy luận tiếp theo (như dòng DeepSeek-R1 năm 2025) huấn luyện mô hình tạo chuỗi suy luận dài kết hợp tìm kiếm trước khi trả lời. Kết quả thực nghiệm cho thấy sai số có thể giảm theo lượng tính toán lúc suy luận:

$$\text{error} \approx c \cdot C_{\mathrm{test}}^{-\gamma}$$

bổ sung cho hướng chỉ tăng số tham số $$N$$ khi tiền huấn luyện ([Snell et al., 2024](https://arxiv.org/abs/2408.03314), *Scaling LLM Test-Time Compute*). Hướng này bổ trợ cho quy luật tối ưu lúc *huấn luyện* của Chinchilla (xem bài tùy chọn Chương 00/01).

## 2. Ứng dụng cụ thể

### Tác tử sử dụng công cụ (Tool-Using Agents)

Các hệ thống “tác tử” (*agent*) trong thực tế giai đoạn 2024–2026 vận hành theo vòng lặp: mô hình ngôn ngữ → gọi công cụ (SQL, trình duyệt, thực thi mã) → quan sát kết quả và lặp lại. Hệ sinh thái công cụ phổ biến gồm Agents / Responses API của OpenAI, [LangGraph](https://github.com/langchain-ai/langgraph) và [smolagents](https://github.com/huggingface/smolagents). Thách thức nghiên cứu chính hiện nay nằm ở độ tin cậy và kiểm soát lỗi, hơn là thiết kế một loại nơ-ron hoàn toàn mới.

### Mô hình đa phương thức nguyên bản

Các hệ thống như GPT-4V, Gemini và các mô hình thị giác–ngôn ngữ mở (LLaVA, Qwen2-VL, PaliGemma) xử lý trực tiếp văn bản, hình ảnh và âm thanh trong cùng một kiến trúc. Nếu CLIP và Flamingo trong bài cốt lõi là nền tảng nghiên cứu ban đầu, thì đây là các sản phẩm và API hoàn chỉnh phát triển từ nền tảng đó.

### Tính bền vững trở thành ràng buộc kỹ thuật

Việc dùng mô hình 70B cho mọi yêu cầu người dùng không còn là câu chuyện tương lai mà là bài toán chi phí và năng lượng trực tiếp. Lượng tử hóa (xem bài tùy chọn Chương 23), định tuyến truy vấn đơn giản sang mô hình nhỏ và kỹ thuật gom lô (*batching*) là những giải pháp thực tế đang được triển khai để giải quyết bài toán năng lượng.

## 3. Phần mềm phổ biến

- [vLLM](https://github.com/vllm-project/vllm) và [SGLang](https://github.com/sgl-project/sglang) — phục vụ thông lượng cao, hỗ trợ giải mã suy luận và xuất định dạng có cấu trúc.
- Các bản mô hình suy luận công khai trên Hugging Face (như các phiên bản chưng cất từ DeepSeek-R1).
- Bộ khung đánh giá mở: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## 4. Trích dẫn (2022–2026)

- [Scaling LLM Test-Time Compute (Snell et al., 2024)](https://arxiv.org/abs/2408.03314).
- [DeepSeek-R1 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948) — huấn luyện năng lực suy luận quy mô lớn bằng học tăng cường (xem thêm bài tùy chọn Chương 21).
- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — công thức xây dựng mô hình thị giác–ngôn ngữ mở được áp dụng rộng rãi trong giai đoạn 2024–2026.

## 5. Bài này bổ sung gì cho ghi chú cốt lõi

Phần thảo luận về định luật mở rộng, NAS, học liên tục và an toàn trong bài cốt lõi vẫn giữ vai trò định hướng. Bài này cập nhật ba mảng — **tính toán lúc suy luận, tác tử dùng công cụ và mô hình đa phương thức mở** — đã chuyển từ hướng nghiên cứu sang hệ thống thực tế trong giai đoạn 2024–2026.

## Video tham khảo

- **Xem trước** [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — Andrej Karpathy, tiếng Anh. Bài giảng chuyên sâu năm 2025 của Karpathy, bao gồm phần mở rộng tính toán lúc suy luận bổ sung trên nền tiền huấn luyện.
- [Chinchilla Explained: Compute-Optimal Massive Language Models](https://www.youtube.com/watch?v=PZXN7jm9IC0) — Edan Meyer, tiếng Anh. Ôn lại định luật Chinchilla lúc huấn luyện để đối chiếu với trục mở rộng lúc suy luận.
- [MIT 6.S191 (2024): Language Models and New Frontiers](https://www.youtube.com/watch?v=N1fbskTpwZ0) — Alexander Amini, tiếng Anh. Tổng quan về tác tử và hệ thống đa phương thức được đề cập trong bài.
