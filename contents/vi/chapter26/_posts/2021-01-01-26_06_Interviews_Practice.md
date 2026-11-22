---
layout: post
title: 26-06 Luyện phỏng vấn
chapter: '26'
order: 8
owner: Deep Learning Course
lang: vi
categories:
- chapter26
lesson_type: optional
---

# Tùy chọn: Câu hỏi luyện phỏng vấn về LLM

> Bài này là **tùy chọn** và **không** thay thế nội dung các bài **26-01** đến **26-05** hay **26-07**. Bộ câu hỏi dưới đây được biên soạn riêng cho khóa học để tự kiểm tra sau khi hoàn thành chương.

Hãy luyện tập khi đã nắm vững hàm mất mát dự đoán token kế tiếp, sự khác biệt giữa ICL và SFT, quy trình SFT → tối ưu sở thích, KV-cache, ba nhóm chưng cất và các công thức ước lượng FLOPs/bộ nhớ ở bài **26-07**. Lộ trình luyện phỏng vấn toàn khóa nằm ở bài **01-98**.

## P1. Tiền huấn luyện thực sự tối ưu điều gì?

Trong khoảng hai phút, hãy định nghĩa LLM theo phạm vi chương này trong một câu, nêu hàm mất mát tiền huấn luyện và những tính chất mà hàm mất mát đó *không* tự đảm bảo.

**Gợi ý.** Kiến trúc chỉ-decoder, dự đoán token kế tiếp, không đảm bảo tính đúng đắn sự kiện.

**Thảo luận.** LLM là mô hình ngôn ngữ Transformer nhân quả được huấn luyện bằng cross-entropy dự đoán token kế tiếp trên kho văn bản lớn. Hàm mất mát tối ưu hóa xác suất xuất hiện theo thống kê ngữ liệu, chứ không trực tiếp đảm bảo tính xác thực sự kiện, khả năng tuân thủ chỉ dẫn hay chi phí bộ nhớ thấp — những yêu cầu đó cần các bước thích nghi và tối ưu ở phía sau (**26-02** đến **26-04**).

## P2. Prompting, tinh chỉnh và truy hồi (RAG)

Một đồng nghiệp yêu cầu “mô hình phải luôn trả về đúng cấu trúc JSON hóa đơn của công ty”. Hãy nêu một giải pháp bằng *prompting*, một giải pháp cập nhật trọng số và một lý do vẫn có thể cần truy hồi thông tin.

**Gợi ý.** Khuôn mẫu / ICL; SFT hoặc LoRA; phân biệt định dạng với dữ liệu sự kiện.

**Thảo luận.** Ràng buộc định dạng JSON cố định có thể giải quyết bằng prompt có cấu trúc kèm ví dụ *few-shot*, hoặc bằng SFT/LoRA nếu cần độ ổn định cao và prompt gọn. Trong khi đó, truy hồi (**18-99**) vẫn cần thiết cho các dữ liệu thay đổi theo thời gian như danh mục mã sản phẩm (SKU) hay điều khoản hiện hành, vì prompting và căn chỉnh không thể tự tạo ra dữ liệu nghiệp vụ mới.

## P3. Vì sao không phải lúc nào cũng chưng cất theo logit?

Nêu hai lý do khiến một nhóm kỹ sư chỉ sử dụng API hội thoại thương mại sẽ chọn **chưng cất qua phản hồi** thay vì chưng cất đầy đủ phân bố token như công thức năm 2015.

**Gợi ý.** Quyền truy cập và kích thước từ vựng $$V$$.

**Thảo luận.** Thứ nhất, hầu hết API chỉ trả về văn bản (hoặc vài *logprob* đầu), không cung cấp toàn bộ vectơ phân bố $$V$$ chiều tại mỗi bước. Thứ hai, ngay cả khi có logit, việc lưu trữ và tính toán trên vectơ kích thước $$V$$ cho các chuỗi dài vẫn rất tốn kém; do đó văn bản đầu ra là tín hiệu khả thi hơn về mặt kỹ thuật (và chỉ được phép sử dụng khi điều khoản dịch vụ cho phép).

## P4. Vai trò của nhiệt độ $$T$$ trong chưng cất

Một mô hình giáo viên đạt xác suất 99,9% cho lớp đúng trên ImageNet. Vì sao cần tăng nhiệt độ $$T$$ trước khi tính phân kỳ KL, và điều gì xảy ra nếu tăng $$T$$ quá lớn đồng thời bỏ hẳn số hạng nhãn cứng?

**Gợi ý.** Tri thức ẩn đối lập với phân bố đều.

**Thảo luận.** Tăng $$T$$ làm nổi bật xác suất của các lớp ngoài lớp đúng, giúp bộc lộ mối quan hệ tương đồng giữa các lớp (tri thức ẩn). Tuy nhiên, khi $$T \to \infty$$, phân bố tiến về phân bố đều, khiến mô hình học sinh mất thông tin phân biệt. Giữ số hạng cross-entropy trên nhãn cứng (hoặc chọn $$T$$ vừa phải) giúp neo giữ đỉnh phân bố chính xác. Với phân bố token rất nhọn trong LLM, thực tế thường kết hợp cắt *top-k* thay vì chỉ tăng $$T$$ thật lớn.

## P5. Tối ưu phục vụ trước khi chưng cất

Nêu hai biện pháp bạn sẽ thử **trước khi** huấn luyện một mô hình học sinh 8B từ mô hình giáo viên 70B tự sở hữu, và một tình huống vẫn cần chưng cất.

**Gợi ý.** Thứ tự ưu tiên từ **26-04** sang **26-05**.

**Thảo luận.** Trước hết nên thử lượng tử hóa (4-bit / GGUF), cắt giảm ngữ cảnh thừa, áp dụng *continuous batching* hoặc *speculative decoding*. Chưng cất trở nên cần thiết khi *bản thân kiến trúc* bắt buộc phải thu nhỏ (ví dụ triển khai trên thiết bị biên, cần phân khúc chi phí thấp hơn hoặc cần một mô hình nền nhỏ để gắn nhiều *adapter* LoRA) — điều mà lượng tử hóa trên cùng bộ trọng số 70B không đáp ứng được.

## P6. Phân biệt chưng cất hợp lệ và sử dụng trái điều khoản

Trong bốn câu, hãy phân biệt giữa chưng cất hợp lệ và việc thu thập dữ liệu đầu ra trái phép.

**Gợi ý.** Quyền sở hữu mô hình giáo viên và mục đích của giao diện API.

**Thảo luận.** Khi bạn sở hữu mô hình giáo viên hoặc sử dụng mô hình nguồn mở có giấy phép cho phép chưng cất và sinh dữ liệu tổng hợp, chưng cất là kỹ thuật nén và tối ưu sản phẩm hoàn toàn hợp lệ. Ngược lại, với API dịch vụ của bên thứ ba, văn bản đầu ra chịu ràng buộc của điều khoản sử dụng và thường bị cấm dùng để huấn luyện mô hình cạnh tranh. Mâu thuẫn mang tính cấu trúc này xuất phát từ việc một API hữu ích tự nhiên phát ra chuỗi token mang giá trị như tín hiệu huấn luyện.

## P7. Đạo hàm softmax và cơ chế softmax trực tuyến

Viết biểu thức $$\partial\mathcal{L}/\partial z$$ cho cross-entropy theo token sau softmax. Sau đó nêu ngắn gọn vai trò của *log-sum-exp* và đóng góp của *online softmax* trong FlashAttention.

**Gợi ý.** $$p - t$$ và giá trị lớn nhất cập nhật theo khối.

**Thảo luận.** Với nhãn *one-hot* $$t$$, đạo hàm là $$\partial\mathcal{L}/\partial z = p - t$$. Phép *log-sum-exp* trừ đi giá trị lớn nhất của hàng để tránh tràn số khi tính $$e^z$$. *Online softmax* duy trì giá trị cực đại tạm thời, tổng chuẩn hóa và tổng có trọng số của $$V$$ khi duyệt qua từng khối, giúp FlashAttention không phải lưu toàn bộ ma trận $$L \times L$$ xuống HBM (**26-01**, **26-07**).

## P8. Cấu trúc khối decoder hiện đại và ước lượng chi phí

Phác thảo một tầng decoder hiện đại (RMSNorm → attention nhân quả + RoPE → residual → RMSNorm → SwiGLU → residual), sau đó nêu ước lượng thứ tự độ lớn cho FLOPs lan truyền xuôi mỗi token và hai thành phần chính của bộ nhớ suy luận.

**Gợi ý.** $$2N$$ FLOPs; trọng số cộng KV-cache.

**Thảo luận.** Lan truyền xuôi tốn khoảng $$2 N_{\mathrm{params}}$$ FLOPs cho mỗi token, lan truyền ngược tốn khoảng gấp đôi ($$\approx 4 N_{\mathrm{params}}$$), nên toàn bộ huấn luyện xấp xỉ $$6 N T$$ FLOPs. Bộ nhớ suy luận gồm dung lượng lưu trọng số cộng KV-cache; phần KV-cache tăng tuyến tính theo độ dài chuỗi $$L$$ và tỉ lệ với số đầu $$n_{\mathrm{kv}}$$ (giảm nhờ GQA). Khi ước lượng cụ thể, luôn tính từ $$d$$, $$n_{\mathrm{layers}}$$ và $$V$$ thay vì đoán số.

## P9. Speculative Decoding khác gì với chưng cất?

Một đồng nghiệp nói: “*Speculative decoding* là cách chúng ta huấn luyện mô hình học sinh 8B”. Hãy chỉnh lại nhận định này trong bốn câu và viết công thức lấy mẫu theo nhiệt độ cùng *top-p*.

**Gợi ý.** Giữ nguyên phân bố mô hình đích so với huấn luyện bộ trọng số $$\theta$$ mới.

**Thảo luận.** *Speculative decoding* là kỹ thuật tối ưu ở tầng *phục vụ*: mô hình nháp nhỏ đề xuất trước vài token và mô hình đích lớn kiểm định song song, giữ nguyên chính xác phân bố đầu ra của mô hình đích ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)). Ngược lại, chưng cất huấn luyện một bộ trọng số $$\theta$$ hoàn toàn mới cho mô hình nhỏ hơn. Khi lấy mẫu, phân bố theo nhiệt độ là $$p_i^{(T)} \propto e^{z_i / T}$$, còn *top-p* giữ lại tập token nhỏ nhất có tổng xác suất tích lũy $$\ge p$$ rồi chuẩn hóa lại (**26-04**).

## P10. Phân bổ tối ưu theo tính toán (Kaplan và Chinchilla)

Với ngân sách FLOP huấn luyện cố định $$C \approx 6NT$$, kết luận ban đầu của Kaplan et al. nhấn mạnh điều gì và nghiên cứu Chinchilla của Hoffmann et al. đã điều chỉnh ra sao (không cần nêu số mũ thực nghiệm)?

**Gợi ý.** Tỉ lệ cân đối giữa số tham số $$N$$ và số token $$T$$.

**Thảo luận.** Kaplan et al. (2020) cho thấy trong thiết lập của họ, tăng số tham số $$N$$ nhanh hơn số token $$T$$ có vẻ hiệu quả hơn. Hoffmann et al. (2022) chạy lại thí nghiệm theo đường đồng mức FLOP (*isoFLOP*) và chỉ ra nhiều mô hình trước đó bị thiếu dữ liệu huấn luyện, nên cần mở rộng **đồng thời cả $$N$$ và $$T$$**. Trong thực tế sản phẩm sau này, nhiều mô hình còn được huấn luyện *vượt* ngưỡng Chinchilla (nhiều token hơn cho cùng $$N$$) để đổi lấy mô hình nhỏ hơn và rẻ hơn khi phục vụ suy luận (**26-07**).

## Cách sử dụng bài luyện

Hãy tự trình bày câu trả lời thành lời trong khoảng 90 giây trước khi đối chiếu với phần thảo luận. Nếu cần ôn lại toán RL, xem **21-99**; công thức KD cổ điển nằm ở **23-01** và **26-05**; sơ đồ khối decoder và hạch toán FLOPs nằm ở **26-07**.

## Video tham khảo

- **Xem trước** [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — Andrej Karpathy, tiếng Anh. Hệ thống hóa quá trình huấn luyện và phục vụ LLM trước khi tự trả lời các câu hỏi phỏng vấn.
- [But what is cross-entropy? | Compression is Intelligence Part 2](https://www.youtube.com/watch?v=GlYgs6v2YfU) — 3Blue1Brown, tiếng Anh. Ôn trực quan cross-entropy và đạo hàm softmax cho câu P7.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, tiếng Anh. Minh họa cấu trúc khối Transformer phục vụ câu hỏi P8.
