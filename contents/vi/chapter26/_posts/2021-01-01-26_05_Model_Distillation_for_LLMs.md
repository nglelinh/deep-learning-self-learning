---
layout: post
title: 26-05 Chưng cất mô hình cho LLM
chapter: '26'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter26
---

Ở Chương 23, chưng cất tri thức (Knowledge Distillation) trông rất gọn gàng: chỉ cần lấy toàn bộ vector logit của mô hình thầy chia nhiệt độ $$T$$ rồi dạy mô hình học sinh. Nhưng bước sang thế giới LLM, công thức đó lập tức vấp phải ba bức tường thực tế: từ điển thầy–trò có thể khác nhau, ma trận logit cho chuỗi 4.096 token với từ điển 128.000 từ nặng hàng gigabyte, và nhiều mô hình thầy mạnh nhất chỉ mở qua API "hộp đen"! Bài học này giải mã ba trường phái chưng cất LLM hiện đại.

## Kiến thức cần nhớ: Bốn trụ cột của Chưng cất Mô hình cho LLM

- **Ôn lại KD cổ điển và điểm nghẽn trên LLM:** Khi $$|V| = 128{,}000$$, việc lưu và truyền phân phối Softmax đầy đủ trên mỗi token trở nên quá đắt đỏ, chưa kể bài toán lệch bộ tách từ (*tokenizer mismatch*).
- **Nhóm 1 — Chưng cất hộp đen qua dữ liệu tổng hợp (Sequence-Level / CoT Distillation):** Dùng mô hình thầy sinh lời giải kèm chuỗi tư duy chi tiết, lọc chất lượng bằng bộ kiểm chứng (verifier), rồi huấn luyện SFT cho học sinh.
- **Nhóm 2 — Chưng cất hộp trắng cấp Logit (Forward KL, Reverse KL, Skew KL):** Chỉ dùng Top-$$k$$ logit hoặc đảo chiều phân kỳ KL (Reverse KL tập trung vào mode chính thay vì trải phẳng *mode-covering* như Forward KL).
- **Nhóm 3 — Chưng cất On-Policy (GKD / MiniLLM):** Để chính mô hình học sinh tự sinh chuỗi rồi nhờ thầy chấm logit ngay trên quỹ đạo đó, khắc phục triệt để lệch phân phối (*exposure bias*) lúc suy luận.

### Giải thích: Tính dung lượng lưu trữ ma trận Logit đầy đủ vs. Top-$$k$$ và so sánh Forward KL vs. Reverse KL

**Phần A — Vì sao không thể lưu toàn bộ Logit của LLM?**
Xét chỉ **1 chuỗi** dài $$L = 4{,}096$$ token với từ điển $$|V| = 128{,}000$$ lưu ở FP16 ($$2\text{ byte/số}$$):
- Lưu toàn bộ logit: $$4{,}096 \times 128{,}000 \times 2\text{ byte} \approx 1{,}05\text{ GB}$$ cho *một* ví dụ duy nhất! Với 1 triệu ví dụ sẽ tốn hơn **1 Petabyte**!
- Chỉ lưu Top-$$k = 64$$ logit cao nhất (kèm chỉ số token 4 byte): $$4{,}096 \times 64 \times (2 + 4)\text{ byte} \approx 1{,}57\text{ MB}$$ — **nhỏ hơn 668 lần** mà vẫn giữ gần trọn khối xác suất.

**Phần B — Khác biệt hành vi giữa Forward KL và Reverse KL:**
Giả sử thầy đặt xác suất lên 2 câu trả lời hợp lệ $$p = [0{,}5,\; 0{,}5,\; 0{,}0]$$ (token 3 sai hoàn toàn), còn học sinh nhỏ chỉ đủ sức học 1 kiểu trả lời:
- **Forward KL** $$\sum p_i \ln(p_i/q_i)$$: nếu học sinh gán $$q_2 \to 0$$ thì $$0{,}5\ln(0{,}5/0) \to +\infty$$ → ép học sinh trải đều xác suất (dễ rớt vào vùng trung bình vô nghĩa).
- **Reverse KL** $$\sum q_i \ln(q_i/p_i)$$: cho phép học sinh tập trung vào 1 mode tốt ($$q = [0{,}98, 0{,}02, 0]$$ chỉ chịu phạt hữu hạn $$\approx 0{,}60$$) nhưng phạt cực nặng nếu học sinh lỡ gán xác suất cho token sai $$p_3 = 0$$ — giúp LLM học sinh ít bịa đặt hơn!

# Chưng cất mô hình cho LLM

Chương 23 đã giới thiệu **chưng cất tri thức (KD)** như một phương pháp nén mô hình: huấn luyện một mạng học sinh nhỏ khớp theo dự đoán của mạng giáo viên lớn. Nguyên lý đó vẫn giữ nguyên đối với LLM, nhưng có hai điểm thay đổi quan trọng: *không gian đầu ra* là một từ vựng rất lớn dự đoán tuần tự từng bước, và *mức độ truy cập* tới giáo viên có thể chỉ là đầu ra văn bản thay vì toàn bộ trọng số.

Bài này ôn lại KD cổ điển, phân tích lý do công thức năm 2015 trở nên khó áp dụng trực tiếp cho mô hình ngôn ngữ tự hồi quy, sau đó phân loại ba nhóm phương pháp chưng cất hiện đại kèm mã minh họa.

## 1. Ôn tập KD cổ điển (nền tảng Chương 23)

[Hinton, Vinyals và Dean, 2015](https://arxiv.org/abs/1503.02531) (*Distilling the Knowledge in a Neural Network*) huấn luyện mô hình học sinh trên phân bố xác suất **mềm** của mô hình giáo viên thay vì chỉ dùng nhãn cứng *one-hot*.

**Lợi ích của mục tiêu mềm:** Nhãn cứng chỉ cho biết “ảnh này thuộc lớp 7”. Một mô hình giáo viên đã huấn luyện có thể dự đoán “lớp 7: 0,80; lớp 3: 0,15; lớp 8: 0,04…”. Phần xác suất phân bổ trên các lớp còn lại thể hiện *mức độ tương đồng giữa các lớp* — điều mà Hinton et al. gọi là **tri thức ẩn** (*dark knowledge*), cung cấp tín hiệu gradient giàu thông tin hơn so với nhãn đúng/sai đơn thuần.

**Vai trò của nhiệt độ $$T$$:** Ở $$T = 1$$, phân bố của giáo viên thường quá nhọn (gần như *one-hot*), làm mờ phần tri thức ẩn. Softmax có nhiệt độ:

$$p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

làm phẳng phân bố khi tăng $$T > 1$$. Mô hình học sinh khớp theo $$p_{\mathrm{teacher}}^{(T)}$$ bằng phân kỳ KL kết hợp với hàm mất mát trên nhãn cứng:

$$\mathcal{L} = \alpha\, T^{2}\, \mathrm{KL}\big(p_{\mathrm{teacher}}^{(T)} \,\|\, p_{\mathrm{student}}^{(T)}\big) + (1-\alpha)\,\mathrm{CE}(y, z_{\mathrm{student}}).$$

Hệ số $$T^{2}$$ bù lại sự suy giảm biên độ gradient khi tăng $$T$$, còn $$\alpha$$ cân bằng giữa việc học theo giáo viên và khớp nhãn gốc. Đây là thiết lập chuẩn cho bài toán **phân loại với số lớp $$K$$ cố định** (như ImageNet; xem lại bài **23-01** nếu cần).

## 2. Vì sao KD logit cổ điển khó áp dụng trực tiếp cho LLM?

Một LLM chỉ-decoder thực hiện phân loại *tại mọi vị trí trong chuỗi* với số lớp $$K = V$$ lên tới hàng chục hoặc hàng trăm nghìn. Điều này gây ra bốn trở ngại thực tế:

- **Chi phí lưu trữ và tính toán:** Lưu hoặc truyền toàn bộ phân bố giáo viên tại mỗi token đòi hỏi bộ nhớ và I/O cỡ $$O(V)$$, nhanh chóng vượt trội chi phí tính toán của chính mô hình học sinh trên chuỗi dài.
- **Tính thưa của phân bố:** Dù có tăng nhẹ nhiệt độ $$T$$, phần lớn xác suất vẫn dồn vào một số ít token; khớp toàn bộ $$V$$ phần tử gây lãng phí trừ khi cắt theo *top-k* hoặc lấy mẫu.
- **Giới hạn quyền truy cập:** KD theo logit đòi hỏi quyền truy cập **hộp trắng** (hoặc ít nhất truy cập đầy đủ logit) vào mô hình giáo viên, trong khi nhiều giao diện API chỉ trả về văn bản (hoặc một vài *logprob* đầu bảng).
- **Tính tuần tự của chuỗi:** Sai số tích lũy dần qua từng bước sinh; khớp phân bố token của giáo viên *trên tiền tố của giáo viên* không hoàn toàn đồng nghĩa với khớp *toàn bộ câu trả lời* khi học sinh tự sinh.

Vì vậy, KD theo logit vẫn được dùng trong nội bộ các tổ chức sở hữu cả giáo viên lẫn học sinh (thường kết hợp cắt *top-k*), nhưng hiếm khi được dùng trực tiếp như công thức mặc định cho mọi bài toán LLM.

## 3. Ba nhóm phương pháp chưng cất hiện đại

Các phương pháp chưng cất LLM có thể phân nhóm theo **mục tiêu học sinh cần khớp** và **mức độ truy cập giáo viên yêu cầu**.

### 3.1 Chưng cất qua dữ liệu tổng hợp / phản hồi (hộp đen)

Mô hình giáo viên sinh các câu trả lời mẫu — gồm đáp án, chuỗi suy luận từng bước, mã nguồn hoặc bản viết lại tài liệu. Ta lưu tập cặp `(prompt, teacher_text)` rồi huấn luyện mô hình học sinh bằng SFT dự đoán token kế tiếp thông thường (có thể trộn thêm dữ liệu do con người viết).

Đây là hướng tiếp cận **hộp đen** phổ biến nhất vì chỉ cần mẫu văn bản đầu ra từ giáo viên mà không cần logit hay trạng thái ẩn. Nhiều dự án tinh chỉnh mô hình nhỏ tuân thủ chỉ dẫn, cũng như các mô hình học sinh học theo chuỗi suy luận dài từ mô hình suy luận lớn, đều thuộc nhóm này.

**Giới hạn cần lưu ý:** Học sinh học được hành vi bề mặt và phong cách xử lý trong phạm vi các prompt đã phủ, nhưng không tự động kế thừa độ hiệu chỉnh xác suất trên toàn từ vựng hay các năng lực không xuất hiện trong tập dữ liệu sinh. Vì giáo viên có thể sinh các mẫu khác nhau cho cùng một prompt, khâu thiết kế tập prompt, lọc chất lượng và trộn dữ liệu thật đóng vai trò quyết định không kém hàm mất mát.

### 3.2 Chưng cất đặc trưng / trạng thái ẩn (hộp trắng)

Mô hình học sinh được huấn luyện để khớp các biểu diễn trung gian của giáo viên như trạng thái ẩn, bản đồ *attention* hoặc phép chiếu luồng phần dư (*residual stream*). Cách này đòi hỏi **quyền truy cập hộp trắng** vào mô hình giáo viên cùng các lớp chiếu trung gian (*adapter*) khi số chiều hoặc độ sâu khác nhau, tương tự các phương pháp khớp đặc trưng trong thị giác máy tính.

### 3.3 Chưng cất logit / phân bố token (hộp trắng)

Áp dụng trực tiếp ý tưởng năm 2015 tại từng vị trí chuỗi: khớp $$p_{\mathrm{student}}(\cdot \mid x_{<t})$$ với $$p_{\mathrm{teacher}}(\cdot \mid x_{<t})$$, thường kết hợp nhiệt độ, cắt *top-k* hoặc biến thể *reverse-KL* để mô hình học sinh tập trung vào các chế độ xác suất chính. [Sanh et al., 2019](https://arxiv.org/abs/1910.01108) (DistilBERT) là ví dụ kinh điển cho kiến trúc encoder, còn với mô hình sinh tự hồi quy có các nghiên cứu như MiniLLM ([Gu et al., 2024](https://arxiv.org/abs/2306.08543)).

Đây là cách chuyển giao phân bố sát nhất nhưng đòi hỏi điều kiện chặt chẽ nhất: dùng chung tokenizer (hoặc có ánh xạ tương thích), đủ tài nguyên tính hoặc lưu logit giáo viên và thống nhất chính sách chọn tiền tố.

| Nhóm phương pháp | Mục tiêu học sinh khớp | Quyền truy cập giáo viên | Ứng dụng phổ biến |
| --- | --- | --- | --- |
| Phản hồi / dữ liệu tổng hợp | Văn bản đầu ra (và chuỗi suy luận nếu có) | Chỉ cần mẫu đầu ra | Giáo viên nguồn mở, tập SFT sản phẩm |
| Đặc trưng / trạng thái ẩn | Giá trị kích hoạt ở các tầng trung gian | Hộp trắng | Nội bộ, kiến trúc cùng họ |
| Logit / phân bố token | Phân bố softmax trên từ vựng $$V$$ | Hộp trắng (hoặc đủ *logprob*) | Nội bộ, dùng chung tokenizer |

Trong thực tế, nhiều quy trình kết hợp hai hướng: tạo tập SFT tổng hợp rồi bổ sung thêm số hạng khớp logit trên một tập con token khi cả hai mô hình cùng chạy nội bộ.

## 4. Sử dụng hợp lệ và giới hạn điều khoản

Chưng cất là kỹ thuật nén và tối ưu hóa tiêu chuẩn khi bạn **có quyền hợp lệ để huấn luyện từ trọng số hoặc đầu ra của mô hình giáo viên**, chẳng hạn:

- Thu nhỏ chính mô hình lớn của tổ chức mình thành phiên bản nhẹ hơn cho thiết bị biên, xử lý lô hoặc triển khai nội bộ.
- Sử dụng các mô hình **giấy phép mở** cho phép rõ ràng việc chưng cất hoặc sinh dữ liệu tổng hợp.
- Nghiên cứu trên các mô hình công khai trong phạm vi điều khoản cho phép.

Vấn đề pháp lý và sản phẩm phát sinh từ đặc điểm cố hữu: **đầu ra văn bản của một API đủ hữu ích cũng đồng thời là tín hiệu huấn luyện**. Do đó, nhiều nhà cung cấp đặt ra **điều khoản dịch vụ** và biện pháp kỹ thuật (giới hạn tần suất, phát hiện lạm dụng) nhằm ngăn việc thu thập đầu ra API để huấn luyện mô hình cạnh tranh.

Nguyên tắc thực hành rõ ràng: nếu bạn không sở hữu mô hình giáo viên và điều khoản dịch vụ cấm dùng đầu ra để huấn luyện mô hình khác, không nên xây dựng tập dữ liệu chưng cất từ API đó; hãy sử dụng mô hình giáo viên có giấy phép mở phù hợp, mô hình tự huấn luyện hoặc tập dữ liệu công khai. Nếu muốn đọc thêm góc nhìn tổng quan về chủ đề này, có thể tham khảo bài viết [A Gentle Introduction to Model Distillation](https://machinelearningmastery.com/a-gentle-introduction-to-model-distillation/) trên Machine Learning Mastery (Chugani, 2026).

## 5. Mã minh họa

### 5.1 Chưng cất có nhiệt độ cho bài toán phân loại cố định

Đoạn mã minh họa thiết lập của Chương 23, làm rõ vai trò hệ số $$T^{2}$$ và chiều tính phân kỳ KL:

```python
import torch
import torch.nn.functional as F

def classification_kd_loss(student_logits, teacher_logits, labels,
                           temperature=4.0, alpha=0.7):
    """Soft-target KD for a shared, small class set.

    student_logits, teacher_logits: (batch, num_classes)
    labels: (batch,) integer class ids
    """
    t = temperature
    hard = F.cross_entropy(student_logits, labels)
    log_p_s = F.log_softmax(student_logits / t, dim=-1)
    p_t = F.softmax(teacher_logits / t, dim=-1)
    # KL(teacher || student); T^2 restores gradient scale as T grows
    soft = F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)
    return alpha * soft + (1.0 - alpha) * hard
```

Khi mở rộng sang LLM, `num_classes` trở thành `vocab_size` lặp qua từng bước thời gian kèm `ignore_index` cho phần đệm — từ đó thấy rõ vì sao thực tế thường chuyển sang cắt *top-k* hoặc SFT trên văn bản.

### 5.2 Chưng cất qua phản hồi dưới dạng tập dữ liệu SFT

Không cần định nghĩa hàm mất mát mới; dữ liệu do giáo viên sinh được tổ chức như một tập tinh chỉnh chỉ dẫn thông thường:

```python
# Mỗi hàng là văn bản giáo viên sinh; học sinh phải gán likelihood cao.
sft_rows = [
    {
        "prompt": "Give a one-paragraph intuition for dropout.",
        "completion": "Dropout randomly masks units at train time so ...",
    },
    {
        "prompt": "Now show a tiny numeric example.",
        "completion": "Suppose a hidden layer has four units and p=0.5 ...",
    },
]
# Huấn luyện bằng CE token kế trên `completion` (và khuôn chat), như 26-01 / 26-02.
# Tùy chọn: giữ một phần do người viết để học sinh không chỉ chép tật của giáo viên.
```

Nếu giáo viên sinh kèm chuỗi lập luận, phần lập luận đó được lưu ngay trong `completion` (hoặc một trường riêng nối trước đáp án); thuật toán huấn luyện vẫn là SFT chuẩn, điểm khác biệt nằm ở nguồn tạo dữ liệu mục tiêu.

## 6. Đọc thêm

- Hinton, G., Vinyals, O. và Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — bài báo gốc về mục tiêu mềm, nhiệt độ và tri thức ẩn.
- Machine Learning Mastery — [A Gentle Introduction to Model Distillation](https://machinelearningmastery.com/a-gentle-introduction-to-model-distillation/) (Chugani, 2026) — tổng quan nhập môn so sánh chưng cất cổ điển và chưng cất thời LLM.
- Các bài liên quan trong khóa học: **23-01 Nén mô hình** (cắt tỉa, lượng tử hóa, KD cổ điển), **23-99** (GPTQ, AWQ, *speculative decoding*), **26-04** (tối ưu phục vụ trước khi chưng cất) và **26-07** (ước lượng FLOPs và bộ nhớ khi chọn kích thước học sinh).
- Tài liệu kỹ thuật mở rộng: DistilBERT ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)), Distilling Step-by-Step ([Hsieh et al., 2023](https://arxiv.org/abs/2305.02301)) và MiniLLM ([Gu et al., 2024](https://arxiv.org/abs/2306.08543)).

## Điểm chính cần nhớ

- KD cổ điển huấn luyện học sinh khớp phân bố lớp mềm của giáo viên; tham số nhiệt độ giúp bộc lộ tri thức ẩn giữa các lớp.
- Với LLM tự hồi quy, khớp đầy đủ logit trên toàn bộ từ vựng vừa tốn chi phí vừa thường không khả thi nếu thiếu quyền truy cập hộp trắng.
- Các phương pháp chưng cất LLM hiện đại chia thành ba nhóm chính: **phản hồi (dữ liệu tổng hợp)**, **đặc trưng ẩn** và **logit**.
- Chưng cất hợp lệ dựa trên mô hình tự sở hữu hoặc giấy phép mở cho phép; thu thập đầu ra API trái điều khoản là vấn đề pháp lý và hợp đồng.
- Phân kỳ KL có nhiệt độ giúp hiểu bản chất lý thuyết, còn tập SFT sinh bởi giáo viên là định dạng huấn luyện phổ biến nhất cho các mô hình học sinh LLM.

## Video tham khảo

- **Xem trước** [Knowledge Distillation: How LLMs train each other](https://www.youtube.com/watch?v=jrJKRYAdh7I) — Julia Turc, tiếng Anh. Giải thích cách mô hình ngôn ngữ lớn huấn luyện mô hình nhỏ hơn, bám sát phần chưng cất hiện đại trong bài.
- [2.3 Knowledge Distillation | Geoffery Hinton | Soft Labels | Softmax with Temperature](https://www.youtube.com/watch?v=g1r4Esohw-g) — AI_Snippets, tiếng Anh. Ôn lại chưng cất nhãn mềm và nhiệt độ của Hinton từ Chương 23.
- [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, tiếng Anh. Giúp thấy bối cảnh chi phí bộ nhớ khi làm việc với từ vựng lớn, làm rõ hạn chế của KD logit thuần túy.
