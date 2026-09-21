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

# Chưng cất mô hình cho LLM

Chương 23 đã giới thiệu **chưng cất tri thức (KD)** như một công cụ nén: học sinh nhỏ khớp giáo viên lớn. Bức tranh đó vẫn đúng. Điều đổi với LLM là *không gian đầu ra* (từ vựng token khổng lồ, từng bước một) và *giao diện* bạn có tới giáo viên (đầy đủ trọng số so với chỉ văn bản).

Bài này ôn KD cổ điển, giải thích vì sao công thức 2015 trở nên vụng với LM tự hồi quy, rồi trình bày ba họ hiện đại bằng ngôn ngữ giảng dạy gốc. Đây **không** phải bản in lại blog hay bài báo nào.

## 1. Ôn KD cổ điển (phần cốt lõi Chương 23)

[Hinton, Vinyals và Dean, 2015](https://arxiv.org/abs/1503.02531), *Distilling the Knowledge in a Neural Network*, huấn luyện học sinh trên phân bố lớp **mềm** của giáo viên, không chỉ trên nhãn one-hot.

**Vì sao mục tiêu mềm giúp.** Nhãn cứng nói “ảnh này là lớp 7.” Giáo viên đã huấn luyện có thể nói “7 với 0,80, 3 với 0,15, 8 với 0,04, …” Khối xác suất ngoài đường chéo mã hóa *sai lầm nào là hợp lý* — cấu trúc tương tự mà Hinton et al. gọi là **tri thức tối** (*dark knowledge*). Học sinh nhận gradient giàu hơn “đúng so với sai.”

**Nhiệt độ.** Ở $$T = 1$$ giáo viên tự tin gần như one-hot, nên tín hiệu thêm biến mất. Softmax với nhiệt độ

$$p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

làm dẹt phân bố khi $$T > 1$$ tăng. Học sinh khớp $$p_{\mathrm{teacher}}^{(T)}$$ (thường bằng divergence KL) và thường vẫn khớp nhãn cứng. Một phối trộn phổ biến là

$$\mathcal{L} = \alpha\, T^{2}\, \mathrm{KL}\big(p_{\mathrm{teacher}}^{(T)} \,\|\, p_{\mathrm{student}}^{(T)}\big) + (1-\alpha)\,\mathrm{CE}(y, z_{\mathrm{student}}).$$

Hệ số $$T^{2}$$ giữ thang gradient không sụp khi bạn tăng $$T$$ (xem ghi chú 2015 và đoạn mã Chương 23). $$\alpha$$ đánh đổi “bắt chước giáo viên” với “trúng nhãn tập dữ liệu.”

Đó là mô hình đầu đúng cho **phân loại $$K$$ cố định** (ImageNet, senone tiếng nói, đầu 10 lớp). Đọc lại **23-01** nếu đoạn này còn mới.

## 2. Vì sao KD logit cổ điển vụng với LM từ vựng lớn

LM decoder là bộ phân loại *tại mọi vị trí*, với $$K = V$$ hàng chục hoặc hàng trăm nghìn, và “một mẫu” là một *chuỗi* các quyết định đó.

Ma sát thực tế:

- **Chi phí.** Lưu hoặc phát toàn bộ phân bố giáo viên mỗi token tốn thêm bộ nhớ và I/O $$O(V)$$. Với chuỗi dài, phần đó lấn át cả logit của học sinh.
- **Thưa.** Sau một $$T$$ nhỏ, hầu như toàn bộ khối lượng $$V$$ nằm trên một nắm token. Bạn trả tiền để khớp các gần-không trừ khi cắt top-$$k$$ hoặc lấy mẫu.
- **Quyền truy cập.** Huấn luyện trên logit giáo viên thường cần giáo viên **hộp trắng** (hoặc ít nhất lộ logit). API chat công khai thường trả *văn bản*, đôi khi vài logprob, không phải vector $$V$$ mỗi bước.
- **Chuỗi, không phải ảnh i.i.d.** Lỗi cộng dồn theo vòng sinh. Khớp phân bố token giáo viên *cho trước tiền tố của giáo viên* khác với khớp *câu trả lời đã hoàn thiện* của giáo viên.

Vì thế các nhóm vẫn dùng KD logit **trong nhà** khi họ sở hữu cả hai mô hình (và thường giới hạn top-$$k$$ hoặc một số vị trí). Họ không coi công thức 2015 như thứ thả vào là “làm bản sao rẻ của mô hình API đóng.”

## 3. Ba họ hiện đại (cách nhóm gốc)

Tên gọi thay đổi giữa các bài báo. Hãy nhóm theo **học sinh được yêu cầu khớp cái gì** và **bạn cần quyền truy cập nào**.

### 3.1 Chưng cất dữ liệu tổng hợp / phản hồi (vào văn bản, ra văn bản)

Giáo viên sinh phần hoàn thiện — đáp án, chuỗi lập luận từng bước, mã, tài liệu viết lại. Bạn lưu `(prompt, teacher_text)` và huấn luyện học sinh bằng SFT token kế thông thường trên văn bản đó (đôi khi trộn dữ liệu người).

Đây là mẫu **hộp đen** chiếm ưu thế: bạn chỉ cần mẫu từ giáo viên, không cần $$z$$ hay hidden state. Các bản clone tuân thủ chỉ dẫn đầu những năm 2020 (các dự án nghiên cứu tinh chỉnh LM nhỏ trên đầu ra của mô hình chat mạnh hơn) thuộc họ này. Công trình sau này nhờ giáo viên viết *lập luận* và dạy học sinh sinh cả token suy luận lẫn đáp án, để học sinh nhận mục tiêu dài và có cấu trúc hơn một chuỗi cuối ngắn.

**Thứ được chuyển.** Hành vi bề mặt và, nếu prompt phủ tác vụ, một phần *quyết định* của giáo viên. **Thứ không tự động chuyển.** Hiệu chỉnh của toàn bộ $$p(\cdot\mid x_{<t})$$, đặc trưng nội tại, hay năng lực mà tập prompt không bao giờ khêu ra.

Vì học sinh chỉ thấy chuỗi đã lấy mẫu, hai mẫu giáo viên cho cùng prompt có thể mâu thuẫn. Thiết kế tập (nhiệt độ, lọc, trộn nhãn thật) quan trọng không kém loss.

### 3.2 Chưng cất đặc trưng / trạng thái ẩn (hộp trắng)

Học sinh khớp kích hoạt trung gian: hidden state, bản đồ attention, hoặc một phép chiếu residual stream của giáo viên. Việc này cần **độ sâu khớp hoặc adapter học được** và quyền vào nội tại giáo viên. Đó là analogue LM của các loss gợi ý dùng trong KD thị giác.

Dùng khi bạn kiểm soát checkpoint giáo viên và muốn học sinh chia sẻ *hình học biểu diễn*, không chỉ chuỗi cuối. Không hợp với API bạn không thể gắn đo.

### 3.3 Chưng cất logit / phân bố token (hộp trắng)

Áp ý tưởng 2015 tại mỗi vị trí: khớp $$p_{\mathrm{teacher}}(\cdot \mid x_{<t})$$ và $$p_{\mathrm{student}}(\cdot \mid x_{<t})$$, thường với nhiệt độ và thường với biến thể reverse-KL hoặc top-$$k$$ để học sinh không phí dung lượng vào đuôi. [Sanh et al., 2019](https://arxiv.org/abs/1910.01108) (DistilBERT) là ví dụ *encoder* kinh điển; các bài LM sinh khảo sát loss liên quan dưới tên như MiniLLM ([Gu et al., 2024](https://arxiv.org/abs/2306.08543)).

Đây là bản sao *phân bố* trung thành nhất, và đòi hỏi nhất: tokenizer giống (hoặc ánh xạ được), logit giáo viên lưu sẵn hoặc tính tại chỗ, và thường cùng chính sách tiền tố.

| Họ | Mục tiêu học sinh | Quyền giáo viên | Dùng điển hình |
| --- | --- | --- | --- |
| Phản hồi / dữ liệu tổng hợp | *Văn bản* giáo viên (và lập luận tùy chọn) | Chỉ mẫu | Giáo viên mở, tập SFT sản phẩm, hầu hết công thức công khai |
| Đặc trưng / trạng thái ẩn | Kích hoạt tầng | Hộp trắng | Trong nhà, kiến trúc họ hàng |
| Logit / phân bố token | Softmax trên $$V$$ | Hộp trắng (hoặc đủ logprob) | Trong nhà, cùng tokenizer |

Nhiều pipeline sản xuất **trộn**: sinh tập SFT tổng hợp, rồi thêm số hạng logit rẻ trên một tập con token nếu cả hai mô hình chạy nội bộ.

## 4. Dùng hợp lệ trong nhà so với gặt không được phép

Chưng cất là kỹ thuật thông thường khi bạn **có quyền huấn luyện trên đầu ra hoặc trọng số của giáo viên**:

- thu nhỏ mô hình frontier của chính bạn thành tầng nhanh hơn (di động, batch, on-prem),
- theo **giấy phép mở** liệt kê chưng cất hoặc dữ liệu tổng hợp là mục đích dự kiến (một số checkpoint mở lớn được phát hành với câu chuyện đó),
- nghiên cứu trên mô hình công khai mà điều khoản cho phép.

Căng thẳng cấu trúc không phải bí ẩn về một công ty cụ thể. **Một mô hình đủ hữu ích qua API sẽ phát ra văn bản cũng là tín hiệu huấn luyện.** Nhà cung cấp vì thế đặt **điều khoản dịch vụ** và kiểm soát kỹ thuật (giới hạn tốc độ, phát hiện lạm dụng) quanh việc dùng đầu ra để dựng mô hình cạnh tranh. Đó là vấn đề hợp đồng và toàn vẹn sản phẩm: năng lực rò qua cùng kênh bạn mở cho khách.

Khóa này **không** tóm lại cáo buộc theo vòng tin, yêu sách kiện tụng, hay số liệu sự cố chưa kiểm chứng. Những thứ đó đổi rất nhanh và dễ sai. Nếu bạn cần cuộc tranh luận công nghiệp như đọc thêm, hãy dùng một bài hướng dẫn đặt *kỹ thuật* lên trước — ví dụ [A Gentle Introduction to Model Distillation](https://machinelearningmastery.com/a-gentle-introduction-to-model-distillation/) trên Machine Learning Mastery (Chugani, 2026) — và coi mọi mục tranh cãi là **tin cần tự kiểm**, không phải nguồn sơ cấp.

**Quy tắc tự học.** Nếu bạn không sở hữu giáo viên và điều khoản cấm huấn luyện trên đầu ra, đừng dựng tập chưng cất từ API đó. Dùng giáo viên giấy phép mở, mô hình của bạn, hoặc tập dữ liệu công khai.

## 5. Phác thảo mã ngắn

### 5.1 KD có nhiệt độ cho bộ phân loại *cố định*

Đây là bối cảnh Chương 23, viết sao cho $$T^{2}$$ và chiều KL rõ. Đây **không** phải bộ huấn luyện LM đầy đủ.

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

Để hình dung analogue LM, nghĩ `num_classes = vocab_size` và vòng theo thời gian, với `ignore_index` trên padding — rồi nhớ vì sao người ta chuyển sang top-$$k$$ hoặc SFT văn bản.

### 5.2 Chưng cất phản hồi *chính là* một tập tinh chỉnh

Không cần loss mới. Bạn dựng các hàng mà học sinh sẽ thấy như dữ liệu chỉ dẫn thông thường:

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

Nếu giáo viên cũng viết lập luận, lưu nó *trong* `completion` (hoặc trường thứ hai bạn nối). Học sinh vẫn đang làm SFT; điều mới là **ai viết mục tiêu**.

## 6. Đọc thêm

- Hinton, G., Vinyals, O., và Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). Nguồn lịch sử cho mục tiêu mềm, nhiệt độ, và tri thức tối.
- Machine Learning Mastery — [A Gentle Introduction to Model Distillation](https://machinelearningmastery.com/a-gentle-introduction-to-model-distillation/) (Chugani, 2026). Khung hướng dẫn thêm về họ cổ điển so với thời LLM; **trích như đọc thêm, không sao chép ở đây**.
- Khóa này: **23-01 Nén mô hình** (cắt tỉa, lượng tử, KD cổ điển) và **23-99** (GPTQ / AWQ / speculative decoding — bổ sung, không thay học sinh).
- Con trỏ kỹ thuật tùy chọn (không bắt buộc): DistilBERT ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)); chưng cất kiểu lập luận như Distilling Step-by-Step ([Hsieh et al., 2023](https://arxiv.org/abs/2305.02301)); MiniLLM ([Gu et al., 2024](https://arxiv.org/abs/2306.08543)).

## Điểm chính cần nhớ

- KD cổ điển = khớp *phân bố lớp mềm* của giáo viên; nhiệt độ làm lộ tri thức tối.
- LM tự hồi quy khiến khớp đủ logit vừa đắt vừa thường không truy cập được.
- Thực hành hiện đại tụ thành chưng cất **phản hồi (tổng hợp)**, **đặc trưng**, và **logit**.
- Chưng cất trong nhà / có giấy phép là việc hiệu năng chuẩn; huấn luyện trên API bị cấm là vấn đề điều khoản và rò năng lực, không phải thuật toán mới.
- KL có nhiệt độ là *hình dung* đúng; tệp SFT do giáo viên viết là thứ hầu hết học sinh LLM thực sự huấn luyện trên đó.
