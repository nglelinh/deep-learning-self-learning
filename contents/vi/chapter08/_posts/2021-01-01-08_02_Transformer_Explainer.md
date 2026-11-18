---
layout: post
title: 08-02 Transformer Explainer
chapter: '08'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter08
---

# Transformer Explainer

Các phương trình trong chương này nói *Transformer tính gì*. [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) cho bạn *nhìn thấy* những phép tính đó trên một mô hình ngôn ngữ thật, không cần cài đặt và không cần GPU. Công cụ chạy trực tiếp trên trình duyệt một mô hình **GPT-2 (small)** còn sống (124 triệu tham số, 12 khối Transformer xếp chồng), nên mọi con số bạn thấy đều là lượt truyền xuôi thật cho prompt bạn gõ.

![Tổng quan Transformer Explainer](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_summary.png)
*Giao diện tương tác: prompt được tách token, rồi đi qua embedding, multi-head self-attention, và MLP. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

Hãy mở công cụ trước, rồi dùng bài này như bản đồ hướng dẫn:

- **Demo trực tiếp:** [poloclub.github.io/transformer-explainer](https://poloclub.github.io/transformer-explainer/)
- **Mã nguồn:** [github.com/poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer)
- **Bài báo (CHI 2026):** [Cho et al., “Transformer Explainer…”](https://doi.org/10.1145/3772318.3791725) ([PDF](https://poloclub.github.io/papers/26-chi-transformer-explainer.pdf), [arXiv:2408.04619](https://arxiv.org/abs/2408.04619))
- **Video demo:** [YouTube](https://www.youtube.com/watch?v=TFUc41G2ikY)

Trang được thiết kế để khám phá cả màn hình (hover trọng số attention, mở rộng tầng, kéo thanh trượt). Khóa học vì thế dẫn ra demo trực tiếp, thay vì nhúng công cụ vào một khung nhỏ.

## 1. Tổng quan khái niệm

GPT-2 là Transformer **chỉ có decoder**: nó dự đoán token kế tiếp khi đã biết mọi token bên trái. Đây là kiến trúc hẹp hơn một chút so với sơ đồ encoder–decoder trong “Attention Is All You Need,” vốn còn self-attention phía encoder và cross-attention giữa encoder với decoder. Những khối bạn đã học vẫn xuất hiện, chỉ được sắp xếp cho mô hình ngôn ngữ nhân quả:

1. **Embedding** — token trở thành vector, cộng thêm tín hiệu vị trí.
2. **Các khối Transformer** — mỗi khối gồm **self-attention** đa đầu (giao tiếp giữa các vị trí) rồi đến **MLP** (tính toán trên từng vị trí), kèm kết nối thặng dư (*residual*).
3. **Xác suất đầu ra** — một lớp tuyến tính cộng softmax biến trạng thái ẩn cuối thành phân phối trên từ vựng, rồi lấy mẫu token kế tiếp từ phân phối đó.

GPT-2 (small) dùng từ vựng 50.257 token và embedding 768 chiều. Mười hai khối có cùng cấu trúc; phần trực quan hóa thường mở **Block 1** và ghi chú rằng còn mười một khối giống vậy phía sau. Mô hình trong trình duyệt được chuyển từ nanoGPT của Andrej Karpathy và chạy bằng ONNX Runtime, nên các giá trị cập nhật tức thì khi bạn sửa prompt hoặc chỉnh tham số lấy mẫu.

![Hình teaser có chú thích của toàn bộ giao diện](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_teaser.jpg)
*Hình teaser chính thức của toàn bộ giao diện (đã đổi kích thước cho trang khóa học). (A) gõ prompt và sinh token kế tiếp; (B) đổi temperature và cách lấy mẫu; (C) mở rộng phép tính attention (tích vô hướng, scale và mask nhân quả, softmax). Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

## 2. Từng phần của bản trực quan hóa cho thấy gì

### Embedding, tokenization và mã hóa vị trí

Hãy mở cột **Embedding**. Prompt trước hết được **tách token**. GPT-2 dùng từ vựng đơn vị con (*subword*): những từ trọn vẹn như `Data` hay `visualization` có thể là một token, trong khi `empowers` bị tách thành `em` và `powers`. Mỗi token có một id nguyên và một **token embedding** 768 chiều đã được học.

![Embedding: token, token embedding và mã hóa vị trí](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_embedding.png)
*Bốn bước embedding cho prompt “Data visualization empowers users to”: (1) token, (2) token embedding kèm id trong từ vựng, (3) mã hóa vị trí, (4) tổng cuối. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

**Mã hóa vị trí được cộng, không phải nối.** GPT-2 *không* dùng bảng sine/cosine cố định của bài báo Transformer gốc; nó học một ma trận embedding vị trí khi huấn luyện rồi cộng vào token embedding. Điểm cần nắm vẫn giống bài lý thuyết: self-attention vốn không biết thứ tự, nên phải “tiêm” thông tin vị trí bằng cách nào đó. Sau phép cộng, mỗi hàng là một vector mang cả “token này là gì” và “nó đứng ở đâu trong prompt.”

### Query, key, value và self-attention đa đầu

Từ các embedding đó, mô hình tạo các vector **query (Q)**, **key (K)** và **value (V)**. Với GPT-2 (small), đây chỉ là một phép nhân ma trận: embedding kích thước $$(6, 768)$$ nhân với ma trận trọng số $$(768, 2304)$$ đã học, cộng bias, cho ra tensor $$(6, 2304)$$. 2304 cột chính là $$3 \times 768$$ — Q, K và V được nối lại — rồi được tách ra cho **12 đầu attention**.

![Tính Q, K và V từ embedding](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_qkv.png)
*Embedding nhân với ma trận trọng số QKV gộp (cộng bias) cho ra Q, K và V đã nối. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

Một phép so sánh hữu ích, cũng được chính công cụ dùng: **query** là câu bạn gõ vào ô tìm kiếm, **key** là tiêu đề các trang có thể khớp, còn **value** là nội dung trang bạn thực sự đọc. Attention là phép tra cứu mềm: mỗi token hỏi “những token trước nào liên quan đến mình?” rồi trộn các value tương ứng.

![Scaled dot-product attention với mask nhân quả](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_attention.png)
*Một đầu attention: tích vô hướng Q–K, scale và mask nhân quả, rồi softmax (và dropout khi huấn luyện). Cấu trúc tam giác dưới là quy tắc của decoder: một token không được nhìn tương lai. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

Đi từ trái sang phải qua ba ma trận:

1. **Tích vô hướng** $$QK^{\top}$$ đo độ tương đồng thô giữa mọi query và mọi key.
2. **Scale và mask** chia cho $$\sqrt{d_k}$$ (đúng hệ số $$1/\sqrt{d_k}$$ ở Chương 07) và áp **mask nhân quả**, gán $$-\infty$$ cho các vị trí tương lai để chúng không được chú ý khi dự đoán token kế tiếp.
3. **Softmax** biến mỗi hàng thành phân phối xác suất trên các vị trí được phép. Các trọng số đó nhân với $$V$$ để ra đầu ra của đầu attention.

Bạn có thể đổi đầu (1 trong 12) và hover lên token để xem một token đang chú ý những vị trí nào. Các đầu thường chuyên môn hóa khác nhau: đầu này bắt cú pháp gần, đầu kia bắt liên hệ ngữ nghĩa dài hơn. Đầu ra của các đầu được nối lại rồi chiếu qua một lớp tuyến tính trước khi tiếp tục trên dòng residual.

### MLP

Sau khi attention đã *định tuyến* thông tin giữa các token, **MLP** (mạng feed-forward hai tầng) *tinh chỉnh* từng token một cách độc lập. Ở GPT-2 (small), độ rộng ẩn nở từ 768 lên $$4 \times 768 = 3072$$ với kích hoạt GELU, rồi chiếu trở lại 768. Kết nối thặng dư nằm quanh khối để biểu diễn gốc vẫn đi tiếp được.

![MLP nở lên 3072 chiều với đường residual](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_mlp.png)
*MLP theo từng vị trí nở mỗi token thành vector ẩn 3072 chiều, rồi ghi trở lại dòng residual. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

### Xác suất đầu ra, temperature và lấy mẫu

Sau cả 12 khối, một lớp tuyến tính ánh xạ vị trí cuối thành **logit** — một điểm số cho mỗi phần tử trong từ vựng. Softmax biến logit thành phân phối xác suất của token kế tiếp.

Temperature là một phép tính rất nhỏ nhưng ảnh hưởng lớn: các logit được chia cho $$T$$ *trước* softmax.

- $$T = 1$$: phân phối không đổi.
- $$T < 1$$: phân phối nhọn hơn; mô hình trông tự tin và xác định hơn.
- $$T > 1$$: phân phối dẹt hơn; các token xếp hạng thấp có xác suất cao hơn — người ta thường gọi đó là “sáng tạo” hơn (và cũng dễ sai hơn).

Hai bộ lọc phổ biến rồi hạn chế tập token được lấy mẫu:

- **Top-k:** chỉ giữ $$k$$ token có xác suất cao nhất; phần còn lại bị gán $$-\infty$$ và nhận xác suất 0.
- **Top-p** (nucleus sampling): giữ tập token nhỏ nhất sao cho tổng xác suất vượt ngưỡng $$p$$.

![Temperature, lấy mẫu top-k và xác suất token kế tiếp](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_softmax.png)
*Với temperature 0.8 và top-k = 5, logit được scale, các token ngoài top năm bị loại, rồi softmax cho phân phối token kế tiếp. Ở đây “visualize” dẫn đầu với 54,67% cho prompt “…users to”. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

Trong hình, temperature 0,8 biến logit $$-135.91$$ thành giá trị đã scale $$-169.89$$ (đúng bằng $$-135.91 / 0.8$$). Sau top-k, `quickly`, `explore` và `find` không còn là ứng viên. Lấy mẫu không phải lúc nào cũng chọn $$\arg\max$$: ngay với thiết lập này, “create” vẫn còn khoảng 21% xác suất.

## 3. Bài tập hướng dẫn

Mở [demo trực tiếp](https://poloclub.github.io/transformer-explainer/) và làm lần lượt các bước sau, ghi câu trả lời ngắn khi đi. Prompt mặc định “Data visualization empowers users to” là điểm xuất phát tốt.

1. **Đọc cách tách token.** Mở Embedding. Những từ nào còn nguyên một token, từ nào bị tách? Vì sao từ vựng đơn vị con lại làm vậy?
2. **Kiểm tra phép cộng vị trí.** Xác nhận mỗi embedding cuối là token embedding cộng vector vị trí (0, 1, 2, …). Nếu bỏ phép cộng này thì điều gì sẽ sai? (Phần ablation trong **08-01-02-02** là phiên bản lập trình của đúng câu hỏi này.)
3. **Quan sát một đầu attention.** Mở một đầu ở Block 1, hover lên token cuối, và ghi những token trước nào nhận trọng số lớn nhất. Đổi sang đầu khác. Hai đầu có đồng ý với nhau không?
4. **Tìm mask nhân quả.** Trong ma trận “Scaling · Mask”, chỉ ra những vị trí bị chặn. Vì sao mô hình chỉ có decoder phải giấu tương lai, ngay cả khi bạn đang xem một *prompt* mà bạn đã gõ sẵn?
5. **Quét temperature.** Để temperature gần 0,1 rồi gần 1,5, không đổi prompt. Mô tả thanh xác suất thay đổi thế nào. Bạn sẽ chọn mức nào khi cần một câu hoàn thành đáng tin, và mức nào khi muốn đa dạng hơn?
6. **So sánh top-k và top-p.** Giữ temperature cố định (thử 0,8), đặt top-k bằng 1, rồi bằng 5, sau đó chuyển sang top-p. Điều gì xảy ra với các token nằm ngoài ngưỡng? Điều đó hiện thế nào ở cột Softmax?
7. **Dùng câu của bạn.** Gõ một prompt tiếng Anh ngắn (hoặc trộn tiếng Anh và tiếng Việt) rồi nhấn Generate. Danh sách token kế tiếp có khớp kỳ vọng không? Nếu không, hãy nhớ đây là GPT-2 small, không phải LLM hiện đại đã tinh chỉnh theo chỉ dẫn — cùng họ kiến trúc, nhưng không cùng năng lực.

## 4. Khái niệm liên quan

- **Chương 07** — scaled dot-product attention, multi-head attention, và mask nhân quả chính là những ma trận mà explainer vẽ ra.
- **08-01-01 Lý thuyết Transformer** — mã hóa vị trí và toàn bộ khối (attention + FFN + residual + norm). Chi tiết GPT-2 cần nhớ: PE tuyệt đối được học, không phải sine/cosine.
- **08-01-02-02 Demo, bài báo và bẫy thường gặp** — một mô hình nhỏ bạn tự huấn luyện; dùng explainer khi muốn *soi* một GPT-2 đã tiền huấn luyện mà không phải viết vòng lặp huấn luyện.
- **Chương 26** — tokenization, giải mã và lấy mẫu ở quy mô LLM; bài này là phần xem trước trực quan của những lựa chọn giải mã đó.

## 5. Các bài báo nền tảng

1. **[Transformer Explainer (Cho et al., CHI 2026)](https://doi.org/10.1145/3772318.3791725)**  
   Hệ thống tương tác mà bài học này xoay quanh: GPT-2 chạy trực tiếp trên trình duyệt, với tổng quan luồng dữ liệu và các góc nhìn khi cần về embedding, attention, và xác suất token kế tiếp. Bản thảo cũng có trên [arXiv:2408.04619](https://arxiv.org/abs/2408.04619).

2. **[Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**  
   Họ mô hình chỉ có decoder đang chạy bên trong bản trực quan hóa.

3. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   Transformer encoder–decoder gốc; hãy so sánh với chồng chỉ có decoder mà bạn thấy trong explainer.

## 6. Điểm then chốt

1. Transformer sinh văn bản là bộ dự đoán token kế tiếp: embedding → các khối xếp chồng (attention + MLP) → softmax trên từ vựng.
2. Trọng số attention không phải đồ trang trí — chúng là độ tương đồng đã chuẩn hóa và đã mask, quyết định mỗi token thu thập ngữ cảnh thế nào.
3. Temperature, top-k và top-p không huấn luyện lại mô hình; chúng chỉ định hình lại cùng một vector logit thành một phân phối lấy mẫu khác.
4. GPT-2 small đủ nhỏ để chạy trên trình duyệt và đủ lớn để khiến kiến trúc trở nên cụ thể. Hãy coi nó là kính hiển vi cho Chương 08, không phải vật thay thế cho một LLM hiện đại.

## Ghi nguồn và giấy phép

Transformer Explainer do Aeree Cho, Grace C. Kim, Alexander Karpekov, Alec Helbling, Jay Wang, Seongmin Lee, Benjamin Hoover và Polo Chau tạo ra tại Georgia Institute of Technology (Polo Club of Data Science). Phần mềm được phát hành theo [giấy phép MIT](https://github.com/poloclub/transformer-explainer/blob/main/LICENSE). Các hình trong bài được sao chép hoặc đổi kích thước từ các tệp chính thức `static/preview` và `static/article_assets` trong [kho công khai](https://github.com/poloclub/transformer-explainer); hình teaser được thu từ 8135×3895 xuống 2400×1149 cho trang khóa học. Nếu dùng công cụ trong bài viết, hãy trích dẫn bài báo CHI 2026:

```bibtex
@inproceedings{cho2026transformer,
  title={Transformer Explainer: Learning LLM Transformers with Interactive Visual Explanation and Experimentation},
  author={Cho, Aeree and Kim, Grace C and Karpekov, Alexander and Lee, Seongmin and Helbling, Alec and Hoover, Benjamin and Wang, Zijie J and Kahng, Minsuk and Chau, Duen Horng},
  booktitle={Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems},
  pages={1--21},
  year={2026}
}
```

## Video tham khảo

- **xem trước** [Transformer Explainer Demo](https://www.youtube.com/watch?v=TFUc41G2ikY) — Polo Club / Georgia Tech, tiếng Anh. Một lượt đi ngắn qua bản trực quan hóa GPT-2 trực tiếp mà bài này dùng.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, tiếng Anh. Cùng câu chuyện embedding → attention → MLP → softmax, ở dạng hoạt hình thay vì mô hình sống.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, tiếng Anh. Bản cài nanoGPT mà mô hình trong trình duyệt được chuyển từ đó.
