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

Đọc công thức GPT-2 trên giấy giống như đọc bản vẽ kiến trúc của một tòa nhà, còn tương tác trực tiếp với **Transformer Explainer** (công cụ trực quan hóa chạy ngay GPT-2 trong trình duyệt của nhóm nghiên cứu Georgia Tech) giống như bước thẳng vào bên trong tòa nhà đó và nhìn dòng điện chạy qua từng dây dẫn: từ lúc một câu tiếng Anh bị cắt thành token, đi qua 12 tầng Multi-Head Attention và MLP, cho tới khi biến thành xác suất chọn từ tiếp theo!

## Kiến thức cần nhớ: Bốn chặng quan sát cốt lõi trên Transformer Explainer

1. **Từ văn bản thô sang Token ID và Vector Embedding + Vị trí**:
   Thấy trực tiếp cách bộ tách từ BPE (Byte-Pair Encoding) chia từ hiếm thành nhiều mảnh (*subwords*) và cộng vector ngữ nghĩa với vector vị trí thành đầu vào `768` chiều của GPT-2 Small.
2. **Dòng chảy Q, K, V và bản đồ chú ý nhân quả trên 12 Head**:
   Nhìn thấy rõ ràng đường chéo tam giác dưới của Causal Self-Attention: mỗi token chỉ nối dây về các token đứng trước nó, và các head khác nhau tạo ra các đường nối sáng đậm ở những từ hoàn toàn khác nhau.
3. **Vai trò mở rộng $$4\times$$ của khối MLP (768 $$\to$$ 3072 $$\to$$ 768)**:
   Quan sát quá trình biến đổi phi tuyến qua GELU tinh chỉnh biểu diễn của từng token sau khi đã thu thập ngữ cảnh từ Attention.
4. **Logits $$\to$$ Temperature $$\to$$ Top-k / Top-p (Nucleus) Sampling**:
   Thấy ngay trong thời gian thực: chỉ cần kéo thanh trượt Temperature hoặc Top-p, danh sách các từ ứng viên tiếp theo lập tức co gọn (chính xác, ít bất ngờ) hoặc giãn rộng (sáng tạo, đa dạng).

### Giải thích: Tính tay bước lọc Top-p (Nucleus Sampling) với $$p = 0{,}80$$

Giả sử sau khi qua `softmax` (tại nhiệt độ $$\tau = 1{,}0$$), GPT-2 gán xác suất cho 5 từ ứng viên đứng đầu (đã sắp xếp giảm dần) như sau:

| Thứ hạng | Từ ứng viên | Xác suất gốc $$P(w)$$ | Xác suất tích lũy | Giữ lại với Top-p ($$p=0{,}80$$)? | Xác suất sau chuẩn hóa lại |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `"data"` | $$0{,}45$$ | $$0{,}45$$ | **Giữ** | $$0{,}45 / 0{,}85 = \mathbf{52{,}9\%}$$ |
| 2 | `"models"` | $$0{,}25$$ | $$0{,}70$$ | **Giữ** | $$0{,}25 / 0{,}85 = \mathbf{29{,}4\%}$$ |
| 3 | `"learning"` | $$0{,}15$$ | $$0{,}85 \ge 0{,}80$$ | **Giữ** (đạt ngưỡng!) | $$0{,}15 / 0{,}85 = \mathbf{17{,}6\%}$$ |
| 4 | `"pizza"` | $$0{,}10$$ | $$0{,}95$$ | Loại bỏ | $$0\%$$ |
| 5 | `"cloud"` | $$0{,}05$$ | $$1{,}00$$ | Loại bỏ | $$0\%$$ |

Top-p tự động co nhóm ứng viên lại khi mô hình đã rất chắc chắn (đôi khi chỉ giữ 1–2 từ) và mở rộng nhóm khi có nhiều câu trả lời hợp lý — khắc phục nhược điểm cứng nhắc của Top-k cố định!

# Transformer Explainer

Các phương trình trong chương này mô tả *những gì Transformer tính toán*, còn [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) cho phép bạn *quan sát trực tiếp* các phép tính đó trên một mô hình ngôn ngữ thực tế mà không cần cài đặt hay dùng GPU. Công cụ chạy trực tiếp trên trình duyệt mô hình **GPT-2 (small)** (124 triệu tham số, 12 khối Transformer xếp chồng), nên mọi con số hiển thị đều là kết quả truyền xuôi thật từ prompt bạn nhập.

![Tổng quan Transformer Explainer](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_summary.png)
*Giao diện tương tác: prompt được tách token, sau đó đi qua embedding, multi-head self-attention và MLP. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

Bạn nên mở công cụ trước rồi dùng bài này như tài liệu hướng dẫn đối chiếu:

- **Demo trực tiếp:** [poloclub.github.io/transformer-explainer](https://poloclub.github.io/transformer-explainer/)
- **Mã nguồn:** [github.com/poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer)
- **Bài báo (CHI 2026):** [Cho et al., “Transformer Explainer…”](https://doi.org/10.1145/3772318.3791725) ([PDF](https://poloclub.github.io/papers/26-chi-transformer-explainer.pdf), [arXiv:2408.04619](https://arxiv.org/abs/2408.04619))
- **Video demo:** [YouTube](https://www.youtube.com/watch?v=TFUc41G2ikY)

Do giao diện được thiết kế để tương tác toàn màn hình (di chuột xem trọng số attention, mở rộng từng tầng, kéo thanh trượt), khóa học dẫn liên kết trực tiếp tới trang demo thay vì nhúng vào khung nhỏ.

## 1. Tổng quan khái niệm

GPT-2 là Transformer **chỉ có decoder**: mô hình dự đoán token tiếp theo dựa trên toàn bộ các token đứng bên trái. Kiến trúc này gọn hơn sơ đồ encoder–decoder trong *"Attention Is All You Need"* (vốn có thêm self-attention phía encoder và cross-attention giữa hai phía), nhưng vẫn giữ nguyên các khối cơ bản:

1. **Embedding** — ánh xạ token thành vectơ và cộng thêm tín hiệu vị trí.
2. **Các khối Transformer** — mỗi khối gồm **self-attention** đa đầu (trao đổi thông tin giữa các vị trí) tiếp nối bởi **MLP** (biến đổi cục bộ từng vị trí), đi kèm kết nối phần dư (*residual*).
3. **Xác suất đầu ra** — lớp tuyến tính kết hợp softmax chuyển trạng thái ẩn cuối thành phân phối xác suất trên toàn bộ từ vựng để lấy mẫu token kế tiếp.

GPT-2 (small) sử dụng bộ từ vựng 50.257 token và chiều embedding 768. Cả 12 khối đều có cùng cấu trúc; giao diện thường mở chi tiết **Block 1** và chú thích 11 khối còn lại phía sau. Mô hình trong trình duyệt được chuyển đổi từ nanoGPT của Andrej Karpathy và chạy qua ONNX Runtime, giúp các giá trị cập nhật tức thì khi bạn đổi prompt hoặc điều chỉnh tham số lấy mẫu.

![Hình teaser có chú thích của toàn bộ giao diện](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_teaser.jpg)
*Hình tổng quan giao diện chính thức (đã điều chỉnh kích thước cho trang khóa học): (A) nhập prompt và sinh token kế tiếp; (B) điều chỉnh nhiệt độ và chế độ lấy mẫu; (C) mở rộng chi tiết phép tính attention (tích vô hướng, chia tỉ lệ kèm mask nhân quả, softmax). Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

## 2. Ý nghĩa từng phần trên giao diện

### Embedding, tách token và mã hóa vị trí

Khi mở cột **Embedding**, prompt trước tiên được tách thành các token con (*subword*): những từ phổ biến như `Data` hay `visualization` giữ nguyên một token, trong khi `empowers` được tách thành `em` và `powers`. Mỗi token ứng với một chỉ số nguyên và một vectơ **token embedding** 768 chiều đã học.

![Embedding: token, token embedding và mã hóa vị trí](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_embedding.png)
*Bốn bước embedding cho prompt "Data visualization empowers users to": (1) tách token, (2) tra cứu token embedding theo id, (3) mã hóa vị trí, (4) cộng hai vectơ. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

**Mã hóa vị trí được cộng trực tiếp thay vì ghép nối.** Khác với bảng sin–cos cố định của Transformer gốc, GPT-2 học một ma trận embedding vị trí trong lúc huấn luyện rồi cộng vào token embedding. Nguyên tắc cốt lõi vẫn không đổi: vì self-attention thuần không tự nhận biết thứ tự, mỗi vectơ đầu vào sau phép cộng cần mang đồng thời thông tin nội dung và vị trí của token.

### Query, Key, Value và multi-head self-attention

Từ các vectơ embedding đó, mô hình tính **query (Q)**, **key (K)** và **value (V)**. Ở GPT-2 (small), thao tác này gom thành một phép nhân ma trận duy nhất: đầu vào kích thước $$(6, 768)$$ nhân với ma trận trọng số $$(768, 2304)$$ cộng độ lệch, cho ra tensor $$(6, 2304)$$. Con số 2304 bằng $$3 \times 768$$ (tương ứng Q, K, V ghép lại), sau đó được tách thành **12 đầu attention**.

![Tính Q, K và V từ embedding](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_qkv.png)
*Nhân embedding với ma trận trọng số QKV gộp (cộng độ lệch) để thu đồng thời Q, K và V. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

Có thể hình dung theo ẩn dụ tìm kiếm mà chính công cụ sử dụng: **query** là từ khóa bạn nhập, **key** là tiêu đề các trang để đối chiếu, còn **value** là nội dung trang bạn đọc sau khi khớp. Attention thực hiện phép tra cứu mềm: mỗi token xác định các token đứng trước có liên quan nhất rồi lấy tổ hợp có trọng số của các value tương ứng.

![Scaled dot-product attention với mask nhân quả](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_attention.png)
*Một đầu attention: tích vô hướng Q–K, chia tỉ lệ kèm mask nhân quả, rồi softmax (kèm dropout khi huấn luyện). Dạng tam giác dưới thể hiện ràng buộc của decoder: không được nhìn thấy token tương lai. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

Đi từ trái sang phải qua ba ma trận:

1. **Tích vô hướng** $$QK^{\top}$$ đo độ tương đồng thô giữa mọi cặp query và key.
2. **Chia tỉ lệ và áp mask** chia cho $$\sqrt{d_k}$$ (hệ số $$1/\sqrt{d_k}$$ ở Chương 07) và áp **mặt nạ nhân quả** gán $$-\infty$$ cho các vị trí tương lai.
3. **Softmax** chuẩn hóa từng hàng thành phân phối xác suất trên các vị trí hợp lệ, sau đó nhân với $$V$$ để ra đầu ra của đầu attention.

Bạn có thể chuyển đổi giữa 12 đầu và di chuột lên từng token để quan sát mức chú ý của nó. Các đầu thường chuyên biệt hóa theo vai trò khác nhau (như cú pháp cục bộ hoặc liên kết ngữ nghĩa xa). Cuối cùng, đầu ra của 12 đầu được nối lại và chiếu qua lớp tuyến tính trước khi cộng vào luồng phần dư.

### Khối MLP

Sau khi attention định tuyến thông tin giữa các vị trí, **MLP** (mạng feed-forward hai tầng) xử lý độc lập từng token. Trong GPT-2 (small), lớp ẩn mở rộng từ 768 lên $$4 \times 768 = 3072$$ với hàm kích hoạt GELU, rồi chiếu trở lại 768 chiều, bao bọc bởi kết nối phần dư.

![MLP nở lên 3072 chiều với đường residual](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_mlp.png)
*MLP theo từng vị trí mở rộng mỗi token lên 3072 chiều rồi chiếu lại 768 chiều và cộng vào đường phần dư. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

### Xác suất đầu ra, nhiệt độ (temperature) và lấy mẫu

Sau 12 khối Transformer, lớp tuyến tính cuối ánh xạ trạng thái ở vị trí cuối thành **logit** — một điểm số cho mỗi từ trong từ vựng — rồi softmax chuyển thành phân phối xác suất cho token tiếp theo.

Tham số **temperature** ($$T$$) chia toàn bộ logit cho $$T$$ *trước* bước softmax:

- $$T = 1$$: giữ nguyên phân phối gốc.
- $$T < 1$$: phân phối sắc nét hơn, ưu tiên các ứng viên xác suất cao (ổn định và ít ngẫu nhiên hơn).
- $$T > 1$$: phân phối phẳng hơn, tăng cơ hội cho các ứng viên xếp hạng thấp (đa dạng hơn nhưng cũng dễ lệch ngữ cảnh hơn).

Hai bộ lọc phổ biến giúp giới hạn tập ứng viên trước khi lấy mẫu:

- **Top-k:** chỉ giữ $$k$$ token có điểm cao nhất; phần còn lại bị gán $$-\infty$$ để nhận xác suất 0.
- **Top-p** (*nucleus sampling*): giữ tập token nhỏ nhất có tổng xác suất tích lũy đạt ngưỡng $$p$$.

![Temperature, lấy mẫu top-k và xác suất token kế tiếp](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_softmax.png)
*Với temperature 0.8 và top-k = 5, logit được chia tỉ lệ, các token ngoài top 5 bị loại rồi áp softmax cho phân phối token kế tiếp. Với prompt “…users to”, từ "visualize" dẫn đầu với 54,67%. Nguồn: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (giấy phép MIT).*

Trong hình minh họa, temperature 0.8 chuyển logit $$-135.91$$ thành $$-169.89$$ ($$-135.91 / 0.8$$). Sau lọc top-k, các từ `quickly`, `explore` và `find` bị loại khỏi danh sách ứng viên. Vì bước cuối là lấy mẫu xác suất chứ không luôn lấy $$\arg\max$$, từ *"create"* vẫn có khoảng 21% cơ hội được chọn.

## 3. Bài tập thực hành có hướng dẫn

Hãy mở [demo trực tiếp](https://poloclub.github.io/transformer-explainer/) và thực hiện lần lượt các bước sau với prompt mặc định *"Data visualization empowers users to"*:

1. **Quan sát tách token:** Mở cột Embedding. Từ nào giữ nguyên một token, từ nào bị tách nhỏ? Lý do bộ từ vựng *subword* hoạt động như vậy là gì?
2. **Kiểm tra phép cộng vị trí:** Xác nhận mỗi vectơ đầu vào là tổng của token embedding và vectơ vị trí (0, 1, 2, …). Điều gì sẽ sai nếu bỏ bước này? (Đối chiếu với phần *ablation* trong bài **08-01-02-02**.)
3. **So sánh các đầu attention:** Mở một đầu ở Block 1, di chuột lên token cuối và ghi lại các token trước có trọng số cao nhất, rồi đổi sang một đầu khác để so sánh mức độ tương đồng.
4. **Kiểm tra mặt nạ nhân quả:** Trong ma trận *"Scaling · Mask"*, xác định vùng bị che. Giải thích vì sao mô hình chỉ-decoder vẫn phải che các vị trí tương lai ngay cả khi toàn bộ prompt đã được nhập sẵn.
5. **Thử nghiệm temperature:** Giữ nguyên prompt, lần lượt đặt temperature gần 0.1 rồi gần 1.5 và nhận xét sự thay đổi của biểu đồ xác suất.
6. **So sánh top-k và top-p:** Giữ temperature ở 0.8, thử `top-k = 1`, `top-k = 5`, rồi chuyển sang `top-p`. Quan sát cách các token ngoài ngưỡng biến mất ở cột Softmax.
7. **Thử với prompt riêng:** Nhập một câu tiếng Anh ngắn (hoặc trộn tiếng Anh và tiếng Việt) rồi nhấn Generate. Lưu ý rằng đây là GPT-2 small chứ không phải LLM hiện đại đã tinh chỉnh theo chỉ dẫn, nên năng lực dự đoán có giới hạn dù dùng chung nền tảng kiến trúc.

## 4. Khái niệm liên quan

- **Chương 07**: scaled dot-product attention, multi-head attention và mặt nạ nhân quả chính là các ma trận hiển thị trong công cụ.
- **08-01-01 Lý thuyết Transformer**: mã hóa vị trí và cấu trúc khối hoàn chỉnh (lưu ý GPT-2 dùng PE tuyệt đối học được thay vì sin–cos cố định).
- **08-01-02-02 Demo, bài báo và bẫy thường gặp**: tự huấn luyện mô hình nhỏ; kết hợp với Explainer để quan sát một mô hình đã tiền huấn luyện thực tế.
- **Chương 26**: tokenization, giải mã và chiến lược lấy mẫu ở quy mô LLM.

## 5. Các bài báo nền tảng

1. **[Transformer Explainer (Cho et al., CHI 2026)](https://doi.org/10.1145/3772318.3791725)**  
   Công trình giới thiệu công cụ trực quan hóa tương tác GPT-2 chạy trên trình duyệt này (bản thảo trên [arXiv:2408.04619](https://arxiv.org/abs/2408.04619)).

2. **[Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**  
   Dòng mô hình chỉ-decoder đang vận hành bên trong công cụ.

3. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   Kiến trúc Transformer encoder–decoder gốc dùng để đối chiếu với kiến trúc chỉ-decoder trong Explainer.

## 6. Điểm chính

1. Transformer sinh văn bản hoạt động theo quy trình dự đoán token kế tiếp: embedding → các khối attention + MLP xếp chồng → softmax trên từ vựng.
2. Trọng số attention là độ tương đồng đã chuẩn hóa và áp mặt nạ, quyết định trực tiếp cách mỗi token tổng hợp ngữ cảnh.
3. Temperature, top-k và top-p không thay đổi trọng số mô hình mà chỉ biến đổi phân phối xác suất lấy mẫu từ cùng một vectơ logit.
4. GPT-2 small đủ gọn để chạy trực tiếp trong trình duyệt và đủ rõ để minh họa cụ thể từng phép toán của Chương 08.

## Ghi nguồn và giấy phép

Transformer Explainer do Aeree Cho, Grace C. Kim, Alexander Karpekov, Alec Helbling, Jay Wang, Seongmin Lee, Benjamin Hoover và Polo Chau phát triển tại Georgia Institute of Technology (Polo Club of Data Science), phát hành theo [giấy phép MIT](https://github.com/poloclub/transformer-explainer/blob/main/LICENSE). Các hình trong bài được trích hoặc điều chỉnh kích thước từ thư mục chính thức `static/preview` và `static/article_assets` trong [kho mã nguồn công khai](https://github.com/poloclub/transformer-explainer) (hình teaser được thu từ 8135×3895 xuống 2400×1149 cho trang khóa học). Khi trích dẫn công cụ, sử dụng mục BibTeX sau:

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

- **Xem trước:** [Transformer Explainer Demo](https://www.youtube.com/watch?v=TFUc41G2ikY) — Polo Club / Georgia Tech (tiếng Anh). Giới thiệu nhanh cách tương tác với công cụ trực quan hóa GPT-2 dùng trong bài.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown (tiếng Anh). Minh họa bằng hoạt hình cùng luồng embedding → attention → MLP → softmax.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Mã nguồn nanoGPT làm nền chuyển đổi cho mô hình chạy trên trình duyệt.
