---
layout: post
title: 07-01 Cơ sở về Cơ chế Chú ý
chapter: '07'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter07
---

# Cơ sở về Cơ chế Chú ý

![Multi-Head Attention](/deep-learning-self-learning/img/chapter_img/chapter07/multihead_attention.png)
*Sơ đồ multi-head attention được sử dụng trong Transformer. Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm

**Cơ chế chú ý** (*attention*) cho phép mô hình *tập trung động* vào những phần phù hợp nhất của đầu vào khi sinh từng phần đầu ra. Thay vì nén toàn bộ chuỗi qua một vectơ tóm tắt có kích thước cố định, attention thực hiện cơ chế **chọn lọc mềm** (*soft selection*): tại mỗi bước tính toán, mô hình gán trọng số cho các vị trí đầu vào rồi lấy tổ hợp có trọng số của các biểu diễn tương ứng.

Ý tưởng này tuy gọn gàng nhưng đã giải quyết được hạn chế cấu trúc của các mô hình chuỗi thời kỳ đầu và trở thành nền tảng cho xử lý ngôn ngữ tự nhiên cũng như nhiều bài toán thị giác máy tính hiện đại.

### Nút thắt thông tin tạo động lực cho attention

Mô hình **encoder–decoder** (seq2seq) cổ điển trong dịch máy vận hành qua ba bước:

1. **Encoder** (RNN) đọc lần lượt từng từ của câu nguồn.
2. Encoder nén toàn bộ câu nguồn vào một trạng thái ẩn cuối cùng có kích thước cố định.
3. **Decoder** (RNN) nhận vectơ duy nhất đó để sinh câu đích.

Sơ đồ:

```text
"The cat sat on the mat"
        │
        ▼
   [Encoder RNN]
        │
        ▼
   fixed context vector  z ∈ R^d     ← bottleneck
        │
        ▼
   [Decoder RNN] → "Le chat était assis sur le tapis"
```

Cách thiết kế này bộc lộ hai điểm yếu rõ rệt:

**Nút thắt thông tin.** Dù câu nguồn dài 5 từ hay 50 từ, toàn bộ thông tin đều bị ép vào một vectơ $$z$$ duy nhất. Với câu dài, nhiều chi tiết quan trọng dễ bị mờ nhạt hoặc thất thoát.

**Dùng chung một vectơ tóm tắt cho mọi từ đích.** Khi dịch một câu, ta không chỉ nhìn vào một bản tóm tắt cố định cho mọi từ: khi sinh *"chat"*, ta cần chú ý tới *"cat"*; khi sinh *"tapis"*, ta cần chú ý tới *"mat"*. Một vectơ $$z$$ cố định không hỗ trợ khả năng *căn chỉnh động* (*dynamic alignment*) này.

**Giải pháp của attention:** giữ lại *toàn bộ* các trạng thái encoder $$\mathbf{h}_1, \ldots, \mathbf{h}_T$$, và tại mỗi bước decoder $$t$$, tính riêng một vectơ ngữ cảnh:

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i}\,\mathbf{h}_i$$

trong đó các trọng số $$\alpha_{t,i}$$ được học từ dữ liệu và thay đổi theo trạng thái decoder hiện tại. Nhờ vậy, mỗi từ đích có thể tập trung vào các vị trí nguồn phù hợp nhất.

### Liên hệ trực quan

Có thể hình dung cơ chế này giống như việc trả lời câu hỏi khi đang mở sách giáo khoa trước mặt. Thay vì cố thuộc lòng cả chương thành một câu tóm tắt duy nhất rồi đóng sách lại, với mỗi ý cần viết ta sẽ tra lại đúng đoạn tương ứng. Attention chính là phiên bản khả vi và học được của thao tác tra cứu đó.

### Nội dung bài học này

Bài này xây dựng nền tảng khái niệm và lịch sử của attention gồm:

- Attention trong encoder–decoder (Bahdanau và Luong),
- Khuôn khổ Query–Key–Value,
- Scaled dot-product attention,
- Phân biệt self-attention và cross-attention,
- Giới thiệu multi-head attention.

Các bài sau trong chương sẽ đi sâu vào toán học và cài đặt, trước khi ghép thành kiến trúc Transformer hoàn chỉnh ở Chương 8.

---

## 2. Nền tảng toán học

### 2.1 Attention trong encoder–decoder dạng tổng quát

Tại bước giải mã $$t$$:

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $$\mathbf{s}_t$$ | Trạng thái ẩn của decoder (thông tin phục vụ sinh từ ở bước hiện tại) |
| $$\mathbf{h}_i$$ | Trạng thái ẩn của encoder tại vị trí nguồn $$i$$ |
| $$e_{t,i}$$ | **Điểm căn chỉnh** (*alignment score*) chưa chuẩn hóa giữa bước decoder $$t$$ và vị trí nguồn $$i$$ |
| $$\alpha_{t,i}$$ | **Trọng số attention** biểu thị mức độ chú ý vào vị trí nguồn $$i$$ |
| $$\mathbf{c}_t$$ | **Vectơ ngữ cảnh** (*context vector*) tóm lược mềm câu nguồn cho bước $$t$$ |

**Bước 1 — Tính điểm số.** Đánh giá mức độ phù hợp giữa trạng thái decoder và từng trạng thái encoder:

$$e_{t,i} = \mathrm{score}(\mathbf{s}_t, \mathbf{h}_i)$$

**Bước 2 — Chuẩn hóa bằng softmax.**

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T}\exp(e_{t,j})}$$

Khi đó $$\alpha_{t,i} \ge 0$$ và $$\sum_i \alpha_{t,i} = 1$$, tạo thành một phân phối xác suất trên các vị trí nguồn.

**Bước 3 — Tổng hợp vectơ ngữ cảnh.**

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i}\,\mathbf{h}_i$$

**Bước 4 — Dự đoán đầu ra.** Kết hợp vectơ ngữ cảnh với trạng thái decoder để dự đoán token tiếp theo:

$$\tilde{\mathbf{s}}_t = \tanh\!\big(\mathbf{W}_c[\mathbf{c}_t;\mathbf{s}_t]\big), \qquad
p(y_t \mid y_{<t}, x) = \mathrm{softmax}(\mathbf{W}_o\tilde{\mathbf{s}}_t)$$

Tùy bài báo, cách ghép $$\mathbf{c}_t$$ và $$\mathbf{s}_t$$ có thể khác nhau đôi chút, nhưng nguyên tắc chung vẫn là **dự đoán dựa trên vectơ ngữ cảnh động $$\mathbf{c}_t$$ thay vì một nút thắt cố định**.

### 2.2 Attention Bahdanau (attention cộng tính, 2015)

[Bahdanau et al.](https://arxiv.org/abs/1409.0473) đề xuất hàm chấm điểm học được:

$$e_{t,i} = \mathbf{v}_a^\top \tanh\!\big(\mathbf{W}_a\mathbf{s}_t + \mathbf{U}_a\mathbf{h}_i\big)$$

**Cơ chế hoạt động:**

- $$\mathbf{W}_a$$ và $$\mathbf{U}_a$$ chiếu trạng thái decoder và encoder về cùng một không gian chung trước khi kết hợp.
- Hàm $$\tanh$$ tạo tính phi tuyến để đo mức độ liên quan linh hoạt hơn phép nhân vô hướng thuần túy.
- Vectơ $$\mathbf{v}_a$$ ánh xạ biểu diễn đó thành một điểm số vô hướng.

Kiểu này được gọi là attention **cộng tính** (*additive*) do thành phần query và key được cộng với nhau bên trong hàm $$\tanh$$. Đây là dạng attention được ứng dụng rộng rãi đầu tiên trong dịch máy neuron (NMT), giúp cải thiện rõ rệt chất lượng dịch trên các câu dài.

### 2.3 Attention Luong (dạng nhân và tổng quát, 2015)

[Luong et al.](https://arxiv.org/abs/1508.04025) khảo sát các hàm chấm điểm gọn hơn và nhanh hơn trong thực tế:

**Tích vô hướng** (*dot product*):

$$e_{t,i} = \mathbf{s}_t^\top \mathbf{h}_i$$

**Dạng tổng quát / song tuyến** (*general / bilinear*):

$$e_{t,i} = \mathbf{s}_t^\top \mathbf{W}_a \mathbf{h}_i$$

**Dạng ghép nối (*concat*)** (tương tự Bahdanau):

$$e_{t,i} = \mathbf{v}_a^\top \tanh\!\big(\mathbf{W}_a[\mathbf{s}_t;\mathbf{h}_i]\big)$$

Tích vô hướng đo độ tương đồng hướng giữa hai vectơ: khi trạng thái decoder và trạng thái encoder cùng hướng, điểm số sẽ cao. Dạng song tuyến chèn thêm ma trận học được $$\mathbf{W}_a$$ để linh hoạt hơn khi $$\mathbf{s}_t$$ và $$\mathbf{h}_i$$ không nằm trong cùng không gian biểu diễn.

### 2.4 Khuôn khổ Query–Key–Value (Q, K, V)

Trong các kiến trúc hiện đại, attention được chuẩn hóa theo ba vai trò:

| Vai trò | Câu hỏi tương ứng | Nguồn thường gặp |
|---------|--------------------|-----------------|
| **Query** $$\mathbf{q}$$ | "Hiện tại đang cần tìm thông tin gì?" | Trạng thái decoder, hoặc vị trí $$i$$ trong self-attention |
| **Key** $$\mathbf{k}_j$$ | "Vị trí $$j$$ có đặc trưng gì để đối chiếu?" | Trạng thái encoder, hoặc mọi vị trí trong chuỗi |
| **Value** $$\mathbf{v}_j$$ | "Nếu vị trí $$j$$ khớp, cần lấy nội dung gì?" | Cùng nguồn với keys, qua ma trận chiếu riêng |

Quy trình chung gồm ba bước: (1) tính điểm giữa query và các keys, (2) chuẩn hóa bằng softmax thành trọng số, (3) tính tổng có trọng số của các values. Cách biểu diễn này bao quát cả **cross-attention** (query từ một chuỗi, key/value từ chuỗi khác) lẫn **self-attention** (cả ba đều xuất phát từ cùng một chuỗi).

Trong các mô hình ngôn ngữ lớn (LLM), quy trình này thường được nhìn như một *mạng quan hệ kết nối toàn phần*: embedding đầu vào → chiếu sang Q/K/V → tính điểm attention → chuẩn hóa Softmax → (nhiều head song song) → tổng hợp ngữ cảnh đầu ra.

![Pipeline attention kiểu LLM](/deep-learning-self-learning/img/chapter_img/chapter07/llm_attn_qkv_pipeline_active.jpg)
*Hình: Luồng xử lý Q–K–V → Softmax → multi-head → tổng hợp ngữ cảnh. (Minh họa từ video LLM Attention Mechanism)*

#### Ví dụ số minh họa: Câu 3 token → Q, K, V

Xét một câu ví dụ gồm 3 từ ("tôi / yêu / bạn"), mỗi từ biểu diễn bằng vectơ embedding 2 chiều để dễ quan sát (trong thực tế thường dùng $$d_{\mathrm{model}}=512$$):

![Embedding 3 từ](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_toy_sentence_embeddings.jpg)
*Hình: Ma trận đầu vào $$\mathbf{X}$$ kích thước 3 token × $$d$$. (Minh họa từ video công thức Attention QKV)*

Ba ma trận trọng số học được $$W^Q, W^K, W^V$$ chiếu $$\mathbf{X}$$ sang ba không gian query, key và value:

![W^Q, W^K, W^V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_weights_wq_wk_wv.jpg)
*Hình: Ba ma trận chiếu riêng biệt cho Q, K và V. (Minh họa từ video công thức Attention QKV)*

![X × W → Q, K, V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_project_x_to_qkv.jpg)
*Hình: Phép chiếu $$\mathbf{Q}=\mathbf{X}W^Q$$, $$\mathbf{K}=\mathbf{X}W^K$$, $$\mathbf{V}=\mathbf{X}W^V$$. (Minh họa từ video công thức Attention QKV)*

### 2.5 Scaled dot-product attention

Phép tính attention tiêu chuẩn trong Transformer là:

$$\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
= \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

![Q, K, V và công thức Attention](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_matrices_and_formula.jpg)
*Hình: Ba ma trận Q, K, V sau phép chiếu cùng công thức scaled dot-product attention. (Minh họa từ video công thức Attention QKV)*

Với một chuỗi đơn (tạm bỏ chiều batch), kích thước các ma trận như sau:

- $$\mathbf{Q} \in \mathbb{R}^{L_q \times d_k}$$
- $$\mathbf{K} \in \mathbb{R}^{L_k \times d_k}$$
- $$\mathbf{V} \in \mathbb{R}^{L_k \times d_v}$$
- Đầu ra $$\in \mathbb{R}^{L_q \times d_v}$$

**Các bước tính trên ví dụ 3 từ:**

1. **Tích vô hướng**: nhân từng hàng của $$Q$$ với từng cột của $$K^\top$$ để thu điểm tương đồng giữa mọi cặp từ:

![Tính từng phần tử QK^T](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_dot_product_row_example.jpg)
*Hình: Ví dụ hàng "tôi" tính tích vô hướng với tất cả các key. (Minh họa từ video công thức Attention QKV)*

![Ma trận similarity chưa scale](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_unscaled_similarities.jpg)
*Hình: Ma trận độ tương đồng chưa chia tỉ lệ $$\mathbf{Q}K^\top$$. (Minh họa từ video công thức Attention QKV)*

2. **Chia tỉ lệ (scaling)**: chia cho $$\sqrt{d_k}$$ (trong ví dụ này $$d_k=2$$ nên chia cho $$\sqrt{2}$$):

![Chia √d_k](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_scale_by_sqrt_dk.jpg)
*Hình: Chuyển từ độ tương đồng chưa chia tỉ lệ sang đã chia tỉ lệ. (Minh họa từ video công thức Attention QKV)*

**Vì sao cần chia cho $$\sqrt{d_k}$$?**

Nếu các thành phần của query và key gần như độc lập với trung bình 0 và phương sai 1, tích vô hướng $$\mathbf{q}^\top \mathbf{k}$$ sẽ có phương sai bằng $$d_k$$. Khi $$d_k$$ lớn (ví dụ 64 đến 512), độ lớn điểm số tăng mạnh khiến softmax rơi vào vùng bão hòa (gần giống one-hot) và làm gradient suy giảm gần về 0. Chia cho $$\sqrt{d_k}$$ đưa phương sai về khoảng 1 để mô hình học được phân phối trọng số mềm và ổn định hơn.

3. **Softmax theo hàng**: chuẩn hóa từng hàng thành một phân phối xác suất trên các keys:

![Softmax theo hàng](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_softmax_over_rows.jpg)
*Hình: Áp dụng softmax dọc theo từng hàng của ma trận điểm số. (Minh họa từ video công thức Attention QKV)*

![Attention weights](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_attention_weights.jpg)
*Hình: Ma trận trọng số attention với tổng mỗi hàng bằng 1. (Minh họa từ video công thức Attention QKV)*

4. **Nhân với V**: lấy tổ hợp có trọng số của các hàng trong $$V$$ để thu đầu ra attention:

![Weights × V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_weights_times_v.jpg)
*Hình: Nhân ma trận trọng số attention với V để tạo đầu ra. (Minh họa từ video công thức Attention QKV)*

### 2.6 Self-attention so với cross-attention

**Cross-attention (attention encoder–decoder).**  
Query đến từ decoder, còn key và value đến từ encoder. Đây là mô hình chú ý trong dịch máy gốc: khi sinh một từ đích, mô hình tra cứu xem những từ nguồn nào liên quan nhất.

**Self-attention.**  
Query, key và value đều được chiếu từ *cùng một* chuỗi đầu vào, cho phép mọi vị trí thu thập ngữ cảnh từ tất cả các vị trí khác:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}^K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}^V$$

Mỗi token ban đầu là một vectơ nhúng, sau đó sinh ra bộ Q, K, V riêng:

![Token = cột số](/deep-learning-self-learning/img/chapter_img/chapter07/sa_tokens_as_vectors.jpg)
*Hình: Biểu diễn câu dưới dạng các vectơ embedding. (Minh họa từ video Transformer / Self-Attention nhập môn)*

![Mỗi từ một cột Q/K/V](/deep-learning-self-learning/img/chapter_img/chapter07/sa_per_token_qkv_columns.jpg)
*Hình: Self-attention tạo bộ Q, K, V riêng cho từng vị trí trong câu. (Minh họa từ video Transformer / Self-Attention nhập môn)*

![Vai trò Q/K/V theo từ](/deep-learning-self-learning/img/chapter_img/chapter07/sa_qkv_roles_apple_phone_orange.jpg)
*Hình: Quy trình QKV áp dụng cho các từ "táo / điện thoại / cam" nhằm đối chiếu ngữ cảnh và tổng hợp giá trị. (Minh họa từ video Transformer / Self-Attention nhập môn)*

Ví dụ, trong câu:

> The animal didn't cross the street because **it** was too tired.

Self-attention có thể học liên kết trực tiếp giữa *"it"* và *"animal"* mà không cần truyền tín hiệu qua nhiều bước hồi quy trung gian.

### 2.7 Multi-head attention (mức nhập môn)

Một đầu chú ý (*attention head*) thường chỉ tập trung vào một kiểu quan hệ nhất định, trong khi ngôn ngữ chứa đồng thời nhiều loại quan hệ như cú pháp, đồng tham chiếu, lân cận hay ngữ nghĩa.

**Multi-head attention** thực hiện song song $$H$$ phép attention trên các không gian chiếu độc lập, sau đó ghép kết quả lại và chiếu qua ma trận đầu ra:

$$\mathrm{head}_h = \mathrm{Attention}(\mathbf{Q}\mathbf{W}_h^Q, \mathbf{K}\mathbf{W}_h^K, \mathbf{V}\mathbf{W}_h^V)$$

$$\mathrm{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V})
= \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)\,\mathbf{W}^O$$

Thông thường ta chọn $$d_k = d_v = d_{\mathrm{model}} / H$$ để tổng chi phí tính toán tương đương một head đầy đủ nhưng khả năng biểu diễn đa dạng hơn. Chẳng hạn với $$d_{\mathrm{model}}=512$$ và $$H=8$$, mỗi head làm việc với $$d_k=64$$:

![Chia d_model cho số head](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_dk_split.jpg)
*Hình: Chia đều $$512/8=64$$ chiều Q/K/V cho mỗi head. (Minh họa từ video công thức Attention QKV)*

![Stack đầu ra multi-head](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_output_stack.jpg)
*Hình: Ghép 8 head kích thước (L × 64) thành ma trận (L × 512). (Minh họa từ video công thức Attention QKV)*

### 2.8 Masking (mặt nạ chú ý)

Hai loại mặt nạ thường dùng trong thực tế là:

**Padding mask.** Khi gộp nhiều chuỗi có độ dài khác nhau vào một batch, các chuỗi ngắn hơn được đệm thêm token `PAD`. Để mô hình không chú ý vào các vị trí đệm, ta cộng một số âm rất lớn (hoặc $$-\infty$$) vào điểm số của chúng trước softmax để trọng số đầu ra tiến về 0.

**Causal (look-ahead) mask.** Trong bài toán sinh tự hồi quy (mô hình ngôn ngữ, decoder của Transformer), vị trí $$i$$ chỉ được phép nhìn các vị trí từ $$i$$ trở về trước và bị cấm nhìn các vị trí tương lai $$j > i$$. Ta dùng mặt nạ tam giác trên để thực thi ràng buộc này.

Nếu thiếu mặt nạ phù hợp, mô hình có thể rò rỉ thông tin từ tương lai khi huấn luyện hoặc phân bổ trọng số vô nghĩa vào các vị trí đệm.

---

## 3. Ví dụ và Trực giác

### 3.1 Căn chỉnh mềm trong dịch máy

Xét câu nguồn **I love deep learning** (4 token) và bước giải mã đang sinh từ tiếng Pháp **apprentissage** (*"learning"*). Giả sử truy vấn từ decoder cho ra các điểm số sau:

| Token nguồn | Điểm thô $$e$$ | Sau softmax $$\alpha$$ |
|-------------|----------------|------------------------|
| I | 0.2 | 0.05 |
| love | 0.5 | 0.07 |
| deep | 1.5 | 0.19 |
| learning | 2.8 | 0.69 |

Vectơ ngữ cảnh thu được tập trung chủ yếu vào *"learning"* kèm một phần đóng góp từ *"deep"*. Khi chuyển sang sinh từ đích khác (như *"J'aime"* tương ứng *"I love"*), trọng số sẽ dịch chuyển sang hai từ *"I"* và *"love"*.

### 3.2 Self-attention như cơ chế ngữ cảnh hóa

Xét ba token với các vectơ value 2 chiều minh họa:

| Token | Vectơ value |
|-------|-------------|
| The | $$(1, 0)$$ |
| cat | $$(0, 1)$$ |
| sat | $$(1, 1)$$ |

Nếu *"sat"* gán trọng số $$[0.1, 0.6, 0.3]$$ cho ba vị trí *(The, cat, sat)*, biểu diễn mới của nó là:

$$0.1(1,0) + 0.6(0,1) + 0.3(1,1) = (0.4,\ 0.9)$$

Biểu diễn của *"sat"* lúc này đã tích hợp thông tin từ ngữ cảnh xung quanh (đặc biệt là *"cat"*). Đây là bản chất của biểu diễn ngữ cảnh hóa trong các kiến trúc xếp chồng self-attention như BERT và GPT.

### 3.3 Giới hạn của attention

Attention cung cấp cơ chế *truy cập động* vào các biểu diễn sẵn có, chứ không tự tạo ra bộ nhớ trạng thái theo thời gian như ô LSTM. Nếu đặc trưng đầu vào yếu, việc tái trọng số cũng không khắc phục được. Ngoài ra, bản thân self-attention thuần không phân biệt được thứ tự vị trí (có tính đồng biến theo hoán vị), nên Transformer bắt buộc phải bổ sung mã hóa vị trí (*positional encoding*).

---

## 4. Mã minh họa
### 4.1 Scaled dot-product attention (NumPy)

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # ổn định số
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (..., Lq, dk)
    K: (..., Lk, dk)
    V: (..., Lk, dv)
    mask: broadcast được tới (..., Lq, Lk); True/1 nghĩa là "loại bỏ"
    trả về: output (..., Lq, dv), weights (..., Lq, Lk)
    """
    dk = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)  # (..., Lq, Lk)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


# Ví dụ self-attention đồ chơi
L, dk, dv = 4, 8, 8
rng = np.random.default_rng(0)
X = rng.normal(size=(L, dk))

# Demo: dùng X làm Q, K, V (chưa có phép chiếu học)
out, attn = scaled_dot_product_attention(X, X, X)
print("output", out.shape)   # (4, 8)
print("attn", attn.shape)    # (4, 4)
print("rows sum to 1:", np.allclose(attn.sum(axis=-1), 1.0))
```

### 4.2 Causal mask

```python
def causal_mask(L):
    """mask[i, j] = True nếu j > i (cấm chú ý tới tương lai)."""
    return np.triu(np.ones((L, L), dtype=bool), k=1)

L = 4
Q = K = V = np.random.randn(L, 8)
out, attn = scaled_dot_product_attention(Q, K, V, mask=causal_mask(L))
print(np.round(attn, 3))
# Tam giác trên (tương lai) phải ~0
```

### 4.3 Multi-head attention (NumPy khái niệm)

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rng=None):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        rng = np.random.default_rng(0) if rng is None else rng
        scale = 0.02
        self.Wq = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wk = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wv = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wo = rng.normal(scale=scale, size=(d_model, d_model))

    def _split(self, x):
        # x: (B, L, d_model) -> (B, H, L, dk)
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_heads, self.dk)
        return x.transpose(0, 2, 1, 3)

    def _merge(self, x):
        # x: (B, H, L, dk) -> (B, L, d_model)
        B, H, L, dk = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(B, L, H * dk)
        return x

    def forward(self, x, mask=None):
        # Self-attention cho đơn giản
        Q = self._split(x @ self.Wq)
        K = self._split(x @ self.Wk)
        V = self._split(x @ self.Wv)
        out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        out = self._merge(out) @ self.Wo
        return out, weights


B, L, d_model, H = 2, 6, 32, 4
x = np.random.randn(B, L, d_model)
mha = MultiHeadAttention(d_model, H)
y, w = mha.forward(x)
print(y.shape)  # (2, 6, 32)
print(w.shape)  # (2, 4, 6, 6)  — batch, heads, query pos, key pos
```

### 4.4 PyTorch (dùng trong thực tế)

```python
import torch
import torch.nn.functional as F

def torch_sdp_attention(q, k, v, attn_mask=None):
    """
    q,k,v: (B, H, L, d) hoặc (B, L, d)
    attn_mask: mask cộng broadcast được tới scores; dùng -inf cho vị trí cấm
    """
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


# Module có sẵn (ưu tiên cho mô hình thật)
mha = torch.nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
x = torch.randn(2, 10, 64)
y, attn = mha(x, x, x, need_weights=True, average_attn_weights=False)
print(y.shape, attn.shape)
```

Trong mã nguồn thực tế, nên ưu tiên `torch.nn.MultiheadAttention` hoặc hàm tích hợp tối ưu `torch.nn.functional.scaled_dot_product_attention` để đảm bảo tốc độ và độ ổn định số học.

---

## 5. Khái niệm liên quan

**RNN và LSTM.** Attention ban đầu được gắn lên trên bộ mã hóa RNN: RNN tạo các trạng thái tuần tự còn attention chọn lọc thông tin từ chúng. Sau này, Transformer thay thế hoàn toàn phần hồi quy bằng self-attention kết hợp mã hóa vị trí.

**Nút thắt seq2seq.** Attention tháo gỡ trực tiếp giới hạn của vectơ ngữ cảnh kích thước cố định trong mô hình encoder–decoder (Chương 5–6).

**Softmax như cơ chế chọn lọc mềm.** Trọng số attention áp dụng cùng phép biến đổi softmax vốn quen thuộc trong bài toán phân loại, nhưng thực hiện trên các *vị trí* hoặc khe nhớ.

**Góc nhìn bộ nhớ và truy xuất.** Keys đóng vai trò địa chỉ, values là nội dung lưu trữ và queries là yêu cầu truy vấn — cách nhìn này kết nối attention với các hệ thống bộ nhớ khả vi và mô hình tăng cường truy xuất (RAG).

**Transformer.** Một khối Transformer cơ bản gồm multi-head self-attention, mạng feed-forward, kết nối phần dư (residual) và chuẩn hóa lớp (LayerNorm), với QKV là phép toán trung tâm.

**Khả năng diễn giải.** Bản đồ nhiệt trọng số attention rất hữu ích khi gỡ lỗi, nhưng không nên xem là lời giải thích nhân quả tuyệt đối cho quyết định của mô hình.

---

## 6. Các bài báo nền tảng

1. **[Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)**  
   Đề xuất attention cộng tính cho dịch máy neuron, cho phép decoder tìm kiếm mềm trên các trạng thái nguồn khi sinh từng từ đích.

2. **[Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)](https://arxiv.org/abs/1508.04025)**  
   So sánh attention toàn cục và cục bộ cùng các hàm chấm điểm nhân tính gọn nhẹ, dễ triển khai.

3. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   Thay thế hồi quy bằng multi-head scaled dot-product self-attention, định hình kiến trúc Transformer.

4. **[Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)**  
   Ứng dụng attention thị giác vào bài toán sinh chú thích ảnh, mở rộng attention ra ngoài dữ liệu văn bản thuần túy.

5. **[BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)** / **[dòng GPT (Radford et al.)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**  
   Chứng minh sức mạnh của việc xếp chồng nhiều lớp self-attention trong học biểu diễn và sinh ngôn ngữ quy mô lớn.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành

**Nhầm điểm số với trọng số.**  
$$e_{t,i}$$ là điểm thô chưa chuẩn hóa; chỉ sau softmax mới thu được trọng số $$\alpha_{t,i}$$ có tổng bằng 1. Không dùng trực tiếp điểm thô để tính trung bình trạng thái.

**Quên chia $$\sqrt{d_k}$$.**  
Khi $$d_k$$ lớn, thiếu hệ số chia tỉ lệ khiến phân phối softmax quá nhọn và làm huấn luyện mất ổn định.

**Nhầm quy ước cực tính của mask.**  
Cần kiểm tra rõ xem `1` (hoặc `True`) biểu thị vị trí "giữ lại" hay "bị che". Nhầm quy ước có thể âm thầm gây rò rỉ token tương lai hoặc loại bỏ nhầm token hợp lệ.

**Không che vị trí padding.**  
Luôn áp padding mask khi huấn luyện theo batch có độ dài biến thiên để mô hình không phân bổ trọng số vào token `PAD`.

**Diễn giải quá mức bản đồ attention.**  
Trọng số attention là công cụ trực quan hóa hữu ích để kiểm tra tiêu điểm của mô hình, không phải bằng chứng nhân quả hoàn chỉnh.

**Mẹo về nhiệt độ (temperature).**  
Chia điểm số cho tham số nhiệt độ $$\tau$$ trước softmax giúp điều chỉnh độ sắc nét của phân phối attention khi cần phân tích hoặc tinh chỉnh chuyên biệt.

**Mẹo về residual và LayerNorm.**  
Khi xếp nhiều lớp attention sâu, luôn kết hợp kết nối phần dư (residual) và chuẩn hóa lớp (LayerNorm) để đảm bảo gradient lan truyền ổn định.

**Mẹo học theo trình tự.**  
Nếu multi-head self-attention ban đầu khó hình dung, hãy nắm chắc cross-attention ("decoder nhìn sang encoder") trước rồi chuyển sang trường hợp hai phía là cùng một chuỗi.

---

## 8. Tóm tắt các điểm chính

1. **Attention thay thế nút thắt cố định bằng cơ chế truy xuất có trọng số động** trên toàn bộ các trạng thái đầu vào.
2. Quy trình **tính điểm → chuẩn hóa softmax → tổng có trọng số các value** là khung chung cho Bahdanau, Luong và Transformer.
3. Khuôn khổ **Q, K, V** thống nhất cả cross-attention lẫn self-attention.
4. **Chia tỉ lệ cho $$\sqrt{d_k}$$** giúp softmax giữ gradient ổn định khi số chiều lớn.
5. **Multi-head attention** cho phép học đồng thời nhiều mẫu quan hệ khác nhau.
6. **Masking** (padding và causal) là thành phần bắt buộc của thuật toán chứ không phải chi tiết phụ.
7. Attention là cầu nối trực tiếp từ RNN seq2seq sang **Transformer** ở chương tiếp theo.

Khi bạn có thể tự tính bằng tay một ví dụ nhỏ về cách bước decoder tạo ra $$\mathbf{c}_t$$ từ các trạng thái encoder, bạn đã nắm vững nền tảng cho các mô hình attention hiện đại.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến); các đường dẫn dưới đây được lưu lại để tra cứu và ghi nhận nguồn:

- [LLM Tập 4: Attention Mechanism (Q–K–V, quan hệ kết nối toàn phần)](https://www.facebook.com/reel/1467113421464723)
- [Transformer là gì? Self-Attention siêu dễ hiểu (Phần 1)](https://www.facebook.com/reel/930207546288223)
- [Công thức Attention: tự tay tính Q, K, V](https://www.facebook.com/reel/1806844676942638)

## Video tham khảo

- **Xem trước:** [Attention for Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=PSs6nxngL6k) — StatQuest with Josh Starmer (tiếng Anh). Giải thích attention như cơ chế kết nối trực tiếp từ các trạng thái đầu vào tới từng bước giải mã trước khi sang Transformer.
- [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Minh họa chi tiết phép tính Q, K, V tương ứng với phần toán học trong bài.
- [Cách cơ chế Attention trong kiến trúc Transformer](https://www.youtube.com/watch?v=L_2bT6hnGSQ) — namkuner (tiếng Việt). Trình bày cơ chế attention trong Transformer bằng tiếng Việt.
