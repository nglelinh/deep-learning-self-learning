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
*Sơ đồ multi-head attention (như dùng trong Transformer). Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm

**Cơ chế chú ý** (*attention*) cho phép mô hình *tập trung động* vào các phần khác nhau của đầu vào khi sinh từng phần đầu ra. Thay vì ép toàn bộ thông tin qua một tóm tắt kích thước cố định, attention tính một **lựa chọn mềm** (*soft selection*): tại mỗi bước đầu ra, gán trọng số cho các vị trí đầu vào và tạo tổ hợp có trọng số của các biểu diễn tương ứng.

Ý tưởng nghe đơn giản, nhưng nó giải quyết một thất bại cấu trúc của các mô hình chuỗi ban đầu và sau đó trở thành xương sống của NLP hiện đại (và phần lớn thị giác máy tính).

### Nút thắt khiến attention trở nên cần thiết

Mô hình **encoder–decoder** (seq2seq) cổ điển cho dịch máy hoạt động như sau:

1. Một **encoder** RNN đọc câu nguồn từng từ.
2. Encoder nén toàn bộ câu thành một trạng thái ẩn cuối cùng (vectơ kích thước cố định).
3. Một **decoder** RNN bắt đầu từ vectơ đó và sinh câu đích.

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

Hai vấn đề xuất hiện ngay.

**Nút thắt thông tin.** Dù nguồn 5 từ hay 50 từ, mọi thứ phải vừa trong $$z$$. Câu dài mất chi tiết; các từ hiếm nhưng then chốt bị “rửa trôi”.

**Một tóm tắt cố định cho mọi từ đích.** Người dịch không đọc lại *cùng* một tóm tắt cho mỗi từ đích. Khi sinh “chat”, họ tập trung vào “cat”; khi sinh “tapis”, họ tập trung vào “mat”. Một $$z$$ cố định không thể cung cấp *căn chỉnh động* (*dynamic alignment*) đó.

**Giải pháp của attention:** giữ *tất cả* trạng thái encoder $$\mathbf{h}_1, \ldots, \mathbf{h}_T$$, và tại mỗi bước decoder $$t$$, tính một vectơ ngữ cảnh mới

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i}\,\mathbf{h}_i$$

trong đó các trọng số $$\alpha_{t,i}$$ được *học* và *phụ thuộc trạng thái decoder hiện tại*. Các từ đích khác nhau nhận các soft alignment khác nhau trên nguồn.

### Phép so sánh đời thường

Hãy nghĩ đến việc viết câu trả lời khi sách giáo khoa đang mở trên bàn. Ta không ghi nhớ cả chương thành một câu rồi chỉ trả lời từ câu đó. Với mỗi câu viết, ta liếc lại các đoạn khác nhau. Attention chính là “cái liếc” đó: phiên bản khả vi, có học của việc tra đúng chỗ trong đầu vào.

### Vì sao chương bắt đầu từ đây

Bài này xây **lõi khái niệm và lịch sử** của attention:

- attention encoder–decoder (Bahdanau / Luong),
- góc nhìn query–key–value,
- scaled dot-product attention,
- self-attention so với cross-attention,
- multi-head attention ở mức nhập môn.

Các bài sau trong chương đào sâu toán học và cài đặt. Transformer (chương kế) lắp các mảnh này thành kiến trúc đầy đủ.

---

## 2. Nền tảng toán học

### 2.1 Attention encoder–decoder ở dạng tổng quát

Tại bước decoder $$t$$:

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $$\mathbf{s}_t$$ | trạng thái ẩn decoder (những gì đang cố sinh *lúc này*) |
| $$\mathbf{h}_i$$ | trạng thái ẩn encoder tại vị trí nguồn $$i$$ |
| $$e_{t,i}$$ | **điểm căn chỉnh** (*alignment score*) chưa chuẩn hóa giữa bước decoder $$t$$ và nguồn $$i$$ |
| $$\alpha_{t,i}$$ | **trọng số attention** (mức độ tập trung vào nguồn $$i$$) |
| $$\mathbf{c}_t$$ | **vectơ ngữ cảnh** (*context vector*) (tóm tắt mềm của nguồn cho bước $$t$$) |

**Bước 1 — điểm số.** So sánh trạng thái decoder với mọi trạng thái encoder:

$$e_{t,i} = \mathrm{score}(\mathbf{s}_t, \mathbf{h}_i)$$

**Bước 2 — chuẩn hóa bằng softmax.**

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T}\exp(e_{t,j})}$$

vậy $$\alpha_{t,i} \ge 0$$ và $$\sum_i \alpha_{t,i} = 1$$. Các trọng số tạo thành phân phối xác suất trên các vị trí nguồn.

**Bước 3 — ngữ cảnh như tổng có trọng số.**

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i}\,\mathbf{h}_i$$

**Bước 4 — dùng ngữ cảnh để dự đoán.** Thường kết hợp ngữ cảnh và trạng thái decoder, rồi dự đoán token tiếp theo:

$$\tilde{\mathbf{s}}_t = \tanh\!\big(\mathbf{W}_c[\mathbf{c}_t;\mathbf{s}_t]\big), \qquad
p(y_t \mid y_{<t}, x) = \mathrm{softmax}(\mathbf{W}_o\tilde{\mathbf{s}}_t)$$

(Các lớp kết hợp cụ thể thay đổi theo bài báo; ý quan trọng: **dự đoán phụ thuộc $$\mathbf{c}_t$$, không chỉ nút thắt cố định.**)

### 2.2 Attention Bahdanau (attention cộng tính, 2015)

[Bahdanau et al.](https://arxiv.org/abs/1409.0473) đưa ra hàm điểm số có học:

$$e_{t,i} = \mathbf{v}_a^\top \tanh\!\big(\mathbf{W}_a\mathbf{s}_t + \mathbf{U}_a\mathbf{h}_i\big)$$

**Vì sao hoạt động.**

- $$\mathbf{W}_a$$ và $$\mathbf{U}_a$$ chiếu trạng thái decoder và encoder vào không gian chung trước khi so sánh.
- $$\tanh$$ thêm phi tuyến để “liên quan” không chỉ là tích vô hướng thuần.
- $$\mathbf{v}_a$$ co vectơ đã chiếu thành điểm số vô hướng.

Thường gọi là attention **cộng tính** (*additive*) vì đặc trưng query và key được *cộng* bên trong $$\tanh$$.

Về lịch sử, đây là attention neuron được dùng rộng rãi đầu tiên cho NMT và làm dịch câu dài tin cậy hơn rõ rệt so với seq2seq nút thắt thuần.

### 2.3 Attention Luong (dạng nhân / tổng quát, 2015)

[Luong et al.](https://arxiv.org/abs/1508.04025) nghiên cứu các hàm điểm số đơn giản hơn, thường nhanh và hiệu quả trong thực tế:

**Tích vô hướng** (*dot product*)

$$e_{t,i} = \mathbf{s}_t^\top \mathbf{h}_i$$

**Tổng quát (song tuyến)** (*general / bilinear*)

$$e_{t,i} = \mathbf{s}_t^\top \mathbf{W}_a \mathbf{h}_i$$

**Concat** (cùng tinh thần với Bahdanau)

$$e_{t,i} = \mathbf{v}_a^\top \tanh\!\big(\mathbf{W}_a[\mathbf{s}_t;\mathbf{h}_i]\big)$$

**Trực giác.** Tích vô hướng đo căn chỉnh hướng: nếu decoder “muốn” thứ gì cùng hướng với một trạng thái encoder, điểm số lớn. Dạng tổng quát chèn metric học được $$\mathbf{W}_a$$ để alignment linh hoạt hơn khi $$\mathbf{s}_t$$ và $$\mathbf{h}_i$$ sống trong không gian khác nhau.

### 2.4 Góc nhìn query–key–value (Q, K, V)

Attention hiện đại thường được viết với ba vai trò:

| Vai trò | Câu hỏi nó trả lời | Nguồn điển hình |
|---------|--------------------|-----------------|
| **Query** $$\mathbf{q}$$ | “Lúc này ta đang tìm gì?” | trạng thái decoder, hoặc vị trí $$i$$ trong self-attention |
| **Key** $$\mathbf{k}_j$$ | “Phần tử $$j$$ đưa ra gì để khớp?” | trạng thái encoder, hoặc mọi vị trí |
| **Value** $$\mathbf{v}_j$$ | “Nếu khớp $$j$$ thì lấy nội dung gì?” | thường cùng vectơ với keys, hoặc chiếu riêng |

Attention khi đó:

1. chấm điểm query với keys,
2. softmax → trọng số,
3. tổng có trọng số của values.

Trừu tượng này bao trùm **cross-attention** (query từ một chuỗi, keys/values từ chuỗi khác) và **self-attention** (query, key, value đều từ cùng chuỗi).

Trong LLM, cùng pipeline được mô tả là *mạng quan hệ kết nối toàn phần*: embedding vào → tách Q/K/V → điểm attention → Softmax → (nhiều head song song) → trộn ngữ cảnh ra.

![Pipeline attention kiểu LLM](/deep-learning-self-learning/img/chapter_img/chapter07/llm_attn_qkv_pipeline_active.jpg)
*Hình: Luồng Q–K–V → Softmax → multi-head → hợp ngữ cảnh. (Minh họa từ video LLM Attention Mechanism)*

#### Walkthrough số: câu 3 token → Q, K, V

Lấy câu toy 3 từ (ví dụ “tôi / yêu / bạn”), mỗi từ đã là vector embedding 2 chiều (minh họa; thực tế thường $$d_{\mathrm{model}}=512$$):

![Embedding 3 từ](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_toy_sentence_embeddings.jpg)
*Hình: Ma trận đầu vào $$\mathbf{X}$$ — 3 token × $$d$$. (Minh họa từ video công thức Attention QKV)*

Ba ma trận trọng số học được $$W^Q, W^K, W^V$$ chiếu $$\mathbf{X}$$ sang không gian query/key/value:

![W^Q, W^K, W^V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_weights_wq_wk_wv.jpg)
*Hình: Ba ma trận chiếu riêng. (Minh họa từ video công thức Attention QKV)*

![X × W → Q, K, V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_project_x_to_qkv.jpg)
*Hình: $$\mathbf{Q}=\mathbf{X}W^Q$$, $$\mathbf{K}=\mathbf{X}W^K$$, $$\mathbf{V}=\mathbf{X}W^V$$. (Minh họa từ video công thức Attention QKV)*

### 2.5 Scaled dot-product attention

Khối xây dựng chuẩn của Transformer là:

$$\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
= \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

![Q, K, V và công thức Attention](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_matrices_and_formula.jpg)
*Hình: Ba ma trận sau chiếu + công thức scaled SDP. (Minh họa từ video công thức Attention QKV)*

với shape (một chuỗi, tạm bỏ batch):

- $$\mathbf{Q} \in \mathbb{R}^{L_q \times d_k}$$
- $$\mathbf{K} \in \mathbb{R}^{L_k \times d_k}$$
- $$\mathbf{V} \in \mathbb{R}^{L_k \times d_v}$$
- đầu ra $$\in \mathbb{R}^{L_q \times d_v}$$

**Bước tính tay (cùng ví dụ 3 từ).**

1. **Tích vô hướng** — hàng query × cột của $$K^\top$$ (mỗi cặp từ → một điểm số):

![Tính từng phần tử QK^T](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_dot_product_row_example.jpg)
*Hình: Ví dụ hàng “tôi” chấm với mọi key. (Minh họa từ video công thức Attention QKV)*

![Ma trận similarity chưa scale](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_unscaled_similarities.jpg)
*Hình: $$\mathbf{Q}K^\top$$ — Unscaled Dot-Product Similarities. (Minh họa từ video công thức Attention QKV)*

2. **Scale** — chia $$\sqrt{d_k}$$ (ở đây $$d_k=2$$ → $$\sqrt{2}$$):

![Chia √d_k](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_scale_by_sqrt_dk.jpg)
*Hình: Unscaled → Scaled Dot-Product Similarities. (Minh họa từ video công thức Attention QKV)*

**Vì sao chia cho $$\sqrt{d_k}$$?**

Nếu các tọa độ query/key gần như độc lập với phương sai 1, thì tích vô hướng thô có phương sai khoảng $$d_k$$. Với $$d_k$$ lớn (ví dụ 64–512), điểm số trở nên rất lớn, softmax gần như one-hot, và gradient triệt tiêu. Scale giữ điểm số trong vùng lành mạnh hơn để mô hình học được phân phối *mềm*, không chỉ attention kiểu argmax cứng.

3. **Softmax theo hàng** — mỗi query thành phân phối trên keys:

![Softmax theo hàng](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_softmax_over_rows.jpg)
*Hình: Softmax chạy theo chiều hàng. (Minh họa từ video công thức Attention QKV)*

![Attention weights](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_attention_weights.jpg)
*Hình: Ma trận trọng số (mỗi hàng ≈ 1). (Minh họa từ video công thức Attention QKV)*

4. **× V** — trộn values theo trọng số → đầu ra attention:

![Weights × V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_weights_times_v.jpg)
*Hình: Attention Weights × V = Attention Output. (Minh họa từ video công thức Attention QKV)*

### 2.6 Self-attention so với cross-attention

**Cross-attention (attention encoder–decoder).**  
Query đến từ decoder; key và value đến từ encoder. Đây là attention NMT gốc: “khi sinh từ đích này, những từ nguồn nào quan trọng?”

**Self-attention.**  
Query, key và value đều đến từ *cùng* chuỗi. Mọi vị trí có thể thu thập ngữ cảnh từ mọi vị trí khác:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}^K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}^V$$

Mỗi token trước hết là một vector số; sau đó mỗi token sinh bộ Q/K/V riêng:

![Token = cột số](/deep-learning-self-learning/img/chapter_img/chapter07/sa_tokens_as_vectors.jpg)
*Hình: Câu được biểu diễn bằng các vector embedding. (Minh họa từ video Transformer / Self-Attention nhập môn)*

![Mỗi từ một cột Q/K/V](/deep-learning-self-learning/img/chapter_img/chapter07/sa_per_token_qkv_columns.jpg)
*Hình: Self-attention tạo Q, K, V cho *từng* vị trí trong câu. (Minh họa từ video Transformer / Self-Attention nhập môn)*

![Vai trò Q/K/V theo từ](/deep-learning-self-learning/img/chapter_img/chapter07/sa_qkv_roles_apple_phone_orange.jpg)
*Hình: Cùng pipeline QKV áp cho “táo / điện thoại / cam” — khớp ngữ cảnh rồi trộn value. (Minh họa từ video Transformer / Self-Attention nhập môn)*

Ví dụ: trong câu

> The animal didn't cross the street because **it** was too tired.

Self-attention có thể học rằng “it” nên chú ý mạnh tới “animal”. Liên kết tầm xa đó không cần đi từng bước qua RNN.

### 2.7 Multi-head attention (góc nhìn nhập môn)

Một đầu attention (*attention head*) có thể chuyên về một kiểu mẫu. Ngôn ngữ (và dữ liệu có cấu trúc khác) có nhiều quan hệ đồng thời: cú pháp, đồng tham chiếu, lân cận, tương đồng ngữ nghĩa, v.v.

**Multi-head attention** chạy song song $$H$$ phép attention với các phép chiếu khác nhau, rồi nối và trộn:

$$\mathrm{head}_h = \mathrm{Attention}(\mathbf{Q}\mathbf{W}_h^Q, \mathbf{K}\mathbf{W}_h^K, \mathbf{V}\mathbf{W}_h^V)$$

$$\mathrm{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V})
= \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)\,\mathbf{W}^O$$

Thường $$d_k = d_v = d_{\mathrm{model}} / H$$ để tổng tính toán tương đương một head full-width, trong khi năng lực biểu diễn tăng nhờ đa dạng các head.

Ví dụ phổ biến: $$d_{\mathrm{model}}=512$$, $$H=8$$ → mỗi head làm việc với $$d_k=64$$:

![Chia d_model cho số head](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_dk_split.jpg)
*Hình: $$512/8=64$$ — chiều Q/K/V mỗi head. (Minh họa từ video công thức Attention QKV)*

![Stack đầu ra multi-head](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_output_stack.jpg)
*Hình: 8 head × (L × 64) rồi concat → (L × 512). (Minh họa từ video công thức Attention QKV)*

### 2.8 Masking (những gì cần biết sớm)

Hai loại mask xuất hiện thường xuyên:

**Padding mask.** Trong một batch, chuỗi ngắn hơn được đệm. Attention không được đặt khối lượng lên token pad. Cách làm: cộng một số âm rất lớn vào các logit bị mask trước softmax để trọng số tương ứng ≈ 0.

**Causal (look-ahead) mask.** Trong sinh tự hồi quy (mô hình ngôn ngữ, decoder-only Transformer), vị trí $$i$$ không được chú ý tới các vị trí tương lai $$j > i$$. Dùng mask tam giác trên.

Không có mask đúng, mô hình “gian lận” bằng cách nhìn token chưa được phép thấy, hoặc chú ý tới padding vô nghĩa.

---

## 3. Ví dụ / Trực giác

### 3.1 Soft alignment trong dịch máy

Nguồn: **I love deep learning** (4 token)  
Bước đích sinh thứ gì đó như **apprentissage** (“learning”).

Giả sử query decoder tại bước đó chấm bốn vị trí nguồn như sau:

| Token nguồn | điểm thô $$e$$ | sau softmax $$\alpha$$ |
|-------------|----------------|------------------------|
| I | 0.2 | 0.05 |
| love | 0.5 | 0.07 |
| deep | 1.5 | 0.20 |
| learning | 2.8 | 0.68 |

Vectơ ngữ cảnh chủ yếu là “learning”, với đóng góp hữu ích từ “deep”. Mô hình không cần bảng căn chỉnh ký hiệu cứng; nó *học* soft alignment từ dữ liệu.

Khi sinh từ đích khác (ví dụ “J’aime” / “I love”), khối lượng trọng số sẽ dịch về “I” và “love”.

### 3.2 Self-attention như contextualization

Lấy ba token với value 2-D đồ chơi (đã chiếu sẵn cho đơn giản):

| Token | vectơ value |
|-------|-------------|
| The | $$(1, 0)$$ |
| cat | $$(0, 1)$$ |
| sat | $$(1, 1)$$ |

Nếu “sat” chú ý với trọng số $$[0.1, 0.6, 0.3]$$ tới (The, cat, sat), đầu ra trở thành

$$0.1(1,0) + 0.6(0,1) + 0.3(1,1) = (0.4,\ 0.9)$$

Biểu diễn của “sat” không còn cô lập; nó **trộn với ngữ cảnh**, đặc biệt “cat”. Đó là bản chất của embedding theo ngữ cảnh do các stack self-attention sinh ra (BERT, GPT, v.v.).

### 3.3 Attention không phải bộ nhớ thần kỳ

Attention cho *quyền truy cập* các biểu diễn đã tồn tại. Bản thân nó không tạo ra lưu trữ dài hạn kiểu trạng thái ô LSTM. Nếu trạng thái encoder yếu, attention chỉ có thể tái trọng số thông tin yếu. Attention tốt cần đặc trưng nền tốt—và sau này, tín hiệu vị trí (Transformer thêm positional encoding chính vì self-attention thuần nhạy với hoán vị nếu không có chúng).

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

### 4.4 PyTorch (thứ dùng trong thực tế)

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

Trong code production, ưu tiên `torch.nn.MultiheadAttention` hoặc kernel scaled-dot-product fused (`torch.nn.functional.scaled_dot_product_attention`) vì tốc độ và ổn định số.

---

## 5. Khái niệm liên quan

**RNN / LSTM.** Attention ban đầu phổ biến *trên nền* encoder RNN. RNN cung cấp trạng thái tuần tự; attention chọn trong số chúng. Transformer sau đó *thay* hồi quy bằng self-attention + positional encoding.

**Nút thắt seq2seq.** Attention là cách sửa trực tiếp cho vectơ ngữ cảnh độ dài cố định trong encoder–decoder (bối cảnh Chương 5–6).

**Softmax như lựa chọn mềm.** Trọng số attention là softmax trên điểm số—cùng nguyên thủy dùng trong phân loại, nhưng áp trên *vị trí* (hoặc khe nhớ).

**Góc nhìn bộ nhớ / truy xuất.** Keys là địa chỉ, values là nội dung lưu, queries là yêu cầu tra cứu. Góc nhìn này nối attention với bộ nhớ khả vi và mô hình tăng cường truy xuất.

**Transformer.** Một lớp Transformer về cơ bản là: multi-head self-attention + mạng feed-forward + residual + chuẩn hóa. Cơ chế QKV của bài này là lõi của thiết kế đó (chương kế).

**Khả năng diễn giải.** Bản đồ nhiệt attention hữu ích để debug, nhưng không phải giải thích nhân quả đầy đủ của quyết định mô hình. Coi chúng là *bằng chứng về tiêu điểm*, không phải chứng minh suy luận.

---

## 6. Các bài báo nền tảng

1. **[Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)**  
   Đưa ra additive attention cho NMT; decoder soft-search các vị trí nguồn khi sinh mỗi từ đích. Bài báo làm attention neuron trở thành xu hướng chính.

2. **[Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)](https://arxiv.org/abs/1508.04025)**  
   So sánh attention global/local và các hàm chấm điểm nhân tính đơn giản; công thức thực tiễn vẫn được trích dẫn.

3. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   Thay hồi quy bằng multi-head self-attention (Transformer). Scaled dot-product attention và thiết kế multi-head trở thành khuôn mẫu mặc định cho mô hình chuỗi hiện đại.

4. **[Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)**  
   Attention thị giác cho chú thích ảnh: cho thấy attention tổng quát hóa vượt ngoài căn chỉnh text-to-text.

5. **[BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)** / **[dòng GPT (Radford et al.)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**  
   Bằng chứng quy mô lớn rằng stack self-attention sinh ra biểu diễn ngôn ngữ và bộ sinh mạnh.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành
**Cạm bẫy: nhầm điểm số với trọng số.**  
$$e_{t,i}$$ là logit thô; chỉ sau softmax mới có $$\alpha_{t,i}$$ tổng bằng 1. Không bao giờ trung bình các trạng thái encoder bằng điểm số chưa chuẩn hóa rồi gọi đó là attention.

**Cạm bẫy: quên $$\sqrt{d_k}$$.**  
Không scale, $$d_k$$ lớn khiến attention quá sắc và huấn luyện không ổn định.

**Cạm bẫy: cực tính mask sai.**  
Cần rõ: `1` nghĩa là “giữ” hay “loại”? Lỗi quy ước im lặng rò rỉ token tương lai hoặc triệt tiêu token thật.

**Cạm bẫy: chú ý tới padding.**  
Luôn mask pad khi huấn luyện theo batch; nếu không, mô hình lãng phí năng lực vào `PAD` và metric trông tệ một cách khó hiểu.

**Cạm bẫy: coi trọng số attention là giải thích ground-truth.**  
Trọng số là trực quan hóa hữu ích, không phải gán nhân quả đảm bảo.

**Mẹo: nhiệt độ / độ sắc.**  
Chia điểm số cho nhiệt độ $$\tau$$ trước softmax (hoặc dùng biến thể sparsemax) có thể điều khiển độ tập trung của attention—hữu ích cho phân tích và một số mô hình chuyên biệt.

**Mẹo: residual + layer norm quanh attention.**  
Trong stack sâu, luôn bọc khối attention bằng residual (và thường LayerNorm). Lớp attention trần khó huấn luyện sâu.

**Mẹo: bắt đầu từ trực giác cross-attention, rồi self-attention.**  
Nếu multi-head self-attention còn trừu tượng, hãy nắm “decoder nhìn encoder” trước, rồi thay cả hai phía bằng cùng một chuỗi.

---

## 8. Tóm tắt các điểm chính
1. **Attention thay nút thắt cố định bằng đọc ra có trọng số động** trên các trạng thái đầu vào.
2. **Điểm số → trọng số softmax → values có trọng số** là khung xương phổ quát (Bahdanau, Luong, Transformer).
3. **Q, K, V** thống nhất cross-attention và self-attention trong một ngôn ngữ.
4. **Scale bởi $$\sqrt{d_k}$$** giữ softmax ứng xử tốt ở chiều cao.
5. **Multi-head attention** học nhiều mẫu quan hệ song song.
6. **Masking** (padding, causal) là một phần của thuật toán, không phải phần tùy chọn.
7. Attention là cầu khái niệm từ RNN seq2seq tới **Transformer**—chương kế tiếp.

Khi giải thích được, với một ví dụ số nhỏ, cách một bước decoder tạo $$\mathbf{c}_t$$ từ các trạng thái encoder, người học đã có nền tảng cần thiết cho mọi mô hình attention hiện đại trong khóa học này.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [LLM Tập 4: Attention Mechanism (Q–K–V, quan hệ kết nối toàn phần)](https://www.facebook.com/reel/1467113421464723)
- [Transformer là gì? Self-Attention siêu dễ hiểu (Phần 1)](https://www.facebook.com/reel/930207546288223)
- [Công thức Attention: tự tay tính Q, K, V](https://www.facebook.com/reel/1806844676942638)
