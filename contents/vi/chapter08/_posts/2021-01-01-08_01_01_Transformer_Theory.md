---
layout: post
title: 08-01-01 Lý thuyết Transformer
chapter: '08'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter08
---

# Lý thuyết Transformer

![Transformer Architecture](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_architecture.png)
*Transformer encoder–decoder (Vaswani et al., 2017). Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm
Transformer là kiến trúc cho dữ liệu chuỗi (và dữ liệu “giống chuỗi”) được xây dựng chủ yếu từ **attention đa đầu** (*multi-head attention*) và **mạng feed-forward theo vị trí** (*position-wise feed-forward network*), bọc trong kết nối dư (*residual connection*) và chuẩn hóa tầng (*layer normalization*). Được Vaswani et al. (2017) giới thiệu cho dịch máy, nó loại bỏ hồi quy (*recurrence*) và tích chập như các toán tử chuỗi chính.

### Vấn đề của “chỉ cần RNN tốt hơn” là gì?

Ngay cả RNN xuất sắc (LSTM/GRU) vẫn có ràng buộc cấu trúc:

| Ràng buộc | Hiệu ứng |
|------------|--------|
| Cập nhật trạng thái trái-sang-phải | Khó song song hóa theo độ dài trong huấn luyện |
| Đường phụ thuộc dài | Token sớm phải “sống sót” qua nhiều bước để ảnh hưởng dự đoán muộn |
| Trạng thái ẩn kích thước cố định | Nghẽn mềm cho ngữ cảnh rất phong phú |

Attention đã cho thấy mô hình có thể *tra cứu* các trạng thái liên quan. Khẳng định của Transformer sắc hơn: **bạn không cần hồi quy chút nào** nếu mọi tầng cho phép mọi vị trí chú ý đến mọi vị trí khác (có mask khi cần) rồi áp dụng một MLP cục bộ mạnh.

### Ba họ sản phẩm từ một kiến trúc

| Họ | Cấu trúc | Ví dụ |
|--------|-----------|----------|
| Chỉ-encoder | Self-attention hai chiều | BERT, RoBERTa |
| Chỉ-decoder | Self-attention nhân quả (*causal*) | GPT-2/3/4, LLaMA |
| Encoder–decoder | Encoder + decoder có cross-attn | Transformer gốc, T5, BART |

Cùng các khối tầng nguyên thủy; khác mẫu attention và mục tiêu huấn luyện.

### Các khối xây dựng (xem trước)

1. Nhúng token + nhúng vị trí  
2. Chồng tầng encoder: self-attn → FFN (mỗi cái có residual + chuẩn hóa)  
3. Chồng tầng decoder: self-attn *có mask* → cross-attn → FFN  
4. Chiếu đầu ra lên từ vựng (sinh) hoặc đầu tác vụ (phân loại)

![Toàn cảnh kiến trúc + Multi-Head / Scaled Dot-Product](/deep-learning-self-learning/img/chapter_img/chapter08/tf_full_arch_multihead_zoom.jpg)
*Hình: Encoder–decoder với zoom multi-head attention và scaled dot-product. (Minh họa từ video Attention trong Transformer)*

Chương 07 đã trình bày toán attention. Bài này đặt toán đó vào **hệ thống đầy đủ**, với trọng tâm đặc biệt vào **thứ tự** (mã hóa vị trí) và **thành phần tầng**.

---

## 2. Nền tảng toán học
### 2.1 Biểu diễn đầu vào

Gọi các token là chỉ số $$t_1,\ldots,t_L$$. Ma trận nhúng $$E$$ ánh xạ chúng thành vectơ:

$$\mathbf{x}_i^{(0)} = E_{t_i} + \mathbf{p}_i \in \mathbb{R}^{d_{\mathrm{model}}}$$

trong đó $$\mathbf{p}_i$$ là **mã hóa vị trí** (*positional encoding*, cố định hoặc học được). Xếp các hàng:

$$\mathbf{X}^{(0)} \in \mathbb{R}^{L \times d_{\mathrm{model}}}.$$

(Chiều batch $$B$$ luôn có trong mã: $$(B,L,d)$$.)

### 2.2 Self-attention đa đầu có scale (giao diện mô hình)

Với các phép chiếu học được (chi tiết xem Chương 07):

$$
\mathrm{MultiHead}(\mathbf{X})
= \mathrm{Concat}_h\!\left(
  \mathrm{Attention}(\mathbf{X}W_h^Q,\mathbf{X}W_h^K,\mathbf{X}W_h^V)
\right) W^O
$$

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

trong đó $$M$$ là mask cộng tùy chọn (đệm và/hoặc nhân quả).

**Vai trò trong Transformer:** đây là bước *giao tiếp*—các vị trí trao đổi thông tin.

### 2.3 Mạng feed-forward theo vị trí (FFN)

Áp dụng **độc lập** cho từng vị trí (cùng trọng số cho mọi vị trí):

$$
\mathrm{FFN}(\mathbf{z}) = W_2\,\sigma(W_1\mathbf{z}+\mathbf{b}_1)+\mathbf{b}_2
$$

với bề rộng điển hình $$d_{\mathrm{model}} \to d_{\mathrm{ff}} \to d_{\mathrm{model}}$$ và $$d_{\mathrm{ff}} \approx 4\,d_{\mathrm{model}}$$. Hàm kích hoạt $$\sigma$$ là ReLU trong bài báo gốc; các mô hình hiện đại thường dùng GELU hoặc biến thể SwiGLU.

**Vì sao cần FFN:** multi-head attention về bản chất là *trộn tuyến tính phụ thuộc dữ liệu* của các giá trị (cộng phép chiếu). Không có MLP phi tuyến trên mỗi vị trí, xếp chồng các tầng attention có sức tính toán trên từng token hạn chế. Attention trộn; FFN *tính toán*.

### 2.4 Kết nối dư và chuẩn hóa tầng (*LayerNorm*)

Tầng con encoder gốc (“Post-LN”):

$$
\mathbf{Y} = \mathrm{LayerNorm}\big(\mathbf{X} + \mathrm{Sublayer}(\mathbf{X})\big)
$$

trong đó Sublayer là MultiHead hoặc FFN. Nhiều mô hình hiện đại dùng **Pre-LN**:

$$
\mathbf{Y} = \mathbf{X} + \mathrm{Sublayer}\big(\mathrm{LayerNorm}(\mathbf{X})\big)
$$

thường huấn luyện ổn định hơn ở độ sâu lớn.

**Residual** giữ “đường cao tốc” gradient; **LayerNorm** ổn định thang kích hoạt qua token và tầng.

### 2.5 Tầng encoder đầy đủ

Dạng Post-LN (như bài báo gốc):

$$
\begin{aligned}
\mathbf{U} &= \mathrm{LayerNorm}\big(\mathbf{X} + \mathrm{MultiHeadSelfAttn}(\mathbf{X})\big) \\
\mathbf{Z} &= \mathrm{LayerNorm}\big(\mathbf{U} + \mathrm{FFN}(\mathbf{U})\big)
\end{aligned}
$$

Xếp $$N$$ tầng như vậy (thường $$N=6$$ ở mô hình base, nhiều hơn rất nhiều trong LLM).

### 2.6 Tầng decoder đầy đủ

Ba tầng con:

1. **Masked self-attention** trên các token đã sinh (mask nhân quả).  
2. **Cross-attention** trong đó query đến từ luồng decoder và key/value đến từ đầu ra encoder.  
3. **FFN** như trên.

$$
\begin{aligned}
\mathbf{U} &= \mathrm{LN}\big(\mathbf{Y} + \mathrm{MaskedSelfAttn}(\mathbf{Y})\big) \\
\mathbf{V} &= \mathrm{LN}\big(\mathbf{U} + \mathrm{CrossAttn}(\mathbf{U}, \mathbf{H}_{\mathrm{enc}}, \mathbf{H}_{\mathrm{enc}})\big) \\
\mathbf{Z} &= \mathrm{LN}\big(\mathbf{V} + \mathrm{FFN}(\mathbf{V})\big)
\end{aligned}
$$

**Mask nhân quả:** với vị trí query $$i$$, cấm các key $$j > i$$ để mô hình không “nhìn tương lai” khi huấn luyện bằng *teacher forcing*.

### 2.7 Mã hóa vị trí dạng sin-cos

Attention đơn thuần là **đẳng biến hoán vị** theo nội dung: không có tín hiệu vị trí thì thứ tự vô hình. PE gốc:

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin\!\big(pos / 10000^{2i/d_{\mathrm{model}}}\big) \\
PE_{(pos,2i+1)} &= \cos\!\big(pos / 10000^{2i/d_{\mathrm{model}}}\big)
\end{aligned}
$$

![Công thức PE + ma trận token](/deep-learning-self-learning/img/chapter_img/chapter08/pe_formula_and_token_matrix.jpg)
*Hình: Mỗi token có vector $$d_{\mathrm{model}}$$; PE dùng sin/cos theo $$pos$$ và chỉ số tần $$i$$. (Minh họa từ video Position Encoding)*

![pos = chỉ số từ trong câu](/deep-learning-self-learning/img/chapter_img/chapter08/pe_pos_index_sentence.jpg)
*Hình: $$pos = 0,1,2,\ldots$$ theo thứ tự token. (Minh họa từ video Position Encoding)*

**Cách đọc chỉ số chiều.** Với $$d_{\mathrm{model}}=512$$, chỉ số tần $$i$$ chạy $$0\ldots 255$$: mỗi $$i$$ sinh **hai** chiều — chẵn $$2i$$ dùng **sin**, lẻ $$2i+1$$ dùng **cos**.

![Bảng i → chiều sin/cos](/deep-learning-self-learning/img/chapter_img/chapter08/pe_i_range_sin_cos_dims.jpg)
*Hình: $$i=0\to$$ chiều 0 (sin), 1 (cos); …; $$i=255\to$$ chiều 510, 511. (Minh họa từ video Position Encoding)*

![Chiều chẵn sin / lẻ cos](/deep-learning-self-learning/img/chapter_img/chapter08/pe_even_odd_dims.jpg)
*Hình: Gán sin vào chiều chẵn, cos vào chiều lẻ của vector PE. (Minh họa từ video Position Encoding)*

![Sin và cos](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sin_cos_formula_plot.jpg)
*Hình: Cặp sin/cos cùng tần số — nền sóng cho PE. (Minh họa từ video Position Encoding)*

![Nhiều tần số theo i](/deep-learning-self-learning/img/chapter_img/chapter08/pe_multiple_frequencies_i.jpg)
*Hình: Mỗi $$i$$ một tần số khác; $$i$$ nhỏ = sóng “chậm”, $$i$$ lớn = sóng “nhanh”. (Minh họa từ video Position Encoding)*

**Lấy mẫu vector PE tại một vị trí.** Cố định $$pos$$, với mỗi $$i$$ đọc một điểm trên sin và một điểm trên cos → xếp thành vector dài $$d_{\mathrm{model}}$$.

![Lấy PE tại pos=0](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sample_values_pos0.jpg)
*Hình: Cột tại $$pos=0$$ — các thành phần PE cho token đầu. (Minh họa từ video Position Encoding)*

![Lấy PE tại pos=3](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sample_values_pos3.jpg)
*Hình: Cùng cách lấy, $$pos=3$$ — vector khác vì sóng đã trôi pha. (Minh họa từ video Position Encoding)*

**Cộng vào embedding.** Token embedding và PE cùng chiều; input thực tế thường là:

$$
\mathbf{x}_{pos} = \mathrm{TokenEmb}(w_{pos}) + \mathrm{PE}_{pos}
$$

![Token emb vs PE (cùng 512-d)](/deep-learning-self-learning/img/chapter_img/chapter08/pe_input_emb_vs_positional.jpg)
*Hình: Hai vector cùng $$d_{\mathrm{model}}$$ — cộng phần tử để “dán” thứ tự vào nội dung. (Minh họa từ video Position Encoding)*

![Cột token emb (tím) + cột PE (xanh)](/deep-learning-self-learning/img/chapter_img/chapter08/pe_token_emb_plus_pe_columns.jpg)
*Hình: Mỗi vị trí có một cột embedding và một cột PE tương ứng. (Minh họa từ video Position Encoding)*

Tính chất:

- Xác định; không có tham số học.  
- “Dấu vân tay” duy nhất cho mỗi vị trí qua sin-cos đa tần.  
- Độ lệch tương đối là biến đổi tuyến tính của vectơ PE, hỗ trợ học mẫu khoảng cách tương đối.  
- Về nguyên tắc có thể ngoại suy vượt độ dài huấn luyện tốt hơn nhúng tuyệt đối học được ngây thơ (dù ngoại suy thực tế vẫn cần cẩn trọng).

**Thay thế:** nhúng tuyệt đối học được (BERT/GPT), bias tương đối (T5, Shaw), RoPE (LLM hiện đại). *Bài toán* luôn giống nhau: tiêm thứ tự vào bộ trộn dựa trên nội dung.

### 2.8 Độ phức tạp

Self-attention dày đặc tốn $$O(L^2 d)$$ thời gian và $$O(L^2)$$ bộ nhớ cho điểm số. Đó là “thuế” trả cho giao tiếp toàn cục. Với tài liệu dài, các biến thể attention thưa/tuyến tính trở nên cần thiết; với độ dài câu/đoạn thông thường, attention dày đặc là mặc định.

### 2.9 Phác thảo mục tiêu huấn luyện (mô hình MT gốc)

Encoder đọc token nguồn; decoder dự đoán token đích kế tiếp bằng cross-entropy dưới *teacher forcing*. Mô hình chỉ-encoder hiện đại dùng masked LM; chỉ-decoder dùng next-token LM. Kiến trúc vẫn liên quan; **mục tiêu + mẫu mask** thay đổi hành vi sản phẩm.

---

## 3. Ví dụ / Trực giác

### 3.1 Hành trình của một token trong encoder

Câu: `I love deep learning`. Tập trung vào vị trí **learning**.

Ở self-attention tầng 1, query của nó có thể đặt trọng số lớn lên `deep` và chính nó, tạo ra hỗn hợp đã mang nghĩa “deep learning” thay vì “learning” trần. FFN sau đó định hình lại hỗn hợp một cách phi tuyến. Các tầng sâu hơn tinh chỉnh thêm vai trò (cú pháp, ngữ nghĩa) khi mẫu attention trở nên trừu tượng hơn.

Cả bốn vị trí làm việc này **song song** qua nhân ma trận—khác RNN sẽ cần bốn bước tuần tự.

### 3.2 Một bước decoder

Sinh tiếng Pháp `profond` sau `J' aime l'apprentissage`:

1. **Masked self-attn** chỉ chú ý đến các token Pháp trước đó (tính nhất quán của giả thuyết đến hiện tại).  
2. **Cross-attn** nhìn các trạng thái encoder tiếng Anh; khối lượng xác suất nên tập trung vào `deep`.  
3. **FFN + softmax trên từ vựng** đề xuất `profond`.

Nếu mask nhân quả sai, mô hình “gian lận” khi huấn luyện và sụp đổ khi suy luận tự hồi quy thật.

### 3.3 Chỉ-encoder so với chỉ-decoder

- **BERT (chỉ-encoder):** self-attn hai chiều + masked LM → biểu diễn *hiểu* mạnh.  
- **GPT (chỉ-decoder):** self-attn nhân quả + next-token LM → *sinh* mạnh.  

Cùng khối MultiHead/FFN; khác mask và luồng dữ liệu.

---

## 4. Mã minh họa (bám theo phương trình)
Shape tối thiểu cho một tầng encoder Pre-LN:

```python
import torch
import torch.nn as nn

class TinyEncoderLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, d)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x

B, L, d = 2, 10, 64
layer = TinyEncoderLayer(d)
y = layer(torch.randn(B, L, d))
print(y.shape)  # (2, 10, 64)
```

Các module viết từ đầu đầy đủ xuất hiện trong các bài cài đặt.

---

## 5. Khái niệm liên quan
- **Seq2seq RNN** — cùng ý tưởng encode/decode; Transformer thay hồi quy bằng attention.  
- **Tích chập** — kernel cục bộ cố định so với trộn toàn cục động.  
- **Mạng residual** — độ sâu mà không “chết” gradient.  
- **Các paradigm tiền huấn luyện** — MLM so với CLM so với span corruption (T5).  
- **Vision Transformer** — patch ảnh như token; cùng toán encoder.

---

## 6. Các bài báo nền tảng
1. [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762) — kiến trúc gốc.  
2. [Devlin et al., 2019 — BERT](https://arxiv.org/abs/1810.04805) — tiền huấn luyện encoder hai chiều.  
3. [Radford et al., 2019 — GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — LM nhân quả lớn + zero-shot.  
4. [Dosovitskiy et al., 2021 — ViT](https://arxiv.org/abs/2010.11929) — Transformer cho ảnh.  
5. [Phuong & Hutter, 2022 — Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238) — định nghĩa tham chiếu chính xác.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành
- Quên thông tin **vị trí** → mô hình mù thứ tự.  
- Mask nhân quả dùng **0** thay vì **−∞** trước softmax → rò rỉ tương lai.  
- Thiếu mask **đệm** (*pad*) khi huấn luyện theo batch.  
- Bỏ $$1/\sqrt{d_k}$$ ở bề rộng lớn.  
- Nhầm **Post-LN so với Pre-LN** khi chuyển công thức (tốc độ học khác nhau).  
- **Mẹo:** kiểm thử đơn vị causal attention với `triu(weights,1)≈0`.  
- **Mẹo:** bắt đầu với mô hình nhỏ 2 tầng trước khi scale độ sâu/bề rộng.

---

## 8. Tóm tắt các điểm chính
1. Một tầng Transformer là **attend (giao tiếp) + FFN (tính toán)** với residual/chuẩn hóa.  
2. **Mask** định nghĩa họ sản phẩm: hai chiều, nhân quả, hoặc chéo.  
3. **Mã hóa vị trí** khắc phục tính mù thứ tự của attention thuần.  
4. Chồng encoder/decoder ghép cùng loại tầng thành hệ dịch máy hoặc seq2seq khác.  
5. BERT/GPT/ViT là các lựa chọn mask + mục tiêu + dữ liệu trên cùng khuôn mẫu này.

Tiếp theo: cài đặt các khối này cẩn thận trong **08-01-02**.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Attention trong Transformer (Self, Masked, Cross, Multi-Head)](https://www.facebook.com/reel/1007473105556936)
- [Position Encoding trong Transformer (từng bước)](https://www.facebook.com/reel/1306224801415127)
