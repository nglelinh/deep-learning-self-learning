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

Nếu Self-Attention là "phòng họp" nơi các từ trao đổi thông tin với nhau, thì câu hỏi đặt ra là: *tại sao sau mỗi phòng họp, Transformer lại phải cho từng từ đi vào một "phòng làm việc riêng" gọi là FFN, rồi lại phải chuẩn hóa bằng LayerNorm và cộng đường tắt Residual?* Bài học lý thuyết này giải mã chính xác từng mắt xích toán học giữ cho cả cỗ máy Transformer 100 tầng vẫn huấn luyện mượt mà.

## Bản đồ kiến thức nền: Bốn trụ cột lý thuyết của Transformer

1. **Mã hóa vị trí lượng giác (Sinusoidal Positional Encoding)**:
   Dùng cặp $$\sin(\text{pos}/10000^{2i/d})$$ và $$\cos(\text{pos}/10000^{2i/d})$$ ở nhiều tần số khác nhau — nhờ công thức cộng góc lượng giác, vị trí $$\text{pos} + k$$ luôn là một phép **biến đổi tuyến tính** (ma trận xoay 2D) của vị trí $$\text{pos}$$, giúp mô hình dễ dàng học khoảng cách tương đối!
2. **Pre-LN vs. Post-LN (Vị trí đặt LayerNorm)**:
   Bài báo gốc đặt LayerNorm *sau* phép cộng residual (Post-LN: $$\text{LN}(x + F(x))$$) nên rất nhạy với learning rate và bắt buộc phải có warmup kỹ. Hầu hết LLM hiện đại chuyển sang **Pre-LN** ($$x + F(\text{LN}(x))$$) để giữ đường cao tốc gradient thông suốt tuyệt đối từ tầng cuối về tầng đầu.
3. **Position-wise Feed-Forward Network (FFN)**:
   Nếu chỉ xếp chồng các tầng Attention thuần túy, sự kết hợp giữa các giá trị chủ yếu vẫn mang tính tuyến tính theo trọng số; FFN chiếu lên chiều $$4 d_{\text{model}}$$ qua ReLU/GELU/SwiGLU để tạo sức biểu đạt phi tuyến giàu có cho từng token.
4. **Liên kết trọng số (Weight Tying) và hệ số $$\sqrt{d_{\text{model}}}$$**:
   Chia sẻ chung ma trận giữa bảng Token Embedding đầu vào và tầng Linear dự đoán từ vựng đầu ra (LM Head) giúp tiết kiệm hàng chục triệu tham số, đồng thời nhân embedding với $$\sqrt{d_{\text{model}}}$$ để tín hiệu ngữ nghĩa không bị lấn át bởi Positional Encoding có biên độ $$\in [-1, 1]$$.

### Khung giải thích (nên đọc chậm): Tính tay vector Positional Encoding lượng giác tại 3 vị trí đầu tiên ($$d_{\text{model}} = 4$$)

Với $$d_{\text{model}} = 4$$, ta có 2 cặp chỉ số tần số:
- Cặp $$i = 0$$ (tần số cao): $$\omega_0 = 1 / 10000^{0/4} = 1$$ $$\to$$ $$(\sin(\text{pos}), \; \cos(\text{pos}))$$
- Cặp $$i = 1$$ (tần số thấp): $$\omega_1 = 1 / 10000^{2/4} = 1/100 = 0{,}01$$ $$\to$$ $$(\sin(0{,}01\,\text{pos}), \; \cos(0{,}01\,\text{pos}))$$

| Vị trí `pos` | Chiều 0: $$\sin(\text{pos})$$ | Chiều 1: $$\cos(\text{pos})$$ | Chiều 2: $$\sin(0{,}01\,\text{pos})$$ | Chiều 3: $$\cos(0{,}01\,\text{pos})$$ |
| :---: | :---: | :---: | :---: | :---: |
| **`pos = 0`** | $$0{,}0000$$ | $$1{,}0000$$ | $$0{,}0000$$ | $$1{,}0000$$ |
| **`pos = 1`** | $$0{,}8415$$ | $$0{,}5403$$ | $$0{,}0100$$ | $$0{,}9999$$ |
| **`pos = 2`** | $$0{,}9093$$ | $$-0{,}4161$$ | $$0{,}0200$$ | $$0{,}9998$$ |

Hãy tưởng tượng một chiếc đồng hồ nhị phân liên tục: cặp chiều `(0, 1)` đóng vai trò "kim giây" quay rất nhanh để phân biệt các từ kề nhau, còn cặp chiều `(2, 3)` đóng vai trò "kim giờ" quay chậm rãi để phân biệt các đoạn văn cách nhau hàng trăm từ!

# Lý thuyết Transformer

![Transformer Architecture](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_architecture.png)
*Kiến trúc Transformer encoder–decoder (Vaswani et al., 2017). Nguồn: Wikimedia Commons*

## 1. Tổng quan khái niệm

Transformer là kiến trúc xử lý dữ liệu chuỗi (và các dạng dữ liệu biểu diễn được thành chuỗi) dựa chủ yếu trên **attention đa đầu** (*multi-head attention*) và **mạng feed-forward theo từng vị trí** (*position-wise feed-forward network*), kết hợp với kết nối phần dư (*residual connection*) và chuẩn hóa lớp (*layer normalization*). Được Vaswani và cộng sự (2017) đề xuất cho dịch máy, Transformer loại bỏ hoàn toàn các tầng hồi quy và tích chập khỏi luồng xử lý chuỗi chính.

### Vì sao cải tiến RNN là chưa đủ

Ngay cả các kiến trúc hồi quy tiên tiến như LSTM hay GRU vẫn chịu ba giới hạn cấu trúc:

| Giới hạn | Hệ quả |
|------------|--------|
| Cập nhật trạng thái tuần tự từ trái sang phải | Khó song song hóa theo độ dài chuỗi khi huấn luyện |
| Đường truyền phụ thuộc dài | Tín hiệu từ token đầu chuỗi phải đi qua nhiều bước mới ảnh hưởng đến vị trí cuối |
| Trạng thái ẩn kích thước cố định | Tạo nút thắt mềm khi cần lưu giữ ngữ cảnh rất phong phú |

Cơ chế attention cho thấy mô hình có thể trực tiếp truy xuất các trạng thái liên quan. Transformer đi thêm một bước quyết định: loại bỏ hẳn phần hồi quy, để mọi tầng cho phép các vị trí chú ý trực tiếp tới nhau (kèm mặt nạ khi cần) rồi biến đổi phi tuyến độc lập trên từng vị trí bằng mạng MLP.

### Ba dòng kiến trúc chính

| Dòng kiến trúc | Cấu trúc attention | Mô hình tiêu biểu |
|--------|-----------|----------|
| Chỉ-encoder (*Encoder-only*) | Self-attention hai chiều | BERT, RoBERTa |
| Chỉ-decoder (*Decoder-only*) | Self-attention nhân quả (*causal*) | GPT-2/3/4, LLaMA |
| Encoder–decoder | Encoder kết hợp decoder có cross-attention | Transformer gốc, T5, BART |

Cả ba dòng đều dùng chung các khối nền tảng, chỉ khác nhau ở mẫu mặt nạ attention và mục tiêu huấn luyện.

### Cấu trúc tổng thể

1. Nhúng token kết hợp mã hóa vị trí.
2. Các tầng encoder xếp chồng: self-attention → FFN (mỗi tầng con kèm kết nối phần dư và chuẩn hóa lớp).
3. Các tầng decoder xếp chồng: masked self-attention → cross-attention → FFN.
4. Lớp chiếu tuyến tính đầu ra sang không gian từ vựng (cho tác vụ sinh) hoặc đầu phân loại (cho tác vụ hiểu).

![Toàn cảnh kiến trúc + Multi-Head / Scaled Dot-Product](/deep-learning-self-learning/img/chapter_img/chapter08/tf_full_arch_multihead_zoom.jpg)
*Hình: Kiến trúc encoder–decoder cùng chi tiết multi-head attention và scaled dot-product. (Minh họa từ video Attention trong Transformer)*

Chương 07 đã trình bày chi tiết toán học của attention; bài này ghép các phép toán đó thành **hệ thống hoàn chỉnh**, tập trung vào **mã hóa vị trí** và **cấu trúc từng tầng**.

---

## 2. Nền tảng toán học
### 2.1 Biểu diễn đầu vào

Với chuỗi chỉ số token $$t_1,\ldots,t_L$$, ma trận nhúng $$E$$ ánh xạ từng token thành vectơ, sau đó cộng với vectơ mã hóa vị trí $$\mathbf{p}_i$$:

$$\mathbf{x}_i^{(0)} = E_{t_i} + \mathbf{p}_i \in \mathbb{R}^{d_{\mathrm{model}}}$$

Xếp các hàng theo thứ tự ta thu được ma trận đầu vào:

$$\mathbf{X}^{(0)} \in \mathbb{R}^{L \times d_{\mathrm{model}}}.$$

(Trong mã nguồn thực tế có thêm chiều batch $$B$$, tức kích thước $$(B, L, d)$$.)

### 2.2 Scaled multi-head self-attention

Với các ma trận chiếu học được (chi tiết xem Chương 07):

$$
\mathrm{MultiHead}(\mathbf{X})
= \mathrm{Concat}_h\!\left(
  \mathrm{Attention}(\mathbf{X}W_h^Q,\mathbf{X}W_h^K,\mathbf{X}W_h^V)
\right) W^O
$$

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

trong đó $$M$$ là ma trận mặt nạ cộng (padding và/hoặc causal). Trong tầng Transformer, đây là bước **trao đổi thông tin** giữa các vị trí.

### 2.3 Mạng feed-forward theo vị trí (FFN)

Mạng FFN áp dụng **độc lập** lên từng vị trí với cùng một bộ trọng số:

$$
\mathrm{FFN}(\mathbf{z}) = W_2\,\sigma(W_1\mathbf{z}+\mathbf{b}_1)+\mathbf{b}_2
$$

Kích thước trung gian thường mở rộng theo tỷ lệ $$d_{\mathrm{model}} \to d_{\mathrm{ff}} \to d_{\mathrm{model}}$$ với $$d_{\mathrm{ff}} \approx 4\,d_{\mathrm{model}}$$. Hàm kích hoạt $$\sigma$$ là ReLU trong bài báo gốc, còn các mô hình mới hơn thường dùng GELU hoặc SwiGLU.

**Vai trò của FFN:** Multi-head attention chủ yếu thực hiện phép trộn tuyến tính có trọng số động giữa các vectơ value. Nếu thiếu lớp MLP phi tuyến trên từng vị trí, năng lực biến đổi đặc trưng trên từng token sẽ bị hạn chế. Có thể tóm tắt ngắn gọn: attention đảm nhiệm *trộn thông tin*, còn FFN đảm nhiệm *tính toán phi tuyến*.

### 2.4 Kết nối phần dư và chuẩn hóa lớp (*LayerNorm*)

Bài báo gốc bố trí chuẩn hóa sau phép cộng phần dư (**Post-LN**):

$$
\mathbf{Y} = \mathrm{LayerNorm}\big(\mathbf{X} + \mathrm{Sublayer}(\mathbf{X})\big)
$$

trong đó Sublayer là MultiHead hoặc FFN. Nhiều kiến trúc hiện đại chuyển sang bố cục **Pre-LN**:

$$
\mathbf{Y} = \mathbf{X} + \mathrm{Sublayer}\big(\mathrm{LayerNorm}(\mathbf{X})\big)
$$

vì giúp quá trình huấn luyện ổn định hơn khi mạng có nhiều tầng sâu. Kết nối phần dư tạo đường truyền trực tiếp cho gradient, còn LayerNorm giữ thang giá trị kích hoạt ổn định qua các tầng.

### 2.5 Tầng encoder đầy đủ

Ở dạng Post-LN (như bài báo gốc):

$$
\begin{aligned}
\mathbf{U} &= \mathrm{LayerNorm}\big(\mathbf{X} + \mathrm{MultiHeadSelfAttn}(\mathbf{X})\big) \\
\mathbf{Z} &= \mathrm{LayerNorm}\big(\mathbf{U} + \mathrm{FFN}(\mathbf{U})\big)
\end{aligned}
$$

Toàn bộ encoder gồm $$N$$ tầng như vậy xếp nối tiếp (mô hình gốc dùng $$N=6$$, các LLM hiện đại có thể dùng hàng chục đến hơn một trăm tầng).

### 2.6 Tầng decoder đầy đủ

Mỗi tầng decoder gồm ba tầng con:

1. **Masked self-attention** trên các token đích đã sinh (áp dụng mặt nạ nhân quả).
2. **Cross-attention** với query lấy từ decoder, còn key và value lấy từ đầu ra của encoder.
3. **FFN** tương tự phía encoder.

$$
\begin{aligned}
\mathbf{U} &= \mathrm{LN}\big(\mathbf{Y} + \mathrm{MaskedSelfAttn}(\mathbf{Y})\big) \\
\mathbf{V} &= \mathrm{LN}\big(\mathbf{U} + \mathrm{CrossAttn}(\mathbf{U}, \mathbf{H}_{\mathrm{enc}}, \mathbf{H}_{\mathrm{enc}})\big) \\
\mathbf{Z} &= \mathrm{LN}\big(\mathbf{V} + \mathrm{FFN}(\mathbf{V})\big)
\end{aligned}
$$

**Mặt nạ nhân quả (causal mask):** Với query ở vị trí $$i$$, các key ở vị trí $$j > i$$ bị chặn để mô hình không nhìn trước đáp án khi huấn luyện bằng *teacher forcing*.

### 2.7 Mã hóa vị trí dạng sin–cos

Vì attention thuần có tính đồng biến theo hoán vị (*permutation equivariant*), thứ tự token sẽ vô nghĩa nếu không có tín hiệu vị trí. Công thức mã hóa vị trí gốc:

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin\!\big(pos / 10000^{2i/d_{\mathrm{model}}}\big) \\
PE_{(pos,2i+1)} &= \cos\!\big(pos / 10000^{2i/d_{\mathrm{model}}}\big)
\end{aligned}
$$

![Công thức PE + ma trận token](/deep-learning-self-learning/img/chapter_img/chapter08/pe_formula_and_token_matrix.jpg)
*Hình: Mỗi token tương ứng một vectơ kích thước $$d_{\mathrm{model}}$$; PE dùng hàm sin/cos theo vị trí $$pos$$ và chỉ số tần số $$i$$. (Minh họa từ video Position Encoding)*

![pos = chỉ số từ trong câu](/deep-learning-self-learning/img/chapter_img/chapter08/pe_pos_index_sentence.jpg)
*Hình: Chỉ số vị trí $$pos = 0,1,2,\ldots$$ đánh số theo thứ tự token trong câu. (Minh họa từ video Position Encoding)*

**Cách ánh xạ chỉ số chiều:** Với $$d_{\mathrm{model}}=512$$, chỉ số $$i$$ chạy từ $$0$$ đến $$255$$; mỗi giá trị $$i$$ tạo **hai** chiều — chiều chẵn $$2i$$ dùng **sin** và chiều lẻ $$2i+1$$ dùng **cos**.

![Bảng i → chiều sin/cos](/deep-learning-self-learning/img/chapter_img/chapter08/pe_i_range_sin_cos_dims.jpg)
*Hình: $$i=0$$ tương ứng chiều 0 (sin) và 1 (cos); …; $$i=255$$ tương ứng chiều 510 và 511. (Minh họa từ video Position Encoding)*

![Chiều chẵn sin / lẻ cos](/deep-learning-self-learning/img/chapter_img/chapter08/pe_even_odd_dims.jpg)
*Hình: Gán hàm sin vào các chiều chẵn và hàm cos vào các chiều lẻ của vectơ PE. (Minh họa từ video Position Encoding)*

![Sin và cos](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sin_cos_formula_plot.jpg)
*Hình: Cặp sóng sin/cos cùng tần số làm cơ sở cho mã hóa vị trí. (Minh họa từ video Position Encoding)*

![Nhiều tần số theo i](/deep-learning-self-learning/img/chapter_img/chapter08/pe_multiple_frequencies_i.jpg)
*Hình: Mỗi chỉ số $$i$$ ứng với một tần số khác nhau: $$i$$ nhỏ ứng với sóng dao động nhanh (tần số cao), $$i$$ lớn ứng với sóng dao động chậm (tần số thấp). (Minh họa từ video Position Encoding)*

**Tạo vectơ PE tại một vị trí:** Cố định $$pos$$, với mỗi chỉ số $$i$$ ta lấy giá trị sin và cos tương ứng rồi ghép thành vectơ dài $$d_{\mathrm{model}}$$.

![Lấy PE tại pos=0](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sample_values_pos0.jpg)
*Hình: Cột giá trị tại $$pos=0$$ tạo thành vectơ PE cho token đầu tiên. (Minh họa từ video Position Encoding)*

![Lấy PE tại pos=3](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sample_values_pos3.jpg)
*Hình: Tại $$pos=3$$, các sóng đã lệch pha nên tạo ra vectơ PE khác biệt. (Minh họa từ video Position Encoding)*

**Cộng vào token embedding:** Vì có cùng số chiều, vectơ PE được cộng trực tiếp theo từng phần tử vào token embedding:

$$
\mathbf{x}_{pos} = \mathrm{TokenEmb}(w_{pos}) + \mathrm{PE}_{pos}
$$

![Token emb vs PE (cùng 512-d)](/deep-learning-self-learning/img/chapter_img/chapter08/pe_input_emb_vs_positional.jpg)
*Hình: Token embedding và PE có cùng chiều $$d_{\mathrm{model}}$$ và được cộng theo từng phần tử để gắn thông tin thứ tự vào biểu diễn nội dung. (Minh họa từ video Position Encoding)*

![Cột token emb (tím) + cột PE (xanh)](/deep-learning-self-learning/img/chapter_img/chapter08/pe_token_emb_plus_pe_columns.jpg)
*Hình: Mỗi vị trí gồm một cột token embedding cộng với cột PE tương ứng. (Minh họa từ video Position Encoding)*

Các đặc điểm chính của PE dạng sin–cos:

- Xác định sẵn, không cần tham số học.
- Tạo biểu diễn riêng biệt cho từng vị trí nhờ tổ hợp nhiều tần số.
- Độ dịch vị trí tương đối $$PE_{pos+k}$$ có thể biểu diễn dưới dạng phép biến đổi tuyến tính của $$PE_{pos}$$, thuận lợi cho việc học quan hệ khoảng cách.
- Về lý thuyết có khả năng ngoại suy sang độ dài lớn hơn lúc huấn luyện tốt hơn so với embedding vị trí tuyệt đối học trực tiếp (dù trong thực tế vẫn có giới hạn).

**Các phương án thay thế:** embedding vị trí tuyệt đối học được (BERT, GPT-2), độ lệch vị trí tương đối (T5, Shaw et al.) và RoPE (phổ biến trong các LLM hiện đại). Tất cả đều giải quyết cùng một nhu cầu là đưa thông tin thứ tự vào cơ chế trộn theo nội dung.

### 2.8 Độ phức tạp tính toán

Self-attention đầy đủ tốn $$O(L^2 d)$$ phép tính và $$O(L^2)$$ bộ nhớ cho ma trận điểm số — đây là chi phí đổi lấy khả năng tương tác trực tiếp toàn cục. Với độ dài câu hoặc đoạn thông thường, chi phí này hoàn toàn khả thi; với văn bản rất dài, các biến thể attention thưa hoặc tuyến tính được sử dụng để giảm tải.

### 2.9 Mục tiêu huấn luyện

Trong bài toán dịch máy gốc, encoder đọc chuỗi nguồn còn decoder dự đoán từng token đích bằng hàm mất mát cross-entropy theo cơ chế *teacher forcing*. Các mô hình chỉ-encoder hiện đại dùng mục tiêu mô hình ngôn ngữ có che (Masked LM), còn mô hình chỉ-decoder dùng dự đoán token kế tiếp (Causal LM). Kiến trúc cốt lõi giữ nguyên; sự thay đổi nằm ở mẫu mặt nạ và mục tiêu huấn luyện.

---

## 3. Ví dụ và Trực giác

### 3.1 Đường đi của một token trong encoder

Xét câu `I love deep learning` tại vị trí từ **learning**. Ở tầng self-attention đầu tiên, query của `learning` có thể gán trọng số lớn cho `deep` và chính nó, tạo ra biểu diễn mang nghĩa cụm *"deep learning"* thay vì nghĩa rời rạc của từ *"learning"*. Sau đó FFN biến đổi phi tuyến biểu diễn này. Quá trình trên diễn ra **song song** cho cả bốn từ chỉ trong một lượt nhân ma trận, thay vì bốn bước tuần tự như RNN.

### 3.2 Một bước giải mã trong decoder

Khi sinh từ tiếng Pháp `profond` sau chuỗi `J' aime l'apprentissage`:

1. **Masked self-attention** chỉ chú ý tới các từ tiếng Pháp đã sinh trước đó để giữ mạch câu nhất quán.
2. **Cross-attention** tra cứu các trạng thái encoder tiếng Anh và tập trung trọng số vào từ `deep`.
3. **FFN cùng lớp softmax trên từ vựng** dự đoán từ tiếp theo là `profond`.

Nếu mặt nạ nhân quả cài đặt sai, mô hình sẽ đọc được đáp án tương lai lúc huấn luyện nhưng hoạt động kém hẳn khi suy luận tự hồi quy thực tế.

### 3.3 So sánh chỉ-encoder và chỉ-decoder

- **BERT (chỉ-encoder):** dùng self-attention hai chiều cùng Masked LM, phù hợp các bài toán hiểu và trích xuất biểu diễn.
- **GPT (chỉ-decoder):** dùng self-attention nhân quả cùng mục tiêu dự đoán token kế tiếp, phù hợp các bài toán sinh văn bản.

Cả hai đều sử dụng cùng khối MultiHead Attention và FFN, chỉ khác mặt nạ và luồng dữ liệu.

---

## 4. Mã minh họa bám theo phương trình

Cài đặt tối giản cho một tầng encoder theo bố cục Pre-LN:

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

Các module cài đặt đầy đủ từ đầu sẽ được trình bày ở các bài thực hành tiếp theo.

---

## 5. Khái niệm liên quan

- **Seq2seq dùng RNN**: cùng mô hình mã hóa–giải mã nhưng Transformer thay tầng hồi quy bằng attention.
- **Tích chập (CNN)**: bộ lọc cục bộ có trọng số cố định, đối lập với phép trộn toàn cục có trọng số động của attention.
- **Mạng phần dư (ResNet)**: giúp xếp chồng nhiều tầng sâu mà không làm suy giảm gradient.
- **Các mục tiêu tiền huấn luyện**: Masked LM (BERT), Causal LM (GPT) và khôi phục đoạn bị che (T5).
- **Vision Transformer (ViT)**: chia ảnh thành các mảng (patch) rồi xử lý như chuỗi token bằng chính kiến trúc encoder Transformer.

---

## 6. Các bài báo nền tảng

1. [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762) — đề xuất kiến trúc Transformer gốc.
2. [Devlin et al., 2019 — BERT](https://arxiv.org/abs/1810.04805) — tiền huấn luyện encoder hai chiều.
3. [Radford et al., 2019 — GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — mô hình ngôn ngữ nhân quả quy mô lớn và khả năng zero-shot.
4. [Dosovitskiy et al., 2021 — ViT](https://arxiv.org/abs/2010.11929) — ứng dụng Transformer cho thị giác máy tính.
5. [Phuong & Hutter, 2022 — Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238) — đặc tả thuật toán chính thức, rõ ràng cho các biến thể Transformer.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành

- **Thiếu mã hóa vị trí** khiến mô hình không phân biệt được thứ tự từ.
- **Dùng giá trị `0` thay vì $$-\infty$$** cho causal mask trước softmax gây rò rỉ thông tin tương lai.
- **Quên padding mask** khi huấn luyện theo batch gồm các chuỗi có độ dài khác nhau.
- **Thiếu hệ số chia $$1/\sqrt{d_k}$$** khi chiều head lớn.
- **Nhầm cấu hình Post-LN với Pre-LN** khi sao chép lịch tốc độ học và warmup từ tài liệu khác.
- **Mẹo:** luôn viết unit-test kiểm tra `triu(weights, 1) ≈ 0` cho causal attention.
- **Mẹo:** thử nghiệm trước với mô hình nhỏ 2 tầng rồi mới mở rộng độ sâu và bề rộng.

---

## 8. Tóm tắt các điểm chính

1. Mỗi tầng Transformer kết hợp **attention (trao đổi thông tin)** và **FFN (biến đổi cục bộ)** cùng kết nối phần dư và chuẩn hóa lớp.
2. **Mẫu mặt nạ** quyết định dòng kiến trúc: hai chiều (encoder), nhân quả (decoder) hay chéo (cross-attention).
3. **Mã hóa vị trí** bù đắp việc self-attention thuần không tự nhận biết thứ tự.
4. Encoder và decoder ghép các tầng cùng loại để giải quyết dịch máy và nhiều bài toán chuỗi khác.
5. BERT, GPT và ViT đều xây dựng trên cùng một khuôn mẫu kiến trúc, chỉ khác mặt nạ, mục tiêu huấn luyện và loại dữ liệu đầu vào.

Tiếp theo, ta sẽ cài đặt chi tiết từng khối trong bài **08-01-02**.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến); các đường dẫn dưới đây được lưu lại để tra cứu và ghi nhận nguồn:

- [Attention trong Transformer (Self, Masked, Cross, Multi-Head)](https://www.facebook.com/reel/1007473105556936)
- [Position Encoding trong Transformer (từng bước)](https://www.facebook.com/reel/1306224801415127)

## Video tham khảo

- **Xem trước:** [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown (tiếng Anh). Trình bày trọn vẹn các thành phần embedding, attention, MLP, softmax và nhiệt độ tương ứng với bài lý thuyết.
- [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Đi sâu vào phép tính attention bên trong khối Transformer.
- [Tất cả những gì bạn cần biết về Transformer trong 35 phút](https://www.youtube.com/watch?v=Gc0HXCWT0IA) — Việt Nguyễn AI (tiếng Việt). Tổng quan đầy đủ về kiến trúc Transformer bằng tiếng Việt.
