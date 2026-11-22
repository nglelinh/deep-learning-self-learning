---
layout: post
title: 07-02-02-02 Multi-Head Attention, Papers và Cạm bẫy
chapter: '07'
order: 7
owner: Deep Learning Course
lang: vi
categories:
- chapter07
---

# Multi-Head Attention, Papers và Cạm bẫy

Cài đặt được single-head attention mới chỉ là bước đầu; trong các hệ thống Transformer thực tế, hiệu năng phụ thuộc chặt chẽ vào **nhiều head chú ý song song**, thao tác tách/ghép chiều tensor chính xác, mặt nạ nhân quả và khả năng phát hiện các lỗi ẩn khi mở rộng quy mô. Bài này hoàn thiện phần cài đặt cho Chương 7.

---

## 1. Tổng quan khái niệm

Multi-head attention không đơn thuần là lặp lại attention cho chậm hơn mà là cơ chế mở rộng năng lực biểu diễn. Mỗi head sử dụng bộ ma trận chiếu riêng $$(W_h^Q, W_h^K, W_h^V)$$, cho phép nó học một tiêu chí tương đồng và một mẫu định tuyến thông tin khác nhau. Các đầu ra sau đó được nối lại theo chiều đặc trưng và trộn qua ma trận $$W^O$$ để trở về chiều của mô hình.

![Kiến trúc Transformer + zoom Multi-Head / Scaled Dot-Product](/deep-learning-self-learning/img/chapter_img/chapter08/tf_full_arch_multihead_zoom.jpg)
*Hình: Multi-head gồm nhiều nhánh Linear(Q, K, V) → scaled attention → nối (concat) → lớp tuyến tính đầu ra. (Minh họa từ video Attention trong Transformer)*

![$$d_k = d_{\mathrm{model}}/H$$](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_dk_split.jpg)
*Hình: Ví dụ $$512/8=64$$ chiều cho mỗi head. (Minh họa từ video công thức Attention QKV)*

![Đầu ra multi-head](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_output_stack.jpg)
*Hình: Ghép đầu ra của các head trước khi nhân ma trận $$W^O$$. (Minh họa từ video công thức Attention QKV)*

Quy trình thực hiện multi-head attention gồm 5 bước:

1. Chiếu $$\mathbf{X}$$ sang Q, K, V với đầy đủ chiều $$d_{\mathrm{model}}$$.
2. Tách và hoán vị chiều thành $$(B, H, L, d_k)$$ với $$d_k=d_{\mathrm{model}}/H$$.
3. Chạy scaled dot-product attention song song trên tất cả các head.
4. Ghép lại các head thành $$(B, L, d_{\mathrm{model}})$$.
5. Áp dụng phép chiếu tuyến tính cuối $$W^O$$.

Nếu thao tác reshape hoặc transpose bị sai thứ tự, tensor đầu ra vẫn có thể giữ nguyên kích thước mong đợi nhưng dữ liệu bên trong bị trộn lẫn. Do đó, kiểm thử đơn vị cho bước tách/ghép head là rất cần thiết.

---

## 2. Nền tảng toán học

$$
\mathrm{head}_h = \mathrm{Attention}(X W_h^Q,\ X W_h^K,\ X W_h^V)
$$

$$
\mathrm{MultiHead}(X)=\mathrm{Concat}_h(\mathrm{head}_h)\,W^O
$$

Trong cài đặt thực tế, thay vì thực hiện $$H$$ phép nhân nhỏ tách rời, ta nhân một lần với ma trận lớn kích thước $$d_{\mathrm{model}}$$ rồi tách chiều cuối thành $$H \times d_k$$:

$$
Q = X W^Q \in \mathbb{R}^{B\times L\times (H d_k)}
\ \xrightarrow{\text{reshape/transpose}}\ 
\mathbb{R}^{B\times H\times L\times d_k}.
$$

Trong giải mã tự hồi quy (causal), mặt nạ:

$$
M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}
$$

được cộng vào ma trận điểm số và broadcast đồng loạt trên mọi batch và mọi head.

---

## 3. Ví dụ và Trực giác

Trên một câu ngắn như *"The cat sat"*, sau quá trình huấn luyện các head thường tự chuyên biệt hóa theo những vai trò khác nhau:

- Head A theo dõi quan hệ lân cận cục bộ (`cat` ↔ `The`, `sat` ↔ `cat`).
- Head B nắm bắt liên kết cú pháp (`sat` chú ý vào chủ ngữ `cat`).
- Head C tập trung gần đường chéo chính (chủ yếu theo vị trí).

Các vai trò này không được lập trình cứng mà hình thành tự nhiên từ tham số khởi tạo riêng biệt và áp lực tối ưu hóa hàm mất mát. Trực quan hóa bản đồ trọng số của từng head giúp quan sát rõ sự phân hóa này.

---

## 4. Mã minh họa
### 4.1 Multi-head self-attention (PyTorch, phiên bản giảng dạy)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.H = num_heads
        self.dk = d_model // num_heads

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        # (B, L, d) -> (B, H, L, dk)
        B, L, _ = x.shape
        x = x.view(B, L, self.H, self.dk).transpose(1, 2)
        return x

    def _merge(self, x):
        # (B, H, L, dk) -> (B, L, d)
        B, H, L, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * dk)

    def forward(self, x, attn_mask=None, need_weights=False):
        """
        x: (B, L, d_model)
        attn_mask: mask cộng broadcast được tới (B, 1, L, L) hoặc (B, H, L, L)
                   0 = giữ, -inf = cấm
        """
        B, L, _ = x.shape
        qkv = self.Wqkv(x)  # (B, L, 3d)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        out = torch.matmul(weights, v)
        out = self.Wo(self._merge(out))
        if need_weights:
            return out, weights  # (B, H, L, L)
        return out


def build_causal_mask(L, device=None):
    # (1, 1, L, L) để broadcast trên batch và heads
    m = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
    return m.view(1, 1, L, L)


# Smoke test
B, L, d, H = 2, 6, 32, 4
x = torch.randn(B, L, d)
mha = MultiHeadSelfAttention(d, H)
y, w = mha(x, attn_mask=build_causal_mask(L), need_weights=True)
assert y.shape == (B, L, d)
assert w.shape == (B, H, L, L)
assert torch.allclose(w.triu(diagonal=1), torch.zeros_like(w), atol=1e-6)
print("multi-head causal OK", y.shape, w.shape)
```

### 4.2 Dùng module có sẵn

```python
official = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
# key_padding_mask: (B, L) True = bỏ qua (padding)
# attn_mask: (L, L) mask True/float tùy phiên bản — đọc docs cẩn thận
y2, w2 = official(x, x, x, need_weights=True, average_attn_weights=False)
print(y2.shape, w2.shape)
```

Cần kiểm tra kỹ tài liệu theo đúng phiên bản PyTorch đang sử dụng vì quy ước kiểu dữ liệu của `attn_mask` (`bool` so với `float` cộng) có thể khác nhau.

### 4.3 Phác thảo seq2seq-with-attention tối giản

```python
class Seq2SeqWithAttention(nn.Module):
    """
    Encoder-decoder giáo dục nhỏ:
    - encoder GRU trên source embeddings
    - bước decoder GRU
    - attention dot kiểu Luong trên encoder outputs
    Không phải NMT production; shape là bài học.
    """

    def __init__(self, src_vocab, tgt_vocab, d_model=64):
        super().__init__()
        self.d = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        self.out = nn.Linear(2 * d_model, tgt_vocab)

    def attend(self, dec_h, enc_out):
        # dec_h: (B, 1, d), enc_out: (B, Ls, d)
        # scores: (B, 1, Ls)
        scores = torch.matmul(dec_h, enc_out.transpose(1, 2)) / math.sqrt(self.d)
        alpha = F.softmax(scores, dim=-1)
        ctx = torch.matmul(alpha, enc_out)  # (B, 1, d)
        return ctx, alpha

    def forward(self, src, tgt):
        # src: (B, Ls), tgt: (B, Lt)  — đầu vào teacher forcing
        enc_out, _ = self.encoder(self.src_emb(src))
        dec_in = self.tgt_emb(tgt)
        dec_out, _ = self.decoder(dec_in)
        # mỗi vị trí decoder chú ý tới encoder (theo batch)
        scores = torch.matmul(dec_out, enc_out.transpose(1, 2)) / math.sqrt(self.d)
        alpha = F.softmax(scores, dim=-1)           # (B, Lt, Ls)
        ctx = torch.matmul(alpha, enc_out)          # (B, Lt, d)
        logits = self.out(torch.cat([dec_out, ctx], dim=-1))
        return logits, alpha


model = Seq2SeqWithAttention(100, 120)
src = torch.randint(0, 100, (2, 4))
tgt = torch.randint(0, 120, (2, 3))
logits, alpha = model(src, tgt)
print(logits.shape, alpha.shape)  # (2,3,120), (2,3,4)
```

### 4.4 Trực quan hóa một head (tùy chọn)

```python
import matplotlib.pyplot as plt

def plot_attention(weights_2d, x_labels=None, y_labels=None, title="Attention"):
    """weights_2d: mảng numpy (Lq, Lk)"""
    plt.figure(figsize=(6, 5))
    plt.imshow(weights_2d, aspect="auto", interpolation="nearest")
    plt.colorbar()
    if x_labels is not None:
        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    if y_labels is not None:
        plt.yticks(range(len(y_labels)), y_labels)
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.show()

# Ví dụ với trọng số multi-head causal ở trên
# plot_attention(w[0, 0].detach().numpy())  # batch 0, head 0
```

---

## 5. Khái niệm liên quan

- **Residual và LayerNorm**: cách bố trí Pre-LN hay Post-LN quanh khối multi-head ảnh hưởng trực tiếp tới độ ổn định khi huấn luyện mạng sâu.
- **Độ lệch vị trí tương đối (Relative position bias)**: các mô hình như T5 hay Shaw et al. cộng thêm hạng tử phụ thuộc khoảng cách tương đối vào điểm số attention.
- **Cross-attention**: trong Transformer encoder–decoder, cross-attention dùng lại đúng cấu trúc multi-head nhưng lấy Q từ decoder và K, V từ encoder.
- **FlashAttention / SDPA**: tối ưu hóa truy cập bộ nhớ bằng cách hợp nhất chuỗi phép tính matmul–softmax–matmul trong khi giữ nguyên kết quả toán học.
- **Attention thưa (Sparse attention)**: các kiến trúc như Longformer hay BigBird giới hạn tập vị trí có trọng số $$\alpha$$ khác 0 để xử lý chuỗi rất dài.

---

## 6. Các bài báo nền tảng

1. **[Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473)** — *Neural Machine Translation by Jointly Learning to Align and Translate.*  
   Giới thiệu attention cộng tính giúp decoder tự học căn chỉnh mềm với các trạng thái nguồn mà không cần nhãn căn chỉnh tường minh.

2. **[Luong et al., 2015](https://arxiv.org/abs/1508.04025)** — *Effective Approaches to Attention-based NMT.*  
   Trình bày các hàm chấm điểm nhân tính, phân biệt attention toàn cục và cục bộ cùng kỹ thuật *input feeding* cho decoder.

3. **[Xu et al., 2015](https://arxiv.org/abs/1502.03044)** — *Show, Attend and Tell.*  
   Áp dụng attention thị giác (cả dạng mềm và dạng cứng) cho bài toán sinh chú thích ảnh, mở rộng phạm vi ứng dụng đa phương thức.

4. **[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)** — *Attention Is All You Need.*  
   Đề xuất multi-head scaled self-attention làm phép toán xử lý chuỗi trung tâm cho kiến trúc Transformer.

5. **[Shaw et al., 2018](https://arxiv.org/abs/1803.02155)** — *Self-Attention with Relative Position Representations.*  
   Đưa biểu diễn khoảng cách tương đối vào điểm số attention, giúp cải thiện khả năng tổng quát hóa theo độ dài chuỗi.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành
### Các lỗi nghiêm trọng cần tránh

1. **Softmax sai trục** — phải luôn chuẩn hóa dọc theo trục key/bộ nhớ.
2. **Áp mask sau softmax** — cần cộng `-inf` trực tiếp vào điểm số trước softmax.
3. **Quên hệ số $$1/\sqrt{d_k}$$** — gây bão hòa softmax và suy giảm gradient khi chiều head lớn.
4. **Reshape head sai thứ tự** — luôn kiểm thử chuỗi `view → transpose → attention → transpose → view` trên tensor đã biết giá trị.
5. **Rò rỉ padding** — luôn truyền `key_padding_mask` tường minh khi huấn luyện theo batch có độ dài khác nhau.
6. **Rò rỉ thông tin tương lai (causal)** — kiểm tra bằng unit-test để đảm bảo phần tam giác trên của trọng số bằng 0.
7. **Nhầm vị trí đặt dropout** — cần nhất quán xem dropout áp lên trọng số sau softmax hay ở đầu ra, đồng thời đảm bảo chuyển đúng chế độ `train`/`eval`.

### Danh mục kiểm tra gỡ lỗi

```text
[ ] In shapes: Q,K,V,scores,weights,output
[ ] weights.sum(dim=keys) == 1
[ ] causal: triu(weights,1) == 0
[ ] pad positions: weights[..., pad] == 0
[ ] Không NaN sau chạy dài (kiểm scale + trừ max trong softmax tự viết)
[ ] Gradient chảy tới Wq/Wk/Wv (hook hoặc .grad sau backward)
```

### Mẹo thực hành

- Bắt đầu kiểm chứng với **một head** và attention đầy đủ; chỉ mở rộng sang nhiều head sau khi phiên bản đơn giản đã hoạt động ổn định.
- Theo dõi **entropy của phân phối attention** theo từng lớp: entropy quá gần 0 cho thấy phân phối bị tập trung cực đoan, còn quá cao có thể báo hiệu trọng số gần như đều nhau (chưa học được mẫu).
- Với bài toán dịch máy hoặc seq2seq, trực quan hóa bản đồ attention trên vài cặp câu mẫu sau mỗi epoch để phát hiện sớm lỗi mask.
- Dùng các kernel SDPA tối ưu khi chạy thực tế, nhưng giữ bản cài đặt chuẩn đơn giản để làm mốc đối chiếu khi kiểm thử.

---

## 8. Tóm tắt các điểm chính

1. **Multi-head attention** kết hợp nhiều cơ chế định tuyến mềm chạy song song rồi trộn đầu ra qua lớp tuyến tính $$W^O$$.
2. Thao tác tách và ghép head là bài toán sắp xếp chiều tensor nên cần kiểm thử cẩn thận.
3. **Causal mask và padding mask** là bộ phận cấu thành của thuật toán trong mọi hệ thống thực tế.
4. Chuỗi bài báo từ Bahdanau, Luong đến Vaswani đánh dấu bước chuyển từ thành phần bổ trợ cho RNN sang kiến trúc độc lập.
5. Phần lớn lỗi khi attention không hội tụ xuất phát từ **sai trục softmax, sai mask hoặc thiếu chia tỉ lệ**, hơn là do bộ tối ưu.

Đến đây, bạn đã đi trọn mạch từ trực giác, toán học, mã nguồn cốt lõi đến multi-head attention. Chương 08 sẽ kết hợp các khối này cùng mạng feed-forward, kết nối phần dư và mã hóa vị trí để xây dựng kiến trúc Transformer hoàn chỉnh.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến); các đường dẫn dưới đây được lưu lại để tra cứu và ghi nhận nguồn:

- [Công thức Attention: tự tay tính Q, K, V](https://www.facebook.com/reel/1806844676942638)
- [Attention trong Transformer (Self, Masked, Cross, Multi-Head)](https://www.facebook.com/reel/1007473105556936)

## Video tham khảo

- **Xem trước:** [A Dive Into Multihead Attention, Self-Attention and Cross-Attention](https://www.youtube.com/watch?v=mmzRYGCfTzc) — Machine Learning Studio (tiếng Anh). Làm rõ bản chất của multi-head là nhiều bản đồ attention song song thay vì một lớp softmax lớn hơn.
- [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Trình bày trực quan kỹ thuật che (masking) cho ràng buộc nhân quả trong phần bẫy thường gặp.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Cài đặt mặt nạ nhân quả trực tiếp trong mã nguồn, giúp thấy rõ hậu quả nếu thiếu bước này.
