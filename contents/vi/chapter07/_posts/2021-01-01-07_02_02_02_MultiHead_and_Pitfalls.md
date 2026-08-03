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

Có thể cài single-head attention mà vẫn bỏ lỡ những gì khiến Transformer hoạt động trong thực tế: **nhiều mẫu attention song song**, đóng gói head đúng, giải mã causal, và các chế độ hỏng chỉ xuất hiện khi scale lên. Bài này hoàn tất nhánh cài đặt.

---

## 1. Tổng quan khái niệm

Multi-head attention không phải “attention nhưng chậm hơn.” Đó là cơ chế dung lượng: mỗi head dùng phép chiếu riêng $$(W_h^Q, W_h^K, W_h^V)$$, nên mỗi head có thể học một khái niệm tương đồng khác và một soft routing khác trên các vị trí. Nối các head giữ các kênh đó; ma trận đầu ra $$W^O$$ trộn chúng vào chiều mô hình.

![Kiến trúc Transformer + zoom Multi-Head / Scaled Dot-Product](/deep-learning-self-learning/img/chapter_img/chapter08/tf_full_arch_multihead_zoom.jpg)
*Hình: Multi-head = nhiều nhánh Linear(Q,K,V) → scaled attention → concat → Linear. (Minh họa từ video Attention trong Transformer)*

![$$d_k = d_{\mathrm{model}}/H$$](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_dk_split.jpg)
*Hình: Ví dụ $$512/8=64$$ — chiều mỗi head. (Minh họa từ video công thức Attention QKV)*

![Đầu ra multi-head](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_output_stack.jpg)
*Hình: Ghép các head trước linear $$W^O$$. (Minh họa từ video công thức Attention QKV)*

Về thao tác, multi-head attention là:

1. Chiếu $$\mathbf{X}$$ thành Q, K, V ở full $$d_{\mathrm{model}}$$.  
2. Reshape thành $$(B, H, L, d_k)$$ với $$d_k=d_{\mathrm{model}}/H$$.  
3. Chạy scaled attention **theo head** (theo batch).  
4. Gộp heads → $$(B, L, d_{\mathrm{model}})$$.  
5. Ánh xạ tuyến tính cuối $$W^O$$.

Reshape/transpose sai thì shape vẫn “đúng” trong khi thông tin bị xáo trộn—do đó có các test bên dưới.

---

## 2. Nền tảng toán học

$$
\mathrm{head}_h = \mathrm{Attention}(X W_h^Q,\ X W_h^K,\ X W_h^V)
$$

$$
\mathrm{MultiHead}(X)=\mathrm{Concat}_h(\mathrm{head}_h)\,W^O
$$

Với ma trận lớn dùng chung (phổ biến trong code), ta chiếu một lần lên $$d_{\mathrm{model}}$$, rồi **view** chiều cuối như $$H \times d_k$$:

$$
Q = X W^Q \in \mathbb{R}^{B\times L\times (H d_k)}
\ \xrightarrow{\text{reshape/transpose}}\ 
\mathbb{R}^{B\times H\times L\times d_k}.
$$

Giải mã causal dùng

$$
M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}
$$

cộng vào điểm số trong mọi head (cùng mask broadcast qua các head).

---

## 3. Ví dụ / Trực giác

Trên câu như “The cat sat,” các head khác nhau thường chuyên biệt sau huấn luyện:

- Head A: bigram cục bộ (`cat`↔`The`, `sat`↔`cat`)  
- Head B: cú pháp yếu (`sat` chú ý chủ ngữ `cat`)  
- Head C: gần như vị trí (đường chéo mạnh)

Ta không lập trình các vai trò đó; tham số riêng + áp lực đa nhiệm từ loss khuyến khích đa dạng. Trực quan hóa bản đồ theo head là cách thực tiễn để thấy chuyên biệt hóa.

---

## 4. Mã minh họa
### 4.1 Multi-head self-attention (PyTorch, phiên bản dạy)

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

Luôn kiểm tra kỹ quy ước mask theo phiên bản PyTorch (`bool` so với float cộng).

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

- **Residual + LayerNorm** quanh khối multi-head (Pre-LN so với Post-LN) chi phối khả năng huấn luyện.  
- **Relative position bias** (T5, Shaw et al.) thêm hạng khoảng cách vào điểm số.  
- **Cross-attention** trong Transformer encoder–decoder tái sử dụng cùng code multi-head với Q từ decoder và K,V từ encoder.  
- **FlashAttention / SDPA** hợp nhất matmul-softmax-matmul để tiết kiệm bộ nhớ; toán học không đổi.  
- **Sparse attention** (Longformer, BigBird) hạn chế support của $$\alpha$$ khi $$L$$ dài.

---

## 6. Các bài báo nền tảng

1. **[Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473)** — *Neural Machine Translation by Jointly Learning to Align and Translate.*  
   Additive attention nền tảng; decoder soft-search các trạng thái nguồn. Thiết lập alignment học được không cần giám sát tường minh.

2. **[Luong et al., 2015](https://arxiv.org/abs/1508.04025)** — *Effective Approaches to Attention-based NMT.*  
   Điểm số nhân tính, attention global so với local, mẹo decoder thực tiễn (input feeding).

3. **[Xu et al., 2015](https://arxiv.org/abs/1502.03044)** — *Show, Attend and Tell.*  
   Attention thị giác cho chú thích ảnh; soft so với hard attention; tính tổng quát đa phương thức của ý tưởng.

4. **[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)** — *Attention Is All You Need.*  
   Multi-head scaled self-attention như toán tử chuỗi chính; khuôn mẫu cho các stack hiện đại.

5. **[Shaw et al., 2018](https://arxiv.org/abs/1803.02155)** — *Self-Attention with Relative Position Representations.*  
   Khoảng cách tương đối trong logit attention; cải thiện tổng quát hóa độ dài ở nhiều biến thể.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành
### Lỗi nghiêm trọng

1. **Softmax sai trục** — phải là trục key/bộ nhớ.  
2. **Mask sau softmax** — dùng `-inf` cộng trên logit.  
3. **Quên $$1/\sqrt{d_k}$$** — attention sắc, gradient thấp khi chiều rộng lớn.  
4. **Reshape head sai** — luôn test `view → transpose → attention → transpose → view` trên tensor biết trước.  
5. **Rò rỉ padding** — `key_padding_mask` / vị trí pad tường minh trong mọi lần chạy batch.  
6. **Rò rỉ causal** — unit-test tam giác trên của weights ~0.  
7. **Dropout trên scores so với weights** — biết đường code dùng cái nào; giữ mode train/eval nhất quán.

### Checklist debug

```text
[ ] In shapes: Q,K,V,scores,weights,output
[ ] weights.sum(dim=keys) == 1
[ ] causal: triu(weights,1) == 0
[ ] pad positions: weights[..., pad] == 0
[ ] Không NaN sau chạy dài (kiểm scale + trừ max trong softmax tự viết)
[ ] Gradient chảy tới Wq/Wk/Wv (hook hoặc .grad sau backward)
```

### Mẹo thực tiễn

- Bắt đầu với **một head** và full attention; chỉ thêm heads khi huấn luyện single-head đã ổn.  
- Ghi **entropy của attention** theo lớp: entropy gần zero nghĩa là sụp; rất cao có thể là nhiễu đều.  
- Với mô hình kiểu NMT, trực quan hóa vài cặp câu mỗi epoch—mask hỏng lộ ngay.  
- Ưu tiên SDPA fused trong production; giữ cài đặt tham chiếu chậm cho test.

---

## 8. Tóm tắt các điểm chính
1. **Multi-head attention = các soft router song song + trộn đầu ra**, không phải ensemble trang trí.  
2. Đóng gói heads là **bài toán layout**; coi test reshape là bắt buộc.  
3. **Causal và padding masks** là một phần định nghĩa thuật toán trong hệ thống thật.  
4. Papers cổ điển (Bahdanau → Luong → Vaswani) đánh dấu đường đi từ phần bổ sung RNN tới kiến trúc độc lập.  
5. Hầu hết lỗi “attention hỏng” là **trục, mask, hoặc scale**—không phải phép thuật optimizer.

Giờ đã có đường đầy đủ: cơ sở → toán học → code cốt lõi → thực hành multi-head. Chương 08 sẽ xếp chồng các khối này với lớp feed-forward, residual và positional encoding thành Transformer đầy đủ.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Công thức Attention: tự tay tính Q, K, V](https://www.facebook.com/reel/1806844676942638)
- [Attention trong Transformer (Self, Masked, Cross, Multi-Head)](https://www.facebook.com/reel/1007473105556936)
