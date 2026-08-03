---
layout: post
title: 08-01-02-01 Cài đặt Cốt lõi Transformer
chapter: '08'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter08
---

# Cài đặt Cốt lõi Transformer

Bài này cài đặt các **khối xây dựng** của một encoder Transformer (và lõi attention dùng chung với decoder). Mã được viết ưu tiên rõ ràng và shape đúng trước; hệ thống sản xuất sau đó thay bằng kernel gộp (`scaled_dot_product_attention`) và cấu hình lớn hơn.

---

## 1. Tổng quan khái niệm
Chúng ta sẽ cài đặt, từ dưới lên:

1. Scaled dot-product attention  
2. Multi-head attention  
3. Mã hóa vị trí dạng sin-cos  
4. FFN theo vị trí  
5. Tầng encoder + chồng encoder  

Mỗi mảnh là một `nn.Module` với chữ ký forward tường minh để bạn có thể unit-test độc lập.

---

## 2. Nhắc lại toán học
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

$$
\mathrm{PE}_{(pos,2i)}=\sin(pos/10000^{2i/d}),\quad
\mathrm{PE}_{(pos,2i+1)}=\cos(pos/10000^{2i/d})
$$

Tầng encoder (kiểu Pre-LN dùng bên dưới để ổn định ở chồng sâu):

$$
\begin{aligned}
X &\leftarrow X + \mathrm{MHA}(\mathrm{LN}(X)) \\
X &\leftarrow X + \mathrm{FFN}(\mathrm{LN}(X))
\end{aligned}
$$

---

## 3. Ví dụ / Trực giác cho các demo

Chúng ta chạy id token ngẫu nhiên qua encoder và kiểm tra:

- shape đầu ra bằng $$(B, L, d_{\mathrm{model}})$$,  
- mask nhân quả/đệm triệt tiêu trọng số bị cấm,  
- một forward ngắn khả vi (`loss.backward()` hoạt động).

Huấn luyện một mô hình ngôn ngữ *thật* không phải mục tiêu ở đây—đúng “đường ống” mới là mục tiêu.

---

## 4. Mã minh họa
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    q,k,v: (B, H, L, d_k)  [v last dim may be d_v == d_k here]
    mask:  broadcastable to (B, H, Lq, Lk); True = keep, False = mask out
           (we use bool keep-mask; convert to -inf fill)
    """

    def forward(self, q, k, v, mask=None):
        dk = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        # (B, L, d) -> (B, H, L, dk)
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.dk).transpose(1, 2)

    def _merge(self, x):
        # (B, H, L, dk) -> (B, L, d)
        B, H, L, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * dk)

    def forward(self, q, k, v, mask=None):
        # q,k,v: (B, L, d_model)
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k, v = self._split(q), self._split(k), self._split(v)
        out, weights = self.attn(q, k, v, mask=mask)
        out = self.dropout(self.Wo(self._merge(out)))
        return out, weights


class PositionalEncoding(nn.Module):
    """
    PE sin-cos cố định (Vaswani et al.).
    - pos: 0..L-1
    - i: 0..d_model/2-1  →  chiều chẵn 2i = sin, chiều lẻ 2i+1 = cos
    - forward: token_emb + PE  (cùng shape)
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)  # 2i
        pe[:, 1::2] = torch.cos(pos * div)  # 2i+1
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        # x: (B, L, d)  — thường đã là token embedding
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Pre-LN encoder layer."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFFN(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        a, _ = self.self_attn(h, h, h, mask=mask)
        x = x + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, token_ids, mask=None):
        # token_ids: (B, L)
        x = self.embed(token_ids) * math.sqrt(self.d_model)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.ln_f(x)


def subsequent_mask(L, device=None):
    """Bool keep-mask for causal attention: (1, 1, L, L)."""
    m = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    return m.view(1, 1, L, L)


# -------------------- smoke tests --------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, V, d, H = 2, 12, 1000, 64, 4
    model = TransformerEncoder(V, d_model=d, n_heads=H, n_layers=2, d_ff=128)
    ids = torch.randint(0, V, (B, L))
    out = model(ids)
    print("encoder out", out.shape)
    assert out.shape == (B, L, d)

    # causal mask path through MHA directly
    mha = MultiHeadAttention(d, H)
    x = torch.randn(B, L, d)
    y, w = mha(x, x, x, mask=subsequent_mask(L))
    assert y.shape == (B, L, d)
    # forbidden future: upper triangle of weights ~ 0
    future = w.triu(diagonal=1)
    assert torch.allclose(future, torch.zeros_like(future), atol=1e-5)
    print("causal multi-head OK")

    loss = out.mean()
    loss.backward()
    print("backward OK")
```

### Ghi chú quy ước mask

Mã này dùng **boolean keep-mask** (`True` = được phép attend). Nhiều API dùng nghĩa bool ngược hoặc mask float cộng. Luôn đọc docstring của hàm bạn gọi—lỗi cực tính mask là footgun số 1 của Transformer.

### Móc cho padding

Với độ dài biến đổi, xây mask shape $$(B,1,1,L)$$ (broadcast qua head và query) với `False` trên key pad, hoặc kết hợp với causal bằng AND logic cho decoder.

---

## 5. Khái niệm liên quan
- **`nn.TransformerEncoder`** — tương đương thư viện của chồng này.  
- **Tầng decoder** — thêm khối MHA thứ hai cho cross-attention + mask nhân quả trên self-attn.  
- **Weight tying** — chia sẻ nhúng đầu vào và chiếu đầu ra trong LM.  
- **SDPA / FlashAttention** — cùng toán, kernel gộp.

---

## 6. Các bài báo nền tảng (liên quan cài đặt)
1. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — danh sách module tham chiếu và siêu tham số base.  
2. [Phuong & Hutter, 2022](https://arxiv.org/abs/2207.09238) — thuật toán hình thức; xuất sắc khi cài lại.  
3. PyTorch: [`torch.nn.Transformer`](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — API hướng sản xuất.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành
| Bẫy | Cách sửa |
|---------|-----|
| Độ dài PE < độ dài chuỗi | Tăng `max_len` hoặc dùng lược đồ tương đối/RoPE |
| Dropout vẫn bật lúc eval | `model.eval()` + `torch.no_grad()` |
| Quên scale embedding với $$\sqrt{d}$$ | Khớp bài báo gốc; giữ nhất quán |
| Nhánh residual sai | Cộng *đầu vào* của tầng con, không chỉ đường đã chuẩn hóa mà không residual |
| Toán tử in-place trên tensor cần cho grad | Tránh in-place trên view dùng cho backward |

**Mẹo:** kiểm tra từng module bằng `torch.autograd.gradcheck` trên đầu vào float64 nhỏ khi viết attention tùy chỉnh.

---

## 8. Tóm tắt các điểm chính
1. Cài đặt theo thứ tự **attention → multi-head → PE → FFN → chồng tầng**.  
2. Khối residual Pre-LN là mặc định ổn định cho chồng sâu.  
3. Mask là một phần hợp đồng API—ghi rõ cực tính.  
4. Smoke test về shape, cấu trúc nhân quả, và backward là bắt buộc trước khi huấn luyện.

Tiếp theo: demo, bài báo dạng tường thuật, và bẫy vận hành.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến). Giữ URL để tra cứu / ghi công nguồn:

- [Position Encoding trong Transformer (từng bước)](https://www.facebook.com/reel/1306224801415127)
