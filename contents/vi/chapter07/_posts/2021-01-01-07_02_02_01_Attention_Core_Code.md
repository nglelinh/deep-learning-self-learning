---
layout: post
title: 07-02-02-01 Cài đặt Attention Cốt lõi
chapter: '07'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter07
---

# Cài đặt Attention Cốt lõi

Bài này biến các phương trình attention thành **code chạy được**. Ta cài đặt attention cộng tính (Bahdanau) cho mô hình encoder–decoder và scaled self-attention bằng cả NumPy lẫn PyTorch, với shape tường minh và chi tiết ổn định số.

---

## 1. Tổng quan khái niệm

Một cài đặt attention đúng phải làm bốn việc mỗi lần:

1. Xây **điểm số** giữa queries và keys.  
2. Áp **mask** tùy chọn trên logit.  
3. **Softmax** trên keys.  
4. Nhân trọng số với **values**.

Phần còn lại là thiết kế API: class so với hàm, NumPy so với `torch.nn`, batching, và đóng gói multi-head (bài kế).

Ta bắt đầu với **Bahdanau** vì vòng lặp trên trạng thái encoder làm thuật toán rõ ràng. Sau đó chuyển sang **scaled attention véc-tơ hóa**, thứ dùng trong Transformer.

---

## 2. Nhắc lại toán phục vụ cài đặt
**Điểm số Bahdanau**

$$e_{i} = \mathbf{v}^\top \tanh(\mathbf{W}_h\mathbf{h}_i + \mathbf{W}_s\mathbf{s}).$$

**Scaled attention**

$$\mathbf{A}=\mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}+\mathbf{M}\right),\quad
\mathbf{Y}=\mathbf{A}\mathbf{V}.$$

**Softmax ổn định** cho vectơ $$\mathbf{z}$$:

$$\mathrm{softmax}(\mathbf{z})_i=\frac{\exp(z_i-\max_j z_j)}{\sum_j\exp(z_j-\max_j z_j)}.$$

Luôn trừ max trên trục softmax trong code viết tay.

---

## 3. Ví dụ / Trực giác cho các demo

Ta mô phỏng thiết lập kiểu dịch máy nhỏ:

- Từ nguồn: `I love deep learning` → bốn trạng thái encoder.  
- Với mỗi từ đích đang sinh, một trạng thái decoder truy vấn bốn trạng thái đó.  
- Trọng số attention in ra cho thấy token nguồn nào nhận khối lượng.

Trọng số ngẫu nhiên lúc khởi tạo—điểm của demo là **đường ống**, không phải alignment đã huấn luyện. Sau khi huấn luyện mô hình thật, cùng một bảng in trở nên có ý nghĩa.

---

## 4. Mã minh họa
### 4.1 Tiện ích

```python
import numpy as np


def softmax_last(x):
    """Softmax trên trục cuối với trừ max."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)
```

### 4.2 Attention Bahdanau (rõ ràng, dạng vòng lặp)

```python
class BahdanauAttention:
    """
    Additive attention cho mô hình encoder-decoder.

    decoder_state: (d,) hoặc (d, 1)
    encoder_states: (L, d)   — một hàng mỗi vị trí nguồn
    """

    def __init__(self, hidden_dim, rng=None):
        self.d = hidden_dim
        rng = np.random.default_rng(0) if rng is None else rng
        s = 0.01
        self.Wh = rng.normal(scale=s, size=(hidden_dim, hidden_dim))
        self.Ws = rng.normal(scale=s, size=(hidden_dim, hidden_dim))
        self.v = rng.normal(scale=s, size=(hidden_dim,))

    def __call__(self, decoder_state, encoder_states):
        s = np.asarray(decoder_state).reshape(-1)          # (d,)
        H = np.asarray(encoder_states)                    # (L, d)
        assert H.ndim == 2 and H.shape[1] == self.d

        # scores[i] = v^T tanh(Wh h_i + Ws s)
        proj_h = H @ self.Wh.T                            # (L, d)
        proj_s = self.Ws @ s                              # (d,)
        u = np.tanh(proj_h + proj_s)                      # broadcast (L, d)
        scores = u @ self.v                               # (L,)

        alpha = softmax_last(scores)                      # (L,)
        context = alpha @ H                               # (d,)
        return context, alpha
```

### 4.3 Demo: bảng soft alignment

```python
rng = np.random.default_rng(42)
d, L = 8, 4
encoder_states = rng.normal(size=(L, d))
source = ["I", "love", "deep", "learning"]
target = ["J'", "aime", "l'apprentissage", "profond"]

attn = BahdanauAttention(d, rng=rng)

print(f"{'Target':<18} | attention over source")
print("-" * 60)
for y in target:
    decoder_state = rng.normal(size=(d,))
    context, alpha = attn(decoder_state, encoder_states)
    pairs = " ".join(f"{w}:{a:.2f}" for w, a in zip(source, alpha))
    print(f"{y:<18} | {pairs}")
    assert np.isclose(alpha.sum(), 1.0)
    assert context.shape == (d,)
```

### 4.4 Scaled self-attention (NumPy véc-tơ hóa)

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Hỗ trợ shape:
      (Lq, d), (Lk, d), (Lk, dv)
      hoặc batch (..., Lq, d), (..., Lk, d), (..., Lk, dv)
    mask: boolean mask broadcast được, True = key bị cấm
    """
    dk = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax_last(scores)
    return weights @ V, weights


def causal_mask(L):
    return np.triu(np.ones((L, L), dtype=bool), k=1)


# Self-attention trên chuỗi ngắn
L, d = 5, 16
X = rng.normal(size=(L, d))
Wq = rng.normal(scale=0.1, size=(d, d))
Wk = rng.normal(scale=0.1, size=(d, d))
Wv = rng.normal(scale=0.1, size=(d, d))
Q, K, V = X @ Wq, X @ Wk, X @ Wv

Y, A = scaled_dot_product_attention(Q, K, V, mask=causal_mask(L))
print("Y", Y.shape, "A", A.shape)
print("no future mass:", np.allclose(np.triu(A, 1), 0, atol=1e-6))
```

### 4.5 Module PyTorch (scaled attention một head)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask=None):
        """
        Q,K,V: (B, L, d)  — một head cho rõ
        attn_mask: mask cộng broadcast được tới (B, Lq, Lk)
                   dùng 0 để giữ, -inf để cấm
        """
        d = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights


class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProductAttention()

    def forward(self, x, attn_mask=None):
        # x: (B, L, d)
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        return self.attn(Q, K, V, attn_mask=attn_mask)


B, L, d = 2, 7, 32
x = torch.randn(B, L, d)
# causal mask cộng: (1, L, L)
mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
y, w = SimpleSelfAttention(d)(x, attn_mask=mask)
print(y.shape, w.shape)  # (2,7,32), (2,7,7)
```

### 4.6 Debug shape (dùng khi kẹt)

```python
def show_attn_shapes(Q, K, V):
    print("Q", tuple(Q.shape), "K", tuple(K.shape), "V", tuple(V.shape))
    print("scores expected", Q.shape[:-1] + K.shape[-2:-1])


show_attn_shapes(torch.randn(2, 5, 16), torch.randn(2, 9, 16), torch.randn(2, 9, 16))
# scores: (2, 5, 9)  — mỗi trong 5 queries chú ý trên 9 keys
```

---

## 5. Khái niệm liên quan

- **Teacher forcing** trong seq2seq: decoder nhận token vàng trước đó; attention vẫn chạy mỗi bước.  
- **Input feeding** (Luong): đưa vectơ ngữ cảnh trước vào bước decoder tiếp theo—thường giúp NMT.  
- **`torch.nn.MultiheadAttention` / SDPA**: kernel production; hiểu bài này trước khi coi chúng là phép thuật.  
- **Batch packing**: padding mask bắt buộc khi batch size > 1 với độ dài biến thiên.

---

## 6. Các bài báo nền tảng (liên quan cài đặt)
1. [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473) — thuật toán tham chiếu cho additive attention.  
2. [Luong et al., 2015](https://arxiv.org/abs/1508.04025) — điểm số đơn giản dùng rộng trong RNNToolkit.  
3. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — scaled attention véc-tơ hóa + đóng gói multi-head.  
4. Tài liệu PyTorch: [`torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — batch_first, masks, average weights.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành
| Lỗi | Triệu chứng | Sửa |
|-----|-------------|-----|
| Softmax sai chiều | Trọng số tổng trên queries hoặc batch | Softmax trên **keys** |
| Mask `0` sau softmax | Token tương lai vẫn ảnh hưởng gradient | Cộng `-inf` **trước** softmax |
| Nhầm `Wh @ h` với `h @ Wh` | Lỗi shape im lặng | Cố định quy ước và unit-test |
| Trộn `(d,)` và `(d,1)` | Ác mộng broadcast | Chuẩn hóa bằng `reshape(-1)` |
| Không trừ `max` | `NaN` khi điểm số lớn | Softmax ổn định |

**Mẹo:** với causal self-attention, assert `triu(weights, 1) == 0` mỗi lần đổi code mask.

**Mẹo:** in `alpha` trên chuỗi đồ chơi 3–4 token trước khi scale lên.

---

## 8. Tóm tắt các điểm chính
1. Code attention luôn là **điểm số → mask → softmax → values có trọng số**.  
2. Dạng vòng lặp Bahdanau là cài đặt dạy tốt nhất; matmul có scale là dạng production.  
3. **Ổn định** (trừ max, `sqrt(d_k)`, mask `-inf`) là một phần của tính đúng đắn.  
4. Unit test trên shape nhỏ bắt được 90% lỗi attention.  

Bài kế: **multi-head attention**, demo phong phú hơn, papers, và phác thảo seq2seq.
