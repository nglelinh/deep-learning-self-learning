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

Cách tốt nhất để biến công thức Attention từ một dòng ký hiệu trừu tượng thành công cụ bạn thực sự làm chủ là tự tay viết hàm `bahdanau_attention` và `scaled_dot_product_attention`, chạy trên một ví dụ nhỏ và **in thẳng ma trận trọng số chú ý $$\alpha$$ ra màn hình**. Khi thấy từng con số xác suất tự động sáng lên ở đúng các cặp từ tương ứng, trực giác về soft alignment sẽ khắc sâu ngay lập tức.

## Kiến thức cần nhớ: Bốn kỹ thuật lập trình cốt lõi cho Attention

1. **Ổn định số học cho Softmax (Log-Sum-Exp Trick)**:
   Trừ đi `max` trên trục tính softmax trước khi lấy `exp` giúp tránh tràn số `exp(1000) = inf`, đảm bảo kết quả luôn hữu hạn và chính xác.
2. **Xử lý Mask an toàn với `masked_fill(mask == 0, -1e9)` (hoặc `-float('inf')`)**:
   Lưu ý trường hợp biên: nếu cả một hàng bị mask toàn bộ thành `-inf` (ví dụ một câu toàn `<PAD>`), `softmax` sẽ sinh `NaN` — vì thế cần kiểm tra kỹ mask hoặc dùng `torch.nan_to_num`.
3. **Tách rõ chiều `L_q` (Query) và `L_k` (Key/Value)**:
   Trong Self-Attention thì $$L_q = L_k$$, nhưng trong Cross-Attention (dịch máy, hỏi đáp ảnh) thì $$L_q \ne L_k$$ — viết code luôn phân biệt rõ hai chiều này giúp hàm tái sử dụng được cho cả hai chế độ.
4. **Kiểm chứng Soft Alignment bằng bảng trọng số**:
   Quan sát tổng mỗi hàng bằng `1.000` và kiểm tra xem khi làm sắc nét điểm số (tăng độ lớn query/key), phân phối hội tụ dần về hard alignment (chỉ chọn 1 vị trí).

### Giải thích: Tính tay bước ổn định số học Softmax khi có giá trị lớn và vị trí bị Mask

Giả sử một hàng điểm chú ý trước softmax là $$\mathbf{s} = [100{,}0, \; 102{,}0, \; 500{,}0]$$ nhưng vị trí thứ 3 là token `<PAD>` (bị mask):

1. **Bước 1 — Áp dụng Mask trước Softmax**:
   $$\mathbf{s}_{\text{masked}} = [100{,}0, \; 102{,}0, \; -\infty]$$
2. **Bước 2 — Trừ đi giá trị lớn nhất của các phần tử hợp lệ** ($$m = \max(100{,}0, 102{,}0) = 102{,}0$$):
   $$\mathbf{s}_{\text{shift}} = [-2{,}0, \; 0{,}0, \; -\infty]$$
   (Nếu không trừ $$102{,}0$$, `exp(102)` bằng $$1{,}98 \times 10^{44}$$ sẽ tràn ngay giới hạn `float16` tối đa `65504`!)
3. **Bước 3 — Tính hàm mũ và chuẩn hóa**:
   $$e^{-2} \approx 0{,}1353, \quad e^{0} = 1{,}0000, \quad e^{-\infty} = 0$$
   $$\boldsymbol{\alpha} = \left[\frac{0{,}1353}{1{,}1353}, \; \frac{1{,}0000}{1{,}1353}, \; 0\right] \approx \mathbf{[0{,}1192, \; 0{,}8808, \; 0{,}0000]}$$

Không hề có tràn số FP16, tổng xác suất bằng đúng `1.0000`, và vị trí `<PAD>` nhận trọng số `0.0000` tuyệt đối!

# Cài đặt Attention Cốt lõi

Bài này chuyển các phương trình attention thành mã nguồn chạy được. Ta sẽ cài đặt attention cộng tính (Bahdanau) cho mô hình encoder–decoder và scaled self-attention bằng cả NumPy lẫn PyTorch, chú trọng vào kích thước tensor và độ ổn định số học.

---

## 1. Tổng quan khái niệm

Mọi cài đặt attention chuẩn đều thực hiện bốn bước cơ bản:

1. Tính **điểm số** giữa các queries và keys.
2. Áp **mask** (nếu có) trực tiếp lên điểm số trước softmax.
3. Tính **softmax** dọc theo trục keys.
4. Nhân ma trận trọng số với **values**.

Các phần còn lại thuộc về thiết kế giao diện lập trình: tổ chức hàm hay lớp, dùng NumPy hay `torch.nn`, hỗ trợ theo batch và mở rộng sang multi-head (sẽ trình bày ở bài tiếp theo).

Ta bắt đầu từ **Bahdanau** để thấy rõ cơ chế tính điểm trên từng trạng thái encoder, sau đó chuyển sang **scaled dot-product attention dạng ma trận** dùng trong Transformer.

---

## 2. Nhắc lại công thức phục vụ cài đặt

**Điểm số Bahdanau:**

$$e_{i} = \mathbf{v}^\top \tanh(\mathbf{W}_h\mathbf{h}_i + \mathbf{W}_s\mathbf{s}).$$

**Scaled dot-product attention:**

$$\mathbf{A}=\mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}+\mathbf{M}\right),\quad
\mathbf{Y}=\mathbf{A}\mathbf{V}.$$

**Softmax ổn định số học** cho vectơ $$\mathbf{z}$$:

$$\mathrm{softmax}(\mathbf{z})_i=\frac{\exp(z_i-\max_j z_j)}{\sum_j\exp(z_j-\max_j z_j)}.$$

Khi tự cài đặt softmax, luôn trừ đi giá trị lớn nhất trên trục tính toán để tránh tràn số ở hàm mũ.

---

## 3. Trực giác cho ví dụ minh họa

Ta mô phỏng một bài toán dịch máy thu gọn:

- Câu nguồn `I love deep learning` tạo ra bốn trạng thái encoder.
- Tại mỗi bước sinh từ đích, một trạng thái decoder truy vấn bốn trạng thái này.
- Bảng trọng số attention in ra cho biết từ nguồn nào đang nhận mức chú ý cao nhất.

Do trọng số được khởi tạo ngẫu nhiên và chưa qua huấn luyện, mục tiêu của đoạn minh họa là kiểm chứng **luồng dữ liệu và kích thước tensor**; khi huấn luyện trên dữ liệu thật, bảng căn chỉnh này sẽ phản ánh đúng mối liên hệ ngữ nghĩa giữa hai ngôn ngữ.

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

### 4.6 Kiểm tra kích thước khi gỡ lỗi

```python
def show_attn_shapes(Q, K, V):
    print("Q", tuple(Q.shape), "K", tuple(K.shape), "V", tuple(V.shape))
    print("scores expected", Q.shape[:-1] + K.shape[-2:-1])


show_attn_shapes(torch.randn(2, 5, 16), torch.randn(2, 9, 16), torch.randn(2, 9, 16))
# scores: (2, 5, 9)  — mỗi trong 5 queries chú ý trên 9 keys
```

---

## 5. Khái niệm liên quan

- **Teacher forcing trong seq2seq**: trong huấn luyện, decoder nhận token nhãn đúng của bước trước làm đầu vào, còn cơ chế attention vẫn chạy ở từng bước.
- **Input feeding (Luong)**: đưa vectơ ngữ cảnh của bước trước quay lại làm một phần đầu vào cho bước decoder tiếp theo, thường giúp cải thiện chất lượng dịch máy.
- **`torch.nn.MultiheadAttention` và SDPA**: các cài đặt tối ưu dùng trong sản xuất; nên hiểu rõ cách cài đặt thủ công trong bài trước khi sử dụng chúng như thành phần đóng gói sẵn.
- **Đệm theo batch (batch padding)**: bắt buộc dùng padding mask khi huấn luyện theo batch gồm các chuỗi có độ dài khác nhau.

---

## 6. Tài liệu tham khảo liên quan đến cài đặt

1. [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473) — thuật toán chuẩn cho attention cộng tính.
2. [Luong et al., 2015](https://arxiv.org/abs/1508.04025) — các hàm chấm điểm gọn nhẹ dùng phổ biến trong hệ thống dịch máy RNN.
3. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — scaled dot-product attention dạng ma trận và multi-head.
4. Tài liệu PyTorch: [`torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — hướng dẫn chi tiết về `batch_first`, quy ước mask và `average_attn_weights`.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành

| Lỗi | Triệu chứng | Cách khắc phục |
|-----|-------------|----------------|
| Softmax sai trục | Tổng trọng số bằng 1 theo chiều query hoặc batch thay vì chiều key | Áp softmax trên **trục keys** |
| Gán `0` sau softmax | Token bị cấm vẫn ảnh hưởng đến gradient và phân phối | Cộng `-inf` **trước** softmax |
| Nhầm `Wh @ h` với `h @ Wh` | Lỗi kích thước hoặc nhân nhầm chiều ẩn | Thống nhất quy ước chiều và viết unit-test |
| Trộn `(d,)` với `(d, 1)` | Lỗi broadcast ngoài ý muốn | Chuẩn hóa kích thước bằng `reshape(-1)` |
| Không trừ `max` | Xuất hiện `NaN` khi điểm số lớn | Luôn dùng softmax ổn định số học |

**Mẹo:** với causal self-attention, hãy kiểm tra `triu(weights, 1) == 0` sau mỗi lần sửa mã nguồn mask.

**Mẹo:** in ma trận trọng số `alpha` trên chuỗi ngắn 3–4 token để kiểm chứng trước khi mở rộng sang mô hình lớn.

---

## 8. Tóm tắt các điểm chính

1. Luồng cài đặt attention luôn theo thứ tự: **tính điểm số → áp mask → softmax → nhân với values**.
2. Cài đặt Bahdanau giúp dễ hình dung cơ chế từng bước, còn phép nhân ma trận có chia tỉ lệ là dạng chuẩn trong thực tế.
3. Các bước đảm bảo **ổn định số học** (trừ giá trị cực đại, chia `sqrt(d_k)`, dùng `-inf` cho mask) là phần thiết yếu của thuật toán.
4. Viết kiểm thử đơn vị trên tensor nhỏ giúp phát hiện sớm phần lớn lỗi liên quan đến chiều và mask.

Bài tiếp theo sẽ mở rộng sang **multi-head attention**, minh họa mô hình seq2seq có attention và tổng hợp quy trình gỡ lỗi.

## Video tham khảo

- **Xem trước:** [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Cài đặt chi tiết từng dòng scaled dot-product attention (video dài; phần attention khớp trực tiếp với bài này).
- [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Đối chiếu kích thước của Q, K, V khi viết mã.
- [A Dive Into Multihead Attention, Self-Attention and Cross-Attention](https://www.youtube.com/watch?v=mmzRYGCfTzc) — Machine Learning Studio (tiếng Anh). Gợi ý các bước mở rộng sau khi đã hoàn thiện phiên bản một head.
