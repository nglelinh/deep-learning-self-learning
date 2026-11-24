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

Andrej Karpathy từng nói rằng cách duy nhất để thực sự hiểu một mô hình ngôn ngữ lớn là **tự viết lại lõi Transformer từ đầu trong một tệp Python duy nhất**. Khi bạn tự tay viết `MultiHeadAttention`, `FeedForward`, `TransformerBlock` và kiểm chứng bằng bài test rò rỉ tương lai (*causal leak test*), mọi bí ẩn đằng sau các mô hình hàng trăm tỷ tham số sẽ tan biến — bởi GPT-4 hay Llama về bản chất chỉ là sự nhân bản của chính khối mã nguồn này ở quy mô lớn hơn!

## Bản đồ kiến thức nền: Bốn chi tiết kỹ thuật cốt tử trong mã nguồn Transformer

1. **Đăng ký buffer không huấn luyện (`register_buffer`)**:
   Ma trận Positional Encoding cố định và mặt nạ tam giác nhân quả phải được lưu bằng `self.register_buffer(...)` thay vì thuộc tính thường — nhờ đó chúng tự động di chuyển cùng mô hình khi gọi `.to('cuda')` mà không bị optimizer cập nhật trọng số.
2. **Cấu trúc Pre-LN trong `TransformerBlock`**:
   Viết `x = x + self.drop(self.attn(self.ln1(x)))` và `x = x + self.drop(self.ffn(self.ln2(x)))` giúp mạng sâu ổn định ngay cả khi không chỉnh lịch warmup quá tỉ mỉ.
3. **Kiểm chứng rò rỉ thông tin tương lai (Causal Leak Test)**:
   Chạy lượt xuôi với `x1`, sau đó sửa ngẫu nhiên token cuối cùng `x2[:, -1] = ...` rồi chạy lại: hiệu tuyệt đối `max(|out1[:, :-1] - out2[:, :-1]|)` bắt buộc phải bằng đúng `0.0` (hoặc `< 1e-6` do làm tròn dấu phẩy động).
4. **Phân biệt mặt nạ tự chú ý (Self-Attention) và chú ý chéo (Cross-Attention)**:
   Trong Decoder, tầng thứ nhất dùng `causal_mask` trên chuỗi đích ($$Q, K, V$$ đều từ đích); tầng thứ hai dùng `src_padding_mask` với $$Q$$ từ đích còn $$K, V$$ từ đầu ra Encoder.

### Khung giải thích (nên đọc chậm): Lần theo luồng kích thước Tensor qua trọn vẹn 1 khối Decoder-only Transformer

Cho batch `B = 4`, độ dài chuỗi `L = 16`, kích thước ẩn `d_model = 128`, số head `H = 4` ($$d_k = 32$$), chiều ẩn FFN `d_ff = 512`, từ vựng `V = 1000`:

| Bước xử lý | Phép toán chính | Kích thước Tensor đầu ra |
| :--- | :--- | :---: |
| 1. Token IDs đầu vào | Mảng số nguyên `[0..999]` | `(4, 16)` |
| 2. Embedding + PosEnc | `emb(ids) * sqrt(128) + pe[:, :16]` | `(4, 16, 128)` |
| 3. Tách Head Q, K, V | `Linear(128,128) -> view -> transpose` | `(4, 4, 16, 32)` |
| 4. Ma trận điểm Attention | `(Q @ K^T) / sqrt(32)` | `(4, 4, 16, 16)` |
| 5. Ghép Head + `W_O` + Residual | `transpose -> reshape -> Linear + x` | `(4, 16, 128)` |
| 6. Khối FFN + Residual | `Linear(128,512) -> GELU -> Linear(512,128) + x` | `(4, 16, 128)` |
| 7. LM Head dự đoán từ tiếp theo | `Linear(128, 1000)` | **`(4, 16, 1000)`** |

Khi tính `CrossEntropyLoss`, ta duỗi 2 chiều đầu của logits thành `(4*16, 1000) = (64, 1000)` và nhãn đích thành `(64,)` — khớp chính xác từng chiều một!

# Cài đặt Cốt lõi Transformer

Bài này cài đặt các **khối nền tảng** của một encoder Transformer (cùng lõi attention dùng chung với decoder) bằng PyTorch. Mã nguồn ưu tiên tính minh bạch và kiểm soát đúng kích thước tensor; trong hệ thống sản xuất, các khối này thường được thay bằng hàm tối ưu hóa tích hợp (`scaled_dot_product_attention`) với quy mô lớn hơn.

---

## 1. Tổng quan khái niệm

Ta sẽ xây dựng lần lượt từ dưới lên theo năm thành phần:

1. Scaled dot-product attention
2. Multi-head attention
3. Mã hóa vị trí dạng sin–cos
4. Mạng FFN theo từng vị trí
5. Tầng encoder và khối encoder hoàn chỉnh

Mỗi thành phần là một `nn.Module` độc lập với giao diện `forward` rõ ràng để dễ dàng kiểm thử riêng lẻ.

---

## 2. Nhắc lại công thức

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

$$
\mathrm{PE}_{(pos,2i)}=\sin(pos/10000^{2i/d}),\quad
\mathrm{PE}_{(pos,2i+1)}=\cos(pos/10000^{2i/d})
$$

Một tầng encoder theo bố cục Pre-LN (được dùng trong mã nguồn bên dưới để tăng ổn định huấn luyện):

$$
\begin{aligned}
X &\leftarrow X + \mathrm{MHA}(\mathrm{LN}(X)) \\
X &\leftarrow X + \mathrm{FFN}(\mathrm{LN}(X))
\end{aligned}
$$

---

## 3. Kiểm chứng luồng dữ liệu

Ta chạy các chuỗi chỉ số token ngẫu nhiên qua encoder để kiểm tra ba tiêu chí cơ bản:

- Đầu ra giữ đúng kích thước $$(B, L, d_{\mathrm{model}})$$.
- Mặt nạ nhân quả / đệm triệt tiêu hoàn toàn trọng số tại các vị trí bị cấm.
- Lượt lan truyền ngược (`loss.backward()`) thực thi suôn sẻ, chứng tỏ đồ thị tính toán khả vi hoàn chỉnh.

Mục tiêu của bước này là đảm bảo tính chính xác của **kiến trúc và luồng tensor** trước khi bước sang huấn luyện thực tế.

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

### Lưu ý về quy ước mặt nạ (mask)

Đoạn mã trên dùng **boolean keep-mask** (`True` nghĩa là vị trí được phép chú ý). Nhiều thư viện lại dùng quy ước ngược lại (`True` là bị che) hoặc dùng mask số thực cộng (`0` / `-inf`). Luôn đọc kỹ tài liệu của hàm đang gọi vì nhầm quy ước mặt nạ là lỗi phổ biến hàng đầu khi cài đặt Transformer.

### Mở rộng cho padding mask

Với các chuỗi có độ dài khác nhau trong một batch, hãy tạo mặt nạ kích thước $$(B, 1, 1, L)$$ (tự broadcast qua các head và vị trí query) gán `False` ở các vị trí `PAD`, sau đó kết hợp bằng phép `AND` logic với mặt nạ nhân quả khi dùng ở decoder.

---

## 5. Khái niệm liên quan

- **`nn.TransformerEncoder`**: phiên bản tích hợp sẵn trong PyTorch tương ứng với khối vừa viết.
- **Tầng decoder**: bổ sung thêm một khối MHA cho cross-attention và mặt nạ nhân quả cho self-attention.
- **Chia sẻ trọng số (Weight tying)**: dùng chung ma trận nhúng đầu vào và ma trận chiếu đầu ra trong mô hình ngôn ngữ.
- **SDPA / FlashAttention**: giữ nguyên kết quả toán học nhưng tối ưu hóa tốc độ và bộ nhớ ở tầng kernel.

---

## 6. Tài liệu tham khảo liên quan đến cài đặt

1. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — danh sách module và bộ siêu tham số chuẩn.
2. [Phuong & Hutter, 2022](https://arxiv.org/abs/2207.09238) — mô tả thuật toán chính thức, rất hữu ích khi cài đặt lại từ đầu.
3. Tài liệu PyTorch: [`torch.nn.Transformer`](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — giao diện chuẩn dùng cho sản xuất.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành

| Lỗi thường gặp | Cách khắc phục |
|---------|-----|
| Chiều dài bảng PE nhỏ hơn độ dài chuỗi | Tăng `max_len` hoặc chuyển sang mã hóa vị trí tương đối / RoPE |
| Quên tắt dropout khi đánh giá | Luôn gọi `model.eval()` kết hợp `torch.no_grad()` |
| Không nhất quán hệ số nhân $$\sqrt{d}$$ cho embedding | Giữ nhất quán theo chuẩn bài báo gốc (nhân $$\sqrt{d_{\mathrm{model}}}$$ trước khi cộng PE) |
| Cộng sai nhánh phần dư | Cộng trực tiếp vào *đầu vào gốc* của tầng con, không bỏ qua đường tắt |
| Phép toán in-place làm hỏng đồ thị gradient | Tránh dùng thao tác in-place lên tensor đang cần cho `backward()` |

**Mẹo:** khi tự viết lớp attention tùy chỉnh, có thể kiểm tra gradient bằng `torch.autograd.gradcheck` trên tensor `float64` kích thước nhỏ.

---

## 8. Tóm tắt các điểm chính

1. Thứ tự xây dựng tự nhiên là **attention → multi-head → PE → FFN → xếp chồng các tầng**.
2. Bố cục phần dư Pre-LN là lựa chọn mặc định ổn định cho các mô hình nhiều tầng.
3. Quy ước mặt nạ là một phần của giao diện API nên cần ghi chú rõ ràng.
4. Kiểm thử nhanh (smoke test) cho kích thước tensor, ràng buộc nhân quả và lan truyền ngược là bước bắt buộc trước khi huấn luyện.

Bài tiếp theo sẽ chạy thử huấn luyện trên bài toán nhỏ, tổng hợp tài liệu nền tảng và danh mục gỡ lỗi vận hành.

<!-- video-references -->

## Nguồn video (tham chiếu)

Một số hình minh họa trong bài được trích từ các video sau (Machine Learning Thực Chiến); các đường dẫn dưới đây được lưu lại để tra cứu và ghi nhận nguồn:

- [Position Encoding trong Transformer (từng bước)](https://www.facebook.com/reel/1306224801415127)

## Video tham khảo

- **Xem trước:** [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Hướng dẫn cài đặt chi tiết từ đầu các module cốt lõi của Transformer (video dài, rất sát với bài này).
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown (tiếng Anh). Sơ đồ luồng dữ liệu trực quan để đối chiếu song song khi viết mã.
- [Positional embeddings in transformers EXPLAINED | Demystifying positional encodings.](https://www.youtube.com/watch?v=1biZfFLPRSY) — AI Coffee Break with Letitia (tiếng Anh). Giải thích chi tiết vai trò của mã hóa vị trí khi mô hình cần nhận diện thứ tự chuỗi.
