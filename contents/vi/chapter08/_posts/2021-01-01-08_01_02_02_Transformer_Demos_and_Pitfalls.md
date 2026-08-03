---
layout: post
title: 08-01-02-02 Demo, Bài báo và Bẫy Transformer
chapter: '08'
order: 6
owner: Deep Learning Course
lang: vi
categories:
- chapter08
---

# Demo, Bài báo và Bẫy Transformer

Với các module đã sẵn, bài này **chạy** chúng, huấn luyện một tác vụ nhỏ để gradient có ý nghĩa, đặt kiến trúc vào bối cảnh tài liệu, và liệt kê các sai lầm tốn nhiều thời gian kỹ thuật nhất.

---

## 1. Tổng quan khái niệm
Ba kỹ năng thực tiễn phân biệt “tôi copy một Transformer” với “tôi có thể đưa một Transformer vào vận hành”:

1. **Công cụ quan sát** (*instrumentation*) — kiểm tra trọng số attention và chuẩn hóa tầng.  
2. **Vòng huấn luyện tối thiểu** — chứng minh chồng học được *điều gì đó*.  
3. **Kiến thức vận hành** — warmup, KV-cache, mixed precision, giới hạn $$O(L^2)$$.

Chúng ta giữ mô hình nhỏ để demo chạy được trên CPU laptop.

---

## 2. Ghi chú toán và hệ thống
**Mất mát teacher forcing** cho dự đoán token kế tiếp trên chuỗi $$y_{1:L}$$:

$$
\mathcal{L} = -\sum_{t=1}^{L-1} \log p_\theta(y_{t+1} \mid y_{\le t})
$$

Trong tác vụ toy chỉ-encoder bên dưới, ta thay bằng đầu **dự đoán dịch chuyển** (*shift prediction*) đơn giản trên trạng thái encoder (không phải LM đầy đủ)—đủ để vận hành PE + attention + chồng residual.

**KV-cache (suy luận):** khi sinh token $$t$$, tái sử dụng $$K_{1:t-1}, V_{1:t-1}$$ đã tính; chỉ chiếu token mới. Chi phí khấu hao mỗi token mới trở thành $$O(t\,d)$$ thay vì tính lại toàn bộ pass $$O(t^2 d)$$ từ đầu mỗi bước (tổng vẫn $$O(T^2)$$ trên độ dài đầy đủ $$T$$, nhưng ít lãng phí hằng số hơn nhiều).

---

## 3. Ví dụ / Trực giác

Tác vụ toy: cho một chuỗi số nguyên, dự đoán số nguyên kế tiếp (copy-shift). Mô hình có PE + self-attention có thể học “nhìn token cuối / mẫu cục bộ” mà không cần hồi quy. Nếu bỏ PE, hiệu năng sẽ sụp trên các mẫu nhạy thứ tự—dùng đó làm ablation kiểm tra.

---

## 4. Mã minh họa
### 4.1 Demo forward + xem attention

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse modules from the core lesson (paste classes from 08-01-02-01 into the same script,
# or import them from your package).

# --- assume MultiHeadAttention, PositionalEncoding, TransformerEncoder,
# --- subsequent_mask are defined as in 08-01-02-01 ---

torch.manual_seed(0)
V, B, L, d, H = 50, 2, 16, 64, 4
enc = TransformerEncoder(V, d_model=d, n_heads=H, n_layers=2, d_ff=128)
ids = torch.randint(0, V, (B, L))
h = enc(ids)
print("hidden", h.shape)

# Inspect first layer self-attention with causal mask via standalone MHA
mha = MultiHeadAttention(d, H)
x = h.detach()
_, w = mha(x, x, x, mask=subsequent_mask(L))
print("weights", w.shape)  # (B, H, L, L)
print("future mass", w.triu(diagonal=1).abs().max().item())
```

### 4.2 Huấn luyện nhỏ: dự đoán bước kế trên trạng thái encoder

```python
class TinyNextStepModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=4 * d_model, dropout=0.1
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, L) token ids
        h = self.encoder(x)           # (B, L, d)
        return self.head(h)           # (B, L, V)


def make_batch(batch_size, seq_len, vocab_size):
    # sequences 0..seq_len-1 mod vocab, plus noise offset per row
    base = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    offset = torch.randint(0, vocab_size, (batch_size, 1))
    x = (base + offset) % vocab_size
    # predict next id (shift); last position predicts (last+1)
    y = (x + 1) % vocab_size
    return x, y


vocab = 32
model = TinyNextStepModel(vocab)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

model.train()
for step in range(201):
    x, y = make_batch(32, 12, vocab)
    logits = model(x)
    # predict y_t from representation at position t
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

model.eval()
with torch.no_grad():
    x, y = make_batch(4, 12, vocab)
    pred = model(x).argmax(-1)
    print("input    ", x[0].tolist())
    print("target   ", y[0].tolist())
    print("pred     ", pred[0].tolist())
```

Bạn nên thấy loss có xu hướng giảm. Độ chính xác hoàn hảo không bắt buộc; **loss giảm + gradient hữu hạn** là ngưỡng đạt.

### 4.3 Ablation: bỏ mã hóa vị trí

```python
class NoPEEncoder(TransformerEncoder):
    def forward(self, token_ids, mask=None):
        x = self.embed(token_ids) * math.sqrt(self.d_model)
        # intentionally skip self.pos
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.ln_f(x)

# Train briefly with NoPEEncoder inside a similar wrapper and compare final loss.
# Order-sensitive tasks should degrade without PE.
```

### 4.4 Phác thảo: warmup tốc độ học (kiểu Transformer gốc)

```python
def transformer_lr(step, d_model=512, warmup=4000):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

for s in [1, 1000, 4000, 10000]:
    print(s, transformer_lr(s))
```

---

## 5. Khái niệm liên quan
- **Seq2seq RNN** — cùng câu chuyện encode/decode; song song hóa kém hơn.  
- **Mô hình chuỗi CNN** — trường tiếp nhận cục bộ; tầm xa cần độ sâu.  
- **Sparse Transformers / Longformer / Linformer** — attention dưới bậc hai.  
- **Encoder-decoder so với chỉ-decoder** — bề mặt sản phẩm, ruột dùng chung.  
- **Instruction tuning / RLHF** — nằm *trên* các LM Transformer đã tiền huấn luyện (các khóa hệ thống sau).

---

## 6. Các bài báo nền tảng
1. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   Định nghĩa kiến trúc: multi-head attention, PE, chồng encoder/decoder, công thức huấn luyện. Bài gốc của chương này.

2. **[BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)**  
   Tiền huấn luyện encoder hai chiều (MLM + NSP). Thiết lập pretrain → fine-tune cho NLU.

3. **[GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**  
   LM nhân quả lớn; đóng khung tác vụ zero-shot qua prompt; tường thuật scale.

4. **[ViT (Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929)**  
   Ảnh như chuỗi token patch; Transformer vượt ra ngoài văn bản.

5. **[Formal Algorithms for Transformers (Phuong & Hutter, 2022)](https://arxiv.org/abs/2207.09238)**  
   Pseudocode chính xác cho các biến thể—dùng khi cài lại cẩn thận.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành
### Bẫy nghiêm trọng

1. **Không có tín hiệu vị trí** — mô hình trở thành bag-of-vectors chỉ trộn nội dung theo cặp.  
2. **Cực tính mask đảo ngược** — rò rỉ tương lai hoặc hàng toàn zero → `NaN` sau softmax.  
3. **Padding được attend như nội dung** — chỉ số MT/LM không ổn định.  
4. **Thiếu $$1/\sqrt{d_k}$$** — attention sắc, huấn luyện kém ở bề rộng lớn.  
5. **Công thức Post-LN vs Pre-LN không khớp** — copy LR/warmup từ bài báo dùng kiểu kia.  
6. **Dropout huấn luyện vẫn bật lúc suy luận** — sinh nhiễu.  
7. **Tính lại toàn bộ quá khứ mỗi bước decode** — độ trễ không dùng được; thêm KV-cache cho decoding thật.

### Mẹo huấn luyện

- **Grad clip** (ví dụ 1.0) giúp các lần chạy Transformer sớm.  
- **Warmup** rồi giảm; tránh LR lớn ở bước 0.  
- **Mixed precision** với loss scaling cho tốc độ/bộ nhớ.  
- Ghi log **entropy attention** mỗi tầng để phát hiện sụp hoặc đồng đều.  
- Bắt đầu **nhỏ** (2 tầng, d=64–128) trước khi scale.

### Checklist gỡ lỗi

```text
[ ] out.shape == (B, L, d) or (B, L, V)
[ ] causal: triu(attn,1) ≈ 0
[ ] pad keys: weights ≈ 0
[ ] loss finite; grad norms finite
[ ] PE length covers max L
[ ] model.train() vs model.eval() correct
[ ] Ablation: shuffle positions → performance drop (if PE works)
```

---

## 8. Tóm tắt các điểm chính
1. Transformer đúng là **module + mask + kỷ luật residual**, không chỉ mã multi-head.  
2. Demo huấn luyện nhỏ chứng minh chồng trước khi bạn tải checkpoint 7B.  
3. Các nhánh tài liệu (BERT/GPT/ViT) chia sẻ lõi này.  
4. Hầu hết nỗi đau sản xuất nằm ở **mask, PE, precision, và $$O(L^2)$$**, không phải công thức softmax.  

Bạn hiện có lộ trình Chương 08 đầy đủ: giới thiệu → lý thuyết → cài đặt → demo/bẫy. Từ đây có thể chuyên sâu vào mô hình ngôn ngữ lớn, NLU chỉ-encoder, hoặc Vision Transformer trên cùng bộ khung đại số này.
