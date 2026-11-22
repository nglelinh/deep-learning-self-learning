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

Sau khi đã có các module cốt lõi ở bài trước, phần này đưa chúng vào chạy thử trên một bài toán nhỏ để kiểm chứng quá trình học, hệ thống hóa các bài báo nền tảng và tổng hợp những lỗi vận hành dễ tốn nhiều thời gian gỡ lỗi nhất.

---

## 1. Tổng quan khái niệm

Ba kỹ năng thực hành giúp chuyển từ việc sao chép mã nguồn sang làm chủ mô hình Transformer:

1. **Quan sát nội bộ** (*instrumentation*) — theo dõi trọng số attention và chuẩn hóa kích hoạt qua từng tầng.
2. **Vòng huấn luyện tối thiểu** — kiểm chứng trên một bài toán nhỏ rằng mô hình thực sự hội tụ và học được mẫu dữ liệu.
3. **Kỹ thuật vận hành** — nắm vững lịch warmup tốc độ học, KV-cache, tính toán hỗn hợp độ chính xác (*mixed precision*) và giới hạn $$O(L^2)$$.

Mô hình trong bài được giữ ở kích thước nhỏ để chạy nhanh ngay trên CPU cá nhân.

---

## 2. Ghi chú toán học và hệ thống

**Hàm mất mát *teacher forcing*** cho bài toán dự đoán token kế tiếp trên chuỗi $$y_{1:L}$$:

$$
\mathcal{L} = -\sum_{t=1}^{L-1} \log p_\theta(y_{t+1} \mid y_{\le t})
$$

Trong ví dụ minh họa chỉ-encoder bên dưới, ta dùng đầu **dự đoán dịch chuyển** (*shift prediction*) trực tiếp trên trạng thái encoder — đủ gọn để kiểm chứng đồng thời PE, attention và kết nối phần dư.

**KV-cache khi suy luận:** Khi sinh token thứ $$t$$, mô hình lưu lại và tái sử dụng các ma trận $$K_{1:t-1}, V_{1:t-1}$$ đã tính từ các bước trước, chỉ chiếu thêm token mới. Nhờ đó chi phí tại bước $$t$$ giảm xuống $$O(t\,d)$$ thay vì tính lại toàn bộ $$O(t^2 d)$$ từ đầu ở mỗi bước (tổng chi phí cho chuỗi dài $$T$$ vẫn là $$O(T^2)$$, nhưng loại bỏ phần tính toán lặp lại rất lớn).

---

## 3. Ví dụ và Trực giác

Bài toán thử nghiệm: cho một chuỗi số nguyên tăng dần theo modulo, dự đoán số nguyên kế tiếp (dịch chuyển 1 bước). Khi có PE và self-attention, mô hình học được quy luật chuỗi mà không cần tầng hồi quy. Nếu loại bỏ PE trong thử nghiệm cắt giảm (*ablation*), hiệu năng trên các mẫu phụ thuộc thứ tự sẽ giảm rõ rệt — đây là phép kiểm tra trực tiếp xem PE có đang hoạt động đúng hay không.

---

## 4. Mã minh họa
### 4.1 Chạy xuôi và kiểm tra trọng số attention

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

### 4.2 Huấn luyện thử: Dự đoán bước kế tiếp trên trạng thái encoder

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

Khi chạy đoạn mã trên, giá trị mất mát cần giảm đều qua các bước với gradient hữu hạn — đó là dấu hiệu xác nhận kiến trúc hoạt động đúng.

### 4.3 Thử nghiệm cắt giảm (Ablation): Bỏ mã hóa vị trí

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

### 4.4 Lịch warmup tốc độ học theo Transformer gốc

```python
def transformer_lr(step, d_model=512, warmup=4000):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

for s in [1, 1000, 4000, 10000]:
    print(s, transformer_lr(s))
```

---

## 5. Khái niệm liên quan

- **Seq2seq dùng RNN**: cùng mô hình mã hóa–giải mã nhưng khó song song hóa hơn.
- **Mô hình chuỗi CNN**: vùng tiếp nhận mang tính cục bộ, cần nhiều tầng sâu để bao phủ tầm xa.
- **Sparse Transformers / Longformer / Linformer**: các biến thể attention giảm độ phức tạp dưới mức bậc hai.
- **Encoder–decoder so với chỉ-decoder**: khác nhau ở giao diện ứng dụng nhưng chia sẻ chung lõi khối Transformer.
- **Instruction tuning và RLHF**: các giai đoạn tinh chỉnh nằm phía trên mô hình ngôn ngữ Transformer đã tiền huấn luyện.

---

## 6. Các bài báo nền tảng

1. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   Công trình gốc đề xuất multi-head attention, mã hóa vị trí, cấu trúc encoder–decoder và lịch huấn luyện chuẩn.

2. **[BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)**  
   Tiền huấn luyện encoder hai chiều (MLM và NSP), định hình quy trình *pretrain → fine-tune* cho hiểu ngôn ngữ tự nhiên.

3. **[GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**  
   Mô hình ngôn ngữ nhân quả quy mô lớn, mở ra hướng giải quyết tác vụ không cần tinh chỉnh (*zero-shot*) qua prompt.

4. **[ViT (Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929)**  
   Biểu diễn ảnh thành chuỗi các mảng (patch), mở rộng phạm vi ứng dụng của Transformer sang thị giác máy tính.

5. **[Formal Algorithms for Transformers (Phuong & Hutter, 2022)](https://arxiv.org/abs/2207.09238)**  
   Trình bày mã giả chặt chẽ cho các biến thể Transformer, rất thuận tiện khi đối chiếu cài đặt.

---

## 7. Cạm bẫy thường gặp và mẹo thực hành
### Các lỗi nghiêm trọng

1. **Thiếu tín hiệu vị trí** — mô hình trở thành tập hợp vectơ không phân biệt thứ tự.
2. **Đảo ngược quy ước mặt nạ** — gây rò rỉ tương lai hoặc tạo hàng toàn $$-\infty$$ dẫn tới `NaN` sau softmax.
3. **Không che token đệm (`PAD`)** — làm nhiễu quá trình huấn luyện và giảm chất lượng đánh giá.
4. **Quên hệ số $$1/\sqrt{d_k}$$** — gây bão hòa softmax và làm huấn luyện khó hội tụ khi chiều lớn.
5. **Trộn lẫn công thức Post-LN và Pre-LN** — áp dụng tốc độ học và lịch warmup của bố cục này sang bố cục kia.
6. **Quên chuyển sang `model.eval()` khi suy luận** — dropout vẫn hoạt động làm đầu ra bị nhiễu ngẫu nhiên.
7. **Tính lại toàn bộ quá khứ ở mỗi bước giải mã** — gây độ trễ rất lớn; cần bật KV-cache khi suy luận thực tế.

### Mẹo huấn luyện

- Áp dụng **cắt ngưỡng gradient** (ví dụ `max_norm=1.0`) để ổn định giai đoạn đầu.
- Dùng **warmup** tăng dần tốc độ học rồi giảm dần thay vì dùng tốc độ học lớn ngay từ bước 0.
- Sử dụng **mixed precision** (kèm *loss scaling* khi cần) để tăng tốc và tiết kiệm bộ nhớ GPU.
- Theo dõi **entropy của trọng số attention** theo từng tầng để phát hiện sớm hiện tượng tập trung cực đoan hoặc phân bố đều bất thường.
- Luôn kiểm chứng trên cấu hình **nhỏ** (2 tầng, `d = 64–128`) trước khi mở rộng quy mô.

### Danh mục kiểm tra gỡ lỗi

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

1. Một cài đặt Transformer đúng đòi hỏi kết hợp chặt chẽ giữa **module chuẩn, quy ước mặt nạ rõ ràng và cấu trúc kết nối phần dư**.
2. Kiểm chứng qua một bài toán nhỏ giúp đảm bảo kiến trúc vận hành đúng trước khi làm việc với các mô hình lớn.
3. Các dòng mô hình tiêu biểu như BERT, GPT và ViT đều kế thừa cùng một lõi kiến trúc.
4. Phần lớn sự cố trong thực tế xuất phát từ **mặt nạ, mã hóa vị trí, độ chính xác số học và chi phí $$O(L^2)$$**.

Bạn đã hoàn thành mạch kiến thức cốt lõi của Chương 08 từ giới thiệu, lý thuyết, cài đặt đến thử nghiệm và gỡ lỗi. Để quan sát trực tiếp một mô hình GPT-2 đã tiền huấn luyện mà không cần tự huấn luyện, hãy tiếp tục với bài **08-02 Transformer Explainer**.

## Video tham khảo

- **Xem trước:** [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Minh họa quá trình huấn luyện, lấy mẫu và các lỗi thường gặp khi thiếu mặt nạ nhân quả hoặc cấu hình sai.
- [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY) — StatQuest with Josh Starmer (tiếng Anh). Trình bày rõ vai trò của từng kết nối (như đường phần dư) và hậu quả khi thiếu chúng.
- [How might LLMs store facts | Deep Learning Chapter 7](https://www.youtube.com/watch?v=9-Jl0dxWQs8) — 3Blue1Brown (tiếng Anh). Phân tích cách khối MLP bên trong Transformer lưu trữ tri thức, mở rộng góc nhìn sau khi chạy demo.
