---
layout: post
title: 08-01-02-02 Transformer Demos, Papers, and Pitfalls
chapter: '08'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter08
---

# Transformer Demos, Papers, and Pitfalls

With modules in place, this lesson **runs** them, trains a tiny task so gradients mean something, situates the architecture in the literature, and catalogs the mistakes that waste the most engineering time.

---

## 1. Concept Overview

Three practical skills separate “I copied a Transformer” from “I can ship one”:

1. **Instrumentation** — inspect attention weights and layer norms.  
2. **Minimal training loop** — prove the stack learns *something*.  
3. **Operational knowledge** — warmup, KV-cache, mixed precision, $$O(L^2)$$ limits.

We keep models tiny so demos run on a laptop CPU.

---

## 2. Mathematical / System Notes

**Teacher forcing loss** for next-token prediction on a sequence $$y_{1:L}$$:

$$
\mathcal{L} = -\sum_{t=1}^{L-1} \log p_\theta(y_{t+1} \mid y_{\le t})
$$

In a pure encoder toy task below, we instead train a simple **shift prediction** head on encoder states (not a full LM)—enough to exercise PE + attention + residual stack.

**KV-cache (inference):** when generating token $$t$$, reuse $$K_{1:t-1}, V_{1:t-1}$$ already computed; only project the new token. Amortized cost per new token becomes $$O(t\,d)$$ instead of recomputing a full $$O(t^2 d)$$ pass from scratch each step (still total $$O(T^2)$$ over full length $$T$$, but with much less constant-factor waste).

---

## 3. Example / Intuition

Toy task: given a sequence of integers, predict the next integer (copy-shift). A model with PE + self-attention can learn “look at the last token / local pattern” without recurrence. If PE is removed, performance should collapse on order-sensitive patterns—use that as a sanity ablation.

---

## 4. Code Snippet

### 4.1 Forward demo + attention peek

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

### 4.2 Tiny training: next-step prediction on encoder states

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

You should see loss trend down. Perfect accuracy is not required; **downward loss + finite grads** is the bar.

### 4.3 Ablation: remove positional encoding

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

### 4.4 Sketch: learning-rate warmup (original Transformer style)

```python
def transformer_lr(step, d_model=512, warmup=4000):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

for s in [1, 1000, 4000, 10000]:
    print(s, transformer_lr(s))
```

---

## 5. Related Concepts

- **RNN seq2seq** — same encode/decode story; worse parallelism.  
- **CNN seq models** — local receptive fields; long range needs depth.  
- **Sparse Transformers / Longformer / Linformer** — sub-quadratic attention.  
- **Encoder-decoder vs decoder-only** — product surface, shared guts.  
- **Instruction tuning / RLHF** — sit *on top of* pretrained Transformer LMs (later systems courses).

---

## 6. Fundamental Papers

1. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   Defines the architecture: multi-head attention, PE, encoder/decoder stacks, training recipe. The root paper for this chapter.

2. **[BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)**  
   Bidirectional encoder pretraining (MLM + NSP). Established pretrain → fine-tune for NLU.

3. **[GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**  
   Large causal LMs; zero-shot task framing via prompts; scaling narrative.

4. **[ViT (Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929)**  
   Images as patch token sequences; Transformers beyond text.

5. **[Formal Algorithms for Transformers (Phuong & Hutter, 2022)](https://arxiv.org/abs/2207.09238)**  
   Precise pseudocode for variants—use when reimplementing carefully.

---

## 7. Common Pitfalls and Tricks

### Critical pitfalls

1. **No positional signal** — model becomes bag-of-vectors with pairwise content mixing only.  
2. **Mask polarity inverted** — future leakage or all-zero rows → `NaN` after softmax.  
3. **Padding attended as content** — unstable MT/LM metrics.  
4. **Missing $$1/\sqrt{d_k}$$** — sharp attention, poor training at width.  
5. **Post-LN vs Pre-LN recipe mismatch** — copy LR/warmup from a paper using the other.  
6. **Training dropout left on at inference** — noisy generations.  
7. **Recomputing full past every decode step** — unusable latency; add KV-cache for real decoding.

### Training tricks

- **Grad clip** (e.g. 1.0) helps early Transformer runs.  
- **Warmup** then decay; avoid large LR at step 0.  
- **Mixed precision** with loss scaling for speed/memory.  
- Log **attention entropy** per layer to detect collapse or uniformity.  
- Start **small** (2 layers, d=64–128) before scaling.

### Debugging checklist

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

## 8. Key Takeaways

1. A correct Transformer is **modules + masks + residual discipline**, not only multi-head code.  
2. Tiny trainable demos prove the stack before you download a 7B checkpoint.  
3. Literature branches (BERT/GPT/ViT) share this core.  
4. Most production pain is **masks, PE, precision, and $$O(L^2)$$**, not the softmax formula.  

You now have a complete Chapter 08 path: introduction → theory → implementation → demos/pitfalls. From here you can specialize into large language models, encoder-only NLU, or Vision Transformers using the same algebraic skeleton.
