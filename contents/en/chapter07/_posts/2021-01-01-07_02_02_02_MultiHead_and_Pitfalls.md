---
layout: post
title: 07-02-02-02 Multi-Head Attention, Papers, and Pitfalls
chapter: '07'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter07
---

# Multi-Head Attention, Papers, and Pitfalls

You can implement single-head attention and still miss what makes Transformers work in practice: **several attention patterns in parallel**, correct packaging of heads, causal decoding, and the failure modes that appear only after you scale up. This lesson completes the implementation track.

---

## 1. Concept Overview

Multi-head attention is not “attention but slower.” It is a capacity mechanism: each head uses its own projections $$(W_h^Q, W_h^K, W_h^V)$$, so each head can learn a different similarity notion and a different soft routing over positions. Concatenating heads preserves those channels; the output matrix $$W^O$$ mixes them into the model dimension.

![Transformer architecture + Multi-Head / Scaled Dot-Product zoom](/deep-learning-self-learning/img/chapter_img/chapter08/tf_full_arch_multihead_zoom.jpg)
*Figure: Multi-head = several Linear(Q,K,V) branches → scaled attention → concat → Linear. (Illustration from a Transformer Attention video)*

![$$d_k = d_{\mathrm{model}}/H$$](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_dk_split.jpg)
*Figure: Example $$512/8=64$$ — width per head. (Illustration from a QKV Attention formula video)*

![Multi-head output stack](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_output_stack.jpg)
*Figure: Concat heads before linear $$W^O$$. (Illustration from a QKV Attention formula video)*

Operationally, multi-head attention is:

1. Project $$\mathbf{X}$$ to Q, K, V in full $$d_{\mathrm{model}}$$.  
2. Reshape to $$(B, H, L, d_k)$$ with $$d_k=d_{\mathrm{model}}/H$$.  
3. Run scaled attention **per head** (batched).  
4. Merge heads → $$(B, L, d_{\mathrm{model}})$$.  
5. Final linear map $$W^O$$.

Get the reshape/transpose wrong and shapes still look right while information is scrambled—hence the tests below.

---

## 2. Mathematical Foundation

$$
\mathrm{head}_h = \mathrm{Attention}(X W_h^Q,\ X W_h^K,\ X W_h^V)
$$

$$
\mathrm{MultiHead}(X)=\mathrm{Concat}_h(\mathrm{head}_h)\,W^O
$$

With shared big matrices (common in code), one projects to $$d_{\mathrm{model}}$$ once, then **views** the last dimension as $$H \times d_k$$:

$$
Q = X W^Q \in \mathbb{R}^{B\times L\times (H d_k)}
\ \xrightarrow{\text{reshape/transpose}}\ 
\mathbb{R}^{B\times H\times L\times d_k}.
$$

Causal decoding uses

$$
M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}
$$

added to scores inside every head (same mask broadcast across heads).

---

## 3. Example / Intuition

On a sentence like “The cat sat,” different heads often specialize after training:

- Head A: local bigrams (`cat`↔`The`, `sat`↔`cat`)  
- Head B: weakly syntactic (`sat` attends subject `cat`)  
- Head C: almost positional (strong diagonal)

You do not program these roles; separate parameters + multi-task pressure from the loss encourage diversity. Visualization of per-head maps is the practical way to see specialization.

---

## 4. Code Snippet

### 4.1 Multi-head self-attention (PyTorch, teaching version)

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
        attn_mask: additive mask broadcastable to (B, 1, L, L) or (B, H, L, L)
                   0 = keep, -inf = forbid
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
    # (1, 1, L, L) for broadcasting over batch and heads
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

### 4.2 Using the built-in module

```python
official = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
# key_padding_mask: (B, L) True = ignore (padding)
# attn_mask: (L, L) True/float mask depending on version — read docs carefully
y2, w2 = official(x, x, x, need_weights=True, average_attn_weights=False)
print(y2.shape, w2.shape)
```

Always double-check the mask convention for your PyTorch version (`bool` vs additive float).

### 4.3 Minimal seq2seq-with-attention sketch

```python
class Seq2SeqWithAttention(nn.Module):
    """
    Tiny educational encoder-decoder:
    - encoder GRU over source embeddings
    - decoder GRU step
    - Luong-style dot attention over encoder outputs
    Not production NMT; shapes are the lesson.
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
        # src: (B, Ls), tgt: (B, Lt)  — teacher forcing inputs
        enc_out, _ = self.encoder(self.src_emb(src))
        dec_in = self.tgt_emb(tgt)
        dec_out, _ = self.decoder(dec_in)
        # attend each decoder position to encoder (batched)
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

### 4.4 Visualizing one head (optional)

```python
import matplotlib.pyplot as plt

def plot_attention(weights_2d, x_labels=None, y_labels=None, title="Attention"):
    """weights_2d: (Lq, Lk) numpy array"""
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

# Example with the causal multi-head weights from above
# plot_attention(w[0, 0].detach().numpy())  # batch 0, head 0
```

---

## 5. Related Concepts

- **Residuals + LayerNorm** around multi-head blocks (Pre-LN vs Post-LN) dominate trainability.  
- **Relative position bias** (T5, Shaw et al.) adds distance terms into scores.  
- **Cross-attention** in encoder–decoder Transformers reuses the same multi-head code with Q from decoder and K,V from encoder.  
- **FlashAttention / SDPA** fuse the matmul-softmax-matmul for memory efficiency; math is unchanged.  
- **Sparse attention** (Longformer, BigBird) restricts the support of $$\alpha$$ for long $$L$$.

---

## 6. Fundamental Papers

1. **[Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473)** — *Neural Machine Translation by Jointly Learning to Align and Translate.*  
   Foundational additive attention; decoder soft-searches source states. Established learned alignment without explicit supervision.

2. **[Luong et al., 2015](https://arxiv.org/abs/1508.04025)** — *Effective Approaches to Attention-based NMT.*  
   Multiplicative scores, global vs local attention, practical decoder tricks (input feeding).

3. **[Xu et al., 2015](https://arxiv.org/abs/1502.03044)** — *Show, Attend and Tell.*  
   Visual attention for captioning; soft vs hard attention; multimodal generality of the idea.

4. **[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)** — *Attention Is All You Need.*  
   Multi-head scaled self-attention as the primary sequence operator; template for modern stacks.

5. **[Shaw et al., 2018](https://arxiv.org/abs/1803.02155)** — *Self-Attention with Relative Position Representations.*  
   Relative distances inside attention logits; improves length generalization in many variants.

---

## 7. Common Pitfalls and Tricks

### Critical bugs

1. **Softmax over the wrong axis** — must be the key/memory axis.  
2. **Mask after softmax** — use additive `-inf` on logits.  
3. **Forgetting $$1/\sqrt{d_k}$$** — sharp, low-gradient attention at large width.  
4. **Incorrect head reshape** — always test `view → transpose → attention → transpose → view` on a known tensor.  
5. **Padding leakage** — `key_padding_mask` / explicit pad positions in every batched run.  
6. **Causal leakage** — unit-test upper triangle of weights is ~0.  
7. **Dropout on scores vs weights** — know which your code path uses; keep train/eval modes consistent.

### Debugging checklist

```text
[ ] Print shapes: Q,K,V,scores,weights,output
[ ] weights.sum(dim=keys) == 1
[ ] causal: triu(weights,1) == 0
[ ] pad positions: weights[..., pad] == 0
[ ] No NaNs after long runs (check scale + max-subtraction in custom softmax)
[ ] Gradient flows to Wq/Wk/Wv (hook or .grad after backward)
```

### Practical tricks

- Start with **one head** and full attention; add heads only when single-head training is stable.  
- Log **entropy of attention** per layer: near-zero entropy means collapse; very high may mean uniform noise.  
- For NMT-style models, visualize a few sentence pairs every epoch—broken masks show up immediately.  
- Prefer fused SDPA in production; keep a slow reference implementation for tests.

---

## 8. Key Takeaways

1. **Multi-head attention = parallel soft routers + output mix**, not a cosmetic ensemble.  
2. Packaging heads is a **layout problem**; treat reshape tests as mandatory.  
3. **Causal and padding masks** are part of the algorithm’s definition in real systems.  
4. Classic papers (Bahdanau → Luong → Vaswani) mark the path from RNN add-on to standalone architecture.  
5. Most “attention is broken” bugs are **axes, masks, or scale**—not optimizer magic.

You now have a complete path: fundamentals → mathematics → core code → multi-head practice. Chapter 08 will stack these blocks with feed-forward layers, residuals, and positional encodings into the full Transformer.

<!-- video-references -->

## Video references

Some figures in this lesson are screenshots from the following videos (Machine Learning Thực Chiến). URLs kept for attribution and further viewing:

- [Attention formula: hand-compute Q, K, V](https://www.facebook.com/reel/1806844676942638)
- [Attention in Transformers (Self, Masked, Cross, Multi-Head)](https://www.facebook.com/reel/1007473105556936)
