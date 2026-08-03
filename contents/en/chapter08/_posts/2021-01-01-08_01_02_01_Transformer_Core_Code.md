---
layout: post
title: 08-01-02-01 Transformer Core Implementation
chapter: '08'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter08
---

# Transformer Core Implementation

This lesson implements the **building blocks** of a Transformer encoder (and the attention core shared with decoders). Code is written for clarity and correct shapes first; production systems then swap in fused kernels (`scaled_dot_product_attention`) and larger configs.

---

## 1. Concept Overview

We will implement, bottom-up:

1. Scaled dot-product attention  
2. Multi-head attention  
3. Sinusoidal positional encoding  
4. Position-wise FFN  
5. Encoder layer + encoder stack  

Each piece is a `nn.Module` with an explicit forward signature so you can unit-test it in isolation.

---

## 2. Mathematical Reminder

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

$$
\mathrm{PE}_{(pos,2i)}=\sin(pos/10000^{2i/d}),\quad
\mathrm{PE}_{(pos,2i+1)}=\cos(pos/10000^{2i/d})
$$

Encoder layer (Pre-LN style used below for stability in deep stacks):

$$
\begin{aligned}
X &\leftarrow X + \mathrm{MHA}(\mathrm{LN}(X)) \\
X &\leftarrow X + \mathrm{FFN}(\mathrm{LN}(X))
\end{aligned}
$$

---

## 3. Example / Intuition for the demos

We run random token ids through an encoder and check:

- output shape equals $$(B, L, d_{\mathrm{model}})$$,  
- causal/pad masks zero out forbidden weights,  
- a short forward pass is differentiable (`loss.backward()` works).

Training a *real* language model is not the goal here—plumbing correctness is.

---

## 4. Code Snippet

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
    Fixed sin-cos PE (Vaswani et al.).
    - pos: 0..L-1
    - i: 0..d_model/2-1  →  even dim 2i = sin, odd dim 2i+1 = cos
    - forward: token_emb + PE  (same shape)
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
        # x: (B, L, d)  — usually token embeddings
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

### Mask convention note

This code uses a **boolean keep-mask** (`True` = attend allowed). Many APIs use the opposite bool meaning or additive float masks. Always read the docstring of the function you call—mask polarity bugs are the #1 Transformer footgun.

### Hook for padding

For variable lengths, build a mask of shape $$(B,1,1,L)$$ (broadcast over heads and queries) with `False` on pad keys, or combine with causal via logical AND for decoders.

---

## 5. Related Concepts

- **`nn.TransformerEncoder`** — library equivalent of this stack.  
- **Decoder layer** — add a second MHA block for cross-attention + causal mask on self-attn.  
- **Weight tying** — share input embedding and output projection in LMs.  
- **SDPA / FlashAttention** — same math, fused kernels.

---

## 6. Fundamental Papers (implementation-relevant)

1. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — reference module list and base hyperparameters.  
2. [Phuong & Hutter, 2022](https://arxiv.org/abs/2207.09238) — formal algorithms; excellent when reimplementing.  
3. PyTorch: [`torch.nn.Transformer`](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — production-oriented API.

---

## 7. Common Pitfalls and Tricks

| Pitfall | Fix |
|---------|-----|
| PE length < sequence length | Raise `max_len` or use relative/RoPE schemes |
| Dropout left on at eval | `model.eval()` + `torch.no_grad()` |
| Scale embedding by $$\sqrt{d}$$ forgotten | Matches original paper; keep consistent |
| Wrong residual branch | Add *input* of sublayer, not normalized-only path without residual |
| In-place ops on tensors needed for grad | Avoid in-place on views used backward |

**Trick:** test each module with `torch.autograd.gradcheck` on tiny float64 inputs when writing custom attention.

---

## 8. Key Takeaways

1. Implement **attention → multi-head → PE → FFN → layer stack** in that order.  
2. Pre-LN residual blocks are a stable default for deep stacks.  
3. Masks are part of the API contract—document polarity.  
4. Smoke tests on shapes, causal structure, and backward are mandatory before training.

Next: demos, papers in narrative form, and operational pitfalls.

<!-- video-references -->

## Video references

Some figures in this lesson are screenshots from the following videos (Machine Learning Thực Chiến). URLs kept for attribution and further viewing:

- [Position Encoding in Transformers (step by step)](https://www.facebook.com/reel/1306224801415127)
