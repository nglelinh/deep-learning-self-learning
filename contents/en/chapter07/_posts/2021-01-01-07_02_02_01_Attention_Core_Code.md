---
layout: post
title: 07-02-02-01 Attention Core Implementation
chapter: '07'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter07
---

# Attention Core Implementation

This lesson turns the attention equations into **working code**. We implement additive (Bahdanau) attention for encoder–decoder models and scaled self-attention in both NumPy and PyTorch, with explicit shapes and numerical-stability details.

---

## 1. Concept Overview

A correct attention implementation must do four things every time:

1. Build **scores** between queries and keys.  
2. Apply optional **mask** on logits.  
3. **Softmax** over keys.  
4. Multiply weights by **values**.

The rest is API design: classes vs functions, NumPy vs `torch.nn`, batching, and multi-head packing (next lesson).

We start with **Bahdanau** because the loop over encoder states makes the algorithm obvious. Then we move to **vectorized scaled attention**, which is what you will use in Transformers.

---

## 2. Mathematical Reminder (implementation-facing)

**Bahdanau score**

$$e_{i} = \mathbf{v}^\top \tanh(\mathbf{W}_h\mathbf{h}_i + \mathbf{W}_s\mathbf{s}).$$

**Scaled attention**

$$\mathbf{A}=\mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}+\mathbf{M}\right),\quad
\mathbf{Y}=\mathbf{A}\mathbf{V}.$$

**Stable softmax** for a vector $$\mathbf{z}$$:

$$\mathrm{softmax}(\mathbf{z})_i=\frac{\exp(z_i-\max_j z_j)}{\sum_j\exp(z_j-\max_j z_j)}.$$

Always subtract the max on the softmax axis in handwritten code.

---

## 3. Example / Intuition for the demos

We will simulate a tiny translation-like setting:

- Source words: `I love deep learning` → four encoder states.  
- For each target word being generated, a decoder state queries those four states.  
- Printed attention weights show which source tokens receive mass.

Weights are random at initialization—the point of the demo is the **plumbing**, not trained alignments. After you train a real model, the same printout becomes meaningful.

---

## 4. Code Snippet

### 4.1 Utilities

```python
import numpy as np


def softmax_last(x):
    """Softmax on the last axis with max-subtraction."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)
```

### 4.2 Bahdanau attention (clear, loop-based)

```python
class BahdanauAttention:
    """
    Additive attention for encoder-decoder models.

    decoder_state: (d,) or (d, 1)
    encoder_states: (L, d)   — one row per source position
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

### 4.3 Demo: soft alignment table

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

### 4.4 Scaled self-attention (vectorized NumPy)

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Supports shapes:
      (Lq, d), (Lk, d), (Lk, dv)
      or batched (..., Lq, d), (..., Lk, d), (..., Lk, dv)
    mask: broadcastable boolean mask, True = forbidden key
    """
    dk = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax_last(scores)
    return weights @ V, weights


def causal_mask(L):
    return np.triu(np.ones((L, L), dtype=bool), k=1)


# Self-attention on a short sequence
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

### 4.5 PyTorch module (single-head scaled attention)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask=None):
        """
        Q,K,V: (B, L, d)  — single head for clarity
        attn_mask: additive mask broadcastable to (B, Lq, Lk)
                   use 0 for keep, -inf for forbid
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
# additive causal mask: (1, L, L)
mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
y, w = SimpleSelfAttention(d)(x, attn_mask=mask)
print(y.shape, w.shape)  # (2,7,32), (2,7,7)
```

### 4.6 Shape debugger (use this when stuck)

```python
def show_attn_shapes(Q, K, V):
    print("Q", tuple(Q.shape), "K", tuple(K.shape), "V", tuple(V.shape))
    print("scores expected", Q.shape[:-1] + K.shape[-2:-1])


show_attn_shapes(torch.randn(2, 5, 16), torch.randn(2, 9, 16), torch.randn(2, 9, 16))
# scores: (2, 5, 9)  — each of 5 queries attends over 9 keys
```

---

## 5. Related Concepts

- **Teacher forcing** in seq2seq: decoder receives gold previous tokens; attention still runs each step.  
- **Input feeding** (Luong): feed previous context vector into the next decoder step—often helps NMT.  
- **`torch.nn.MultiheadAttention` / SDPA**: production kernels; understand this lesson before treating them as magic.  
- **Batch packing**: padding masks are mandatory once batch size > 1 with variable lengths.

---

## 6. Fundamental Papers (implementation lineage)

1. [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473) — reference algorithm for additive attention.  
2. [Luong et al., 2015](https://arxiv.org/abs/1508.04025) — simpler scores widely used in RNNToolkits.  
3. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — vectorized scaled attention + multi-head packaging.  
4. PyTorch docs: [`torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — batch_first, masks, average weights.

---

## 7. Common Pitfalls and Tricks

| Bug | Symptom | Fix |
|-----|---------|-----|
| Softmax on wrong dimension | Weights sum over queries or batch | Softmax over **keys** |
| Mask value `0` after softmax | Future tokens still affect grads | Add `-inf` **before** softmax |
| `Wh @ h` vs `h @ Wh` confusion | Silent shape errors | Fix a convention and unit-test |
| Mixing `(d,)` and `(d,1)` | Broadcasting nightmares | Canonicalize with `reshape(-1)` |
| No `max` subtraction | `NaN` on large scores | Stable softmax |

**Trick:** for causal self-attention, assert `triu(weights, 1) == 0` every time you change mask code.

**Trick:** print `alpha` on a 3–4 token toy sequence before scaling up.

---

## 8. Key Takeaways

1. Attention code is always **score → mask → softmax → weighted values**.  
2. Bahdanau’s loop form is the best teaching implementation; scaled matmul is the production form.  
3. **Stability** (max-subtraction, `sqrt(d_k)`, `-inf` masks) is part of correctness.  
4. Unit tests on tiny shapes catch 90% of attention bugs.  

Next lesson: **multi-head attention**, richer demos, papers, and a seq2seq sketch.
