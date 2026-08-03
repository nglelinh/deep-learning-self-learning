---
layout: post
title: 08-01-01 Transformer Theory
chapter: '08'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter08
---

# Transformer Theory

![Transformer Architecture](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_architecture.png)
*Encoder–decoder Transformer (Vaswani et al., 2017). Source: Wikimedia Commons*

## 1. Concept Overview

The Transformer is an architecture for sequence (and “sequence-like”) data built primarily from **multi-head attention** and **position-wise feed-forward networks**, wrapped in residual connections and layer normalization. Introduced by Vaswani et al. (2017) for machine translation, it discarded recurrence and convolution as the main sequence operators.

### What was wrong with “just use a better RNN”?

Even excellent RNNs (LSTM/GRU) have structural constraints:

| Constraint | Effect |
|------------|--------|
| Left-to-right state updates | Hard to parallelize over length during training |
| Long dependency path | Early tokens must survive many steps to affect late predictions |
| Fixed-size hidden state | Soft bottleneck for very rich context |

Attention already showed that models can *look up* relevant states. The Transformer’s claim was sharper: **you do not need recurrence at all** if every layer lets every position attend to every other position (with masks when needed) and then applies a strong local MLP.

### Three product families from one architecture

| Family | Structure | Examples |
|--------|-----------|----------|
| Encoder-only | Bidirectional self-attention | BERT, RoBERTa |
| Decoder-only | Causal self-attention | GPT-2/3/4, LLaMA |
| Encoder–decoder | Encoder + cross-attn decoder | Original Transformer, T5, BART |

Same layer primitives; different attention patterns and training objectives.

### Building blocks (preview)

1. Token + positional embeddings  
2. Stacked encoder layers: self-attn → FFN (each with residual + norm)  
3. Stacked decoder layers: *masked* self-attn → cross-attn → FFN  
4. Output projection to vocabulary (generation) or task heads (classification)

![Full architecture + Multi-Head / Scaled Dot-Product zoom](/deep-learning-self-learning/img/chapter_img/chapter08/tf_full_arch_multihead_zoom.jpg)
*Figure: Encoder–decoder with multi-head and scaled dot-product attention zoomed in. (Illustration from a Transformer Attention video)*

Chapter 07 already covered attention math. This lesson places that math inside the **full system**, with special focus on **order** (positional encoding) and **layer composition**.

---

## 2. Mathematical Foundation

### 2.1 Input representation

Let tokens be indices $$t_1,\ldots,t_L$$. An embedding matrix $$E$$ maps them to vectors:

$$\mathbf{x}_i^{(0)} = E_{t_i} + \mathbf{p}_i \in \mathbb{R}^{d_{\mathrm{model}}}$$

where $$\mathbf{p}_i$$ is a **positional encoding** (fixed or learned). Stack rows:

$$\mathbf{X}^{(0)} \in \mathbb{R}^{L \times d_{\mathrm{model}}}.$$

(Batch dimension $$B$$ is always present in code: $$(B,L,d)$$.)

### 2.2 Scaled multi-head self-attention (model interface)

With learned projections (see Chapter 07 for details):

$$
\mathrm{MultiHead}(\mathbf{X})
= \mathrm{Concat}_h\!\left(
  \mathrm{Attention}(\mathbf{X}W_h^Q,\mathbf{X}W_h^K,\mathbf{X}W_h^V)
\right) W^O
$$

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

where $$M$$ is an optional additive mask (padding and/or causal).

**Role in the Transformer:** this is the *communication* step—positions exchange information.

### 2.3 Position-wise feed-forward network (FFN)

Applied **independently** to each position (same weights for all positions):

$$
\mathrm{FFN}(\mathbf{z}) = W_2\,\sigma(W_1\mathbf{z}+\mathbf{b}_1)+\mathbf{b}_2
$$

with typical widths $$d_{\mathrm{model}} \to d_{\mathrm{ff}} \to d_{\mathrm{model}}$$ and $$d_{\mathrm{ff}} \approx 4\,d_{\mathrm{model}}$$. Activation $$\sigma$$ was ReLU in the original paper; modern models often use GELU or SwiGLU variants.

**Why FFN exists:** multi-head attention is essentially a *data-dependent linear mixture* of values (plus projections). Without a nonlinear MLP per position, stacking attention layers has limited per-token computational power. Attention mixes; FFN *computes*.

### 2.4 Residual connections and LayerNorm

Original (“Post-LN”) encoder sublayer:

$$
\mathbf{Y} = \mathrm{LayerNorm}\big(\mathbf{X} + \mathrm{Sublayer}(\mathbf{X})\big)
$$

where Sublayer is MultiHead or FFN. Many modern models use **Pre-LN**:

$$
\mathbf{Y} = \mathbf{X} + \mathrm{Sublayer}\big(\mathrm{LayerNorm}(\mathbf{X})\big)
$$

which often trains more stably at depth.

**Residuals** keep gradient highways; **LayerNorm** stabilizes activation scales across tokens and layers.

### 2.5 Full encoder layer

Post-LN form (as in the original paper):

$$
\begin{aligned}
\mathbf{U} &= \mathrm{LayerNorm}\big(\mathbf{X} + \mathrm{MultiHeadSelfAttn}(\mathbf{X})\big) \\
\mathbf{Z} &= \mathrm{LayerNorm}\big(\mathbf{U} + \mathrm{FFN}(\mathbf{U})\big)
\end{aligned}
$$

Stack $$N$$ such layers (often $$N=6$$ in the base model, far more in LLMs).

### 2.6 Full decoder layer

Three sublayers:

1. **Masked self-attention** over previously generated tokens (causal mask).  
2. **Cross-attention** where queries come from the decoder stream and keys/values come from encoder outputs.  
3. **FFN** as above.

$$
\begin{aligned}
\mathbf{U} &= \mathrm{LN}\big(\mathbf{Y} + \mathrm{MaskedSelfAttn}(\mathbf{Y})\big) \\
\mathbf{V} &= \mathrm{LN}\big(\mathbf{U} + \mathrm{CrossAttn}(\mathbf{U}, \mathbf{H}_{\mathrm{enc}}, \mathbf{H}_{\mathrm{enc}})\big) \\
\mathbf{Z} &= \mathrm{LN}\big(\mathbf{V} + \mathrm{FFN}(\mathbf{V})\big)
\end{aligned}
$$

**Causal mask:** for query position $$i$$, forbid keys $$j > i$$ so the model cannot “see the future” when trained with teacher forcing.

### 2.7 Positional encoding (sinusoidal)

Attention alone is **permutation-equivariant** in content: without position signals, order is invisible. The original PE:

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin\!\big(pos / 10000^{2i/d_{\mathrm{model}}}\big) \\
PE_{(pos,2i+1)} &= \cos\!\big(pos / 10000^{2i/d_{\mathrm{model}}}\big)
\end{aligned}
$$

![PE formulas + token matrix](/deep-learning-self-learning/img/chapter_img/chapter08/pe_formula_and_token_matrix.jpg)
*Figure: Each token has a $$d_{\mathrm{model}}$$ vector; PE uses sin/cos of $$pos$$ and frequency index $$i$$. (Illustration from a Position Encoding video)*

![pos = token index in the sentence](/deep-learning-self-learning/img/chapter_img/chapter08/pe_pos_index_sentence.jpg)
*Figure: $$pos = 0,1,2,\ldots$$ in token order. (Illustration from a Position Encoding video)*

**Reading dimension indices.** For $$d_{\mathrm{model}}=512$$, frequency index $$i$$ runs $$0\ldots 255$$: each $$i$$ produces **two** dimensions — even $$2i$$ uses **sin**, odd $$2i+1$$ uses **cos**.

![Table i → sin/cos dimensions](/deep-learning-self-learning/img/chapter_img/chapter08/pe_i_range_sin_cos_dims.jpg)
*Figure: $$i=0\to$$ dims 0 (sin), 1 (cos); …; $$i=255\to$$ dims 510, 511. (Illustration from a Position Encoding video)*

![Even dims sin / odd dims cos](/deep-learning-self-learning/img/chapter_img/chapter08/pe_even_odd_dims.jpg)
*Figure: Sin fills even slots, cos fills odd slots of the PE vector. (Illustration from a Position Encoding video)*

![Sin and cos](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sin_cos_formula_plot.jpg)
*Figure: Sin/cos pair at the same frequency — the wave basis for PE. (Illustration from a Position Encoding video)*

![Multiple frequencies by i](/deep-learning-self-learning/img/chapter_img/chapter08/pe_multiple_frequencies_i.jpg)
*Figure: Each $$i$$ has a different frequency; small $$i$$ = slow waves, large $$i$$ = fast waves. (Illustration from a Position Encoding video)*

**Sampling a PE vector at one position.** Fix $$pos$$; for each $$i$$ read one point on sin and one on cos → stack into a length-$$d_{\mathrm{model}}$$ vector.

![Sample PE at pos=0](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sample_values_pos0.jpg)
*Figure: Column at $$pos=0$$ — PE components for the first token. (Illustration from a Position Encoding video)*

![Sample PE at pos=3](/deep-learning-self-learning/img/chapter_img/chapter08/pe_sample_values_pos3.jpg)
*Figure: Same procedure at $$pos=3$$ — a different vector because the waves have phase-shifted. (Illustration from a Position Encoding video)*

**Add to the embedding.** Token embedding and PE share the same width; the actual input is usually:

$$
\mathbf{x}_{pos} = \mathrm{TokenEmb}(w_{pos}) + \mathrm{PE}_{pos}
$$

![Token emb vs PE (same 512-d)](/deep-learning-self-learning/img/chapter_img/chapter08/pe_input_emb_vs_positional.jpg)
*Figure: Two same-$$d_{\mathrm{model}}$$ vectors — elementwise sum “glues” order onto content. (Illustration from a Position Encoding video)*

![Token emb columns (purple) + PE columns (green)](/deep-learning-self-learning/img/chapter_img/chapter08/pe_token_emb_plus_pe_columns.jpg)
*Figure: Each position has a token-embedding column and a matching PE column. (Illustration from a Position Encoding video)*

Properties:

- Deterministic; no learned parameters.  
- Unique “fingerprint” per position via multi-frequency sinusoids.  
- Relative offsets are linear transforms of PE vectors, which helps learning relative-distance patterns.  
- Can, in principle, extrapolate beyond training lengths better than naive learned absolute embeddings (though real extrapolation still needs care).

**Alternatives:** learned absolute embeddings (BERT/GPT), relative bias (T5, Shaw), RoPE (modern LLMs). The *problem* is always the same: inject order into a content-based mixer.

### 2.8 Complexity

Dense self-attention costs $$O(L^2 d)$$ time and $$O(L^2)$$ memory for scores. That is the tax paid for global communication. For long documents, sparse/linear attention variants become necessary; for typical sentence/paragraph lengths, dense attention is the default.

### 2.9 Training objective sketch (original MT model)

Encoder reads source tokens; decoder predicts next target token with cross-entropy under teacher forcing. Modern encoder-only models use masked LM; decoder-only models use next-token LM. Architecture stays related; **objective + mask pattern** change the product behavior.

---

## 3. Example / Intuition

### 3.1 One encoder token’s journey

Sentence: `I love deep learning`. Focus on position **learning**.

In layer 1 self-attention, its query may put large weight on `deep` and itself, producing a mixture that already means “deep learning” rather than bare “learning.” The FFN then nonlinearly reshapes that mixture. Deeper layers further refine roles (syntax, semantics) as attention patterns become more abstract.

All four positions do this **in parallel** via matrix multiplies—unlike an RNN that would need four sequential steps.

### 3.2 One decoder step

Generating French `profond` after `J' aime l'apprentissage`:

1. **Masked self-attn** attends only to earlier French tokens (coherence of the hypothesis so far).  
2. **Cross-attn** looks at English encoder states; mass should concentrate on `deep`.  
3. **FFN + softmax over vocab** proposes `profond`.

If the causal mask is wrong, the model cheats during training and collapses at true autoregressive inference.

### 3.3 Encoder-only vs decoder-only

- **BERT (encoder-only):** bidirectional self-attn + masked LM → strong *understanding* representations.  
- **GPT (decoder-only):** causal self-attn + next-token LM → strong *generation*.  

Same MultiHead/FFN blocks; different masks and data flow.

---

## 4. Code Snippet (equation-facing)

Minimal shapes for one Pre-LN encoder layer:

```python
import torch
import torch.nn as nn

class TinyEncoderLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, d)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ff(h)
        return x

B, L, d = 2, 10, 64
layer = TinyEncoderLayer(d)
y = layer(torch.randn(B, L, d))
print(y.shape)  # (2, 10, 64)
```

Full from-scratch modules appear in the implementation lessons.

---

## 5. Related Concepts

- **Seq2seq RNNs** — same encode/decode idea; Transformer replaces recurrence with attention.  
- **Convolution** — fixed local kernels vs dynamic global mixing.  
- **Residual nets** — depth without gradient death.  
- **Pretraining paradigms** — MLM vs CLM vs span corruption (T5).  
- **Vision Transformer** — image patches as tokens; same encoder math.

---

## 6. Fundamental Papers

1. [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original architecture.  
2. [Devlin et al., 2019 — BERT](https://arxiv.org/abs/1810.04805) — bidirectional encoder pretraining.  
3. [Radford et al., 2019 — GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — large causal LMs + zero-shot.  
4. [Dosovitskiy et al., 2021 — ViT](https://arxiv.org/abs/2010.11929) — Transformers for images.  
5. [Phuong & Hutter, 2022 — Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238) — precise reference definitions.

---

## 7. Common Pitfalls and Tricks

- Forgetting **positional** information → order-blind model.  
- Causal mask with **0** instead of **−∞** before softmax → future leakage.  
- Missing **pad** masks in batched training.  
- Omitting $$1/\sqrt{d_k}$$ at large width.  
- Confusing **Post-LN vs Pre-LN** when porting recipes (learning rates differ).  
- **Trick:** unit-test causal attention with `triu(weights,1)≈0`.  
- **Trick:** start with a 2-layer tiny model before scaling depth/width.

---

## 8. Key Takeaways

1. A Transformer layer is **attend (communicate) + FFN (compute)** with residual/norm.  
2. **Masks** define the product family: bidirectional, causal, or cross.  
3. **Positional encodings** fix order blindness of pure attention.  
4. Encoder/decoder stacks compose the same layer types into translation or other seq2seq systems.  
5. BERT/GPT/ViT are mask + objective + data choices on top of this template.

Next: implement these blocks carefully in **08-01-02**.

<!-- video-references -->

## Video references

Some figures in this lesson are screenshots from the following videos (Machine Learning Thực Chiến). URLs kept for attribution and further viewing:

- [Attention in Transformers (Self, Masked, Cross, Multi-Head)](https://www.facebook.com/reel/1007473105556936)
- [Position Encoding in Transformers (step by step)](https://www.facebook.com/reel/1306224801415127)
