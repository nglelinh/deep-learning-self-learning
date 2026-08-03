---
layout: post
title: 07-01 Attention Mechanism Fundamentals
chapter: '07'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter07
---

# Attention Mechanism Fundamentals

![Multi-Head Attention](/deep-learning-self-learning/img/chapter_img/chapter07/multihead_attention.png)
*Multi-head attention schematic (as used in Transformers). Source: Wikimedia Commons*

## 1. Concept Overview

**Attention** is a mechanism that lets a model *dynamically focus* on different parts of its input when producing each piece of output. Instead of forcing all information through a single fixed-size summary, attention computes a **soft selection**: for every output step, it assigns weights to input positions and forms a weighted combination of their representations.

That idea sounds simple, but it solved a structural failure of early sequence models and later became the backbone of modern NLP (and much of vision).

### The bottleneck that made attention necessary

Classic **encoder–decoder** (seq2seq) models for machine translation work as follows:

1. An **encoder** RNN reads the source sentence word by word.
2. The encoder compresses the whole sentence into one final hidden state (a fixed-size vector).
3. A **decoder** RNN starts from that vector and generates the target sentence.

Schematically:

```text
"The cat sat on the mat"
        │
        ▼
   [Encoder RNN]
        │
        ▼
   fixed context vector  z ∈ R^d     ← bottleneck
        │
        ▼
   [Decoder RNN] → "Le chat était assis sur le tapis"
```

Two problems appear immediately.

**Information bottleneck.** Whether the source is 5 words or 50, everything must fit in $$z$$. Long sentences lose detail; rare but critical words get washed out.

**Fixed “one summary for every target word.”** Human translators do not re-read the *same* summary for every target word. When producing “chat,” they focus on “cat”; when producing “tapis,” they focus on “mat.” A single fixed $$z$$ cannot provide that *dynamic alignment*.

**Attention’s solution:** keep *all* encoder states $$\mathbf{h}_1, \ldots, \mathbf{h}_T$$, and at each decoder step $$t$$, compute a new context vector

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i}\,\mathbf{h}_i$$

where the weights $$\alpha_{t,i}$$ are *learned* and *depend on the current decoder state*. Different target words get different soft alignments over the source.

### Everyday analogy

Think of writing an answer while an open textbook lies on the desk. You do not memorize the whole chapter into one sentence and then answer from that sentence alone. For each sentence you write, you glance back at different paragraphs. Attention is that “glance”: a differentiable, learned version of looking up the right place in the input.

### Why this chapter starts here

This lesson builds the **conceptual and historical core** of attention:

- encoder–decoder attention (Bahdanau / Luong),
- the query–key–value view,
- scaled dot-product attention,
- self-attention vs cross-attention,
- multi-head attention at an introductory level.

Later lessons in this chapter deepen the mathematics and implementations. Transformers (next chapter) assemble these pieces into a full architecture.

---

## 2. Mathematical Foundation

### 2.1 Encoder–decoder attention in general form

At decoder step $$t$$:

| Symbol | Meaning |
|--------|---------|
| $$\mathbf{s}_t$$ | decoder hidden state (what we are trying to generate *now*) |
| $$\mathbf{h}_i$$ | encoder hidden state at source position $$i$$ |
| $$e_{t,i}$$ | unnormalized **alignment score** between decoder step $$t$$ and source $$i$$ |
| $$\alpha_{t,i}$$ | **attention weight** (how much to focus on source $$i$$) |
| $$\mathbf{c}_t$$ | **context vector** (soft summary of the source for step $$t$$) |

**Step 1 — scores.** Compare the decoder state to every encoder state:

$$e_{t,i} = \mathrm{score}(\mathbf{s}_t, \mathbf{h}_i)$$

**Step 2 — normalize with softmax.**

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T}\exp(e_{t,j})}$$

so $$\alpha_{t,i} \ge 0$$ and $$\sum_i \alpha_{t,i} = 1$$. The weights form a probability distribution over source positions.

**Step 3 — context as a weighted sum.**

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i}\,\mathbf{h}_i$$

**Step 4 — use context for prediction.** Typically combine context and decoder state, then predict the next token:

$$\tilde{\mathbf{s}}_t = \tanh\!\big(\mathbf{W}_c[\mathbf{c}_t;\mathbf{s}_t]\big), \qquad
p(y_t \mid y_{<t}, x) = \mathrm{softmax}(\mathbf{W}_o\tilde{\mathbf{s}}_t)$$

(Exact combination layers vary by paper; the important idea is: **prediction depends on $$\mathbf{c}_t$$, not only on a fixed bottleneck.**)

### 2.2 Bahdanau attention (additive attention, 2015)

[Bahdanau et al.](https://arxiv.org/abs/1409.0473) introduced a trainable score:

$$e_{t,i} = \mathbf{v}_a^\top \tanh\!\big(\mathbf{W}_a\mathbf{s}_t + \mathbf{U}_a\mathbf{h}_i\big)$$

**Why it works.**

- $$\mathbf{W}_a$$ and $$\mathbf{U}_a$$ project decoder and encoder states into a shared space before comparison.
- $$\tanh$$ adds nonlinearity so “relevance” is not a pure linear dot product.
- $$\mathbf{v}_a$$ collapses the projected vector to a scalar score.

This is often called **additive** attention because query and key features are *added* inside the $$\tanh$$.

Historically, this was the first widely used neural attention for NMT and made long-sentence translation dramatically more reliable than pure bottleneck seq2seq.

### 2.3 Luong attention (multiplicative / general forms, 2015)

[Luong et al.](https://arxiv.org/abs/1508.04025) studied simpler score functions that are often faster and work well in practice:

**Dot product**

$$e_{t,i} = \mathbf{s}_t^\top \mathbf{h}_i$$

**General (bilinear)**

$$e_{t,i} = \mathbf{s}_t^\top \mathbf{W}_a \mathbf{h}_i$$

**Concat** (similar spirit to Bahdanau)

$$e_{t,i} = \mathbf{v}_a^\top \tanh\!\big(\mathbf{W}_a[\mathbf{s}_t;\mathbf{h}_i]\big)$$

**Intuition.** Dot product measures directional alignment: if the decoder “wants” something in the same direction as an encoder state, the score is large. The general form inserts a learned metric $$\mathbf{W}_a$$ so alignment can be more flexible when $$\mathbf{s}_t$$ and $$\mathbf{h}_i$$ live in different spaces.

### 2.4 The query–key–value (Q, K, V) view

Modern attention is usually written with three roles:

| Role | Question it answers | Typical source |
|------|---------------------|----------------|
| **Query** $$\mathbf{q}$$ | “What am I looking for right now?” | decoder state, or position $$i$$ in self-attention |
| **Key** $$\mathbf{k}_j$$ | “What does item $$j$$ offer for matching?” | encoder states, or all positions |
| **Value** $$\mathbf{v}_j$$ | “What content should I retrieve if I match $$j$$?” | often same vectors as keys, or a separate projection |

Attention then does:

1. score queries against keys,
2. softmax → weights,
3. weighted sum of values.

This abstraction covers **cross-attention** (query from one sequence, keys/values from another) and **self-attention** (query, key, value all from the same sequence).

In LLMs the same pipeline is often described as a *fully connected relational network*: embeddings in → split Q/K/V → attention scores → Softmax → (many heads in parallel) → fused context out.

![LLM-style attention pipeline](/deep-learning-self-learning/img/chapter_img/chapter07/llm_attn_qkv_pipeline_active.jpg)
*Figure: Q–K–V flow → Softmax → multi-head → context fusion. (Illustration from an LLM Attention Mechanism video)*

#### Numeric walkthrough: 3-token sentence → Q, K, V

Take a toy 3-word sentence (e.g. “I / love / you”), each word already a 2-d embedding (illustration; real models often use $$d_{\mathrm{model}}=512$$):

![3-word embeddings](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_toy_sentence_embeddings.jpg)
*Figure: Input matrix $$\mathbf{X}$$ — 3 tokens × $$d$$. (Illustration from a QKV Attention formula video)*

Three learned weight matrices $$W^Q, W^K, W^V$$ project $$\mathbf{X}$$ into query/key/value spaces:

![W^Q, W^K, W^V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_weights_wq_wk_wv.jpg)
*Figure: Three separate projection matrices. (Illustration from a QKV Attention formula video)*

![X × W → Q, K, V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_project_x_to_qkv.jpg)
*Figure: $$\mathbf{Q}=\mathbf{X}W^Q$$, $$\mathbf{K}=\mathbf{X}W^K$$, $$\mathbf{V}=\mathbf{X}W^V$$. (Illustration from a QKV Attention formula video)*

### 2.5 Scaled dot-product attention

The standard building block of Transformers is:

$$\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
= \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

![Q, K, V and the Attention formula](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_matrices_and_formula.jpg)
*Figure: Projected matrices + scaled SDP formula. (Illustration from a QKV Attention formula video)*

with shapes (for one sequence, ignoring batch for a moment):

- $$\mathbf{Q} \in \mathbb{R}^{L_q \times d_k}$$
- $$\mathbf{K} \in \mathbb{R}^{L_k \times d_k}$$
- $$\mathbf{V} \in \mathbb{R}^{L_k \times d_v}$$
- output $$\in \mathbb{R}^{L_q \times d_v}$$

**Hand calculation (same 3-word example).**

1. **Dot products** — each query row × columns of $$K^\top$$ (one score per token pair):

![Per-entry QK^T](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_dot_product_row_example.jpg)
*Figure: Example: the “I” query dotted with every key. (Illustration from a QKV Attention formula video)*

![Unscaled similarity matrix](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_unscaled_similarities.jpg)
*Figure: $$\mathbf{Q}K^\top$$ — Unscaled Dot-Product Similarities. (Illustration from a QKV Attention formula video)*

2. **Scale** — divide by $$\sqrt{d_k}$$ (here $$d_k=2$$ → $$\sqrt{2}$$):

![Divide by √d_k](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_scale_by_sqrt_dk.jpg)
*Figure: Unscaled → Scaled Dot-Product Similarities. (Illustration from a QKV Attention formula video)*

**Why divide by $$\sqrt{d_k}$$?**

If query/key coordinates are roughly independent with variance 1, then a raw dot product has variance about $$d_k$$. For large $$d_k$$ (e.g. 64–512), scores become huge, softmax becomes nearly one-hot, and gradients vanish. Scaling keeps scores in a healthier range so the model can learn *soft* distributions, not only hard argmax-like attention.

3. **Row-wise softmax** — each query becomes a distribution over keys:

![Softmax over rows](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_softmax_over_rows.jpg)
*Figure: Softmax runs along each row. (Illustration from a QKV Attention formula video)*

![Attention weights](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_attention_weights.jpg)
*Figure: Weight matrix (each row sums to ≈ 1). (Illustration from a QKV Attention formula video)*

4. **× V** — mix values by those weights → attention output:

![Weights × V](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_weights_times_v.jpg)
*Figure: Attention Weights × V = Attention Output. (Illustration from a QKV Attention formula video)*

### 2.6 Self-attention vs cross-attention

**Cross-attention (encoder–decoder attention).**  
Queries come from the decoder; keys and values come from the encoder. This is the original NMT attention: “when generating this target word, which source words matter?”

**Self-attention.**  
Queries, keys, and values all come from the *same* sequence. Every position can gather context from every other position:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q,\quad
\mathbf{K} = \mathbf{X}\mathbf{W}^K,\quad
\mathbf{V} = \mathbf{X}\mathbf{W}^V$$

Each token starts as a numeric vector; then every token builds its own Q/K/V:

![Tokens as numeric columns](/deep-learning-self-learning/img/chapter_img/chapter07/sa_tokens_as_vectors.jpg)
*Figure: A sentence represented as embedding vectors. (Illustration from an intro Transformer / Self-Attention video)*

![Per-token Q/K/V columns](/deep-learning-self-learning/img/chapter_img/chapter07/sa_per_token_qkv_columns.jpg)
*Figure: Self-attention forms Q, K, V for *each* position. (Illustration from an intro Transformer / Self-Attention video)*

![Q/K/V roles by word](/deep-learning-self-learning/img/chapter_img/chapter07/sa_qkv_roles_apple_phone_orange.jpg)
*Figure: Same QKV pipeline on “apple / phone / orange” — match context, then mix values. (Illustration from an intro Transformer / Self-Attention video)*

Example: in the sentence

> The animal didn't cross the street because **it** was too tired.

Self-attention can learn that “it” should attend strongly to “animal.” That long-range link does not need to travel step-by-step through an RNN.

### 2.7 Multi-head attention (introductory view)

A single attention head can specialize in one kind of pattern. Language (and other structured data) has many simultaneous relations: syntax, coreference, adjacency, semantic similarity, etc.

**Multi-head attention** runs $$H$$ attention operations in parallel with different projections, then concatenates and mixes them:

$$\mathrm{head}_h = \mathrm{Attention}(\mathbf{Q}\mathbf{W}_h^Q, \mathbf{K}\mathbf{W}_h^K, \mathbf{V}\mathbf{W}_h^V)$$

$$\mathrm{MultiHead}(\mathbf{Q},\mathbf{K},\mathbf{V})
= \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)\,\mathbf{W}^O$$

Often $$d_k = d_v = d_{\mathrm{model}} / H$$ so total compute stays comparable to one full-width head, while representational capacity increases through diversity of heads.

Common numbers: $$d_{\mathrm{model}}=512$$, $$H=8$$ → each head uses $$d_k=64$$:

![Split d_model across heads](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_dk_split.jpg)
*Figure: $$512/8=64$$ — Q/K/V width per head. (Illustration from a QKV Attention formula video)*

![Stacked multi-head outputs](/deep-learning-self-learning/img/chapter_img/chapter07/qkv_multihead_output_stack.jpg)
*Figure: 8 heads × (L × 64) then concat → (L × 512). (Illustration from a QKV Attention formula video)*

### 2.8 Masking (what you must know early)

Two masks appear constantly:

**Padding mask.** In a batch, shorter sequences are padded. Attention must not put mass on pad tokens. Implementation: add a large negative number to masked logits before softmax so those weights ≈ 0.

**Causal (look-ahead) mask.** In autoregressive generation (language modeling, decoder-only Transformers), position $$i$$ must not attend to future positions $$j > i$$. Use an upper-triangular mask.

Without the right mask, models “cheat” by looking at tokens they should not see yet, or by attending to meaningless padding.

---

## 3. Example / Intuition

### 3.1 Soft alignment in translation

Source: **I love deep learning** (4 tokens)  
Target step producing something like **apprentissage** (“learning”).

Suppose the decoder query at that step scores the four source positions as:

| Source token | raw score $$e$$ | after softmax $$\alpha$$ |
|--------------|-----------------|---------------------------|
| I | 0.2 | 0.05 |
| love | 0.5 | 0.07 |
| deep | 1.5 | 0.20 |
| learning | 2.8 | 0.68 |

The context vector is mostly “learning,” with a useful contribution from “deep.” The model did not need a hard symbolic alignment table; it *learned* a soft alignment from data.

When generating a different target word (e.g. “J’aime” / “I love”), the weight mass would shift toward “I” and “love.”

### 3.2 Self-attention as contextualization

Take three tokens with toy 2-D values (already projected for simplicity):

| Token | value vector |
|-------|----------------|
| The | $$(1, 0)$$ |
| cat | $$(0, 1)$$ |
| sat | $$(1, 1)$$ |

If “sat” attends with weights $$[0.1, 0.6, 0.3]$$ to (The, cat, sat), its output becomes

$$0.1(1,0) + 0.6(0,1) + 0.3(1,1) = (0.4,\ 0.9)$$

The representation of “sat” is no longer isolated; it is **mixed with context**, especially “cat.” That is the essence of contextual embeddings produced by self-attention stacks (BERT, GPT, etc.).

### 3.3 Attention is not magic memory

Attention gives *access* to representations that already exist. It does not by itself invent long-term storage like an LSTM cell state. If the encoder states are weak, attention can only reweight weak information. Good attention needs good underlying features—and later, positional signals (Transformers add positional encodings precisely because pure self-attention is permutation-sensitive without them).

---

## 4. Code Snippet

### 4.1 Scaled dot-product attention (NumPy)

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (..., Lq, dk)
    K: (..., Lk, dk)
    V: (..., Lk, dv)
    mask: broadcastable to (..., Lq, Lk); True/1 means "mask out"
    returns: output (..., Lq, dv), weights (..., Lq, Lk)
    """
    dk = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)  # (..., Lq, Lk)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


# Toy self-attention example
L, dk, dv = 4, 8, 8
rng = np.random.default_rng(0)
X = rng.normal(size=(L, dk))

# For demo: use X as Q, K, V (no learned projections yet)
out, attn = scaled_dot_product_attention(X, X, X)
print("output", out.shape)   # (4, 8)
print("attn", attn.shape)    # (4, 4)
print("rows sum to 1:", np.allclose(attn.sum(axis=-1), 1.0))
```

### 4.2 Causal mask

```python
def causal_mask(L):
    """mask[i, j] = True if j > i (forbid attending to the future)."""
    return np.triu(np.ones((L, L), dtype=bool), k=1)

L = 4
Q = K = V = np.random.randn(L, 8)
out, attn = scaled_dot_product_attention(Q, K, V, mask=causal_mask(L))
print(np.round(attn, 3))
# Upper triangle (future) should be ~0
```

### 4.3 Multi-head attention (conceptual NumPy)

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rng=None):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        rng = np.random.default_rng(0) if rng is None else rng
        scale = 0.02
        self.Wq = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wk = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wv = rng.normal(scale=scale, size=(d_model, d_model))
        self.Wo = rng.normal(scale=scale, size=(d_model, d_model))

    def _split(self, x):
        # x: (B, L, d_model) -> (B, H, L, dk)
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_heads, self.dk)
        return x.transpose(0, 2, 1, 3)

    def _merge(self, x):
        # x: (B, H, L, dk) -> (B, L, d_model)
        B, H, L, dk = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(B, L, H * dk)
        return x

    def forward(self, x, mask=None):
        # Self-attention for simplicity
        Q = self._split(x @ self.Wq)
        K = self._split(x @ self.Wk)
        V = self._split(x @ self.Wv)
        out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        out = self._merge(out) @ self.Wo
        return out, weights


B, L, d_model, H = 2, 6, 32, 4
x = np.random.randn(B, L, d_model)
mha = MultiHeadAttention(d_model, H)
y, w = mha.forward(x)
print(y.shape)  # (2, 6, 32)
print(w.shape)  # (2, 4, 6, 6)  — batch, heads, query pos, key pos
```

### 4.4 PyTorch (what you will use in practice)

```python
import torch
import torch.nn.functional as F

def torch_sdp_attention(q, k, v, attn_mask=None):
    """
    q,k,v: (B, H, L, d) or (B, L, d)
    attn_mask: additive mask broadcastable to scores; use -inf on forbidden positions
    """
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


# Built-in module (preferred for real models)
mha = torch.nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
x = torch.randn(2, 10, 64)
y, attn = mha(x, x, x, need_weights=True, average_attn_weights=False)
print(y.shape, attn.shape)
```

In production code, prefer `torch.nn.MultiheadAttention` or fused scaled-dot-product kernels (`torch.nn.functional.scaled_dot_product_attention`) for speed and numerical stability.

---

## 5. Related Concepts

**RNNs / LSTMs.** Attention was first popularized *on top of* RNN encoders. RNNs provide sequential states; attention selects among them. Transformers later *replace* recurrence with self-attention + positional encoding.

**Seq2seq bottleneck.** Attention is the direct fix for fixed-length context vectors in encoder–decoder models (Chapters 5–6 background).

**Softmax as soft selection.** Attention weights are a softmax over scores—the same primitive used in classification, but applied over *positions* (or memory slots).

**Memory / retrieval view.** Keys are addresses, values are stored contents, queries are lookup requests. This view connects attention to differentiable memory and retrieval-augmented models.

**Transformers.** A Transformer layer is largely: multi-head self-attention + feed-forward network + residual connections + normalization. This lesson’s QKV machinery is the core of that design (next chapter).

**Interpretability.** Attention heatmaps are a useful debugging tool, but they are not a complete causal explanation of model decisions. Treat them as *evidence of focus*, not proof of reasoning.

---

## 6. Fundamental Papers

1. **[Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)**  
   Introduced additive attention for NMT; decoder soft-searches source positions while generating each target word. The paper that made neural attention mainstream.

2. **[Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)](https://arxiv.org/abs/1508.04025)**  
   Compared global/local attention and simple multiplicative scoring functions; practical recipes still cited today.

3. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   Replaced recurrence with multi-head self-attention (the Transformer). Scaled dot-product attention and multi-head design became the default template for modern sequence models.

4. **[Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)**  
   Visual attention for image captioning: demonstrated that attention generalizes beyond text-to-text alignment.

5. **[BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)** / **[GPT lineage (Radford et al.)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**  
   Large-scale proof that stacks of self-attention produce powerful language representations and generators.

---

## 7. Common Pitfalls and Tricks

**Pitfall: confusing scores with weights.**  
$$e_{t,i}$$ are raw logits; only after softmax do you get $$\alpha_{t,i}$$ that sum to 1. Never average encoder states with unnormalized scores and call it attention.

**Pitfall: forgetting $$\sqrt{d_k}$$.**  
Without scaling, large $$d_k$$ makes attention overly sharp and training unstable.

**Pitfall: wrong mask polarity.**  
Be explicit: does `1` mean “keep” or “mask out”? Off-by-convention bugs silently leak future tokens or kill real tokens.

**Pitfall: attending to padding.**  
Always mask pads in batched training; otherwise the model wastes capacity on `PAD` and metrics look mysteriously bad.

**Pitfall: treating attention weights as ground-truth explanations.**  
Weights are useful visualizations, not guaranteed causal attributions.

**Trick: temperature / sharpness.**  
Dividing scores by a temperature $$\tau$$ before softmax (or using sparsemax variants) can control how peaked attention is—useful for analysis and some specialized models.

**Trick: residual + layer norm around attention.**  
In deep stacks, always wrap attention blocks with residuals (and usually LayerNorm). Bare attention layers are harder to train deeply.

**Trick: start with cross-attention intuition, then self-attention.**  
If multi-head self-attention feels abstract, first master “decoder looks at encoder,” then replace both sides with the same sequence.

---

## 8. Key Takeaways

1. **Attention replaces a fixed bottleneck with a dynamic weighted readout** over input states.
2. **Scores → softmax weights → weighted values** is the universal skeleton (Bahdanau, Luong, Transformer).
3. **Q, K, V** unify cross-attention and self-attention in one language.
4. **Scaling by $$\sqrt{d_k}$$** keeps softmax well-behaved in high dimensions.
5. **Multi-head attention** learns multiple relation patterns in parallel.
6. **Masking** (padding, causal) is part of the algorithm, not an optional extra.
7. Attention is the conceptual bridge from RNN seq2seq to **Transformers**—the next chapter.

When you can explain, with a small numerical example, how a decoder step forms $$\mathbf{c}_t$$ from encoder states, you have the foundation needed for every modern attention model in this course.

<!-- video-references -->

## Video references

Some figures in this lesson are screenshots from the following videos (Machine Learning Thực Chiến). URLs kept for attribution and further viewing:

- [LLM Ep.4: Attention Mechanism (Q–K–V, fully connected relations)](https://www.facebook.com/reel/1467113421464723)
- [What is a Transformer? Self-Attention for beginners (Part 1)](https://www.facebook.com/reel/930207546288223)
- [Attention formula: hand-compute Q, K, V](https://www.facebook.com/reel/1806844676942638)
