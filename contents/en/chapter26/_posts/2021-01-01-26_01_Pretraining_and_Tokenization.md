---
layout: post
title: 26-01 Pretraining and Tokenization
chapter: '26'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter26
---

# Pretraining and Tokenization

A modern LLM is pretrained as a **next-token predictor**. Everything later — prompting, instruction-following, tools — sits on top of a model that, given tokens $$x_1,\ldots,x_{t-1}$$, outputs a distribution over the next token $$x_t$$.

This lesson stays conceptual. You do not need a full training stack; you need the objective, the tokenizer that defines the symbol set, and the context window that bounds what the model can “see.”

## 1. Next-token prediction

Let $$x = (x_1,\ldots,x_T)$$ be a token sequence. A causal language model factorizes

$$p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}).$$

Training minimizes the average negative log-likelihood (cross-entropy) on a corpus:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).$$

At each position the network produces logits $$z \in \mathbb{R}^{V}$$ ($$V$$ is the vocabulary size). Softmax turns them into $$p_\theta(\cdot \mid x_{<t})$$. The “label” is simply the observed next token — no human annotator is required. That is why this is **self-supervised** in the sense of Chapter 16, even though the loss looks like ordinary classification.

Two consequences matter later in the chapter:

- The output space is **huge** (often $$V \sim 32\mathrm{k}$$–$$200\mathrm{k}+$$). Classical knowledge distillation that matches a 1,000-class softmax (Chapter 23) does not copy over cheaply — see **26-05**.
- Training is **teacher-forced**: the model always conditions on the *true* prefix, not on its own samples. At decode time it must condition on its own tokens. That train/serve mismatch is normal; alignment and decoding tricks try to live with it, not erase it.

A tiny PyTorch-shaped sketch of the loss (shapes only):

```python
import torch.nn.functional as F

def next_token_loss(logits, input_ids):
    # logits: (batch, seq, vocab) for positions 0..T-1
    # targets: next token at each position
    targets = input_ids[:, 1:]
    pred = logits[:, :-1, :]
    return F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,  # padding
    )
```

The rest of this lesson is the **math the last layer actually uses**, then the tokenizer and the window. If a formula already lives in Chapters 02–03 or 10, we only restate the LM-shaped version.

## 2. Softmax, cross-entropy, and $$\partial\mathcal{L}/\partial z = p - t$$

At one position the model emits logits $$z \in \mathbb{R}^{V}$$. Softmax is

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.$$

The LM loss on a one-hot target $$t$$ (the observed next token) is cross-entropy

$$\mathcal{L} = -\sum_{i=1}^{V} t_i \log p_i = -\log p_{t^*},$$

where $$t^*$$ is the index of the true token. Differentiating through softmax gives the interview identity

$$\frac{\partial \mathcal{L}}{\partial z} = p - t.$$

That is the same cancellation Chapter 03 shows for CE + softmax on a small classifier. The LM does it at every time step, with $$V$$ in the tens or hundreds of thousands.

**CE, KL, and entropy.** For any two discrete distributions $$p^*$$ and $$p$$,

$$\mathrm{CE}(p^*, p) = H(p^*) + \mathrm{KL}(p^* \| p).$$

A one-hot $$p^*$$ has $$H(p^*) = 0$$, so token-level CE **is** $$\mathrm{KL}(\text{one-hot} \| p)$$ **is** negative log-likelihood. When the target is a *soft* teacher (distillation, **26-05**), $$H(p^*)$$ is a constant w.r.t. the student and the extra signal is the KL. Temperature-scaled softmax in 26-05 is this identity with a flatter $$p^*$$.

## 3. Log-sum-exp and why FlashAttention talks about “online softmax”

Naive $$e^{z_i}$$ overflows when logits are large. The stable rewrite subtracts the row max $$m = \max_j z_j$$:

$$\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}, \qquad p_i = \exp\big(z_i - \mathrm{LSE}(z)\big).$$

That is **log-sum-exp**. The same algebra can be run *incrementally*: if you have already seen a prefix of logits with running max $$m$$ and running sum $$s = \sum e^{z-m}$$, and a new block arrives with max $$m'$$, you rescale

$$s \leftarrow s\, e^{m - m_{\mathrm{new}}} + \sum_{\text{new}} e^{z - m_{\mathrm{new}}}, \quad m_{\mathrm{new}} = \max(m, m').$$

Keep a matching running weighted sum of values and you have **online softmax** — the numerical story inside [FlashAttention](https://arxiv.org/abs/2205.14135), without any CUDA. Lesson **26-07** uses this as the reason the $$L\times L$$ matrix never has to land in HBM.

## 4. The linear layer: math $$XW$$ vs PyTorch $$W$$

A batch of hidden states $$X \in \mathbb{R}^{B \times d_{\mathrm{in}}}$$ and a math textbook write

$$Y = X W + b, \qquad W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}.$$

`torch.nn.Linear(n_in, n_out)` stores `weight` with shape **$$(n_{\mathrm{out}}, n_{\mathrm{in}})$$** and computes $$X W^\top + b$$. Same affine map; the stored matrix is the textbook $$W$$ *transposed*. When you count parameters in **26-07**, use $$d_{\mathrm{in}} \times d_{\mathrm{out}}$$ and do not double-count because of the layout. When you port a paper, check which convention the authors drew.

## 5. Activations the FFN actually uses

Chapter 02 already has the pictures. The LM-relevant three:

$$
\begin{aligned}
\mathrm{ReLU}(z) &= \max(0, z), \\
\mathrm{SiLU}(z) = \mathrm{Swish}(z) &= z\,\sigma(z), \\
\mathrm{SwiGLU}(x) &= \big(\mathrm{SiLU}(x W_{\mathrm{gate}}) \odot (x W_{\mathrm{up}})\big) W_{\mathrm{down}}.
\end{aligned}
$$

ReLU is the 2017 FFN. SiLU is the smooth gate. SwiGLU is the *block* (three matrices, Hadamard gate) that **26-07** puts in the decoder diagram. GELU (Chapter 02) still appears in older GPT-style FFNs; it is not a third theory.

## 6. Adam / AdamW: moments, and which parameters decay

Pretraining almost always uses **AdamW** ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)), not raw Adam ([Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980)). The moments are the Chapter 10 ones:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}.
\end{aligned}
$$

Adam applies an adaptive step and, if you add L2 to the *loss*, that L2 gradient is also rescaled by $$\sqrt{\hat{v}}$$ — so “weight decay” is no longer a uniform pull toward zero. AdamW **decouples** the decay:

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} + \lambda \theta_{t-1} \right).$$

**Which parameters get $$\lambda > 0$$?** The usual LLM convention (Hugging Face `AdamW`, Llama-style trainers): apply decay to **2-D weight matrices** (attention and FFN projections, often the embedding / LM head). Do **not** decay 1-D tensors: biases (if any) and RMSNorm / LayerNorm gains. Those scales are not “weights that should shrink”; decaying them fights the normalizer. Details live in **10-01**; this paragraph is only the LM checklist.

## 7. Why we do not train on raw characters (or raw words)

Characters make sequences very long and waste capacity on spelling. Whole words explode the vocabulary and fail on rare or new spellings. **Subword** tokenization sits in between: frequent words stay one token; rare words split into reusable pieces.

Two families you will actually meet:

**Byte Pair Encoding (BPE).** Start from characters (or bytes). Repeatedly merge the most frequent adjacent pair. The merge table *is* the tokenizer. [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909) introduced BPE for neural MT; GPT-2-style tokenizers are a widely used descendant. OpenAI’s [tiktoken](https://github.com/openai/tiktoken) is an engineering implementation of this idea, not a new algorithm.

**Unigram / SentencePiece.** Instead of greedy merges, a unigram model keeps a large candidate vocabulary and drops tokens that hurt a likelihood objective. [Kudo & Richardson, 2018](https://arxiv.org/abs/1808.06226) (SentencePiece) is the usual package: it trains from raw text and can emit either BPE or unigram vocabularies. Many multilingual open LMs use this stack.

Intuition, not a training recipe:

- Tokenization is a **lossy, deterministic front-end**. The LM never sees characters the tokenizer fused, and it cannot invent a token ID that is not in the table.
- The same English word can be 1 token in one vocab and 3 tokens in another. **Cost and context are measured in tokens, not words.**
- Special tokens (`<bos>`, `<eos>`, chat template markers, tool tags) are first-class vocabulary entries. Off-by-one mistakes here waste the context window or break chat formatting.

Chapter 18’s embedding table $$E \in \mathbb{R}^{V \times d}$$ is exactly layer 0 of this model: each token ID becomes a vector, then (in modern decoders) a positional method such as RoPE is applied inside attention (optional lesson **08-99**).

## 8. The context window

Self-attention in a vanilla Transformer mixes every pair of positions in the window. If the window length is $$L$$, one layer costs $$O(L^2)$$ in attention (plus cheap position-wise MLPs). So $$L$$ is both a **capability** limit and a **compute/memory** limit.

What the window actually bounds:

- The model can only condition on the last $$L$$ tokens of the concatenated prompt + generation (plus whatever you retrieve and paste — RAG, Chapter 18-99).
- Long documents must be truncated, summarized, or retrieved in chunks. “The model read my 200-page PDF” usually means *a retrieval system stuffed selected chunks into $$L$$*.
- During decode, keys and values for those $$L$$ positions are the natural thing to **cache** (lesson **26-04**).

Extensions (sliding windows, grouped-query attention, extra-long RoPE scaling) change the constants; they do not remove the idea that the model has a finite working tape.

## 9. Pretraining data, briefly

Pretraining corpora mix web text, books, code, and filtered crawls. You do not need a secret mix to understand the objective. You *do* need to remember:

- The loss rewards **fluency and corpus statistics**, not truth. Hallucination is not a separate bug; it is next-token sampling from an imperfect $$p_\theta$$.
- Data filters and deduplication are part of the model. Two LMs with the same architecture can behave differently because the corpus differed.
- Scaling (Chapter 25; train-time sketch in **26-07**) relates loss to parameters, tokens, and compute. This lesson assumes that curve exists; it does not re-fit it.

## Key takeaways

- Pretraining = maximize next-token likelihood under a causal mask.
- Softmax + CE collapses to $$\partial\mathcal{L}/\partial z = p - t$$. CE = entropy + KL; one-hot targets make that just NLL.
- Log-sum-exp is the stable softmax; the incremental version is the FlashAttention narrative (**26-07**).
- PyTorch `Linear` stores $$W$$ as $$(n_{\mathrm{out}}, n_{\mathrm{in}})$$ and applies $$X W^\top$$. AdamW decays 2-D weights, not norm gains.
- The tokenizer defines the discrete alphabet; BPE and SentencePiece are the two practical stories.
- Context length $$L$$ is the working memory of attention — and the size of the KV-cache you will pay for at serve time.
- The same softmax-over-$$V$$ view that makes pretraining simple is what makes **classical logit distillation expensive** for LMs (continue in **26-05**).
