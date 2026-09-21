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

## 2. Why we do not train on raw characters (or raw words)

Characters make sequences very long and waste capacity on spelling. Whole words explode the vocabulary and fail on rare or new spellings. **Subword** tokenization sits in between: frequent words stay one token; rare words split into reusable pieces.

Two families you will actually meet:

**Byte Pair Encoding (BPE).** Start from characters (or bytes). Repeatedly merge the most frequent adjacent pair. The merge table *is* the tokenizer. [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909) introduced BPE for neural MT; GPT-2-style tokenizers are a widely used descendant. OpenAI’s [tiktoken](https://github.com/openai/tiktoken) is an engineering implementation of this idea, not a new algorithm.

**Unigram / SentencePiece.** Instead of greedy merges, a unigram model keeps a large candidate vocabulary and drops tokens that hurt a likelihood objective. [Kudo & Richardson, 2018](https://arxiv.org/abs/1808.06226) (SentencePiece) is the usual package: it trains from raw text and can emit either BPE or unigram vocabularies. Many multilingual open LMs use this stack.

Intuition, not a training recipe:

- Tokenization is a **lossy, deterministic front-end**. The LM never sees characters the tokenizer fused, and it cannot invent a token ID that is not in the table.
- The same English word can be 1 token in one vocab and 3 tokens in another. **Cost and context are measured in tokens, not words.**
- Special tokens (`<bos>`, `<eos>`, chat template markers, tool tags) are first-class vocabulary entries. Off-by-one mistakes here waste the context window or break chat formatting.

Chapter 18’s embedding table $$E \in \mathbb{R}^{V \times d}$$ is exactly layer 0 of this model: each token ID becomes a vector, then (in modern decoders) a positional method such as RoPE is applied inside attention (optional lesson **08-99**).

## 3. The context window

Self-attention in a vanilla Transformer mixes every pair of positions in the window. If the window length is $$L$$, one layer costs $$O(L^2)$$ in attention (plus cheap position-wise MLPs). So $$L$$ is both a **capability** limit and a **compute/memory** limit.

What the window actually bounds:

- The model can only condition on the last $$L$$ tokens of the concatenated prompt + generation (plus whatever you retrieve and paste — RAG, Chapter 18-99).
- Long documents must be truncated, summarized, or retrieved in chunks. “The model read my 200-page PDF” usually means *a retrieval system stuffed selected chunks into $$L$$*.
- During decode, keys and values for those $$L$$ positions are the natural thing to **cache** (lesson **26-04**).

Extensions (sliding windows, grouped-query attention, extra-long RoPE scaling) change the constants; they do not remove the idea that the model has a finite working tape.

## 4. Pretraining data, briefly

Pretraining corpora mix web text, books, code, and filtered crawls. You do not need a secret mix to understand the objective. You *do* need to remember:

- The loss rewards **fluency and corpus statistics**, not truth. Hallucination is not a separate bug; it is next-token sampling from an imperfect $$p_\theta$$.
- Data filters and deduplication are part of the model. Two LMs with the same architecture can behave differently because the corpus differed.
- Scaling (Chapter 25) relates loss to parameters, tokens, and compute. This chapter assumes that curve exists; it does not re-fit it.

## Key takeaways

- Pretraining = maximize next-token likelihood under a causal mask.
- The tokenizer defines the discrete alphabet; BPE and SentencePiece are the two practical stories.
- Context length $$L$$ is the working memory of attention — and the size of the KV-cache you will pay for at serve time.
- The same softmax-over-$$V$$ view that makes pretraining simple is what makes **classical logit distillation expensive** for LMs (continue in **26-05**).
