---
layout: post
title: 08-02 Transformer Explainer
chapter: '08'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter08
---

# Transformer Explainer

The equations in this chapter tell you *what* a Transformer computes. [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) lets you *watch* those computations on a real language model, with no installation and no GPU. It runs a live **GPT-2 (small)** model (124 million parameters, 12 stacked Transformer blocks) directly in the browser, so every number you see is the actual forward pass for the prompt you type.

![Transformer Explainer overview](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_summary.png)
*Overview of the interactive interface: a prompt is tokenized, then flows through embedding, multi-head self-attention, and the MLP. Source: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (MIT License).*

Open the tool first, then use this lesson as a guided map:

- **Live demo:** [poloclub.github.io/transformer-explainer](https://poloclub.github.io/transformer-explainer/)
- **Source code:** [github.com/poloclub/transformer-explainer](https://github.com/poloclub/transformer-explainer)
- **Paper (CHI 2026):** [Cho et al., “Transformer Explainer…”](https://doi.org/10.1145/3772318.3791725) ([PDF](https://poloclub.github.io/papers/26-chi-transformer-explainer.pdf), [arXiv:2408.04619](https://arxiv.org/abs/2408.04619))
- **Demo video:** [YouTube](https://www.youtube.com/watch?v=TFUc41G2ikY)

The site is built for full-page exploration (hovering on attention weights, expanding layers, moving sliders). This course therefore links out to the live demo rather than embedding it in a small frame.

## 1. Concept Overview

GPT-2 is a **decoder-only** Transformer: it predicts the next token given all tokens to its left. That is a slightly narrower architecture than the encoder–decoder map in “Attention Is All You Need,” which also has encoder self-attention and encoder–decoder cross-attention. The pieces you already studied still appear, just arranged for causal language modeling:

1. **Embedding** — tokens become vectors, plus a positional signal.
2. **Transformer blocks** — each block is multi-head **self-attention** (communicate across positions) followed by an **MLP** (compute per position), with residual connections.
3. **Output probabilities** — a linear head plus softmax turns the last hidden state into a distribution over the vocabulary, from which the next token is sampled.

GPT-2 (small) uses a 50,257-token vocabulary and 768-dimensional embeddings. The 12 blocks are identical in structure; the visualization typically expands **Block 1** and notes that eleven more follow. The in-browser model is adapted from Andrej Karpathy’s nanoGPT and executed with ONNX Runtime, so the values update live as you edit the prompt or the sampling controls.

![Annotated teaser of the full interface](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_teaser.jpg)
*Official teaser of the full interface (resized for the course site). (A) type a prompt and generate the next token; (B) change temperature and sampling; (C) expand the attention computation (dot product, scaling and causal mask, softmax). Source: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (MIT License).*

## 2. What each part of the visualization shows

### Embedding, tokenization, and positional encoding

Expand the **Embedding** column. The prompt is first **tokenized**. GPT-2 uses a subword vocabulary: whole words such as `Data` or `visualization` may be a single token, while `empowers` is split into `em` and `powers`. Each token has an integer id and a learned 768-dimensional **token embedding**.

![Embedding: tokens, token embeddings, and positional encodings](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_embedding.png)
*Four steps of embedding for the prompt “Data visualization empowers users to”: (1) tokens, (2) token embeddings with vocabulary ids, (3) positional encodings, (4) the final sum. Source: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (MIT License).*

**Positional encoding is added, not concatenated.** GPT-2 does *not* use the fixed sine/cosine table from the original Transformer paper; it learns a positional embedding matrix during training and adds it to the token embedding. The teaching point is the same as in the theory lesson: self-attention has no built-in order, so position must be injected somehow. After the sum, each row is a vector that carries both “what this token is” and “where it sits in the prompt.”

### Query, key, value, and multi-head self-attention

From those embeddings the model builds **query (Q)**, **key (K)**, and **value (V)** vectors. In GPT-2 (small) this is one matrix multiply: a $$(6, 768)$$ embedding matrix times a learned $$(768, 2304)$$ weight matrix, plus a bias, produces a $$(6, 2304)$$ tensor. The 2304 columns are simply $$3 \times 768$$ — Q, K, and V concatenated — which are then split across **12 attention heads**.

![Computing Q, K, and V from embeddings](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_qkv.png)
*Embeddings multiplied by the combined QKV weight matrix (plus bias) yield concatenated Q, K, and V. Source: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (MIT License).*

A useful analogy, also used by the tool: the **query** is the search you type, the **keys** are the titles you might match, and the **values** are the page contents you actually read. Attention is a soft lookup: each token asks “which earlier tokens are relevant to me?” and mixes their values accordingly.

![Scaled dot-product attention with a causal mask](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_attention.png)
*One attention head: Q–K dot products, scaling and causal mask, then softmax (and dropout during training). The lower-triangular structure is the decoder rule: a token must not attend to the future. Source: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (MIT License).*

Walk the three matrices from left to right:

1. **Dot product** $$QK^{\top}$$ measures raw similarity between every query and every key.
2. **Scaling and mask** divides by $$\sqrt{d_k}$$ (the same $$1/\sqrt{d_k}$$ factor from Chapter 07) and applies a **causal mask**, setting future positions to $$-\infty$$ so they cannot be attended to when predicting the next token.
3. **Softmax** turns each row into a probability distribution over allowed positions. Those weights multiply $$V$$ to produce the head output.

You can switch heads (1 of 12) and hover on tokens to see which positions a given token attends to. Different heads often specialize: some pick up local syntax, others longer semantic links. The head outputs are concatenated and projected before the residual stream continues.

### MLP

After attention has *routed* information between tokens, the **MLP** (a two-layer feed-forward network) *refines* each token on its own. In GPT-2 (small) the hidden width expands from 768 to $$4 \times 768 = 3072$$ with a GELU activation, then projects back to 768. Residual connections sit around the block so the original representation can pass through.

![MLP expansion to 3072 dimensions with residual path](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_mlp.png)
*The position-wise MLP expands each token to a 3072-dimensional hidden vector, then writes back into the residual stream. Source: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (MIT License).*

### Output probabilities, temperature, and sampling

After all 12 blocks, a linear layer maps the last position to **logits** — one score per vocabulary item. Softmax converts logits into a probability distribution over the next token.

Temperature is a surprisingly small operation with a large effect: the logits are divided by $$T$$ *before* softmax.

- $$T = 1$$: the distribution is unchanged.
- $$T < 1$$: the distribution sharpens; the model looks more confident and deterministic.
- $$T > 1$$: the distribution flattens; lower-ranked tokens become more likely, which people often describe as more “creative” (and more error-prone).

Two common filters then restrict what may be sampled:

- **Top-k:** keep only the $$k$$ tokens with the highest probability; the rest are set to $$-\infty$$ and receive probability 0.
- **Top-p** (nucleus sampling): keep the smallest set of tokens whose cumulative probability exceeds a threshold $$p$$.

![Temperature, top-k sampling, and next-token probabilities](/deep-learning-self-learning/img/chapter_img/chapter08/transformer_explainer_softmax.png)
*With temperature 0.8 and top-k = 5, logits are scaled, tokens outside the top five are zeroed out, and softmax yields the next-token distribution. Here “visualize” leads at 54.67% for the prompt “…users to”. Source: Polo Club of Data Science, Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (MIT License).*

In the figure, a temperature of 0.8 turns a logit of $$-135.91$$ into a scaled value of $$-169.89$$ (exactly $$-135.91 / 0.8$$). After top-k, `quickly`, `explore`, and `find` are no longer candidates. Sampling is not the same as always taking $$\arg\max$$: even at this setting, “create” still has about 21% probability.

## 3. Guided exercise

Open the [live demo](https://poloclub.github.io/transformer-explainer/) and work through the following, writing short answers as you go. The default prompt “Data visualization empowers users to” is a good starting point.

1. **Read the tokenization.** Expand Embedding. Which words stay as one token, and which word is split? Why might a subword vocabulary do that?
2. **Confirm the positional sum.** Check that each final embedding is the token embedding plus a position vector (0, 1, 2, …). What would go wrong if this addition were skipped? (The ablation in **08-01-02-02** is the coding counterpart of this question.)
3. **Inspect one attention head.** Open a head in Block 1, hover over the last token, and note which earlier tokens receive the largest weights. Switch to another head. Do the two heads agree?
4. **Find the causal mask.** In the “Scaling · Mask” matrix, identify the positions that are blocked. Why must a decoder-only model hide the future even during a visualization of a *prompt* that you already typed?
5. **Sweep temperature.** Set temperature near 0.1, then near 1.5, without changing the prompt. Describe how the probability bar chart changes. Which setting would you use if you needed a reliable completion versus a more diverse one?
6. **Compare top-k and top-p.** With temperature fixed (try 0.8), set top-k to 1, then to 5, then switch to top-p. What happens to tokens that fall outside the cutoff? How is that visible in the Softmax column?
7. **Use your own sentence.** Type a short English prompt (or a mix of English and Vietnamese) and press Generate. Does the next-token list match your expectation? If not, remember that this is GPT-2 small, not a modern instruction-tuned LLM — the architecture is the same family, the capacity is not.

## 4. Related Concepts

- **Chapter 07** — scaled dot-product attention, multi-head attention, and causal masks are exactly the matrices the explainer draws.
- **08-01-01 Transformer Theory** — positional encodings and the full block (attention + FFN + residual + norm). Note the GPT-2 detail: learned absolute PE rather than sine/cosine PE.
- **08-01-02-02 Demos, papers, and pitfalls** — a tiny model you train yourself; use the explainer when you want a *pretrained* GPT-2 you can inspect without writing a training loop.
- **Chapter 26** — tokenization, decoding, and sampling at LLM scale; this lesson is the visual preview of those decoding choices.

## 5. Fundamental Papers

1. **[Transformer Explainer (Cho et al., CHI 2026)](https://doi.org/10.1145/3772318.3791725)**  
   The interactive system this lesson is built around: a live in-browser GPT-2 with an overview of data flow and on-demand views of embedding, attention, and next-token probabilities. Also on [arXiv:2408.04619](https://arxiv.org/abs/2408.04619).

2. **[Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**  
   The decoder-only model family running inside the visualization.

3. **[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)**  
   The original encoder–decoder Transformer; compare it with the decoder-only stack you see in the explainer.

## 6. Key Takeaways

1. A text-generative Transformer is a next-token predictor: embedding → stacked blocks (attention + MLP) → softmax over the vocabulary.
2. Attention weights are not decoration — they are the normalized, masked similarities that decide how each token gathers context.
3. Temperature, top-k, and top-p do not retrain the model; they only reshape the same logits into a different sampling distribution.
4. GPT-2 small is small enough to run in a browser and large enough to make the architecture tangible. Treat it as a microscope for Chapter 08, not as a substitute for a modern LLM.

## Attribution and license

Transformer Explainer was created by Aeree Cho, Grace C. Kim, Alexander Karpekov, Alec Helbling, Jay Wang, Seongmin Lee, Benjamin Hoover, and Polo Chau at the Georgia Institute of Technology (Polo Club of Data Science). The software is released under the [MIT License](https://github.com/poloclub/transformer-explainer/blob/main/LICENSE). The figures in this lesson are copied or resized from the project’s official `static/preview` and `static/article_assets` files in the [public repository](https://github.com/poloclub/transformer-explainer); the teaser was resized from 8135×3895 to 2400×1149 for the course site. Please cite the CHI 2026 paper if you use the tool in written work:

```bibtex
@inproceedings{cho2026transformer,
  title={Transformer Explainer: Learning LLM Transformers with Interactive Visual Explanation and Experimentation},
  author={Cho, Aeree and Kim, Grace C and Karpekov, Alexander and Lee, Seongmin and Helbling, Alec and Hoover, Benjamin and Wang, Zijie J and Kahng, Minsuk and Chau, Duen Horng},
  booktitle={Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems},
  pages={1--21},
  year={2026}
}
```

## Video tham khảo

- **xem trước** [Transformer Explainer Demo](https://www.youtube.com/watch?v=TFUc41G2ikY) — Polo Club / Georgia Tech, English. A short walk through the live GPT-2 visualization this lesson uses.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, English. The same embedding → attention → MLP → softmax story, in animation rather than in a live model.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, English. The nanoGPT implementation that the in-browser model is derived from.
