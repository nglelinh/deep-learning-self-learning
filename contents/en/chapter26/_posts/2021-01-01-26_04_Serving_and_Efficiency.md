---
layout: post
title: 26-04 Serving and Efficiency for LLMs
chapter: '26'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter26
---

# Serving and Efficiency for LLMs

A pretrained, instructed, aligned decoder is still expensive **per generated token**. Training is a one-off (or rare) bill; serving is the bill that scales with users. This lesson lists the knobs you should name before you reach for distillation.

The compression math — quantization grids, pruning scores, classical KD — stays in **Chapter 23**. The LLM-specific recipes (GPTQ, AWQ, GGUF, speculative decoding) are in **23-99**. Here we only answer: *what is expensive at decode time, and when is a smaller student the right next step?*

## 1. Prefill vs decode

Autoregressive generation has two phases:

**Prefill.** The full prompt (system + history + user + retrieved text) is consumed in parallel. You pay attention over the prompt length once and write a **KV-cache**: for each layer and each prompt token, store the key and value vectors.

**Decode.** You emit one new token at a time. Without a cache you would recompute attention from scratch on the growing prefix (roughly $$O(t^2)$$ work by step $$t$$). With a cache you append one new $$k,v$$ and attend against the stored prefix — about $$O(t)$$ attention work per new token, plus the MLP.

That is the same KV-cache already named in **08-99**. Serving systems (vLLM, SGLang, llama.cpp) are mostly: *manage many caches, batch the matrix multiplies, and overlap I/O*.

Memory, not just FLOPs, often dominates. The cache for a long conversation is

$$\mathrm{bytes} \approx 2 \cdot L \cdot n_{\mathrm{layers}} \cdot n_{\mathrm{kv\_heads}} \cdot d_{\mathrm{head}} \cdot b$$

(the factor 2 is keys and values; $$b$$ is bytes per element; grouped-query attention shrinks $$n_{\mathrm{kv\_heads}}$$). This is why long context is a **product** decision, not only a quality decision.

## 2. Quantization is the first cheap win

Storing weights in 16-bit instead of 32-bit, then 8-bit or 4-bit, cuts memory and can cut bandwidth-bound decode time. Chapter 23 gives the rounding picture; **23-99** names GPTQ / AWQ / GGUF as the recipes people actually run on 7B–70B decoders.

Quantization **does not** train a new model. It approximates the same $$\theta$$ with fewer bits. Use it when:

- you own or may redistribute that checkpoint,
- quality drop is acceptable after a quick eval,
- you need the *same* behavior at lower RAM (laptops, a single GPU, edge).

If 4-bit still misses a latency or cost target, you need fewer layers/width or fewer active parameters (MoE routing — also in 23-99) — or a **student**.

## 3. Other serving levers (so you do not overfit on distillation)

- **Batching.** Continuous batching fills GPU gaps as sequences finish at different times.
- **Speculative decoding.** A cheap draft model proposes several tokens; the large model verifies a prefix in one parallel forward ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)). Same distribution if implemented correctly; extra complexity.
- **Prompt caching / prefix sharing.** Many requests share a system prompt; reuse that prefill.
- **Retrieval vs long context.** Stuffing a book into $$L$$ is often slower and worse than retrieving (18-99).
- **Adapters (LoRA / QLoRA).** Specialize without serving a full extra copy of every expert (Chapter 15 optional).

None of these *transfer knowledge into a smaller architecture*. They make the teacher cheaper to run.

## 4. When to distill

Reach for distillation (**26-05**) when at least one of these is true:

1. **Architecture must shrink.** Quantization cannot delete layers. A 70B teacher that must become a 7–8B (or a mobile) student needs a new $$\theta$$.
2. **You want a cheaper product tier** with similar *behavior*, not a bit-exact clone — in-house, on a teacher you are allowed to train from.
3. **You will fine-tune many specialists.** Distill a general student once, then LoRA each specialist.
4. **White-box internals are available** and you care about matching hidden states or token distributions, not only text.

Do **not** distill first if you have not tried 4-bit weights, a sane context limit, and batching. Distillation is a training project; quantization is often a conversion flag.

Also do not distill a teacher you are not licensed or contracted to train from. The next lesson separates ordinary in-house / open-license distillation from harvesting an API you do not control.

## Key takeaways

- Decode cost = MLP + attending over a growing **KV-cache**.
- Quantization and batching are the default efficiency tools (Chapter 23 / 23-99).
- Distillation is the tool for a *new, smaller* network that *imitates* a teacher.
- Decide “cache / quantize / retrieve” before “train a student.”
