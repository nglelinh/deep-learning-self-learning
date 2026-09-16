---
layout: post
title: 01-99 Modern Applications and Updates (2022–2026)
chapter: '01'
order: 8
owner: Deep Learning Course
lang: en
categories:
- chapter01
lesson_type: optional
---

# Optional: what “doing deep learning” means after foundation models

> This lesson is **optional**. It does **not** replace the introductions to deep learning, MNIST, or the course guide. It updates the *practice* of the field after foundation models became the default starting point.

Chapter 01’s core idea is that deep models learn hierarchical features from data. From 2022 onward, most applied work starts from a **pretrained** model and a serving stack, not from a blank MNIST net — but the same questions (when DL helps, what data you need, what can go wrong) still decide success.

## 1. The new default workflow

A typical 2024–2026 project looks like:

1. Choose a foundation model (vision, language, speech, or multimodal).
2. Adapt it with prompting, retrieval, or parameter-efficient fine-tuning (Chapter 15).
3. Evaluate on a held-out set *and* on safety / cost constraints.

That is still supervised or self-supervised learning. The change is the **initialization** and the **interface** (APIs, tokenizers, chat templates), not the definition of a neural net.

## 2. Concrete applications

### Data science copilots

Tabular and SQL workflows now wrap an LLM around a classical pipeline (pandas, scikit-learn, warehouse queries). The model does not replace the loss or the split; it writes code and explanations. Hugging Face [transformers](https://github.com/huggingface/transformers) plus [datasets](https://github.com/huggingface/datasets) is the common research glue.

### Compute-optimal training

[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) showed that many GPT-3-scale models were **undertrained**: for a fixed FLOP budget $$C$$, the compute-optimal parameter count $$N$$ and token count $$D$$ scale together. Applied teams now ask “how many tokens per parameter?” before buying GPUs.

### Open model ecosystem

Meta’s [LLaMA](https://arxiv.org/abs/2302.13971) / [Llama 2](https://arxiv.org/abs/2307.09288) / [Llama 3](https://ai.meta.com/blog/meta-llama-3/) and later open families (Mistral, Qwen, Gemma) made it normal to run a capable LM on a single workstation with [llama.cpp](https://github.com/ggerganov/llama.cpp) or [vLLM](https://github.com/vllm-project/vllm). The MNIST walkthrough still teaches the loop; these stacks are where the loop is deployed.

## 3. Widely used software

- [PyTorch](https://pytorch.org/) 2.x with `torch.compile` as the default accelerator path.
- [Hugging Face Hub](https://huggingface.co/) — model cards, tokenizers, and evaluation harnesses.
- [JAX](https://github.com/google/jax) / Flax — research training of large models at Google-scale labs.

## 4. Citations (2022–2026)

- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — Chinchilla: scale data with model size.
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) — openly described foundation LM that reset the ecosystem.
- [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — dialogue / RLHF-tuned open weights.

## 5. How this complements the core notes

The “What is deep learning” and MNIST lessons still define the field. This note only updates *where* a newcomer starts in 2026: pretrained weights, hubs, and scaling-law budgeting — not a new definition of a neural network.
