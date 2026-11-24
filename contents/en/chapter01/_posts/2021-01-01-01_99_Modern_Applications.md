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

> *In 2016, "doing deep learning" meant collecting a custom dataset and training a ResNet from random weights. By 2022–2026, the default workflow inverted: you start from a massive pretrained foundation model, evaluate zero-shot or few-shot performance, and only fine-tune—often with parameter-efficient adapters—when the data proves it necessary.*

## Background Knowledge Map: The Foundation-Model Workflow (2022–2026)

1. **Pretrain–Adapt Paradigm**: Shifts compute from bespoke per-task training to massive self-supervised pretraining followed by lightweight adaptation (prompting, RAG, LoRA, DPO/RLHF).
2. **Chinchilla & Inference-Optimal Scaling Laws**: Balances model parameters $$N$$ and training tokens $$D$$ under a fixed compute budget $$C \approx 6ND$$, revealing that many earlier models were severely undertrained relative to their size.
3. **Open-Weight Ecosystem & Quantization**: Enables running billion-parameter models locally via 4-bit/8-bit weight quantization (AWQ, GPTQ, GGUF) and efficient KV-cache serving engines (vLLM, TensorRT-LLM).
4. **Evaluation-Driven Iteration**: Treats rigorous benchmark suites, contamination checks, and slice-level error metrics as the primary bottleneck once model capabilities outpace naive accuracy metrics.

### Step-by-Step Worked Example (Read Slowly): Applying the Chinchilla Compute-Optimal Rule ($$C \approx 6ND$$)

Suppose your lab has a fixed pretraining compute budget of $$C = 1.2 \times 10^{21}$$ FLOPs and follows the Chinchilla ratio of roughly $$D \approx 20 N$$ training tokens per parameter.

1. **Substitute $$D = 20N$$ into the Transformer Training FLOP Formula $$C \approx 6ND$$**:
   $$C \approx 6 N (20 N) = 120 N^2$$
2. **Solve for Optimal Parameter Count $$N$$**:
   $$N^2 = \frac{1.2 \times 10^{21}}{120} = 1.0 \times 10^{19} \implies N = \sqrt{10 \times 10^{18}} \approx 3.16 \times 10^9\text{ parameters (3.16B)}$$
3. **Solve for Optimal Training Token Count $$D$$**:
   $$D = 20 N \approx 20 \times (3.16 \times 10^9) = 6.32 \times 10^{10}\text{ tokens (63.2B tokens)}$$
4. **Sanity Check**:
   $$6 \times (3.16 \times 10^9) \times (6.32 \times 10^{10}) \approx 1.198 \times 10^{21}\text{ FLOPs}$$
   Notice that if you instead trained a 30B model on the same budget, it would only see $$D = \frac{1.2\times 10^{21}}{6 \times 3\times 10^{10}} = 6.67\text{B}$$ tokens—severely undertrained and slower at inference!

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

## Video tham khảo

- **xem trước** [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) — 3Blue1Brown, English. A short account of pretraining, transformers, and RLHF, which is the 2022–2026 default this note describes.
- [\[1hr Talk\] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Andrej Karpathy, English. The longer map of the same workflow: data, training, and what a deployed model is.
- [MIT 6.S191 (2024): Language Models and New Frontiers](https://www.youtube.com/watch?v=N1fbskTpwZ0) — Alexander Amini, English. MIT's 2024 frontier lecture is the 'what shipped recently' companion to the citations in this note.
