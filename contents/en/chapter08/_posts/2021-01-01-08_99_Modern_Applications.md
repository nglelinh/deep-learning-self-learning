---
layout: post
title: 08-99 Modern Applications and Updates (2022–2026)
chapter: '08'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter08
lesson_type: optional
---

# Optional: Transformer stacks that became LLMs

> This lesson is **optional**. It does **not** rewrite encoder/decoder theory or positional encodings. It maps the 2017 block onto the 2022–2026 LLM stack: RoPE, KV-cache, open weights, and serving.

A Transformer layer is still “communicate (attention) then compute (MLP).” After ChatGPT (late 2022), that block is the **default computer** for language, code, and many multimodal systems.

## 1. What actually changed in the block

Most decoder-only LMs (GPT-2 lineage → Llama / Mistral / Qwen / Gemma) use:

- **RoPE** ([Su et al., 2021/2023](https://arxiv.org/abs/2104.09864)) instead of absolute sin–cos added to embeddings,
- **pre-norm + RMSNorm + SwiGLU** (Chapter 02 optional),
- **GQA** and FlashAttention (Chapter 07 optional),
- **KV-cache** at decode time: keys/values for tokens $$1..t-1$$ are reused so each new token costs $$O(t)$$, not $$O(t^2)$$ matmuls from scratch.

RoPE rotates pairs of dimensions by an angle $$\theta_i t$$:

$$\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
\leftarrow
\begin{pmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}.$$

The theory lesson’s sin–cos PE is still the right first model; RoPE is the one you will see in 2025 checkpoints.

## 2. Concrete applications

### Open foundation LMs

[LLaMA](https://arxiv.org/abs/2302.13971), [Llama 2](https://arxiv.org/abs/2307.09288), Llama 3, [Mistral 7B](https://arxiv.org/abs/2310.06825), Qwen2, and Gemma 2 are the reference decoder stacks. Instruction-tuned variants are the ones products call.

### Serving and local inference

- [vLLM](https://github.com/vllm-project/vllm) — continuous batching + PagedAttention.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF quantized inference on CPUs and Apple Silicon.
- Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)) — a small draft model proposes tokens; the large model verifies in parallel.

### Multimodal Transformers

ViT ([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)) plus a language decoder (LLaVA, 2023; later GPT-4V-class systems) reuse the same block on image patches. Architecture homework stays in this chapter; vision-language products are the application.

## 3. Widely used software

- [Hugging Face transformers](https://github.com/huggingface/transformers) — model definitions and `generate()`.
- [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) — high-throughput serving.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — local GGUF.

## 4. Citations (2022–2026)

- [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) and [Llama 2 (2023)](https://arxiv.org/abs/2307.09288).
- [Mistral 7B (Jiang et al., 2023)](https://arxiv.org/abs/2310.06825) — sliding-window + GQA.
- [RoFormer / RoPE (Su et al., 2023 journal version)](https://arxiv.org/abs/2104.09864).
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192).

## 5. How this complements the core notes

Complete the encoder–decoder derivation and the toy PyTorch Transformer first. This lesson only names the **LLM product stack** built on that derivation.
