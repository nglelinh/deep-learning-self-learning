---
layout: post
title: 23-99 Modern Applications and Updates (2022–2026)
chapter: '23'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter23
lesson_type: optional
---

Naive round-to-nearest 4-bit quantization wrecks a 70B LLM because a tiny handful of activation channels (~0.1% of features) carry massive outlier magnitudes that blow up quantization scales. Between 2023 and 2026, efficient LLM serving conquered that outlier problem through Hessian-guided quantization (GPTQ), activation-aware channel scaling (AWQ), portable consumer formats (GGUF), Sparse Mixture-of-Experts (MoE), and lossless speculative decoding.

## Key ideas: Modern LLM Efficiency Stack (2022–2026)

- **Pillar 1 — Outlier-Aware Weight Quantization (GPTQ & AWQ)**: Compensates quantization rounding error across remaining weights using inverse Hessian updates (GPTQ) or rescales the ~1% salient weight channels protected by large input activations (AWQ).
- **Pillar 2 — Consumer Edge Formats (GGUF / K-Quants)**: Packs group-wise quantized blocks (e.g., `Q4_K_M`, 32-weight super-blocks with FP16 scales) into a single memory-mapped file for fast CPU/Apple-Silicon inference (`llama.cpp`).
- **Pillar 3 — Lossless Speculative Decoding**: Uses a tiny draft model to propose $K$ tokens rapidly, then verifies all $K$ in a *single* parallel forward pass of the large target model via rejection sampling—preserving the **exact** target distribution.
- **Pillar 4 — Sparse Mixture-of-Experts (MoE)**: Decouples total knowledge capacity (e.g., 8×7B or 671B total params) from per-token compute (routing each token to only top-2 experts).

### Worked example: Effective Bits-Per-Weight (BPW) of Group-Wise INT4 & Speculative Speedup

**Part A — Group-Wise INT4 Effective Bit-Width**: Suppose weights are stored in 4-bit integers in groups of $g = 128$ weights, where each group stores one FP16 scale ($16$ bits) and one FP16 zero-point ($16$ bits).
1. Total bits per 128 weights: $128 \times 4 + 16 + 16 = 512 + 32 = 544\text{ bits}$.
2. Effective bits per weight: $544 / 128 = 4.25\text{ BPW}$ (only $0.25$ bits overhead for huge accuracy gains!).

**Part B — Speculative Decoding Speedup**: Suppose the target model costs $T = 40\text{ ms}$ per forward pass, a draft model costs $t_d = 4\text{ ms}$ per token, we draft $K = 4$ tokens per round, and on average $3.0$ drafted tokens are accepted (yielding $3.0 + 1 = 4.0$ verified target tokens per round, including the bonus token).
1. Time per round: $4 \times 4\text{ ms} + 40\text{ ms} = 56\text{ ms}$ for $4.0$ tokens $\implies 14.0\text{ ms/token}$.
2. Speedup vs. standard target decoding ($40\text{ ms/token}$): $40 / 14 = 2.86\times$ faster with **zero** quality loss!
3. **Takeaway**: Combining group-wise 4.25-BPW quantization with speculative decoding turns a sluggish 70B server model into a responsive interactive assistant.

# Optional: efficient LLMs — GPTQ, AWQ, GGUF, and speculative decoding

> This lesson is **optional**. It does **not** replace pruning, quantization math, distillation, or MobileNet/EfficientNet in the core note. Those still matter for CNNs and edge vision. This lesson is the **LLM serving** efficiency stack of 2022–2026.

INT8

$$w_q = \mathrm{round}\big((w-z)/s\big)$$

is the same idea. The new work is **weight-only 4-bit** for Transformers and **decoding algorithms** that cut sequential steps.

## 1. Post-training quantization for LMs

- [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323) — layer-wise second-order PTQ; foundation of many Hub `gptq` models.
- [AWQ (Lin et al., 2024)](https://arxiv.org/abs/2306.00978) — protect *salient* channels (activation-aware) before 4-bit rounding; strong on instruction-tuned LMs.
- [GGUF / llama.cpp](https://github.com/ggerganov/llama.cpp) — k-quants (`Q4_K_M`, …) for CPU and Apple Silicon.

These complement, not duplicate, the chapter’s general quantizer: they are **recipe + kernel** for attention blocks.

## 2. Concrete applications

### LoRA + quant (QLoRA)

Already sketched in Chapter 15 optional: 4-bit base, 16-bit adapters. Serving then merges or swaps adapters. Efficiency *and* transfer.

### Speculative decoding

[Leviathan et al., 2023](https://arxiv.org/abs/2211.17192): a cheap draft model proposes $$K$$ tokens; the large model accepts a prefix in one parallel forward. Wall-clock speedup without changing $$\pi$$. vLLM and many APIs ship this.

### MoE routing as efficiency

[Mixtral (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) activates a subset of experts per token: more parameters, similar FLOPs. Complementary to compression: *sparse compute*, not fewer stored weights.

## 3. Widely used software

- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) and [vllm-project/vllm](https://github.com/vllm-project/vllm).

## 4. Citations (2022–2026)

- [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323).
- [AWQ: Activation-aware Weight Quantization (Lin et al., 2024)](https://arxiv.org/abs/2306.00978).
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192).
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088).

## 5. How this complements the core notes

Keep the pruning / KD / MobileNet derivations. This lesson only adds **LLM-specific PTQ, GGUF, speculative decoding, and MoE** — the tools you use when the “model” is a 7B–70B Transformer.

## Video tham khảo

- **xem trước** [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, English. Post-training quantization of language models is this note.
- [QLoRA: Efficient Finetuning of Quantized LLMs | Tim Dettmers](https://www.youtube.com/watch?v=fQirE9N5q_Y) — AMLD Intelligence Summit, English. QLoRA, the fine-tune that stays in 4-bit, from Tim Dettmers.
- [Knowledge Distillation: How LLMs train each other](https://www.youtube.com/watch?v=jrJKRYAdh7I) — Julia Turc, English. Distillation as the other way to get a smaller deployed model.
