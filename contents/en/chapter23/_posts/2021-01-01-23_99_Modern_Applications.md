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
