---
layout: post
title: 08-99 Modern Applications and Updates (2022–2026)
chapter: '08'
order: 8
owner: Deep Learning Course
lang: en
categories:
- chapter08
lesson_type: optional
---

Open the source code of Llama 3, Qwen 2.5, Mistral, or Gemma and compare it line-by-line against the 2017 *Attention Is All You Need* diagram: the macro skeleton is unmistakably the same decoder-only Transformer, yet **five targeted micro-upgrades** replaced almost every original sub-layer. Pre-RMSNorm replaced Post-LayerNorm, Rotary Positional Embeddings (RoPE) replaced additive sinusoids, SwiGLU replaced ReLU FFNs, and Grouped-Query Attention (GQA) replaced standard MHA—together enabling stable 70B–600B+ parameter scaling and 128K+ context windows.

## Key ideas: Five Modern Upgrades Inside Frontier LLM Blocks

- **Pre-RMSNorm**: Dropping mean-centering from LayerNorm leaves $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot g$$—matching LayerNorm's training stability while cutting normalization overhead by ~$$10\text{–}40\%$$.
- **Rotary Positional Embeddings (RoPE)**: Instead of adding position vectors to embeddings, RoPE rotates each 2D pair of $$q_m, k_n$$ by angles $$m\theta_i$$ and $$n\theta_i$$, so their dot product $$q_m^\top k_n$$ depends *purely* on the relative distance $$(m - n)$$!
- **SwiGLU Gated Feed-Forward Layers**: Replacing `W2(ReLU(W1 x))` with `W_down(SiLU(W_gate x) * (W_up x))` adds multiplicative gating; to keep parameter counts equal to a $$4d$$ ReLU FFN, intermediate width is scaled to $$\approx \frac{8}{3} d_{\text{model}}$$.
- **Multimodal & Local Serving Ecosystem**: The exact same Transformer backbone now fuses ViT image patches, audio tokens, and text tokens inside unified models served locally via quantized GGUF/llama.cpp or at scale via vLLM/SGLang.

### Worked example: Why SwiGLU Uses Hidden Width $$\frac{8}{3} d$$ Instead of $$4 d$$

A standard Transformer FFN has **2** matrices ($$W_1, W_2$$) with hidden width $$d_{\text{ff}} = 4 d$$, giving parameter count:
$$P_{\text{ReLU-FFN}} = 2 \cdot d \cdot (4 d) = 8 d^2.$$
A SwiGLU FFN has **3** matrices ($$W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$$) with hidden width $$d_{\text{swiglu}}$$:

1. **Equate parameter budgets**:
   $$3 \cdot d \cdot d_{\text{swiglu}} = 8 d^2 \implies d_{\text{swiglu}} = \frac{8}{3} d \approx 2.667\, d.$$
2. **Concrete Llama-style example ($$d = 4{,}096$$)**:
   $$\frac{8}{3} \times 4{,}096 = 10{,}922.67 \xrightarrow{\text{round to multiple of 256}} 11{,}008.$$
3. **Check RoPE relative-distance property in 2D**:
   If $$R(\theta)$$ is a $$2 \times 2$$ rotation matrix, then $$R(m\theta)^\top R(n\theta) = R((n - m)\theta)$$. Thus $$(R(m\theta) q)^\top (R(n\theta) k) = q^\top R((n - m)\theta) k$$—exact relative position invariance with zero extra parameters!

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

Complete the encoder–decoder derivation and the toy PyTorch Transformer first. This lesson only names the **LLM product stack** built on that derivation. Formulas and Mermaid for RMSNorm → attention + RoPE → SwiGLU, plus parameter / FLOP accounting, live in **26-07**.

## Video tham khảo

- **xem trước** [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, English. The baseline block this note says has since changed in detail but not in outline.
- [How might LLMs store facts | Deep Learning Chapter 7](https://www.youtube.com/watch?v=9-Jl0dxWQs8) — 3Blue1Brown, English. The MLP-as-memory view of what a modern decoder block still computes.
- [How FlashAttention Accelerates Generative AI Revolution](https://www.youtube.com/watch?v=gBMO1JZav44) — Jia-Bin Huang, English. The attention implementation change (FlashAttention) the note treats as a systems upgrade, not a new formula.
