---
layout: post
title: 05-99 Modern Applications and Updates (2022–2026)
chapter: '05'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter05
lesson_type: optional
---

Conventional wisdom in 2020 declared recurrent neural networks obsolete in the shadow of the Transformer. Yet Transformers carry a stubborn physical tax: their KV-cache grows linearly with context length and prefill attention scales quadratically $$O(T^2)$$. Between 2023 and 2026, a wave of linear-time recurrent architectures—most notably **Mamba** (Selective State Space Models) and **RWKV**—returned to the exact state-space recursion of Chapter 05, upgrading it with input-dependent gating and hardware-aware parallel scans so models can train in parallel like Transformers while generating tokens in $$O(1)$$ constant memory like an RNN.

## Key ideas: The Modern Recurrent Revival (SSMs, Mamba, RWKV)

- **Linear Time-Invariant & Selective State Space Models**: Classical continuous-time control theory ($$\dot{h}(t) = A h(t) + B x(t)$$) discretized into $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$ turns recurrence into an associative operation. Making $$\bar{A}_t, \bar{B}_t$$ functions of $$x_t$$ (Mamba's selectivity) lets the state selectively remember or ignore tokens.
- **Parallel Associative Scan (Blelloch Prefix Sum)**: Because linear recurrence is associative, a sequence of length $$T$$ can be computed in $$O(\log T)$$ parallel depth on GPUs during training—breaking the sequential training bottleneck of classical nonlinear RNNs.
- **Constant-Memory Autoregressive Inference**: While a Transformer must attend over an ever-growing KV-cache of $$O(T)$$ past tokens, an SSM or RWKV updates a fixed-size state tensor in $$O(1)$$ time and $$O(1)$$ memory per generated token.
- **Hybrid Transformer–SSM Architectures**: Models like Jamba and Zamba interleave Mamba layers (cheap long-range compression) with periodic Attention layers (exact needle-in-a-haystack recall), combining the strengths of both families.

### Worked example: Why Linear Recurrence Parallelizes via Associativity

Consider a scalar selective SSM step $$h_t = a_t h_{t-1} + b_t$$ with $$h_0 = 0$$ and 4 time steps:
$$(a_1, b_1) = (0.5, 2),\; (a_2, b_2) = (0.5, 4),\; (a_3, b_3) = (0.2, 1),\; (a_4, b_4) = (1.0, 3).$$

1. **Define the associative composition operator $$\oplus$$** on pairs $$(a, b)$$:
   $$(a_i, b_i) \oplus (a_j, b_j) = (a_j a_i,\; a_j b_i + b_j).$$
2. **Parallel Stage 1 (combine pairs $$(1,2)$$ and $$(3,4)$$ simultaneously)**:
   - $$(1 \to 2)$$: $$(0.5 \times 0.5,\; 0.5 \times 2 + 4) = (0.25, 5.0) \implies h_2 = 5.0$$.
   - $$(3 \to 4)$$: $$(1.0 \times 0.2,\; 1.0 \times 1 + 3) = (0.2, 4.0)$$.
3. **Parallel Stage 2 (combine $$(1 \to 2)$$ with $$(3 \to 4)$$ in one step)**:
   - $$(1 \to 4)$$: $$(0.2 \times 0.25,\; 0.2 \times 5.0 + 4.0) = (0.05, 5.0) \implies h_4 = 5.0$$.
4. **Verify sequentially**:
   $$h_1 = 2,\; h_2 = 0.5(2)+4 = 5,\; h_3 = 0.2(5)+1 = 2,\; h_4 = 1.0(2)+3 = 5.$$
   Exact match! A length-$$T$$ sequence collapses from $$T$$ sequential steps to $$\lceil \log_2 T \rceil$$ parallel GPU rounds.

# Optional: recurrent sequence models after Transformers

> This lesson is **optional**. It does **not** replace vanilla RNN theory or the character-LM implementation. It explains where recurrence still appears — and which 2023–2025 models revived the “scan over time” idea.

A vanilla RNN is

$$\mathbf{h}_t = \sigma(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b}).$$

Transformers removed the left-to-right *training* loop. After 2022, **state-space and linear-recurrent** models brought a loop back, but as a *parallel scan* with linear cost in sequence length.

## 1. Why RNNs did not vanish

Transformers cost $$O(T^2)$$ memory/time in sequence length $$T$$ (before approximations). Streaming speech, control, and very long genomics still want $$O(T)$$ steps and a compact state. That is the same motivation as this chapter — with better parameterizations.

## 2. Concrete applications

### Structured state space models and Mamba

[Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) (Mamba) use a selective SSM: the discrete state update is a recurrence

$$\mathbf{h}_t = \overline{A}_t \mathbf{h}_{t-1} + \overline{B}_t \mathbf{x}_t, \quad \mathbf{y}_t = C_t \mathbf{h}_t,$$

where $$\overline{A}_t$$ depends on the input (the “selection” mechanism). Training uses a parallel scan, not a Python `for` loop. This is the 2023–2025 answer to “can recurrence compete with attention on language?”

### RWKV

[Peng et al., 2023](https://arxiv.org/abs/2305.13048) mix a linear attention-style time mix with a channel mix, so inference is RNN-like (constant state) while training is parallel. Used in open chat models when KV-cache memory is the bottleneck.

### Where classical RNNs remain

On-device keyword spotting, some forecasting baselines, and teaching labs still use `nn.RNN` / `nn.GRU`. Production ASR moved to Conformer / Whisper (Chapters 08 and 19), not vanilla Elman RNNs.

## 3. Widely used software

- [state-spaces/mamba](https://github.com/state-spaces/mamba) — official Mamba kernels.
- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — RWKV training and inference.
- PyTorch `nn.RNN` — still the right API for the core homework.

## 4. Citations (2022–2026)

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752).
- [RWKV: Reinventing RNNs for the Transformer Era (Peng et al., 2023)](https://arxiv.org/abs/2305.13048).
- [Efficiently Modeling Long Sequences with Structured State Spaces (Gu et al., 2022)](https://arxiv.org/abs/2111.00396) — S4, the SSM precursor.

## 5. How this complements the core notes

Implement the vanilla RNN and the pitfalls lesson first. This note only contrasts **modern linear recurrences** with Transformers — it does not replace the unfolding / BPTT derivation.

## Video tham khảo

- **xem trước** [MAMBA and State Space Models explained | SSM explained](https://www.youtube.com/watch?v=vrF3MtGwD0Y) — AI Coffee Break with Letitia, English. State-space models are the 2023–2024 reason a recurrence did not disappear behind transformers.
- [Recurrent Neural Networks (RNNs), Clearly Explained!!!](https://www.youtube.com/watch?v=AsNTP8Kwu80) — StatQuest with Josh Starmer, English. The hidden state those newer recurrences still keep.
- [MIT 6.S191 (2024): Recurrent Neural Networks, Transformers, and Attention](https://www.youtube.com/watch?v=dqoEU9Ac3ek) — Alexander Amini, English. Where a 2024 course still teaches RNNs next to attention.
