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
