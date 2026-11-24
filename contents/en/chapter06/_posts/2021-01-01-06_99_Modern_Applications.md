---
layout: post
title: 06-99 Modern Applications and Updates (2022–2026)
chapter: '06'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter06
lesson_type: optional
---

Just when many assumed LSTMs were permanently relegated to historical footnotes, Sepp Hochreiter's own research group published **xLSTM (Extended Long Short-Term Memory)** in 2024—asking a sharp scientific question: *what happens if we fix the three original limitations of the 1997 LSTM?* By replacing sigmoid gates with exponential gating (stabilized in log-space) and upgrading the scalar cell state into a **matrix-valued memory** ($$C_t \in \mathbb{R}^{d \times d}$$ updated via outer products $$v_t k_t^\top$$), xLSTM scales to billions of parameters while preserving linear-time inference—proving gated recurrence still has vital roles in streaming audio, finance, edge robotics, and hybrid LLM stacks.

## Background Knowledge Map: Modern Gated Recurrence (xLSTM & Streaming Systems)

- **Exponential Gating (sLSTM)**: Classical sigmoid gates saturate in $$(0, 1)$$, making it hard to revise a strongly stored memory value. Exponential gates $$i_t = \exp(\tilde{i}_t)$$ and $$f_t = \exp(\tilde{f}_t)$$ (stabilized with a running log-normalizer $$m_t$$) allow decisive memory overrides when a surprising token arrives.
- **Matrix Memory & Associative Retrieval (mLSTM)**: Replacing the vector cell $$c_t \in \mathbb{R}^d$$ with a covariance-like matrix $$C_t = f_t C_{t-1} + i_t v_t k_t^\top$$ expands storage capacity from $$O(d)$$ to $$O(d^2)$$ and lets queries retrieve stored values via $$C_t q_t$$—bridging LSTMs directly to linear attention!
- **Streaming & Edge Domains**: In real-time speech recognition, industrial sensor telemetry, and ultra-low-latency tabular event streams, compact LSTM/GRU models still outperform Transformers on hardware with strict memory and causal latency budgets.
- **Parallelizable Matrix Recurrence**: Because mLSTM removes hidden-to-hidden recurrent mixing inside the cell update, it trains in parallel across sequence length just like Transformers and Mamba.

### Step-by-Step Worked Example (Read Slowly): Associative Matrix Memory Update in mLSTM

Suppose a tiny mLSTM cell has dimension $$d = 2$$, initial matrix memory $$C_0 = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$$, and receives two orthogonal unit key-value pairs with $$f_t = 1, i_t = 1$$:
- Step 1: key $$k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$, value $$v_1 = \begin{bmatrix} 3 \\ 5 \end{bmatrix}$$.
- Step 2: key $$k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$, value $$v_2 = \begin{bmatrix} -2 \\ 4 \end{bmatrix}$$.

1. **Step 1 outer-product write**:
   $$C_1 = C_0 + v_1 k_1^\top = \begin{bmatrix} 3 \\ 5 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 5 & 0 \end{bmatrix}.$$
2. **Step 2 outer-product write**:
   $$C_2 = C_1 + v_2 k_2^\top = \begin{bmatrix} 3 & 0 \\ 5 & 0 \end{bmatrix} + \begin{bmatrix} 0 & -2 \\ 0 & 4 \end{bmatrix} = \begin{bmatrix} 3 & -2 \\ 5 & 4 \end{bmatrix}.$$
3. **Associative read with query $$q = k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$**:
   $$C_2 q = \begin{bmatrix} 3 & -2 \\ 5 & 4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 5 \end{bmatrix} = v_1!$$
4. **Why this is a breakthrough**: Unlike a classical vector LSTM that would blur $$v_1$$ and $$v_2$$ together by addition, the $$d \times d$$ matrix memory stores up to $$d$$ orthogonal key-value associations simultaneously with zero interference!

# Optional: gated recurrence after LSTMs — xLSTM and when gates still win

> This lesson is **optional**. It does **not** rewrite LSTM/GRU gate theory. It covers 2024 gated-RNN revivals and the production niches where LSTM/GRU remain the right tool.

The LSTM cell state

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

is still the cleanest picture of a **writable memory**. Transformers stole large-scale NLP; they did not delete every gated recurrent model.

## 1. xLSTM (2024): exponential gates + matrix memory

[Beck et al., 2024](https://arxiv.org/abs/2405.04517) revisit LSTM with:

- **sLSTM** — scalar memory with exponential input/forget gates and a normalizer (stable revision of the classic cell),
- **mLSTM** — a *matrix* memory updated with a covariance-like rule, trainable in parallel like a Transformer block.

The paper reports language-modeling results that close much of the gap to Transformers at modest scale. Treat xLSTM as evidence that **gating + memory** is not a 1997 idea only — not as a reason to skip attention.

## 2. Concrete applications

### Time series and tabular event streams

Forecasting stacks (retail demand, ICU vitals, industrial sensors) still ship **LSTM/GRU encoders**, often under a Temporal Fusion Transformer or a simple seq2seq head. The gate analysis from this chapter is exactly why they survive noisy, irregular sampling.

### Speech and streaming

On-device wake-word and some streaming ASR encoders keep GRUs because the state is a few kilobytes. Large ASR (Whisper, Chapter 19) is Transformer-based; the split is **latency / memory**, not accuracy on LibriSpeech.

### Hybrid stacks

A common 2024–2026 pattern: Transformer (or Mamba) for the backbone, **LSTM decoder** or **GRU controller** for an action head in robotics and recommendation session models.

## 3. Widely used software

- PyTorch `nn.LSTM` / `nn.GRU` — production-quality CuDNN/cuDNN-like kernels.
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm) — official xLSTM implementations.
- [sktime](https://github.com/sktime/sktime) / [GluonTS](https://github.com/awslabs/gluonts) — forecasting libraries that still expose RNN estimators.

## 4. Citations (2022–2026)

- [xLSTM: Extended Long Short-Term Memory (Beck et al., 2024)](https://arxiv.org/abs/2405.04517).
- [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) — the other “recurrence is back” line; compare with xLSTM rather than duplicating Chapter 05.
- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting (Lim et al., 2021)](https://arxiv.org/abs/1912.09363) — widely deployed 2022–2026; LSTM encoders + attention.

## 5. How this complements the core notes

Derive forget/input/output gates from the theory lesson first. This note only adds **xLSTM** and the remaining industrial LSTM/GRU use cases — it does not change the gate equations.

## Video tham khảo

- **xem trước** [XLSTM - Extended LSTMs with sLSTM and mLSTM (paper explained)](https://www.youtube.com/watch?v=0aWGTNS03PU) — AI Bites, English. xLSTM (exponential gates and a matrix memory) is the 2024 update this note is about.
- [Long Short-Term Memory (LSTM), Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — StatQuest with Josh Starmer, English. The 1997 cell xLSTM is extending, so the new gates have a baseline.
- [Illustrated Guide to LSTM's and GRU's: A step by step explanation](https://www.youtube.com/watch?v=8HyCNIVRbSU) — The AI Hacker, English. The gate diagrams you need in order to see what sLSTM and mLSTM change.
