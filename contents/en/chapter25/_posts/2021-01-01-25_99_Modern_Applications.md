---
layout: post
title: 25-99 Modern Applications and Updates (2022–2026)
chapter: '25'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter25
lesson_type: optional
---

For five years, the scaling game had only one lever: make pretraining bigger. Between 2024 and 2026, a second axis broke wide open—**test-time compute scaling**: letting a reasoning model spend extra thinking tokens, verify intermediate steps, call external tools (code interpreters, search, APIs), and process native images/audio in a single unified stream. Meanwhile, real-world energy and water constraints turned compute efficiency from an academic footnote into a first-class systems requirement.

## Key ideas: What Shipped at the Frontier (2022–2026)

- **Pillar 1 — Test-Time Compute Scaling**: Trades extra inference tokens (long chains of thought, best-of-$N$ verifier search, tree search) for dramatic gains on math, coding, and planning—complementing pretraining scaling.
- **Pillar 2 — Tool-Using Agent Loops**: Connects LLM reasoning to deterministic external execution (Python sandboxes, SQL, browsers, MCP tools) via structured observe–think–act–verify loops.
- **Pillar 3 — Native Multimodal Architectures**: Tokenizes text, image patches, audio frames, and video clips into one shared Transformer backbone instead of bolting fragile adapters onto a text-only LLM.
- **Pillar 4 — Energy, Cost, and Routing Constraints**: Uses model routing (small fast models for easy queries, large reasoning models for hard ones) and batching to bound carbon footprint and serving cost.

### Worked example: Pass@$k$ / Best-of-$N$ Test-Time Compute Trade-off

Suppose on a hard coding problem, a single sample from a small model succeeds with independent probability $p = 0.25$ and costs $c = \$0.001$ per attempt, and we have a unit-test suite that verifies correctness with $100\%$ precision.

1. **Probability that at least 1 of $N$ independent samples passes (`pass@N`)**:
   $$\text{pass@}N = 1 - (1 - p)^N.$$
   - $N = 1$: $25.0\%$ (cost $\$0.001$).
   - $N = 4$: $1 - (0.75)^4 = 1 - 0.3164 = 68.4\%$ (cost $\$0.004$).
   - $N = 8$: $1 - (0.75)^8 = 1 - 0.1001 = 90.0\%$ (cost $\$0.008$).
2. **Compare against a $10\times$ larger model** that achieves $85\%$ on a single pass at cost $\$0.015$:
   Spending $8\times$ test-time compute on the small model reaches **higher accuracy ($90.0\%$ vs. $85.0\%$)** at almost **half the dollar cost ($\$0.008$ vs. $\$0.015$)**!
3. **Takeaway**: Whenever a reliable verifier exists (unit tests, compilers, formal proof checkers), test-time compute scaling can beat brute-force parameter scaling on both accuracy and cost.

# Optional: what actually shipped — test-time compute, agents, multimodal

> This lesson is **optional**. It does **not** rewrite the chapter’s scaling-law / multimodal / safety survey. That note already looks forward; this lesson lists **concrete 2024–2026 deployments** so the “future” section stays tied to systems you can run.

The core chapter’s power-law

$$L \propto N^{-\alpha}$$

still describes pretraining. The new industrial knob is **test-time compute**: spend more FLOPs *per query*.

## 1. Inference-time scaling

[OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/) (2024) and follow-on reasoning models (DeepSeek-R1-class, 2025) train models to emit long chains of thought and search. Empirically, accuracy can scale with extra samples or longer traces, closer to

$$\text{error} \approx c \cdot C_{\mathrm{test}}^{-\gamma}$$

than to a bigger pretrained $$N$$ alone ([Snell et al., 2024](https://arxiv.org/abs/2408.03314), “Scaling LLM Test-Time Compute”). This is complementary to Chinchilla’s *train*-time scaling (Chapter 00/01 optional).

## 2. Concrete applications

### Tool-using agents

Production “agents” (2024–2026) are loops: LM → tool (SQL, browser, code) → observation. Software: [OpenAI Agents / Responses tools](https://platform.openai.com), [LangGraph](https://github.com/langchain-ai/langgraph), [smolagents](https://github.com/huggingface/smolagents). The research risk is still reliability, not a new neuron.

### Native multimodal

GPT-4V-class, Gemini, and open VLMs (LLaVA, Qwen2-VL, PaliGemma) take images/audio in one stack. The chapter’s CLIP/Flamingo pointer was the research seed; these are the APIs.

### Sustainability as a constraint

Serving 70B models for every click is not a “future problem”: quantization (Chapter 23 optional), routing to small models, and batching are the deployed answers to the chapter’s energy paragraph.

## 3. Widely used software

- [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) — high-throughput + speculative / structured decode.
- [huggingface/transformers](https://github.com/huggingface/transformers) reasoning-model cards (DeepSeek-R1 distillations, etc.).
- Open evaluation: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## 4. Citations (2022–2026)

- [Scaling LLM Test-Time Compute Optimally (Snell et al., 2024)](https://arxiv.org/abs/2408.03314).
- [DeepSeek-R1 (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948) — large-scale reasoning with RL (see Chapter 21 optional).
- [Visual Instruction Tuning — LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — the open VLM recipe widely copied in 2024–2026.

## 5. How this complements the core notes

Keep the chapter’s discussion of scaling, NAS, continual learning, and safety. This lesson only pins **test-time compute, agents, and open VLMs** as the items that moved from “future” to “default product” between 2024 and 2026.

## Video tham khảo

- **xem trước** [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — Andrej Karpathy, English. Karpathy's 2025 deep dive includes inference-time reasoning, the knob this note adds on top of pretraining. It is long and canonical.
- [Chinchilla Explained: Compute-Optimal Massive Language Models](https://www.youtube.com/watch?v=PZXN7jm9IC0) — Edan Meyer, English. Chinchilla is the train-time law the note says test-time compute does not replace.
- [MIT 6.S191 (2024): Language Models and New Frontiers](https://www.youtube.com/watch?v=N1fbskTpwZ0) — Alexander Amini, English. The 2024 frontier lecture for agents and multimodal systems the note lists next.
