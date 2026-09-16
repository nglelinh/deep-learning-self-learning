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
