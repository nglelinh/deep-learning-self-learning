---
layout: post
title: 21-99 Modern Applications and Updates (2022–2026)
chapter: '21'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter21
lesson_type: optional
---

# Optional: deep RL after DQN/PPO — RLHF, DPO, and GRPO

> This lesson is **optional**. It does **not** replace DQN, replay, or PPO theory (the core notes already mention RLHF in passing). It is the 2022–2026 **alignment** stack that made deep RL visible outside games.

PPO still maximizes a clipped surrogate. Language alignment added a **reward model** and then, quickly, methods that **skip the RL loop**.

## 1. RLHF with PPO

[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) and [Bai et al., 2022](https://arxiv.org/abs/2204.05862) (Anthropic HH):

1. SFT on demonstrations,
2. train $$r_\phi$$ on pairwise preferences,
3. optimize the policy with PPO against $$r_\phi$$ plus a KL to the SFT model.

That is Chapter 21’s actor–critic applied to tokens. It is expensive (value model, sampling, instability).

## 2. DPO: preferences without a reward model

[Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) rewrite the RLHF optimum as a **classification** loss on pairs $$(y_w, y_l)$$:

$$\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Big).$$

No sampling loop at train time. DPO and cousins (IPO, KTO, ORPO) became the default open-source alignment recipe in 2024–2025 ([trl](https://github.com/huggingface/trl)).

## 3. Concrete applications

### GRPO and reasoning-model training

[Shao et al., 2024](https://arxiv.org/abs/2402.03300) (DeepSeekMath) introduce **Group Relative Policy Optimization**: several samples per prompt, advantages from the group’s own rewards — no learned value net. Later reasoning models (DeepSeek-R1-class, 2025) popularized this family.

### World models

[Hafner et al., 2023](https://arxiv.org/abs/2301.04104) (DreamerV3) learn a latent MDP (VAE-like) and plan inside it. Complementary to LLM-RL: still deep RL, not chat.

### Software

- [huggingface/trl](https://github.com/huggingface/trl) — PPO, DPO, GRPO trainers.
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) / [verl](https://github.com/volcengine/verl) — distributed RLHF.
- [CleanRL](https://github.com/vwxyzjn/cleanrl) and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — classic deep RL, not LLMs.

## 4. Citations (2022–2026)

- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155).
- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290).
- [DeepSeekMath — GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300).
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104).

## 5. How this complements the core notes

Implement DQN/PPO ideas from the chapter first. This lesson only adds **preference optimization for LMs** (RLHF/DPO/GRPO) — not a new Bellman backup.
