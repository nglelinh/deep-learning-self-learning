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

Between 2022 and 2026, Deep RL moved from Atari games to the center of frontier AI: RLHF with PPO aligned ChatGPT, Direct Preference Optimization (DPO) showed you can optimize the exact same KL-regularized reward objective without training a separate reward model, and Group Relative Policy Optimization (GRPO) removed the critic network entirely to train math and coding reasoning models (like DeepSeek-R1) directly from verifiable outcomes.

## Background Knowledge Map: Post-DQN Deep RL for Alignment & Reasoning (2022–2026)

- **Pillar 1 — RLHF with PPO & KL Regularization**: Maximizes $\mathbb{E}[r_\phi(x, y)] - \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ so the policy improves human preference scores without drifting into gibberish reward hacks.
- **Pillar 2 — Direct Preference Optimization (DPO)**: Applies the closed-form optimal policy $r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$ inside the Bradley–Terry preference model, turning RLHF into a pure supervised binary cross-entropy loss on $(y_w, y_l)$ pairs.
- **Pillar 3 — Group Relative Policy Optimization (GRPO)**: Samples a group of $G$ completions $\{y_1, \dots, y_G\}$ for the same prompt, scores them with a verifier (unit tests / math grader), and normalizes group rewards into advantages $\hat{A}_i = \frac{r_i - \bar{r}}{\text{std}(r)}$—cutting memory by dropping the value critic.
- **Pillar 4 — Learned World Models**: Trains latent dynamics models (DreamerV3, Genie) so agents can imagine rollouts inside a learned simulator before acting in the real world.

### Step-by-Step Worked Example (Read Slowly): Computing DPO Loss & GRPO Group Advantages by Hand

**Part A — DPO Loss**: For prompt $x$, chosen response $y_w$, and rejected response $y_l$ with $\beta = 0.1$, suppose log-probability ratios are $\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} = +2.0$ and $\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} = -1.0$.
1. Implicit reward margin: $\Delta = \beta(2.0 - (-1.0)) = 0.1 \times 3.0 = +0.30$.
2. DPO loss: $\mathcal{L}_{\text{DPO}} = -\ln\sigma(0.30) = -\ln(0.5744) \approx 0.5544\text{ nats}$.

**Part B — GRPO Group Advantages**: Suppose for one math prompt we sample $G = 4$ chains of thought and our verifier awards binary correctness rewards $r = [1, 0, 1, 0]$.
1. Mean $\bar{r} = 0.50$, standard deviation $\sigma_r = \sqrt{\frac{4 \times 0.25}{4}} = 0.50$.
2. Group-relative advantages: $\hat{A} = \left[\frac{1-0.5}{0.5}, \frac{0-0.5}{0.5}, \frac{1-0.5}{0.5}, \frac{0-0.5}{0.5}\right] = [+1,\; -1,\; +1,\; -1]$.
3. **Takeaway**: Both DPO and GRPO eliminate heavy components of classic PPO (reward model in DPO; critic network in GRPO) while keeping the exact mathematical backbone of KL-regularized policy optimization.

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

Implement DQN/PPO ideas from the chapter first. This lesson only adds **preference optimization for LMs** (RLHF/DPO/GRPO) — not a new Bellman backup. The Chapter 26 map (SFT → reward / PPO → DPO, with the same DPO display) is **26-03**.

## Video tham khảo

- **xem trước** [Proximal Policy Optimization (PPO) for LLMs Explained Intuitively](https://www.youtube.com/watch?v=8jtAzxUwDj0) — Julia Turc, English. PPO is the policy-gradient method this note uses for RLHF.
- [Direct Preference Optimization (DPO) - How to fine-tune LLMs directly without reinforcement learning](https://www.youtube.com/watch?v=k2pD3k1485A) — Luis Serrano Academy, English. DPO, the preference objective the note sets beside PPO.
- [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained!!!](https://www.youtube.com/watch?v=qPN_XZcJf_s) — StatQuest with Josh Starmer, English. The human-feedback loop both algorithms are optimizing.
