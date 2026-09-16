---
layout: post
title: 20-99 Modern Applications and Updates (2022–2026)
chapter: '20'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter20
lesson_type: optional
---

# Optional: RL fundamentals in production — bandits, recsys, and RLHF’s MDP

> This lesson is **optional**. It does **not** replace MDPs, returns, or policy/value definitions. It shows where those objects showed up in 2022–2026 *data-science* systems — especially preference-based language training.

An MDP $$(S,A,P,R,\gamma)$$ is still the model. The surprising application: **chat alignment** is an MDP whose “environment” is a human (or a reward model) scoring a whole transcript.

## 1. RLHF as an MDP you already know

[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) (InstructGPT) treat each prompt as a start state, tokens as actions, and a learned reward $$r_\phi$$ as $$R$$. PPO (Chapter 21) then maximizes

$$J(\theta) = \mathbb{E}_{\pi_\theta}[r_\phi] - \beta\, D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}}).$$

The KL term is the same “don’t leave the behavior policy” idea as in conservative RL. Details of PPO/DPO wait for Chapter 21’s optional lesson; here the point is that **Chapter 20’s vocabulary is what industry used**.

## 2. Concrete applications

### Contextual bandits in ranking

News feeds and notifications are often **bandits**, not full MDPs: one action, immediate reward (click). Libraries: [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit), [Ray RLlib](https://github.com/ray-project/ray). The exploration–exploitation section of this chapter is the theory.

### Offline / batch RL for logs

When you cannot reset a factory or a market, you learn from logged $$ (s,a,r,s') $$. [Levine et al., 2020](https://arxiv.org/abs/2005.01643) survey offline RL; 2022–2025 applied work uses conservative Q-learning and fitted policies on recommender logs.

### Simulation before the real reward

Robotics and ads still build a simulator so the MDP is cheap. The definition of $$P$$ and $$R$$ in this chapter is the checklist for “is my simulator lying?”

## 3. Widely used software

- [openai/gymnasium](https://github.com/Farama-Foundation/Gymnasium) — the 2023+ fork of Gym.
- [huggingface/trl](https://github.com/huggingface/trl) — RLHF trainers (see Chapter 21).
- [vw](https://github.com/VowpalWabbit/vowpal_wabbit) — production bandits.

## 4. Citations (2022–2026)

- [Training language models to follow instructions with human feedback — InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155).
- [Gymnasium](https://gymnasium.farama.org/) — maintained MDP API used in 2023–2026 courses.
- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives (Levine et al., 2020)](https://arxiv.org/abs/2005.01643) — still the map for log-based DS work.

## 5. How this complements the core notes

Learn states, returns, and Bellman equations first. This lesson only names the **MDPs that shipped** (bandits, offline logs, RLHF) — not a new definition of a value function.
