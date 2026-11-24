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

You might think tabular MDPs only live in toy GridWorlds—until you realize that every modern recommendation feed, dynamic pricing engine, and RLHF LLM post-training run is literally an MDP. In RLHF, the state is the prompt plus tokens generated so far, the action is the next token from the vocabulary, the policy is the LLM itself, and the reward arrives at the end of the response from a preference model or verifier.

## Background Knowledge Map: Classical RL Concepts in Production (2022–2026)

- **Pillar 1 — RLHF as a Token-Level MDP**: Maps prompt-to-response generation directly onto $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ with deterministic string-concatenation transitions and terminal reward plus per-token KL penalty.
- **Pillar 2 — Contextual Bandits in Ranking & RecSys**: Solves 1-step MDPs ($\gamma = 0$) where context $x$ is the user/session, actions $a$ are candidate items, and algorithms like LinUCB or Thompson Sampling balance revenue against exploration.
- **Pillar 3 — Offline / Batch RL & Propensity Weighting**: Learns improved policies strictly from logged historical interactions using Inverse Propensity Scoring (IPS) or Conservative Q-Learning (CQL) when live exploration is risky or expensive.
- **Pillar 4 — Sim-to-Real & Reward Hacking Prevention**: Tests policies in simulators before deployment and guards against Goodhart's Law (where optimizing a proxy metric like clickbait destroys true user satisfaction).

### Step-by-Step Worked Example (Read Slowly): Inverse Propensity Scoring (IPS) for Offline Policy Evaluation

Suppose a logged recommendation system showed Item A with probability $\pi_0(A \mid x) = 0.20$ and observed click reward $r = 1.0$, and showed Item B with probability $\pi_0(B \mid x) = 0.80$ and observed reward $r = 0.0$. We want to evaluate a new candidate policy $\pi_{\text{new}}$ that puts probability $\pi_{\text{new}}(A \mid x) = 0.70$ and $\pi_{\text{new}}(B \mid x) = 0.30$—without deploying it live yet!

1. **Compute importance weights $w(a) = \frac{\pi_{\text{new}}(a \mid x)}{\pi_0(a \mid x)}$**:
   $$w(A) = \frac{0.70}{0.20} = 3.50, \qquad w(B) = \frac{0.30}{0.80} = 0.375.$$
2. **Compute the IPS-estimated expected reward across these 2 logged impressions**:
   $$\hat{V}_{\text{IPS}}(\pi_{\text{new}}) = \frac{1}{2}\big[w(A)\cdot 1.0 + w(B)\cdot 0.0\big] = \frac{3.50 + 0}{2} = 1.75\text{ (unclipped!)}.$$
3. **Self-Normalized IPS (SNIPS) to control variance**:
   $$\hat{V}_{\text{SNIPS}} = \frac{w(A)\cdot 1.0 + w(B)\cdot 0.0}{w(A) + w(B)} = \frac{3.50}{3.50 + 0.375} = \frac{3.50}{3.875} \approx 0.903.$$
4. **Takeaway**: Notice how raw IPS can even exceed the maximum possible reward $1.0$ when logging propensities are small ($0.20$), which is why production offline RL systems rely on Self-Normalized IPS or clipped importance ratios.

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

## Video tham khảo

- **xem trước** [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained!!!](https://www.youtube.com/watch?v=qPN_XZcJf_s) — StatQuest with Josh Starmer, English. RLHF is an MDP whose reward comes from human preferences, which is this note's claim.
- [Reinforcement Learning from Human Feedback (RLHF) Explained](https://www.youtube.com/watch?v=T_X4XFwKX8k) — IBM Technology, English. A shorter corporate-lecture cut of the same loop.
- [RL Course by David Silver - Lecture 2: Markov Decision Process](https://www.youtube.com/watch?v=lfHX2hHRMVQ) — Google DeepMind, English. The MDP vocabulary the note says you already have.
