---
layout: post
title: 10-99 Modern Applications and Updates (2022–2026)
chapter: '10'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter10
lesson_type: optional
---

# Optional: AdamW and the 2023–2025 optimizer wave (Lion, Sophia, Muon)

> This lesson is **optional**. It does **not** replace momentum / RMSprop / Adam theory. The core notes already mention AdamW; this lesson covers what became standard *after* AdamW, and when to stay with AdamW.

AdamW is still the default in Hugging Face and most LLM trainers:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,$$

$$\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} - \eta\lambda \theta.$$

The last term is **decoupled** weight decay (not L2 inside the adaptive step). Everything below is an alternative or a specialized upgrade.

## 1. Lion (2023)

[Chen et al., 2023](https://arxiv.org/abs/2302.06675) (symbolic search at Google) keep only the **sign** of an interpolated momentum:

$$\theta \leftarrow \theta - \eta\,\mathrm{sign}(\beta_1 m + (1-\beta_1)g) - \eta\lambda\theta.$$

Memory is lower than Adam (no second moment). Used in some vision and distillation jobs; less universal than AdamW for LLMs.

## 2. Sophia (2023)

[Liu et al., 2023](https://arxiv.org/abs/2305.14342) estimate a diagonal Hessian (via Hutchinson or a clipped Gauss–Newton) and precondition:

$$\theta \leftarrow \theta - \eta \cdot \mathrm{clip}\!\left(\frac{\hat{m}_t}{\max\{\gamma \hat{h}_t, \varepsilon\}}\right).$$

Reported to cut GPT-style steps versus AdamW at similar quality. Still a research optimizer in most open trainers.

## 3. Muon (2024)

[Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/) apply SGD-momentum to **hidden matrices**, then orthogonalize the update with a few Newton–Schulz iterations (a cheap polar-factor approximation). Embeddings and LM heads typically stay on AdamW. [Moonshot / Kimi, 2025](https://arxiv.org/abs/2502.16982) report scaled LLM training with Muon. This is the most-discussed *new* matrix optimizer of 2024–2025.

## 4. Concrete applications and software

- **Stay on AdamW** for a first LLM or ConvNeXt run (`torch.optim.AdamW`, fused CUDA kernels in Apex / `torch`).
- **Lion**: [lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch); some `timm` recipes.
- **Sophia**: [Liuhongwei95/Sophia](https://github.com/Liuhongwei95/Sophia) (paper code).
- **Muon**: [KellerJordan/Muon](https://github.com/KellerJordan/Muon) and the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun.
- **Schedule-free AdamW** ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682)) — removes explicit warmup/cosine for some workloads.

## 5. Citations (2022–2026)

- [Symbolic Discovery of Optimization Algorithms — Lion (Chen et al., 2023)](https://arxiv.org/abs/2302.06675).
- [Sophia: A Scalable Stochastic Second-order Optimizer (Liu et al., 2023)](https://arxiv.org/abs/2305.14342).
- [Muon (Jordan et al., 2024)](https://kellerjordan.github.io/posts/muon/) — hidden-layer orthogonalized momentum.
- [Muon is Scalable for LLM Training (Liu et al., 2025)](https://arxiv.org/abs/2502.16982) — Kimi/Moonshot scale report.
- [The Road Less Scheduled (Defazio et al., 2024)](https://arxiv.org/abs/2405.15682) — schedule-free AdamW.

## 6. How this complements the core notes

Learn Adam and the Rosenbrock comparison first. This lesson only adds **post-AdamW** methods used in 2023–2026 papers and speedruns — it does not replace the optimizer implementations in this chapter.
