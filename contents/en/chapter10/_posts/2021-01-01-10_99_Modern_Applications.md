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

For nearly a decade, AdamW sat virtually unchallenged atop deep learning optimization—yet training 100B+ parameter LLMs exposed its two biggest costs: **16 bytes/parameter** of optimizer state memory and purely diagonal (coordinate-wise) preconditioning that ignores matrix structure across weight rows and columns. Between 2023 and 2025, a new wave of optimizers challenged that status quo: **Lion** (discovered via symbolic program search, using sign updates and cutting optimizer memory in half), **Sophia** (lightweight diagonal Hessian clipping), and **Muon** (orthogonalizing 2D weight-matrix momentum updates via Newton–Schulz iterations for dramatically faster LLM pretraining).

## Background Knowledge Map: Beyond AdamW — The 2023–2025 Optimizer Frontier

- **Sign-Based Memory Compression (Lion, 2023)**: Replaces Adam's second-moment buffer `v` entirely by taking $$\operatorname{sign}(\beta_1 m_{t-1} + (1 - \beta_1) g_t)$$—halving optimizer memory (only 1 momentum buffer) while requiring ~$$3\times\text{–}10\times$$ smaller learning rates and larger weight decay.
- **Second-Order Curvature Approximation (Sophia, 2023)**: Periodically estimates the diagonal Hessian $$h_t$$ (via Hutchinson's or Gauss–Newton estimators) and clips the update $$\operatorname{clip}(m_t / \max(h_t, \epsilon), -1, 1)$$, accelerating along flat directions while braking aggressively on sharp curvature.
- **Matrix-Aware Orthogonalization (Muon, 2024)**: Recognizes that linear-layer weights $$W \in \mathbb{R}^{m \times n}$$ are **matrices**, not unstructured vectors. Applying fast **Newton–Schulz** iterations to the momentum matrix $$M_t$$ approximates its polar factor $$U V^\top$$ (setting all singular values to $$1$$), updating every spectral direction at equal speed!
- **Hybrid Production Recipes**: Frontier training runs increasingly pair Muon for hidden 2D weight matrices with AdamW for 1D biases, LayerNorm gains, and token embeddings—combining spectral speed with scalar stability.

### Step-by-Step Worked Example (Read Slowly): Why Orthogonalizing a Momentum Matrix (Muon) Rescues Small Singular Directions

Suppose a $$2 \times 2$$ weight layer has momentum matrix $$M = \begin{bmatrix} 10 & 0 \\ 0 & 0.1 \end{bmatrix}$$ (singular values $$\sigma_1 = 10, \sigma_2 = 0.1$$ — a $$100:1$$ anisotropy!).

1. **Standard Momentum update** ($$\Delta W = -\eta M$$ with $$\eta = 0.01$$):
   $$\Delta W_{\text{mom}} = \begin{bmatrix} -0.10 & 0 \\ 0 & -0.001 \end{bmatrix}.$$
   Direction 2 moves $$100\times$$ slower than Direction 1.
2. **Muon polar orthogonalization** (SVD $$M = U \Sigma V^\top \implies \operatorname{orth}(M) = U V^\top$$):
   - Here $$U = I, V = I, \Sigma = \operatorname{diag}(10, 0.1)$$, so $$U V^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$.
   - Muon update ($$\eta = 0.01$$):
     $$\Delta W_{\text{Muon}} = -0.01 \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} -0.01 & 0 \\ 0 & -0.01 \end{bmatrix}.$$
3. **One-step Newton–Schulz check**: Starting from normalized $$X_0 = M / \|M\|_F \approx \operatorname{diag}(0.99995, 0.0099995)$$, the iteration $$X_{k+1} = \frac{3}{2} X_k - \frac{1}{2} X_k^3$$ rapidly inflates tiny singular values toward $$1$$ using only fast GPU matrix multiplications—no expensive SVD required!

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

## Video tham khảo

- **xem trước** [Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg) — DeepBean, English. Lion, Sophia, and Muon are later members of the momentum/Adam family this video lines up.
- [Adam Optimization Algorithm (C2W2L08)](https://www.youtube.com/watch?v=JXQT_vxqwIs) — DeepLearningAI, English. Adam is the update those 2023–2024 methods are measured against.
- [Lecture 7 | Training Neural Networks II](https://www.youtube.com/watch?v=_JB0AO7QxSA) — Stanford University School of Engineering, English. The training-lecture context for why a new optimizer is a small change next to data and schedule.
