---
layout: post
title: 09-99 Modern Applications and Updates (2022–2026)
chapter: '09'
order: 8
owner: Deep Learning Course
lang: en
categories:
- chapter09
lesson_type: optional
---

Walk into a modern frontier LLM pretraining run (Llama, Gemma, Qwen) and you will notice something surprising: **Dropout is almost completely gone** (`dropout = 0.0`), replaced by single-epoch multi-trillion-token data streams, decoupled weight decay (AdamW), and RMSNorm—while in Vision Transformers and ConvNeXt, **DropPath (Stochastic Depth)** and **Sharpness-Aware Minimization (SAM)** have largely superseded neuron-level dropout. Understanding *why* modern recipes shifted from random neuron masking to path dropping and flat-minima optimization connects textbook regularization directly to 2022–2026 research.

## Key ideas: Modern Regularization Recipes (SAM, DropPath, LLM Stacks)

- **Sharpness-Aware Minimization (SAM)**: Instead of minimizing loss only at the current point $$w$$, SAM minimizes the worst-case loss inside an $$\ell_2$$ ball of radius $$\rho$$: $$\min_w \max_{\|\epsilon\|_2 \le \rho} \mathcal{L}(w + \epsilon)$$—explicitly steering SGD into wide, flat basins that generalize better under distribution shift.
- **Stochastic Depth / DropPath**: In deep residual vision backbones (ViT, ConvNeXt, Swin), randomly dropping entire residual branches $$x + \text{DropPath}(F(x))$$ with linearly increasing drop probability across depth regularizes deep pathways far more effectively than dropping individual activations.
- **Why Frontier LLMs Often Use Zero Dropout**: When pretraining on trillions of unique tokens (roughly 1 epoch), the model rarely sees the same example twice—so classic memorization overfitting is minimal, and dropout actually wastes effective model capacity per FLOP.
- **Label Smoothing, Z-Loss, and QK-Norm**: Modern large-scale recipes pair decoupled weight decay with logit stabilization (router/logit z-loss) and Query–Key normalization to prevent attention-logit explosion at billion-parameter scale.

### Worked example: One Two-Step SAM Update on a 1D Sharp vs. Flat Valley

Suppose we evaluate a 1D loss $$\mathcal{L}(w) = 10(w - 1)^2$$ at $$w = 1.2$$ with perturbation radius $$\rho = 0.05$$ and learning rate $$\eta = 0.02$$.

1. **Step 1 — Ascent to the worst-case neighbor $$w_{\text{adv}}$$**:
   - Local gradient: $$g = \mathcal{L}'(1.2) = 20(1.2 - 1) = 4.0$$.
   - Normalized adversarial step: $$\hat{\epsilon} = \rho \frac{g}{|g|} = 0.05 \times (+1) = +0.05$$.
   - Perturbed weight: $$w_{\text{adv}} = 1.2 + 0.05 = 1.25$$.
2. **Step 2 — Compute gradient at the perturbed point and update original $$w$$**:
   - Gradient at $$w_{\text{adv}}$$: $$g_{\text{adv}} = \mathcal{L}'(1.25) = 20(1.25 - 1) = 5.0$$.
   - SAM weight update: $$w_{\text{new}} = w - \eta g_{\text{adv}} = 1.2 - 0.02(5.0) = 1.10$$ (vs. standard SGD's $$1.2 - 0.02(4.0) = 1.12$$).
3. **Intuition**: In a sharp ravine, stepping $$\rho$$ uphill reveals a much steeper slope, pushing the optimizer aggressively away from sharp cliffs at the cost of $$2\times$$ forward/backward passes per step.

# Optional: regularization after dropout — SAM, DropPath, and LLM recipes

> This lesson is **optional**. It does **not** replace dropout or BatchNorm theory. It covers the regularizers that actually appear in 2022–2026 training recipes.

Dropout

$$\tilde{\mathbf{h}} = \frac{\mathbf{m} \odot \mathbf{h}}{1-p}, \quad m_i \sim \mathrm{Bernoulli}(1-p)$$

is still the right mental model. Large Transformers, however, often drop **residual branches** (stochastic depth / DropPath) rather than hidden units, and they lean on **weight decay + data scale** more than on classic dropout.

## 1. Sharpness-Aware Minimization (SAM)

[Foret et al., 2021](https://arxiv.org/abs/2010.01412) (widely adopted after 2022 in vision) seek parameters that stay good in a neighborhood:

$$\min_\theta \max_{\|\boldsymbol{\varepsilon}\|_2 \le \rho} \mathcal{L}(\theta + \boldsymbol{\varepsilon}).$$

The practical step is a two-forward update: climb to $$\theta + \rho \frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|}$$, then step using that “worst-case” gradient. [ASAM](https://arxiv.org/abs/2102.11600) and later SAM-for-LLM papers adapt the radius. This is regularization of the **optimizer trajectory**, complementary to dropout’s noise on activations.

## 2. Concrete applications

### Stochastic depth / DropPath

Vision ConvNeXt and ViT recipes randomly drop an entire residual block during training (Huang et al., 2016; used as default in 2022–2026 `timm` recipes). That is dropout on **paths**, not units — the implementation is a few lines, the effect is implicit ensembling of shallower nets.

### What LLMs actually use

LLaMA-style training: **weight decay** (AdamW), token-level dropout rarely, sometimes attention/dropout on tiny models only. The regularizer that matters at 7B+ is **data mixture + packing + decay**, plus later RLHF/DPO (Chapter 21). Do not expect the homework dropout rate $$p=0.5$$ in a Llama trainer.

### Normalization variants as regularizers

LayerNorm / RMSNorm replaced BatchNorm in sequence models (already in Chapter 09’s BN variants). Residual + norm is part of the regularizer story: they stabilize scale so you can train longer without exploding features.

## 3. Widely used software

- `timm` `DropPath` and SAM implementations in [pytorch-optimizer](https://github.com/jettify/pytorch-optimizer) / [sam](https://github.com/davda54/sam).
- PyTorch `torch.optim.AdamW` + `weight_decay` (see Chapter 10).
- Hugging Face training recipes (`Trainer`, `trl`) — weight decay defaults, little hidden dropout.

## 4. Citations (2022–2026)

- [Sharpness-Aware Minimization (Foret et al., 2021)](https://arxiv.org/abs/2010.01412) — the SAM objective used throughout 2022–2026 vision.
- [ConvNeXt (Liu et al., 2022)](https://arxiv.org/abs/2201.03545) — stochastic depth as a first-class recipe ingredient.
- [AdamW (Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) — decoupled decay; still the LLM default in 2026.

## 5. How this complements the core notes

Keep the dropout inverted-dropout derivation and the BatchNorm placement notes. This lesson only adds **SAM** and **DropPath**, and warns that foundation-model recipes drifted away from heavy unit dropout.

## Video tham khảo

- **xem trước** [Sharpness-Aware Minimization (SAM) in 7 minutes](https://www.youtube.com/watch?v=k6-jJ58MFKU) — Yuxiang "Shawn" Wang, English. Sharpness-aware minimization is the 2021–2026 generalization method this note adds.
- [Overfitting in a Neural Network explained](https://www.youtube.com/watch?v=DEMmkFC6IGM) — deeplizard, English. The overfitting problem SAM is still trying to solve.
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — StatQuest with Josh Starmer, English. Bias and variance, so the sharpness story has the older vocabulary next to it.
