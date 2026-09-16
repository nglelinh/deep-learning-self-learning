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
