---
layout: post
title: 24-99 Modern Applications and Updates (2022–2026)
chapter: '24'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter24
lesson_type: optional
---

# Optional: interpretability after LIME/SHAP — circuits and sparse autoencoders

> This lesson is **optional**. It does **not** replace saliency, LIME, SHAP, or the implementation notes. It covers **mechanistic interpretability** of Transformers — the 2022–2026 research/product line for LLMs.

SHAP still answers “which input feature moved the score?” For a 70B chat model the interesting question is often “which **internal feature** implements refusal / code syntax / a latent fact?”

## 1. Sparse autoencoders on activations

[Cunningham et al., 2023](https://arxiv.org/abs/2309.08600) and [Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Anthropic, *Scaling Monosemanticity*) train a sparse AE on residual-stream activations $$x$$:

$$\hat{x} = W_{\mathrm{dec}}\,\mathrm{ReLU}(W_{\mathrm{enc}}x + b) + b', \quad \mathcal{L}=\|x-\hat{x}\|_2^2 + \lambda\|f\|_1.$$

The sparse code $$f$$ is treated as **monosemantic features**. Teams then steer or ablate those features (gold-standard evals remain messy). This is an autoencoder (Chapter 12) used as an *interpretability instrument*, not a generative model.

## 2. Concrete applications

### Attribution still ships

[Captum](https://captum.ai/) integrated gradients and attention-rollout remain what regulated industries put in model cards. Use the theory lesson’s SHAP/LIME for tabular DS; do not replace them with SAEs on a credit model.

### Circuit-style analyses

[Elhage et al., Transformer Circuits](https://transformer-circuits.pub/) (2022–2024) reverse-engineer induction heads and other motifs. Complementary reading, not a homework replacement for Grad-CAM.

### Safety evals

Interpretability is now a **governance** input: can we locate a “deception” feature? Papers are preliminary; the course takeaway is the *tooling*, not a claim that models are fully explained.

## 3. Widely used software

- [captum](https://github.com/pytorch/captum) — production attribution.
- [openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder) and [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — hook-based circuit work.

## 4. Citations (2022–2026)

- [Sparse Autoencoders Find Highly Interpretable Features in Language Models (Cunningham et al., 2023)](https://arxiv.org/abs/2309.08600).
- [Scaling Monosemanticity (Templeton et al., 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
- [A Mathematical Framework for Transformer Circuits (Elhage et al., 2021/2022)](https://transformer-circuits.pub/2021/framework/index.html).

## 5. How this complements the core notes

Run LIME/SHAP and saliency from the chapter first. This lesson only adds **LLM mechanistic tools** (SAEs, circuits) that appeared after the classic XAI catalog.
