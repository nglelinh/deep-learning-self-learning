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

In 2023–2026, interpretability leaped from highlighting input pixels to reverse-engineering the internal algorithms of frontier LLMs—a field called **Mechanistic Interpretability**. The core obstacle was *polysemanticity*: a single Transformer neuron often fires for unrelated concepts (both French poetry and Python brackets) because models pack more concepts than dimensions via **superposition**. Sparse Autoencoders (SAEs) broke that code by projecting residual-stream activations into an overcomplete sparse basis where individual latents correspond cleanly to monosemantic concepts like "Golden Gate Bridge" or "SQL injection."

## Background Knowledge Map: Mechanistic Interpretability & SAEs (2022–2026)

- **Pillar 1 — Superposition & Polysemanticity**: Explains how an $n$-dimensional residual stream can pack $m \gg n$ nearly orthogonal sparse features using Johnson–Lindenstrauss geometry.
- **Pillar 2 — Sparse Autoencoders (SAEs) on Residual Streams**: Trains an overcomplete encoder $f = \operatorname{ReLU}(W_{\text{enc}} x + b_{\text{enc}})$ with an $L_1$ or Top-$k$ sparsity penalty to decompose polysemantic activations into interpretable monosemantic features.
- **Pillar 3 — Activation Patching & Circuit Discovery**: Performs causal interventions—swapping activations from a corrupted prompt into a clean prompt—to isolate exact attention-head circuits (e.g., Induction Heads, Indirect Object Identification).
- **Pillar 4 — Feature Steering & Safety Evals**: Clamps or amplifies discovered SAE features at inference time to test honesty, refusal, and sycophancy behaviors causally.

### Step-by-Step Worked Example (Read Slowly): Superposition Geometry & Top-$k$ SAE Feature Decomposition

Suppose a 2D residual stream ($d = 2$) packs $m = 3$ unit-norm concept directions $120^\circ$ apart:
$$d_1 = (1, 0), \qquad d_2 = (-0.5, +0.866), \qquad d_3 = (-0.5, -0.866).$$

1. **Check pairwise interference (dot products)**:
   $$d_1^\top d_2 = d_1^\top d_3 = d_2^\top d_3 = -0.5.$$
   Even though 3 vectors cannot be mutually orthogonal in $\mathbb{R}^2$, if at most 1 concept is active at a time, a ReLU bias $b = -0.6$ cleanly separates them!
2. **Evaluate an SAE encoder** $f_j = \operatorname{ReLU}(d_j^\top x - 0.6)$ on an activation $x = 2.0\,d_1 = (2.0, 0.0)$:
   - $f_1 = \operatorname{ReLU}(2.0(1) - 0.6) = 1.4$ (active!).
   - $f_2 = \operatorname{ReLU}(2.0(-0.5) - 0.6) = \operatorname{ReLU}(-1.6) = 0.0$ (suppressed!).
   - $f_3 = \operatorname{ReLU}(2.0(-0.5) - 0.6) = 0.0$ (suppressed!).
3. **Takeaway**: An overcomplete sparse basis ($m > d$) combined with ReLU thresholding cleanly disentangles superposed features that look hopelessly mixed when inspecting raw coordinates $x_1, x_2$.

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

## Video tham khảo

- **xem trước** [A Window  Into LLMs | Sparse Autoencoders Explained](https://www.youtube.com/watch?v=TKozVZoXAYs) — Papers Are Wonderful, English. Sparse autoencoders on activations are the feature-finding method this note describes.
- [The Dark Matter of AI \[Mechanistic Interpretability\]](https://www.youtube.com/watch?v=UGO_Ehywuxc) — Welch Labs, English. Welch Labs on mechanistic interpretability, the program those autoencoders belong to.
- [Lecture 12 | Visualizing and Understanding](https://www.youtube.com/watch?v=6wcs6szJWMY) — Stanford University School of Engineering, English. The older visualization program this line of work is continuing.
