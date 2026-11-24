---
layout: post
title: 24-01 Neural Network Interpretability and Explainability
chapter: '24'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter24
---

Opening the black box of a deep neural network requires both mathematical axioms (so your explanations don't lie to you) and clean PyTorch instrumentation (so you can inspect real models in seconds). This module pairs the theory of saliency, SHAP, and LRP directly with hands-on Grad-CAM and attribution code.

## Background Knowledge Map: Theory-to-Practice Interpretability Guide

- **Pillar 1 — Gradient vs. Perturbation Methods**: Contrasts fast single-pass gradient methods (Saliency, Grad-CAM) with model-agnostic perturbation methods (LIME, KernelSHAP).
- **Pillar 2 — Conservation Axioms**: Uses Completeness and Shapley Efficiency to separate principled attributions from noisy edge-detector visualizations.
- **Pillar 3 — PyTorch Hook Engineering**: Extracts intermediate layer activations and gradients cleanly for visual and tabular diagnostics.
- **Pillar 4 — Pitfall Awareness**: Recognizes confirmation bias, baseline sensitivity, and adversarial manipulation of explanations.

### Step-by-Step Worked Example (Read Slowly): Baseline Sensitivity in Integrated Gradients

Consider a single-pixel model $f(x) = x^2$ evaluated at $x = 2.0$. Compare attributions using two different baselines: black baseline $x'_A = 0.0$ versus gray baseline $x'_B = 1.0$.

1. **Exact integral $\text{IG}(x; x') = (x - x')\int_0^1 f'(x' + \alpha(x-x'))\,d\alpha = f(x) - f(x')$**:
   - With black baseline $x'_A = 0.0$: $\text{IG} = 2^2 - 0^2 = +4.0$.
   - With gray baseline $x'_B = 1.0$: $\text{IG} = 2^2 - 1^2 = +3.0$.
2. **What happens if a black pixel ($x = 0$) is actually the critical feature?**
   With black baseline $x' = 0$, $x - x' = 0 \implies \text{IG} = 0$ identically—making the black feature completely invisible!
3. **Takeaway**: Choosing a semantically neutral baseline (or averaging over multiple random/blurred baselines as in Expected Gradients) is essential so attribution maps never go blind to dark features.

Model interpretability:

1. **Theory** — saliency, SHAP, LRP, intuition
2. **Implementation** — Grad-CAM and SHAP code, papers, pitfalls

## Video tham khảo

- **xem trước** [Lecture 12 | Visualizing and Understanding](https://www.youtube.com/watch?v=6wcs6szJWMY) — Stanford University School of Engineering, English. The difference between looking inside a net and explaining one prediction.
- [SHAP values for beginners | What they mean and their applications](https://www.youtube.com/watch?v=MQ6fFDwjuco) — A Data Odyssey, English. SHAP values as one explanation of a single prediction.
- [Explainable AI explained! | #3 LIME](https://www.youtube.com/watch?v=d6j6bofhj2M) — DeepFindr, English. LIME as the other local explanation.
