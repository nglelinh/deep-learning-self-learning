---
layout: post
title: 24-01-01 Interpretability Theory
chapter: '24'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter24
---

# Model Interpretability: Understanding the Black Box

## 1. Concept Overview

Neural network interpretability addresses one of deep learning's most significant challenges: understanding why models make particular predictions. While neural networks achieve remarkable performance on diverse tasks, their decision-making process often appears opaque—millions of parameters interacting through nonlinear transformations make it difficult to trace how inputs map to outputs. This "black box" nature creates problems in high-stakes domains like medicine (why did the model diagnose this condition?), law (why was this defendant classified as high-risk?), and autonomous vehicles (why did the car brake suddenly?), where we need not just accurate predictions but justifications we can audit, trust, and debug.

Understanding the distinction between interpretability and explainability clarifies what we're seeking. Interpretability means the model's internal workings are transparent—we can understand the computation from inspection. Simple models like linear regression or decision trees are inherently interpretable; we can see exactly how features combine to produce predictions. Explainability means we can provide post-hoc explanations of why a model made specific predictions, even if the model itself isn't inherently transparent. Deep neural networks are rarely interpretable (understanding millions of parameters is infeasible) but can be made explainable through techniques that highlight relevant inputs, visualize learned features, or approximate decisions with interpretable surrogates.

The motivation for interpretability extends beyond satisfying curiosity. In scientific applications, understanding what features models use can generate new hypotheses—if a medical imaging model identifies a subtle pattern doctors missed, investigating this pattern might reveal new diagnostic markers. In debugging, interpretability reveals when models exploit spurious correlations (detecting huskies by snow in background rather than dog features) or fail to use relevant features. In safety-critical applications, interpretability enables verification that models behave reasonably across diverse scenarios. In regulated industries, explainability may be legally required for decisions affecting individuals. Understanding these diverse motivations helps appreciate that interpretability isn't a single goal but multiple related objectives requiring different techniques.

The landscape of interpretability methods is vast, reflecting the multiple ways we might understand neural networks. Saliency methods highlight which input regions most influenced a prediction, answering "what did the model look at?" Activation visualization shows what patterns activate neurons, revealing "what features has the network learned?" Attribution methods decompose predictions into feature contributions, explaining "how much did each input feature matter?" Concept-based explanations identify high-level concepts the model uses, moving beyond pixel or word-level attributions to semantic understanding. Each approach provides different insights, and comprehensive interpretability often requires multiple complementary techniques.

Yet interpretability has fundamental tensions. More interpretable models are often less accurate (linear models vs deep networks). Faithful explanations (accurately describing model behavior) might be complex and hard to understand. Simple explanations might be understandable but unfaithful to actual model behavior. Perfect interpretability might require understanding millions of parameters—as complex as understanding the phenomenon the model learned. These tensions mean interpretability research involves careful tradeoffs between fidelity, simplicity, and utility, without universal solutions that satisfy all desiderata simultaneously.

## 2. Mathematical Foundation

Interpretability methods often formalize the question "which inputs matter most for this prediction?" through attribution. Given input $$\mathbf{x}$$ and model $$f$$, compute attribution $$\mathbf{a}$$ where $$a_i$$ indicates importance of input dimension $$i$$ for the prediction $$f(\mathbf{x})$$.

### Gradient-Based Saliency

The simplest attribution uses gradients:

$$\mathbf{a} = \left|\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}\right|$$

This measures how much the output would change for small changes in each input dimension. Large gradient indicates high sensitivity—that input dimension strongly influences the output. For images, this produces saliency maps highlighting important pixels.

However, gradient can saturate in ReLU networks (gradient is zero or one, not informative about magnitude of change) and doesn't account for baseline (what are we comparing against?). Improvements address these issues:

**Integrated Gradients** accumulates gradients along path from baseline $$\mathbf{x}'$$ to input $$\mathbf{x}$$:

$$\mathbf{a}_i = (x_i - x_i') \int_{\alpha=0}^1 \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

This satisfies desirable axioms: sensitivity (if input doesn't affect output, attribution is zero) and implementation invariance (equivalent networks give same attributions).

### SHAP: Shapley Additive Explanations

SHAP uses Shapley values from cooperative game theory. The contribution of feature $$i$$ is:

$$\phi_i = \sum_{S \subseteq \mathcal{F} \backslash \{i\}} \frac{|S|!(|\mathcal{F}|-|S|-1)!}{|\mathcal{F}|!} [f(S \cup \{i\}) - f(S)]$$

where $$\mathcal{F}$$ is all features, $$S$$ are feature subsets, $$f(S)$$ is model output with only features in $$S$$ present (others set to baseline). This computes the average marginal contribution of feature $$i$$ across all possible feature coalitions—a fair allocation of prediction among features.

Computing exact Shapley values requires $$2^{|\mathcal{F}|}$$ model evaluations (exponential in features), so approximations are used. Kernel SHAP approximates through weighted linear regression. For tree-based models, TreeSHAP computes exactly in polynomial time.

### Layer-wise Relevance Propagation (LRP)

LRP backpropagates relevance from output to input:

$$R_i^{(l)} = \sum_j \frac{z_{ij}}{\sum_k z_{kj}} R_j^{(l+1)}$$

where $$z_{ij} = a_i^{(l)} w_{ij}$$ is contribution of neuron $$i$$ in layer $$l$$ to neuron $$j$$ in layer $$l+1$$. Starting with $$R_{\text{out}} = f(\mathbf{x})$$ at output, relevance propagates backward, decomposing prediction into input contributions satisfying $$\sum_i R_i^{(0)} = f(\mathbf{x})$$ (conservation).

## 3. Example / Intuition

Consider a CNN classifying an image as "dog" with 95% confidence. Without interpretability, we don't know why. Was it the dog's face, body shape, background context, or spurious patterns like grass (if all dogs in training had grass backgrounds)?

**Gradient saliency** computes $$\partial p_{\text{dog}}/\partial \text{pixels}$$. Large gradients highlight pixels that, if changed slightly, would most affect the dog probability. Visualized as a heatmap overlay on the image, we might see high values around the dog's face and ears—good, the model uses actual dog features. If high values appear in background, the model might be exploiting spurious correlations.

**Class Activation Mapping (CAM)** for CNNs with global average pooling shows which regions the final convolutional layer found important. For "dog" class, we compute weighted combination of final conv layer's feature maps using the classification weights:

$$\text{CAM} = \sum_k w_k^{\text{dog}} \cdot \text{FeatureMap}_k$$

This produces a heatmap at feature map resolution showing which spatial regions contributed to the "dog" prediction. Upsampling to input resolution and overlaying on the image reveals the model focused on the dog's head and body—interpretable and reassuring.

**SHAP values** for a particular prediction might show:
- Pixel region containing dog face: +0.35 (strong positive contribution)
- Pixel region with dog body: +0.28
- Background grass: +0.08 (small contribution - concerning if high)
- Sky region: -0.02 (slight negative - expected for irrelevant regions)

If grass has high SHAP value, we've discovered the model uses spurious correlation (dogs often photographed on grass). We can then collect more diverse training data or use data augmentation to fix this.

**Adversarial examples** provide another interpretability lens. By finding minimal input perturbations that change predictions, we reveal model vulnerabilities. If adding imperceptible noise to dog image causes "cat" prediction, the model's representation is fragile—it hasn't learned robust features. Studying these adversarial perturbations reveals what features matter: perturbations often add patterns the model strongly associates with target class, revealing learned (but perhaps spurious) class indicators.

