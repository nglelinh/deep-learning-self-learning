---
layout: post
title: 00-04-01 Basic Probability Theory
chapter: '00'
order: 15
owner: AI Assistant
lang: en
categories:
- chapter00
---

> *Before you can understand why a language model predicts the next token conditioned on 100,000 previous tokens, or why variational autoencoders work, you need the bedrock laws of probability: sample spaces, Kolmogorov's axioms, Bayes' theorem, and random variables.*

## Background Knowledge Map: Three Building Blocks of Probability Theory

1. **Sample Spaces & Kolmogorov Axioms**: Establishes the minimal rules—non-negativity, unit total mass, and countable additivity—from which every probability identity follows without contradiction.
2. **Conditional Probability & Independence**: Formalizes how observing context $$B$$ reshapes the probability of $$A$$ via $$P(A\mid B) = P(A \cap B)/P(B)$$, and defines conditional independence—the structural assumption powering autoregressive factorization.
3. **Random Variables & Moments**: Maps abstract outcomes to real numbers so we can compute means (expected values), variances (spread), and covariances (linear co-movement) across activations and gradients.

### Step-by-Step Worked Example (Read Slowly): Autoregressive Chain Rule on a 3-Token Sequence

Consider a tiny vocabulary $$\{\text{The}, \text{cat}, \text{slept}\}$$ where a language model assigns:
- $$P(w_1 = \text{The}) = 0.40$$
- $$P(w_2 = \text{cat} \mid w_1 = \text{The}) = 0.25$$
- $$P(w_3 = \text{slept} \mid w_1 = \text{The}, w_2 = \text{cat}) = 0.50$$

1. **Apply the Chain Rule of Probability**:
   $$P(w_1, w_2, w_3) = P(w_1)\,P(w_2 \mid w_1)\,P(w_3 \mid w_1, w_2)$$
   $$P(\text{"The cat slept"}) = 0.40 \times 0.25 \times 0.50 = 0.05$$
2. **Convert to Log-Probability and Perplexity**:
   - Log-probability: $$\ln(0.05) = \ln(0.40) + \ln(0.25) + \ln(0.50) \approx -0.9163 - 1.3863 - 0.6931 = -2.9957$$.
   - Average negative log-likelihood per token: $$\text{NLL} = \frac{2.9957}{3} \approx 0.9986$$.
   - Perplexity: $$\text{PPL} = e^{0.9986} = (0.05)^{-1/3} = 20^{1/3} \approx 2.714$$.
   A single identity from basic probability—the chain rule—is literally the training and evaluation backbone of every modern LLM.

This lesson covers basic probability theory for deep learning, split into focused sections:

1. **Sample Spaces and Probability Axioms** — sample spaces, events, Kolmogorov axioms, and basic rules
2. **Conditional Probability and Independence** — conditioning, Bayes-related intuition, and independence
3. **Random Variables and Deep Learning** — random variables, MLE connection, and key takeaways

Interactive demos are included in each section.

## Video tham khảo

- **xem trước** [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, English. The axiom lesson this overview introduces.
- [Conditional Probabilities, Clearly Explained!!!](https://www.youtube.com/watch?v=_IgyaD7vOOA) — StatQuest with Josh Starmer, English. Conditional probability is the second lesson in the block.
- [MaDL - Random Variables and Probability Distributions](https://www.youtube.com/watch?v=bvBukLjHZh8) — Tübingen Machine Learning, English. Random variables are the third lesson, tied to how a network sees data.
