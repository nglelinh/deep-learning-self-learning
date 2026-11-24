---
layout: post
title: 00-04 Probability and Statistics
chapter: '00'
order: 14
owner: AI Assistant
categories:
- chapter00
lang: en
---

> *Real-world data is never clean: sensors jitter, human annotators disagree, and the same prompt admits dozens of valid continuations. Probability theory is what lets neural networks stop chasing impossible deterministic perfection and instead learn calibrated distributions over what is likely.*

## Background Knowledge Map: Probability & Statistics in Deep Learning

1. **Axiomatic Probability & Conditioning (Belief Updating)**: Provides the formal rules—sum rule, product rule, and Bayes' theorem—that govern how evidence updates uncertainty across latent and observed variables.
2. **Random Variables & Expectation (Risk Minimization)**: Frames training as minimizing expected loss $$\mathbb{E}_{(x,y)\sim p_{\text{data}}}[\ell(f_\theta(x), y)]$$, approximated in practice by empirical mini-batch averages.
3. **Parametric Distributions (Modeling Outputs & Noise)**: Links every standard neural loss directly to a likelihood assumption—Gaussian for MSE, Bernoulli for binary cross-entropy, and Categorical for softmax cross-entropy.
4. **Statistical Estimation (MLE, MAP, and Generalization)**: Connects maximum likelihood to cross-entropy minimization and shows how placing a prior on weights (MAP estimation) mathematically yields $$L_2$$ (Gaussian prior) and $$L_1$$ (Laplace prior) regularization.

### Step-by-Step Worked Example (Read Slowly): Empirical Risk vs. Expected Risk on a Mini-Batch

Suppose a binary classifier assigns predicted probability $$\hat{p}(y=1\mid x)$$ on a mini-batch of $$N = 4$$ examples with ground-truth labels and predictions:
- $$(y_1, \hat{p}_1) = (1, 0.8)$$, $$(y_2, \hat{p}_2) = (1, 0.5)$$, $$(y_3, \hat{p}_3) = (0, 0.2)$$, $$(y_4, \hat{p}_4) = (0, 0.5)$$.

1. **Compute the Likelihood of Each Sample Under a Bernoulli Model**:
   - Sample 1 ($$y=1$$): $$P(y_1 \mid x_1) = 0.8$$
   - Sample 2 ($$y=1$$): $$P(y_2 \mid x_2) = 0.5$$
   - Sample 3 ($$y=0$$): $$P(y_3 \mid x_3) = 1 - 0.2 = 0.8$$
   - Sample 4 ($$y=0$$): $$P(y_4 \mid x_4) = 1 - 0.5 = 0.5$$
2. **Compute Joint Mini-Batch Likelihood (Assuming i.i.d. Samples)**:
   $$\mathcal{L}(\theta) = 0.8 \times 0.5 \times 0.8 \times 0.5 = 0.16$$
3. **Convert to Average Negative Log-Likelihood (Empirical Binary Cross-Entropy)**:
   Using $$\ln(0.8) \approx -0.2231$$ and $$\ln(0.5) \approx -0.6931$$:
   $$\hat{R}(\theta) = -\frac{1}{4}\ln(0.16) = -\frac{1}{4}\big[2(-0.2231) + 2(-0.6931)\big] = \frac{1.8324}{4} = 0.4581\text{ nats}$$
   Maximizing the joint probability $$0.16$$ is algebraically identical to minimizing the average cross-entropy loss $$0.4581$$—and taking logarithms prevents underflow when $$N = 10^6$$.

## Probability and Statistics for Deep Learning

Probability and statistics form a crucial foundation for understanding many deep-learning problems, especially in machine learning and data science. This section introduces the essential probabilistic concepts that frequently appear in convex deep-learning, from maximum likelihood estimation to Bayesian deep-learning.

### Why Probability Matters in Deep Learning

Many deep-learning problems arise from statistical modeling:

- **Maximum Likelihood Estimation (MLE)**: Finding parameters that maximize the likelihood of observed data
- **Bayesian Deep Learning**: Using probabilistic models to guide the search for optimal solutions
- **Stochastic Deep Learning**: Dealing with uncertainty and randomness in objective functions
- **Regularization**: Adding probabilistic priors to prevent overfitting
- **Risk Minimization**: Optimizing expected loss over probability distributions

### Key Topics Covered

1. **Basic Probability Theory**: Sample spaces, events, and probability axioms
2. **Common Probability Distributions**: Normal, exponential, and other distributions crucial for deep-learning
3. **Expectation and Variance**: Computing and optimizing expected values
4. **Bayes' Theorem**: Foundation for Bayesian deep-learning and inference
5. **Statistical Estimation**: Connecting probability theory to deep-learning problems

### Connection to Deep Learning

Understanding probability helps you:

- **Formulate Problems**: Convert real-world uncertainty into mathematical deep-learning problems
- **Choose Objective Functions**: Select appropriate loss functions based on probabilistic assumptions
- **Interpret Results**: Understand confidence intervals and statistical significance of solutions
- **Handle Noise**: Deal with measurement errors and stochastic processes
- **Design Algorithms**: Develop robust deep-learning methods that work under uncertainty

This foundation will be essential as we explore how probabilistic models lead to convex deep-learning problems in machine learning, statistics, and engineering applications.

<div style="background: #e8f4fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0;">
<strong>💡 Learning Path:</strong> Start with basic probability concepts, then explore how they connect to deep-learning through maximum likelihood estimation and Bayesian methods. Each lesson builds toward understanding how uncertainty and randomness create deep-learning problems.
</div>

## Video tham khảo

- **xem trước** [L01.4 Probability Axioms](https://www.youtube.com/watch?v=pA83XtLeVig) — MIT OpenCourseWare, English. The axioms are the floor under every distribution in this block.
- [MaDL - Random Variables and Probability Distributions](https://www.youtube.com/watch?v=bvBukLjHZh8) — Tübingen Machine Learning, English. Random variables and distributions, in the notation used for deep learning.
- [MaDL - Common Probability Distributions](https://www.youtube.com/watch?v=NbzVwvE_syQ) — Tübingen Machine Learning, English. The named families (Bernoulli through Gaussian) that the distribution lessons specialize.
