---
layout: post
title: 00-04-02 Common Probability Distributions
chapter: '00'
order: 19
owner: AI Assistant
lang: en
categories:
- chapter00
---

> *Every time you choose a loss function, initialize a layer, or design a generative decoder, you are implicitly picking a probability distribution. Knowing the canonical distribution families turns ad-hoc hyperparameter guessing into principled statistical modeling.*

## Key ideas: The Distribution Toolkit of Deep Learning

1. **Discrete Families (Bernoulli, Binomial, Categorical, Poisson)**: Model binary labels, dropout masks, multi-class token predictions, and event counts.
2. **Continuous Families (Uniform, Gaussian, Exponential, Beta)**: Model weight initializations, continuous regression noise, latent spaces in VAEs/diffusion models, and calibrated probabilities on $$[0, 1]$$.
3. **Multivariate Families (Multivariate Normal & Covariance Geometry)**: Capture feature correlations across dimensions, underpinning PCA, Mahalanobis outlier detection, and Gaussian diffusion transitions.
4. **Distribution-to-Loss Duality**: Every standard deep learning loss is the negative log-likelihood (NLL) of a specific distribution—matching your model's output distribution to the data geometry prevents severe bias.

### Worked example: Matching a Distribution to Its Exact Neural Loss

Suppose a network predicts a scalar mean $$\mu_\theta(x) = 3.0$$ for a continuous target $$y = 5.0$$ under a Gaussian model $$y \mid x \sim \mathcal{N}(\mu_\theta(x), \sigma^2)$$ with fixed $$\sigma = 2.0$$.

1. **Write the Gaussian Density at $$y = 5.0$$**:
   $$p(y \mid x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{(y - \mu_\theta(x))^2}{2\sigma^2}\right)$$
2. **Take the Negative Log-Likelihood**:
   $$-\ln p(y \mid x) = \ln(\sqrt{2\pi}\sigma) + \frac{(y - \mu_\theta(x))^2}{2\sigma^2}$$
3. **Substitute Numbers ($$\sigma = 2.0, \mu_\theta = 3.0, y = 5.0$$)**:
   - Constant term (independent of $$\theta$$): $$\ln(2\sqrt{2\pi}) \approx \ln(5.0133) \approx 1.6121$$.
   - Data-fit term: $$\frac{(5.0 - 3.0)^2}{2(2.0)^2} = \frac{4.0}{8.0} = 0.5000$$.
   - Total NLL: $$1.6121 + 0.5000 = 2.1121\text{ nats}$$.
4. **Key Insight**:
   Because $$\ln(\sqrt{2\pi}\sigma)$$ and $$\frac{1}{2\sigma^2}$$ are constants with respect to $$\theta$$, minimizing Gaussian NLL produces the exact same optimal weights as minimizing Mean Squared Error $$(y - \mu_\theta(x))^2$$.

This lesson surveys probability distributions used throughout deep learning:

1. **Discrete Distributions** — Bernoulli, Binomial, Categorical, Poisson (with interactive demos)
2. **Continuous Distributions** — Uniform, Gaussian, Exponential, Laplace, and related densitites
3. **Multivariate Distributions and Deep Learning** — multivariate Gaussians and modeling applications

Work through each section in order; demos are included where they support the material.

## Video tham khảo

- **xem trước** [The Main Ideas behind Probability Distributions](https://www.youtube.com/watch?v=oI3hZJqXJuc) — StatQuest with Josh Starmer, English. What a distribution is, before the discrete, continuous, and multivariate lessons.
- [MaDL - Common Probability Distributions](https://www.youtube.com/watch?v=NbzVwvE_syQ) — Tübingen Machine Learning, English. Names the families this block then splits into their own lessons.
- [The Normal Distribution, Clearly Explained!!!](https://www.youtube.com/watch?v=rzFX5NWojp0) — StatQuest with Josh Starmer, English. The Gaussian is the continuous case the next lessons keep returning to.
