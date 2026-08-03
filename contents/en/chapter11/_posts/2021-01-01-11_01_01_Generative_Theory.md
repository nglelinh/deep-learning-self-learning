---
layout: post
title: 11-01-01 Generative Models Theory
chapter: '11'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter11
---

# Generative Models: Learning to Create

## 1. Concept Overview

Generative models represent a fundamental shift in how we think about machine learning. While discriminative models learn to map inputs to outputs—classifying images into categories, translating sentences between languages, or predicting stock prices from historical data—generative models learn to understand and reproduce the underlying structure of data itself. They ask a more ambitious question: given examples of some data distribution, can we learn to generate new, realistic samples from that distribution? This capability opens remarkable possibilities: creating photorealistic images of people who don't exist, composing music in the style of Bach, designing molecules with desired properties, or augmenting limited datasets with synthetic examples.

Understanding why generative modeling matters requires appreciating what's fundamentally different about generation versus discrimination. A discriminative classifier for dog breeds learns features sufficient to distinguish breeds—the shape of ears, coat patterns, size. It doesn't need to understand how these features combine to form a coherent dog, or what makes a dog anatomically plausible versus impossible. A generative model must learn deeper structure: how pixels organize into textures, how textures form objects, how objects compose into scenes, and crucially, what combinations are realistic versus unrealistic. This deeper understanding means generative models often learn richer representations than discriminative models, making them valuable even when generation itself isn't the end goal.

The mathematical framework for generative models is rooted in probability theory and statistical modeling. We assume data $$\mathbf{x}$$ comes from some unknown distribution $$p_{\text{data}}(\mathbf{x})$$. Our goal is to learn a model distribution $$p_{\text{model}}(\mathbf{x}; \theta)$$ parameterized by $$\theta$$ (neural network weights) that approximates $$p_{\text{data}}$$. If we succeed, sampling from $$p_{\text{model}}$$ should produce data indistinguishable from samples from $$p_{\text{data}}$$. This probabilistic framing connects generative models to maximum likelihood estimation, variational inference, and other foundational concepts in statistics, while the use of neural networks for the model provides unprecedented flexibility in the functional forms we can represent.

Different generative modeling approaches make different tradeoffs between sample quality, training stability, theoretical guarantees, and computational requirements. Autoregressive models like PixelCNN explicitly model $$p(\mathbf{x}) = \prod_i p(x_i | x_{<i})$$, decomposing generation into sequential conditional distributions. They provide exact likelihoods and stable training but generate slowly (one pixel at a time). Variational autoencoders introduce latent variables $$\mathbf{z}$$ and model $$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$$, optimizing a tractable lower bound on likelihood. They enable fast sampling and provide a principled probabilistic framework but often generate somewhat blurry samples. Generative adversarial networks sidestep explicit density modeling entirely, using adversarial training to learn a generator that implicitly samples from $$p_{\text{data}}$$. They often produce the sharpest, most realistic samples but suffer from training instability and mode collapse.

The practical applications of generative models extend far beyond novelty. In computer vision, they enable data augmentation (generating additional training examples), super-resolution (upscaling low-resolution images), inpainting (filling missing regions), and style transfer (applying artistic styles to photographs). In natural language processing, they power text generation, machine translation through generative seq2seq models, and data augmentation for low-resource languages. In drug discovery, they generate molecular structures with desired properties. In creative applications, they assist artists and designers. In anomaly detection, they identify outliers by measuring how well they fit the learned distribution. Understanding generative models opens this vast application space while providing insights into data structure that benefit even purely discriminative tasks.

Yet generative modeling is fundamentally harder than discriminative learning in several ways. The space of possible outputs is exponentially larger than the space of labels ($$2^{784}$$ possible MNIST images vs 10 labels). The learned distribution must capture complex dependencies between output dimensions (pixels aren't independent—nearby pixels are correlated, object parts must be anatomically coherent). Evaluation is challenging—we can't simply compute accuracy as we can for classification. And generation requires understanding not just what separates classes but what makes examples realistic, a higher bar of understanding. These challenges make generative modeling an active research area where major innovations continue to emerge regularly.

## 2. Mathematical Foundation

The mathematical foundation of generative models rests on probability theory, likelihood estimation, and information theory. Let's build these concepts systematically to understand what we're optimizing when training generative models and why different approaches lead to different algorithms.

### Probability Density and the Data Distribution

We assume our training data $$\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(m)}\}$$ consists of independent samples from some unknown distribution $$p_{\text{data}}(\mathbf{x})$$. For images, $$\mathbf{x}$$ might be $$28 \times 28 = 784$$ dimensional (MNIST) or $$224 \times 224 \times 3 = 150,528$$ dimensional (ImageNet). The distribution $$p_{\text{data}}$$ assigns probability density to each possible $$\mathbf{x}$$, with high density for realistic images (actual digits, photographs of objects) and low or zero density for unrealistic ones (random noise, anatomically impossible scenes).

Our goal is to learn a parametric model $$p_{\text{model}}(\mathbf{x}; \theta)$$ that approximates $$p_{\text{data}}$$. The parameters $$\theta$$ (neural network weights) should be set such that the model assigns high probability to training examples and, by generalization, to held-out examples from the same distribution. The standard approach is maximum likelihood estimation:

$$\theta^* = \arg\max_\theta \prod_{i=1}^{m} p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$

Taking logarithms (for numerical stability and mathematical convenience):

$$\theta^* = \arg\max_\theta \sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta) = \arg\max_\theta \frac{1}{m}\sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$

The average log-likelihood $$\frac{1}{m}\sum_{i=1}^{m} \log p_{\text{model}}(\mathbf{x}^{(i)}; \theta)$$ approximates the expected log-likelihood under the data distribution:

$$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\text{model}}(\mathbf{x}; \theta)]$$

Maximizing this expectation is equivalent to minimizing the Kullback-Leibler divergence between data and model distributions:

$$KL(p_{\text{data}} \| p_{\text{model}}) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\text{data}}(\mathbf{x}) - \log p_{\text{model}}(\mathbf{x}; \theta)]$$

Since $$p_{\text{data}}$$ is fixed, minimizing KL divergence is equivalent to maximizing expected log-likelihood. This connects maximum likelihood to information theory and provides a principled measure of how well our model approximates the true distribution.

### Explicit vs Implicit Density Models

The challenge in generative modeling is that for high-dimensional data, explicitly defining $$p_{\text{model}}(\mathbf{x}; \theta)$$ that's both flexible (can approximate complex distributions) and tractable (we can actually compute it and optimize it) is difficult.

**Explicit density models** directly parameterize $$p_{\text{model}}(\mathbf{x}; \theta)$$:

*Autoregressive models* use the chain rule to factorize density:

$$p(\mathbf{x}) = p(x_1) p(x_2|x_1) p(x_3|x_1, x_2) \cdots p(x_d|x_1, \ldots, x_{d-1}) = \prod_{i=1}^{d} p(x_i|\mathbf{x}_{<i})$$

Each conditional $$p(x_i|\mathbf{x}_{<i})$$ is modeled with a neural network. This is exact—we can compute $$p(\mathbf{x})$$ for any $$\mathbf{x}$$—but generation is slow (must generate dimensions sequentially) and the conditional independence assumptions might be restrictive.

*Flow-based models* use invertible transformations $$\mathbf{x} = f(\mathbf{z})$$ where $$\mathbf{z} \sim p_{\mathbf{z}}$$ is simple (Gaussian). The change of variables formula gives:

$$p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f^{-1}(\mathbf{x})) \left|\det \frac{\partial f^{-1}}{\partial \mathbf{x}}\right|$$

This is exact and allows both density evaluation and fast sampling, but requires carefully designed architectures to ensure invertibility and tractable Jacobian determinants.

**Implicit density models** define a stochastic procedure for sampling without explicitly specifying $$p_{\text{model}}(\mathbf{x})$$:

*GANs* learn a generator $$G: \mathcal{Z} \to \mathcal{X}$$ such that if $$\mathbf{z} \sim p_{\mathbf{z}}$$ then $$G(\mathbf{z})$$ has distribution approximating $$p_{\text{data}}$$. We never compute $$p_{\text{model}}$$ but can sample efficiently. Training uses adversarial objective instead of likelihood.

*VAEs* partially explicit: they model $$p(\mathbf{x}|\mathbf{z})$$ explicitly but marginalize over latent $$\mathbf{z}$$ using variational approximation. They maximize a lower bound on log-likelihood (ELBO) instead of likelihood itself.

The choice between explicit and implicit, between different model families, depends on priorities: do we need exact likelihood (for anomaly detection, compression)? Do we need fast sampling (for real-time generation)? Do we prioritize sample quality over training stability? Understanding these tradeoffs guides model selection for specific applications.

### Latent Variable Models

Many generative models introduce latent variables $$\mathbf{z}$$ representing hidden factors of variation. The generative process becomes:

1. Sample latent code: $$\mathbf{z} \sim p(\mathbf{z})$$ (typically $$\mathcal{N}(0, I)$$)
2. Generate data: $$\mathbf{x} \sim p(\mathbf{x}|\mathbf{z}; \theta)$$

The marginal distribution is:

$$p(\mathbf{x}; \theta) = \int p(\mathbf{x}|\mathbf{z}; \theta) p(\mathbf{z}) d\mathbf{z}$$

This framework is powerful because latent variables can represent interpretable factors (for faces: pose, lighting, expression, identity) and low-dimensional latent spaces can capture high-dimensional data manifolds. The challenge is that computing the integral for exact likelihood requires integrating over all possible latent codes, which is intractable for continuous $$\mathbf{z}$$. Different generative models address this differently:

VAEs use variational inference, introducing an encoder $$q(\mathbf{z}|\mathbf{x}; \phi)$$ that approximates the posterior $$p(\mathbf{z}|\mathbf{x})$$ and optimizing the Evidence Lower BOund (ELBO):

$$\log p(\mathbf{x}; \theta) \geq \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x}; \phi)}[\log p(\mathbf{x}|\mathbf{z}; \theta)] - KL(q(\mathbf{z}|\mathbf{x}; \phi) \| p(\mathbf{z}))$$

This lower bound is tractable—we can estimate it via sampling and optimize it via backpropagation through the reparameterization trick.

GANs bypass the likelihood computation entirely, directly training the generator $$G(\mathbf{z}; \theta)$$ to produce samples indistinguishable from data through adversarial training. We never compute $$p(\mathbf{x})$$ but implicitly learn to sample from it.

### Evaluation Metrics

Evaluating generative models is challenging because we care about distribution matching, not just performance on specific examples. Several metrics have been proposed:

**Log-likelihood** (when computable): Measures how well the model assigns probability to test data. Higher is better. However, high likelihood doesn't guarantee good samples (a model memorizing training data has perfect likelihood on training set).

**Inception Score** (IS): Generates samples, classifies them with Inception network, computes:

$$IS = \exp(\mathbb{E}_{\mathbf{x} \sim p_G}[KL(p(y|\mathbf{x}) \| p(y))])$$

Measures both quality (samples should be confidently classified) and diversity (should cover all classes). Higher is better, but IS has issues (biased toward ImageNet classes, doesn't detect memorization).

**Fréchet Inception Distance** (FID): Compares statistics of real and generated samples in Inception feature space, treating them as Gaussians and computing:

$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

Lower FID indicates closer distributions. More reliable than IS but still imperfect (assumes Gaussian features).

Understanding these metrics' limitations is as important as using them. They correlate with perceptual quality but aren't perfect. Visual inspection remains crucial. For specific applications, domain-specific metrics (face identity preservation for face generation, molecular validity for drug design) often matter more than generic metrics.

## 3. Example / Intuition

To build intuition for generative models, let's think about learning to generate handwritten digits. Imagine you've never seen the digit "3" but have seen thousands of other digits. Could you invent a plausible "3"? Probably not—you lack understanding of what makes a valid digit, what "3" specifically looks like, how strokes connect.

Now suppose you see thousands of examples of each digit including "3". You could learn: "3" has two rounded parts, typically connected, orientation upright, strokes smooth. With this understanding, you could generate novel "3"s—not copies of training examples but new variations following the learned pattern. This is what generative models do, but discovered automatically from data rather than described verbally.

Consider the different approaches to this task:

**Autoregressive approach**: Generate the digit pixel by pixel, left-to-right, top-to-bottom. At each position, predict the pixel value conditioned on all previous pixels. This ensures each pixel is consistent with preceding ones (if the top already looks like "3", continue that pattern). The sequential generation provides strong guidance but is slow—784 sequential decisions for MNIST.

**VAE approach**: Learn a latent space where different regions correspond to different digits and variations. To generate a "3", sample a latent code from the "3 region" (learned during training) and decode it through the decoder network. The latent space provides efficient generation (sample once, decode once) and enables interpolation (smoothly morph between digits). However, the reconstruction-based training might produce blurry samples because pixel-wise MSE doesn't capture perceptual quality well.

**GAN approach**: Train a generator to fool a discriminator that's trying to detect fakes. The generator learns whatever mapping from noise to images makes the discriminator unable to detect fakes. This adversarial training doesn't require explicit pixel-wise reconstruction, allowing the generator to prioritize perceptual realism over exact pixel matching. The result is often sharper, more realistic samples, though training can be unstable and mode collapse might occur (generator only learns to create certain types of "3"s).

Let's trace through a concrete example with a simple toy dataset: 2D points forming two clusters (representing two modes of a distribution). The true distribution $$p_{\text{data}}$$ is a mixture of two Gaussians:

$$p_{\text{data}}(\mathbf{x}) = 0.5 \mathcal{N}(\mathbf{x}; [2, 2], I) + 0.5 \mathcal{N}(\mathbf{x}; [-2, -2], I)$$

**Autoregressive model**: Models $$p(x_2|x_1)p(x_1)$$. For the first mode centered at [2, 2], it learns $$p(x_1) \approx \mathcal{N}(2, 1)$$ and $$p(x_2|x_1) \approx \mathcal{N}(2, 1)$$ (roughly independent since we're using Gaussians, but could learn correlations). Generation: sample $$x_1 \sim p(x_1)$$, then $$x_2 \sim p(x_2|x_1)$$.

**VAE**: Introduces latent $$z \in \mathbb{R}$$. Learns that $$z < 0$$ maps to mode at [-2, -2] and $$z > 0$$ maps to mode at [2, 2]. To generate, sample $$z \sim \mathcal{N}(0, 1)$$, decode to $$\mathbf{x}$$. The latent space smoothly varies from one mode to another.

**GAN**: Generator learns to map 1D noise $$z$$ to 2D points such that the discriminator (which sees both real samples from the two Gaussians and generated samples) cannot distinguish real from fake. The generator might learn a nonlinear function that maps $$z \in [-3, 0]$$ to the first mode and $$z \in [0, 3]$$ to the second mode.

Each approach successfully generates from both modes if trained properly, but they differ in how they represent the distribution, training stability, and generation procedure. Understanding these differences through simple examples builds intuition for their behavior on complex data like images.

