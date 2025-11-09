---
layout: post
title: 11-01 Generative Models - Core Concepts
chapter: '11'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter11
lesson_type: required
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

## 4. Code Snippet

Let's implement different generative modeling approaches on a toy dataset to understand their mechanics:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate toy data: mixture of 8 Gaussians in a circle
def generate_mixture_data(n_samples=10000):
    """
    Generate 2D data from mixture of 8 Gaussians arranged in a circle.
    
    This toy dataset lets us visualize learned distributions and compare
    different generative modeling approaches. Each Gaussian represents a
    "mode" - generative models should learn to generate from all modes.
    """
    n_modes = 8
    radius = 2.0
    std = 0.02
    
    # Angles for modes evenly spaced around circle
    thetas = np.linspace(0, 2*np.pi, n_modes, endpoint=False)
    
    # Centers of Gaussians
    centers = np.array([[radius * np.cos(t), radius * np.sin(t)] for t in thetas])
    
    # Sample from mixture
    data = []
    for _ in range(n_samples):
        # Choose mode uniformly
        mode_idx = np.random.randint(n_modes)
        # Sample from chosen Gaussian
        sample = centers[mode_idx] + std * np.random.randn(2)
        data.append(sample)
    
    return np.array(data), centers

# Generate training data
print("="*70)
print("Generative Models on Toy 2D Dataset")
print("="*70)

data_train, true_centers = generate_mixture_data(n_samples=10000)
data_tensor = torch.FloatTensor(data_train)

print(f"Generated {len(data_train)} samples from 8 Gaussian modes")
print(f"Data shape: {data_train.shape}")  # (10000, 2)
print(f"Mode centers:\n{true_centers.round(3)}")

# 1. Simple Autoregressive Model
class AutoregressiveModel(nn.Module):
    """
    Simple 2D autoregressive model: p(x) = p(x2|x1) * p(x1)
    
    Models p(x1) as mixture of logistics, p(x2|x1) as conditional mixture.
    Demonstrates explicit density modeling - we can compute p(x) exactly.
    """
    
    def __init__(self, n_components=10):
        super().__init__()
        
        # p(x1): mixture of logistics
        self.x1_logits = nn.Parameter(torch.randn(n_components))
        self.x1_means = nn.Parameter(torch.randn(n_components))
        self.x1_scales = nn.Parameter(torch.ones(n_components) * 0.1)
        
        # p(x2|x1): neural network outputting mixture parameters
        self.x2_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_components * 3)  # logits, means, scales for mixture
        )
        
        self.n_components = n_components
    
    def log_prob(self, x):
        """
        Compute log p(x) = log p(x1) + log p(x2|x1)
        
        This is what makes it an explicit density model - we can evaluate
        probability of any point, enabling maximum likelihood training.
        """
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        
        # log p(x1): log mixture of logistics
        logits_1 = self.x1_logits.unsqueeze(0)  # (1, n_components)
        means_1 = self.x1_means.unsqueeze(0)
        scales_1 = torch.abs(self.x1_scales).unsqueeze(0) + 0.01
        
        # Logistic log-prob for each component
        z = (x1 - means_1) / scales_1
        log_probs_1 = -z - 2 * torch.nn.functional.softplus(-z) - torch.log(scales_1)
        
        # Mixture log-prob using log-sum-exp
        log_p_x1 = torch.logsumexp(logits_1 + log_probs_1, dim=1) - \
                   torch.logsumexp(logits_1, dim=1)
        
        # log p(x2|x1): conditional mixture
        params_2 = self.x2_net(x1)
        params_2 = params_2.view(-1, self.n_components, 3)
        
        logits_2 = params_2[:, :, 0]
        means_2 = params_2[:, :, 1]
        scales_2 = torch.abs(params_2[:, :, 2]) + 0.01
        
        z_2 = (x2 - means_2) / scales_2
        log_probs_2 = -z_2 - 2 * torch.nn.functional.softplus(-z_2) - torch.log(scales_2)
        
        log_p_x2_given_x1 = torch.logsumexp(logits_2 + log_probs_2, dim=1) - \
                            torch.logsumexp(logits_2, dim=1)
        
        # Total log-prob
        return log_p_x1 + log_p_x2_given_x1
    
    def sample(self, n_samples):
        """
        Generate samples: first sample x1, then x2|x1
        
        Demonstrates sequential generation - characteristic of autoregressive.
        Exact sampling from learned distribution.
        """
        # Sample x1
        probs_1 = torch.softmax(self.x1_logits, dim=0)
        components = torch.multinomial(probs_1, n_samples, replacement=True)
        
        means = self.x1_means[components]
        scales = torch.abs(self.x1_scales[components])
        
        # Logistic samples (approximately using Gaussian)
        x1 = means + scales * torch.randn(n_samples)
        
        # Sample x2|x1
        params_2 = self.x2_net(x1.unsqueeze(1))
        params_2 = params_2.view(n_samples, self.n_components, 3)
        
        # Sample component for each x1
        logits_2 = params_2[:, :, 0]
        probs_2 = torch.softmax(logits_2, dim=1)
        components_2 = torch.multinomial(probs_2, 1).squeeze()
        
        # Get parameters for chosen components
        means_2 = params_2[range(n_samples), components_2, 1]
        scales_2 = torch.abs(params_2[range(n_samples), components_2, 2])
        
        x2 = means_2 + scales_2 * torch.randn(n_samples)
        
        return torch.stack([x1, x2], dim=1)

# Train autoregressive model
print("\n1. Training Autoregressive Model (Explicit Density)")
print("-" * 70)

ar_model = AutoregressiveModel(n_components=10)
ar_optimizer = optim.Adam(ar_model.parameters(), lr=0.001)

ar_model.train()
for epoch in range(200):
    # Shuffle data
    indices = torch.randperm(len(data_tensor))
    
    # Mini-batch training
    batch_size = 128
    epoch_loss = 0
    
    for i in range(0, len(data_tensor), batch_size):
        batch = data_tensor[indices[i:i+batch_size]]
        
        # Compute negative log-likelihood
        log_probs = ar_model.log_prob(batch)
        loss = -log_probs.mean()  # Negative log-likelihood
        
        ar_optimizer.zero_grad()
        loss.backward()
        ar_optimizer.step()
        
        epoch_loss += loss.item()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: NLL = {epoch_loss/(len(data_tensor)/batch_size):.4f}")

# Generate samples
ar_model.eval()
with torch.no_grad():
    samples_ar = ar_model.sample(1000)
    print(f"\nGenerated {len(samples_ar)} samples")
    print(f"Sample mean: {samples_ar.mean(dim=0).numpy()}")
    print(f"Data mean:   {data_tensor.mean(dim=0).numpy()}")

# 2. Simple GAN for comparison
print("\n2. Training GAN (Implicit Density)")
print("-" * 70)

class ToyGenerator(nn.Module):
    """Simple generator for 2D data"""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output 2D points
        )
    
    def forward(self, z):
        return self.net(z)

class ToyDiscriminator(nn.Module):
    """Simple discriminator for 2D data"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

gen = ToyGenerator(latent_dim=2)
disc = ToyDiscriminator()

gen_optimizer = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

print("Training GAN on mixture of Gaussians...")

for epoch in range(1000):
    # Train discriminator
    for _ in range(1):  # k discriminator steps per generator step
        disc.zero_grad()
        
        # Real data
        batch_real = data_tensor[torch.randint(len(data_tensor), (128,))]
        labels_real = torch.ones(128, 1)
        output_real = disc(batch_real)
        loss_d_real = criterion(output_real, labels_real)
        
        # Fake data
        z = torch.randn(128, 2)
        fake = gen(z)
        labels_fake = torch.zeros(128, 1)
        output_fake = disc(fake.detach())
        loss_d_fake = criterion(output_fake, labels_fake)
        
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        disc_optimizer.step()
    
    # Train generator
    gen.zero_grad()
    z = torch.randn(128, 2)
    fake = gen(z)
    output = disc(fake)
    labels_real_for_g = torch.ones(128, 1)
    loss_g = criterion(output, labels_real_for_g)
    
    loss_g.backward()
    gen_optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}: D_loss = {loss_d.item():.4f}, "
              f"G_loss = {loss_g.item():.4f}, "
              f"D(real) = {output_real.mean():.3f}, "
              f"D(fake) = {output_fake.mean():.3f}")

# Generate samples
gen.eval()
with torch.no_grad():
    z_sample = torch.randn(1000, 2)
    samples_gan = gen(z_sample)
    
print(f"\nGAN generated {len(samples_gan)} samples")
print(f"Checking mode coverage (samples should cluster around 8 centers)...")

# Check if GAN covers all modes (mode collapse detection)
# For each true center, count nearby generated samples
for i, center in enumerate(true_centers):
    distances = torch.norm(samples_gan - torch.FloatTensor(center), dim=1)
    nearby = (distances < 0.5).sum().item()
    print(f"  Mode {i} (center {center.round(2)}): {nearby} nearby samples")

if all((torch.norm(samples_gan - torch.FloatTensor(c), dim=1) < 0.5).sum() > 50 
       for c in true_centers):
    print("✓ GAN covers all modes successfully!")
else:
    print("✗ Mode collapse detected - some modes have few/no samples")

print("\n" + "="*70)
print("Generative Modeling Comparison")
print("="*70)
print("\nAutoregressive Model:")
print("  + Exact likelihood computable")
print("  + Stable training")
print("  - Sequential generation (slow)")
print("  - Strong ordering assumptions")

print("\nGAN:")
print("  + Fast parallel generation")
print("  + Often high sample quality")
print("  - No explicit likelihood")
print("  - Training can be unstable")
print("  - Mode collapse risk")

print("\nVAE (next chapter):")
print("  + Explicit latent space")
print("  + Stable training")
print("  + Fast generation")
print("  - Samples sometimes blurry")
```

Demonstrate likelihood-based evaluation:

```python
print("\n" + "="*70)
print("Evaluating Generative Model Quality")
print("="*70)

# For autoregressive model, we can compute exact likelihood
ar_model.eval()
with torch.no_grad():
    # Test set (held-out data from same distribution)
    data_test, _ = generate_mixture_data(n_samples=1000)
    data_test_tensor = torch.FloatTensor(data_test)
    
    # Compute log-likelihood on test set
    test_log_probs = ar_model.log_prob(data_test_tensor)
    avg_test_ll = test_log_probs.mean().item()
    
    print(f"Autoregressive Model Test Log-Likelihood: {avg_test_ll:.4f}")
    print(f"Higher is better - model assigns high probability to test data")
    
    # Generate and evaluate (samples should have similar likelihood to real data)
    samples_ar_eval = ar_model.sample(1000)
    samples_log_probs = ar_model.log_prob(samples_ar_eval)
    avg_sample_ll = samples_log_probs.mean().item()
    
    print(f"Generated Samples Log-Likelihood: {avg_sample_ll:.4f}")
    print(f"Should be similar to test LL if model is good")
    
    diff = abs(avg_test_ll - avg_sample_ll)
    if diff < 0.5:
        print(f"✓ Small difference ({diff:.4f}) indicates good model!")
    else:
        print(f"✗ Large difference ({diff:.4f}) indicates issues")

# For GAN, we can't compute likelihood, so we use proxy metrics
print("\nGAN Evaluation (without explicit likelihood):")
print("  - Visual inspection (do samples look realistic?)")
print("  - Mode coverage (do samples span all modes?)")
print("  - Diversity (are samples varied or repetitive?)")
print("  - Discriminator score (should be ~0.5 for good generator)")

with torch.no_grad():
    disc_scores_real = disc(data_test_tensor)
    disc_scores_fake = disc(samples_gan)
    
    print(f"\nDiscriminator scores:")
    print(f"  Real data: {disc_scores_real.mean():.3f} (should be ~1.0 if D is good)")
    print(f"  Generated: {disc_scores_fake.mean():.3f} (should be ~0.5 at equilibrium)")
    
    if disc_scores_fake.mean() > 0.4 and disc_scores_fake.mean() < 0.6:
        print("✓ Generator successfully fools discriminator!")
```

## 5. Related Concepts

Generative models connect to density estimation, a classical problem in statistics where we try to estimate probability density functions from samples. Traditional methods like kernel density estimation or parametric fitting (Gaussian mixture models) work well in low dimensions but scale poorly to high dimensions due to the curse of dimensionality. Neural network-based generative models overcome this by learning hierarchical features that capture data structure rather than explicitly representing density in raw input space. A deep generative model effectively performs density estimation in a learned feature space where data structure is simpler, then maps back to input space. This perspective helps appreciate why deep generative models succeed where classical methods fail.

The relationship to unsupervised and self-supervised learning is profound. Generative models learn representations without labels, discovering structure purely from data patterns. The features learned during generative modeling often transfer well to downstream supervised tasks—autoencoders provide good initializations, GAN discriminators learn useful features, VAE encoders create meaningful latent spaces. This connects to the broader theme that large-scale unsupervised pre-training (like BERT or GPT language modeling, which are generative tasks) followed by supervised fine-tuning often outperforms purely supervised learning, especially in limited-data regimes. Understanding generative models provides insight into why unsupervised learning works and what representations emerge from generative objectives.

Generative models connect to data augmentation through their ability to generate synthetic training examples. For imbalanced datasets (many examples of common classes, few of rare classes), generative models can synthesize additional minority class examples. For expensive-to-label data (medical images requiring expert annotation), generated examples can augment limited labeled sets. However, care is required: if the generative model hasn't learned the true distribution accurately, synthetic examples might introduce bias. Best practice is to validate that generated examples aid rather than hurt downstream task performance.

The evolution of evaluation metrics for generative models reflects ongoing challenges in measuring quality and diversity. Early GANs used human evaluation (time-consuming, not reproducible) or binary classifier tests (can a classifier distinguish real from fake?). Inception Score and FID provided automated metrics but have known biases and failure modes. Recent work explores learned perceptual metrics (measuring distance in learned feature spaces), precision-recall tradeoffs (quantifying quality vs diversity separately), and likelihood-based methods (for models that provide likelihoods). Understanding that no single metric is perfect guides practitioners to use multiple complementary evaluations rather than optimizing for any single metric.

Finally, generative models connect to the fundamental question of what neural networks learn. By training networks to generate complex data like images or text, we're essentially asking: what patterns, structures, and regularities exist in this data, and can networks discover them automatically? The fact that neural networks can learn to generate photorealistic faces, coherent paragraphs, or valid molecular structures demonstrates they're capturing deep statistical regularities, not just memorizing. This learning of latent structure has implications beyond generation—it suggests neural networks are discovering representations that reflect genuine structure in the world, not just fitting training data.

## 6. Fundamental Papers

**["A tutorial on Energy-Based Learning" (2006)](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)**  
*Author*: Yann LeCun  
While not specifically about modern generative models, this tutorial established the energy-based framework that underlies much generative modeling. LeCun showed how many learning problems can be formulated as learning energy functions that assign low energy to correct/realistic outputs and high energy to incorrect/unrealistic ones. Generative models fit this framework: they learn energy landscapes where data has low energy. The tutorial covered Boltzmann machines, contrastive divergence, and other techniques that influenced later work on deep generative models. Understanding energy-based models provides theoretical foundation for why certain training procedures (like contrastive divergence or score matching) work and connects generative modeling to statistical physics and probabilistic inference. While modern generative models often use different training procedures (backpropagation with reparameterization for VAEs, adversarial training for GANs), the energy-based perspective remains valuable for understanding what these models are fundamentally doing.

**["NADE: The Neural Autoregressive Distribution Estimator" (2011)](http://proceedings.mlr.press/v15/larochelle11a.html)**  
*Authors*: Hugo Larochelle, Iain Murray  
This paper introduced NADE, an efficient autoregressive model that tractably computes $$p(\mathbf{x}) = \prod_i p(x_i|\mathbf{x}_{<i})$$ using neural networks for the conditionals. The key innovation was weight sharing: rather than training separate networks for each conditional, NADE uses a single neural network with shared parameters, making it efficient and preventing overfitting. The paper demonstrated that autoregressive models could compete with more complex approaches like restricted Boltzmann machines while providing exact likelihood computation and stable training. NADE influenced subsequent autoregressive models like PixelRNN/PixelCNN (for images) and WaveNet (for audio), establishing autoregressive modeling as a viable approach for complex, high-dimensional data. The work showed that explicit density modeling—directly parameterizing $$p(\mathbf{x})$$—was practical for deep learning, not just classical statistics.

**["Auto-Encoding Variational Bayes" (2014)](https://arxiv.org/abs/1312.6114)**  
*Authors*: Diederik P. Kingma, Max Welling  
This foundational paper introduced Variational Autoencoders, combining variational inference with neural networks to create a scalable framework for generative modeling with latent variables. The key contribution was the reparameterization trick: instead of sampling $$\mathbf{z} \sim q(\mathbf{z}|\mathbf{x})$$ (which isn't differentiable with respect to $$q$$'s parameters), rewrite sampling as $$\mathbf{z} = \mu + \sigma \odot \boldsymbol{\epsilon}$$ where $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$. This deterministic function of parameters ($$\mu, \sigma$$) and external randomness ($$\boldsymbol{\epsilon}$$) enables backpropagation through sampling, making variational inference trainable via gradient descent. The paper showed VAEs could learn meaningful latent representations and generate novel samples while providing a principled probabilistic framework (unlike GANs which were concurrent but more heuristic initially). VAEs influenced countless subsequent works and established that latent variable models could scale to complex data through careful algorithm design. The ELBO objective and reparameterization trick have become fundamental tools in probabilistic deep learning.

**["Generative Adversarial Networks" (2014)](https://arxiv.org/abs/1406.2661)**  
*Authors*: Ian Goodfellow et al.  
The GAN paper revolutionized generative modeling by introducing adversarial training as an alternative to maximum likelihood. By framing generation as a game between generator and discriminator, GANs enabled learning implicit density models that generate high-quality samples without requiring explicit density computation or intractable integrals. The paper's theoretical analysis—showing that at Nash equilibrium, the generator recovers the data distribution—provided foundation while empirical results demonstrated practical viability. GANs spawned enormous subsequent research addressing training stability, mode collapse, and architecture design, becoming one of the most influential ideas in modern machine learning. The adversarial framework has been applied beyond generation to domain adaptation, robust training, and semi-supervised learning, demonstrating how a novel training paradigm can impact the field broadly.

**["Normalizing Flows for Probabilistic Modeling and Inference" (2019)](https://arxiv.org/abs/1912.02762)**  
*Authors*: George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan  
This comprehensive review unified normalizing flows—generative models based on invertible transformations—explaining their theoretical foundations and practical implementations. Flows learn bijective mappings $$\mathbf{x} = f(\mathbf{z})$$ where $$\mathbf{z}$$ has simple density (Gaussian) and $$f$$ is invertible with tractable Jacobian determinant. This enables exact likelihood computation (unlike GANs) and fast sampling (unlike autoregressive models). The paper covered the landscape of flow architectures (coupling flows, autoregressive flows, continuous flows), their theoretical properties, and applications. Flows are less commonly used than VAEs or GANs for image generation but excel in tasks requiring exact density (anomaly detection, compression) or specific structure (molecular generation where validity constraints matter). Understanding flows completes the generative modeling picture, showing the tradeoff space between likelihood tractability, sampling efficiency, and architectural flexibility.

## Common Pitfalls and Tricks

The most fundamental mistake in generative modeling is evaluating models solely on training set likelihood or reconstruction quality. A model that memorizes training examples achieves perfect training likelihood but generates no novel examples—it fails as a generative model despite optimizing the objective perfectly. Symptoms include generated samples being near-identical to training examples and poor test set likelihood. Detection requires checking nearest neighbors in training set for each generated sample (if always very close, likely memorization) and evaluating on held-out data. Prevention includes proper regularization (weight decay, dropout), using validation set for model selection, and architectures that encourage generalization (bottlenecks in autoencoders, discriminator in GANs forcing novelty).

Choosing inappropriate reconstruction losses causes perceptual mismatches between what the model optimizes and what humans care about. Pixel-wise MSE treats all pixels equally, but human vision is non-uniform—we're more sensitive to structure and edges than to smooth regions. An MSE-optimal reconstruction might be blurry (averaging out details) while looking poor perceptually. Conversely, a reconstruction with slightly shifted edges (high MSE) might look perceptually similar. Solutions include perceptual losses (measuring distance in feature space of a pre-trained network like VGG), adversarial losses (using a discriminator to judge realism), or structured losses (measuring gradient similarity, not just pixel similarity). Understanding that loss functions embody assumptions about what's important guides appropriate choices for specific applications.

For latent variable models, choosing latent dimensionality involves subtle tradeoffs. Too small (2-3 dimensions) enables visualization but may not capture data complexity, causing poor reconstructions. Too large (approaching input dimension) enables perfect reconstruction but may not learn meaningful structure—the model might use each latent dimension for one input dimension, learning identity mapping. The right size depends on data complexity and desired compression. A useful heuristic: start with 10-20× compression, adjust based on reconstruction quality and downstream task performance. For MNIST (784 dimensions), try 32-64 latent dimensions. For ImageNet (224×224×3), try 512-2048.

When generating samples, the sampling temperature often significantly affects quality-diversity tradeoffs. For autoregressive models or VAEs where we sample from learned distributions, we can scale logits by temperature before softmax:

$$p(x_i | \mathbf{x}_{<i}) = \text{softmax}(\mathbf{z}_i / T)$$

Low temperature ($$T < 1$$) makes the distribution sharper—more confident, less diverse. High temperature ($$T > 1$$) makes it more uniform—more diverse but potentially less realistic. Temperature provides a post-training knob for trading off quality and diversity without retraining. Understanding this tradeoff helps generate samples appropriate for different applications.

A powerful technique for improving sample quality is rejection sampling: generate multiple samples, score them with a discriminator or classifier, keep only high-scoring ones. This filters generated samples for quality at the cost of efficiency (must generate more samples than needed). For applications where quality matters more than generation speed (creating artwork, designing molecules), rejection sampling provides an easy win. Understanding that we can post-process generated samples—not just use whatever the model produces—expands the toolkit for practical applications.

## Key Takeaways

Generative models learn to understand and reproduce data distributions, enabling creation of novel, realistic samples from learned patterns. The three main paradigms—autoregressive models providing explicit sequential density factorization, variational autoencoders using latent variables with variational inference, and generative adversarial networks training through adversarial competition—make different tradeoffs between likelihood tractability, sampling efficiency, training stability, and sample quality. Maximum likelihood provides a principled training objective connecting to information theory through KL divergence, though it requires tractable density evaluation or lower bounds. Latent variable models introduce compressed representations capturing factors of variation, enabling fast sampling and interpretable manipulation, though requiring careful inference procedures. Evaluation of generative models is challenging, requiring multiple metrics (likelihood when available, Inception Score, FID, human evaluation) and domain-specific validation rather than single numbers. Applications span data augmentation, super-resolution, style transfer, drug discovery, and creative tools, with choice of approach depending on whether we need likelihood estimation, controlled generation, sample quality, or training stability. Understanding generative modeling deeply means appreciating both the statistical foundations (probability theory, density estimation, variational inference) and the deep learning implementations (neural architectures, training algorithms, practical tricks) that make learning complex distributions tractable.

Generative models demonstrate that neural networks can discover and internalize the statistical structure underlying complex data, learning representations that enable not just recognition but creation—a capability that edges closer to what we might consider genuine understanding.

