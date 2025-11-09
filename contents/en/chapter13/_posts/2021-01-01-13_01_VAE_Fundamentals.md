---
layout: post
title: 13-01 Variational Autoencoders
chapter: '13'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter13
lesson_type: required
---

# Variational Autoencoders: Probabilistic Generative Models

![VAE Architecture](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)
*Hình ảnh: Kiến trúc VAE với phân phối xác suất. Nguồn: Lilian Weng's Blog*

## 1. Concept Overview

Variational Autoencoders represent a beautiful marriage of variational inference from statistics and neural networks from deep learning, creating a principled probabilistic framework for generative modeling. While standard autoencoders learn deterministic encodings and decodings optimized for reconstruction, VAEs learn probability distributions over latent codes and data, enabling them to generate novel samples by sampling from the learned latent distribution. This probabilistic perspective addresses a fundamental limitation of vanilla autoencoders: their latent spaces can have "holes" where no training examples map, making random sampling produce unrealistic outputs. VAEs regularize the latent space to be continuous and complete, ensuring we can sample from any region and decode to realistic data.

The key insight that makes VAEs work is the Evidence Lower BOund (ELBO), a tractable objective that lower-bounds the intractable log-likelihood we actually want to maximize. Computing $$\log p(\mathbf{x})$$ for a latent variable model requires integrating over all possible latent codes $$\mathbf{z}$$, which is generally impossible for continuous latent spaces with neural network decoders. VAEs sidestep this by introducing a recognition network (encoder) that approximates the true posterior $$p(\mathbf{z}|\mathbf{x})$$, then optimizing a lower bound that's tractable. The tightness of this bound depends on how well the encoder approximates the true posterior—better approximation means tighter bound and better model.

Understanding VAEs requires appreciating several sophisticated ideas working together. The reparameterization trick enables backpropagation through stochastic sampling operations, turning an optimization problem that seems to require reinforcement learning into one solvable with standard gradient descent. The KL divergence between the approximate posterior and the prior acts as a regularizer, preventing the latent space from fragmenting into disconnected regions and encouraging smoothness that enables interpolation. The encoder and decoder are trained jointly, creating an architecture where the encoder learns to infer meaningful latent representations while the decoder learns to generate realistic data from these representations.

The beauty of the VAE framework is its generality. The same basic structure—encoder, latent distribution, decoder, ELBO objective—works for images, text, audio, and other data types, with appropriate choices of encoder/decoder architectures and output distributions. The latent space learned by VAEs has remarkable properties: it's continuous (nearby latent codes decode to similar outputs), complete (every region contains valid codes), and often interpretable (different latent dimensions capture different factors of variation like pose, color, or shape). These properties make VAEs valuable not just for generation but for representation learning, interpolation, and manipulation of semantic attributes.

Yet VAEs have characteristic limitations that motivate ongoing research. Samples are often somewhat blurry compared to GANs, a consequence of the reconstruction-based objective and Gaussian assumptions commonly used for the decoder. The prior distribution (typically standard Gaussian) might not match the true aggregate posterior, creating a gap between what the encoder produces and what we sample from during generation. The encoder-decoder architecture creates a potential bottleneck if the latent dimension is too small. Understanding these limitations alongside VAE strengths enables using them appropriately: when you need a principled probabilistic model with tractable training, smooth latent space for interpolation, or explicit density estimation, VAEs excel. When sample quality is paramount and training instability is acceptable, GANs might be preferable.

## 2. Mathematical Foundation

The mathematical framework of VAEs is rooted in variational inference, a powerful technique from Bayesian statistics for approximating intractable posterior distributions. Let's build up the mathematics carefully, understanding why each component is necessary and how they combine to create a trainable generative model.

We assume data $$\mathbf{x}$$ is generated from latent variables $$\mathbf{z}$$ through a probabilistic process:

$$\mathbf{z} \sim p(\mathbf{z})$$ (prior distribution, chosen to be simple)

$$\mathbf{x} \sim p_\theta(\mathbf{x}|\mathbf{z})$$ (likelihood, parameterized by neural network $$\theta$$)

The marginal likelihood of data is:

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

This integral is intractable for continuous $$\mathbf{z}$$ with complex $$p_\theta(\mathbf{x}|\mathbf{z})$$ (neural network decoder). Direct maximum likelihood optimization is impossible because we cannot evaluate the objective we want to maximize.

Variational inference addresses this by introducing an approximate posterior $$q_\phi(\mathbf{z}|\mathbf{x})$$ (the encoder, parameterized by $$\phi$$) and deriving a lower bound on log-likelihood. Starting from the log-likelihood and introducing the encoder:

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

Multiply inside the integral by $$\frac{q_\phi(\mathbf{z}|\mathbf{x})}{q_\phi(\mathbf{z}|\mathbf{x})} = 1$$:

$$= \log \int q_\phi(\mathbf{z}|\mathbf{x}) \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} d\mathbf{z}$$

$$= \log \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})}\left[\frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$$

By Jensen's inequality (log is concave):

$$\geq \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]$$

Rearranging terms:

$$= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})}\right]$$

$$= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

This is the Evidence Lower BOund (ELBO). The first term is reconstruction: how well can we reconstruct $$\mathbf{x}$$ from latent codes sampled from the encoder. The second term is a KL divergence between the approximate posterior and the prior, acting as regularization.

The ELBO provides a tractable training objective. Unlike $$\log p_\theta(\mathbf{x})$$ which requires intractable integration, we can estimate the ELBO through sampling:

$$\mathcal{L}(\theta, \phi; \mathbf{x}) \approx \frac{1}{L}\sum_{l=1}^L \log p_\theta(\mathbf{x}|\mathbf{z}^{(l)}) - KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

where $$\mathbf{z}^{(l)} \sim q_\phi(\mathbf{z}|\mathbf{x})$$. Often $$L=1$$ suffices (single sample per datapoint).

### The Reparameterization Trick

The remaining challenge is computing gradients with respect to $$\phi$$ when the objective involves sampling $$\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$$. Naively, sampling is a non-differentiable operation—we cannot backpropagate through randomness. The reparameterization trick solves this elegantly.

For Gaussian approximate posterior $$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}^2_\phi(\mathbf{x}))$$, instead of sampling directly, we:

1. Sample $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$ (fixed distribution, no parameters)
2. Compute $$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}$$

This is equivalent to sampling from $$q_\phi(\mathbf{z}|\mathbf{x})$$ but expressed as a deterministic function of $$\phi$$ and external randomness $$\boldsymbol{\epsilon}$$. Gradients with respect to $$\phi$$ flow through $$\boldsymbol{\mu}_\phi$$ and $$\boldsymbol{\sigma}_\phi$$, enabling backpropagation.

For the common case where both $$q_\phi(\mathbf{z}|\mathbf{x})$$ and $$p(\mathbf{z})$$ are Gaussian, the KL divergence has a closed form:

$$KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = KL(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \| \mathcal{N}(0, I))$$

$$= \frac{1}{2}\sum_{j=1}^{J} \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

where $$J$$ is latent dimension. This allows exact computation without sampling, making training more stable.

The complete VAE loss for a single datapoint becomes:

$$\mathcal{L}_{\text{VAE}}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0,I)}[\log p_\theta(\mathbf{x}|\boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon})] - \frac{1}{2}\sum_{j=1}^{J} (\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)$$

Maximizing this (or equivalently minimizing the negative) trains the VAE. The reconstruction term encourages accurate reconstruction while the KL term regularizes the latent space.

### Decoder Output Distribution

The choice of $$p_\theta(\mathbf{x}|\mathbf{z})$$ affects what reconstruction loss we use. For continuous data like images:

**Gaussian likelihood**: $$p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2 I)$$

The negative log-likelihood is proportional to MSE: $$-\log p_\theta(\mathbf{x}|\mathbf{z}) \propto \|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2$$

**Bernoulli likelihood** (for binary images): Each pixel independent Bernoulli with probability $$\hat{x}_i$$ from decoder.

The negative log-likelihood is binary cross-entropy: $$-\log p_\theta(\mathbf{x}|\mathbf{z}) = -\sum_i [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$$

The decoder network outputs the parameters of these distributions (means for Gaussian, probabilities for Bernoulli), and we sample from them during generation but use the mean/mode during reconstruction for training.

## 3. Example / Intuition

To understand how VAEs work in practice, let's trace through generating a new handwritten digit. Suppose we've trained a VAE on MNIST with 20-dimensional latent space.

During training, the VAE saw thousands of "3"s. For each "3" image $$\mathbf{x}$$, the encoder computed a Gaussian distribution in latent space: $$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}^2_\phi(\mathbf{x}))$$. The KL penalty in the loss encouraged these distributions to be close to $$\mathcal{N}(0, I)$$, preventing them from spreading arbitrarily far or collapsing to point masses. The result: all "3"s map to overlapping Gaussian distributions in a region of latent space. Different "3"s (thick, thin, slanted) map to slightly different means, but all are close to each other and to the origin.

Similarly, "8"s map to a different but also origin-centered region, "1"s to another, and so on. The latent space self-organizes: different digits occupy different regions, but all regions are near the origin (due to KL penalty), and transitions between regions are smooth (no holes).

Now for generation. We sample $$\mathbf{z} \sim \mathcal{N}(0, I)$$—just random numbers from a standard Gaussian. Suppose we get $$\mathbf{z} = [0.5, -0.3, 0.1, \ldots]$$ (20 numbers). By chance, this $$\mathbf{z}$$ falls in the "3" region of latent space. We decode: $$\hat{\mathbf{x}} = \text{decoder}_\theta(\mathbf{z})$$. The decoder, having seen many "3"s during training whose latent codes were near this $$\mathbf{z}$$, has learned to map this region to "3"-like images. The output is a novel "3"—not a copy of any training example but a new instance following the learned pattern.

The probabilistic nature provides interesting capabilities. If we sample $$\mathbf{z}$$ from exactly $$\mathcal{N}(0, I)$$, we get a diverse mix of all digits. If we want only "3"s, we can sample from the region where "3"s tend to map—but we don't know this region exactly without examining the encoder on "3" training examples. This is a limitation: VAEs don't provide explicit control over what class to generate unless we condition on class labels or discover class regions post-hoc.

Interpolation between images works beautifully. Encode two images to get $$\boldsymbol{\mu}_1$$ and $$\boldsymbol{\mu}_2$$ (using encoder's means, ignoring variances for determinism). Linearly interpolate:

$$\mathbf{z}_t = (1-t)\boldsymbol{\mu}_1 + t\boldsymbol{\mu}_2, \quad t \in [0,1]$$

Decode each $$\mathbf{z}_t$$ to get interpolated images. Because the VAE regularized the latent space to be smooth (through KL penalty), this interpolation produces coherent images throughout—smoothly morphing from one digit to another. This is unlike vanilla autoencoders where interpolation might produce unrealistic images in unexplored latent regions.

The KL divergence penalty's role deserves deep understanding. It serves three purposes: (1) regularization preventing overfitting to training examples, (2) ensuring the latent space is continuous and complete for sampling, (3) encouraging disentanglement where different latent dimensions capture independent factors of variation. The KL term $$KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$ can be decomposed:

$$KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = -H(q_\phi(\mathbf{z}|\mathbf{x})) + H(p(\mathbf{z}), q_\phi(\mathbf{z}|\mathbf{x}))$$

The first term encourages high entropy in $$q_\phi$$ (uncertainty), preventing the encoder from collapsing to deterministic encodings (which would make sampling impossible). The second term encourages closeness to the prior $$p(\mathbf{z})$$, ensuring the latent space structure matches what we'll sample from during generation.

## 4. Code Snippet

Let's implement a complete VAE with all mathematical components explicit:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class VAE(nn.Module):
    """
    Variational Autoencoder for MNIST.
    
    Architecture:
    - Encoder: maps images to latent distribution parameters (μ, σ)
    - Sampler: reparameterization trick for backpropagation through sampling
    - Decoder: maps latent codes to reconstructed images
    
    Loss: ELBO = reconstruction + KL divergence
    """
    
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: outputs parameters of Gaussian distribution
        # We output both μ and log(σ²) rather than σ for numerical stability
        # (σ must be positive, easier to ensure with exp(log σ²))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Separate layers for mean and log-variance
        # This allows encoder to learn both independently
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: maps latent code to reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Output in [0,1] for pixel values
        )
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Returns:
            mu: mean of q(z|x)
            logvar: log variance of q(z|x)
        
        We return log variance instead of variance/std for numerical stability.
        Variance must be positive, so we can ensure this by exponentiating logvar.
        This is more stable than directly predicting σ and squaring it.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ⊙ε where ε ~ N(0,I)
        
        This is THE key innovation making VAEs trainable with backprop.
        Instead of sampling z ~ N(μ,σ²) (not differentiable w.r.t. μ,σ),
        we express z as a deterministic function of μ,σ and external randomness ε.
        
        Gradients can flow through μ and σ to encoder parameters, enabling
        end-to-end training via standard backpropagation.
        """
        # Compute standard deviation from log variance
        # std = exp(log(σ²) / 2) = exp(logvar / 2)
        std = torch.exp(0.5 * logvar)
        
        # Sample epsilon from standard normal
        # During training: random. During generation: can use specific ε
        eps = torch.randn_like(std)
        
        # Reparameterized sample: z = μ + σ * ε
        z = mu + std * eps
        
        return z
    
    def decode(self, z):
        """Map latent code to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """
        Full VAE forward pass.
        
        Returns:
            recon: reconstructed input
            mu: latent mean (for KL computation)
            logvar: latent log-variance (for KL computation)
        
        We return mu and logvar separately because we need them to compute
        the KL divergence in the loss function.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, num_samples):
        """
        Generate new samples by sampling from prior and decoding.
        
        This is how we use the trained VAE for generation:
        1. Sample z ~ N(0, I) (the prior)
        2. Decode to get x
        
        Because we regularized q(z|x) to be close to N(0,I) during training,
        samples from N(0,I) should decode to realistic outputs.
        """
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss: negative ELBO = reconstruction loss + KL divergence
    
    Args:
        recon_x: reconstructed input
        x: original input
        mu: latent mean from encoder
        logvar: latent log-variance from encoder
        beta: weight for KL term (β-VAE uses β≠1 for disentanglement)
    
    The loss has two terms:
    1. Reconstruction: how well we reconstruct input
    2. KL: how much encoder distribution differs from prior
    
    We want to minimize both: good reconstruction AND latent distribution
    close to prior.
    """
    # Reconstruction loss (binary cross-entropy for images in [0,1])
    # Treating each pixel as independent Bernoulli
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence KL(N(μ,σ²) \| N(0,I))
    # Has closed form: 0.5 * Σ(μ² + σ² - log(σ²) - 1)
    # We have logvar = log(σ²), so σ² = exp(logvar)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss (negative ELBO)
    # Minimizing this is equivalent to maximizing ELBO
    return BCE + beta * KLD, BCE, KLD

# Training VAE
print("="*70)
print("Training Variational Autoencoder on MNIST")
print("="*70)

# Load data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Create VAE
vae = VAE(input_dim=784, latent_dim=20)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

print(f"VAE Architecture:")
print(f"  Input dimension: 784 (28×28)")
print(f"  Latent dimension: 20 (compression factor: 39×)")
print(f"  Total parameters: {sum(p.numel() for p in vae.parameters()):,}")

# Training loop
print("\nTraining VAE...")
vae.train()

for epoch in range(10):
    train_loss = 0
    train_bce = 0
    train_kld = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        # Flatten images
        data = data.view(-1, 784)
        
        # Forward pass
        recon, mu, logvar = vae(data)
        
        # Compute loss
        loss, bce, kld = vae_loss(recon, data, mu, logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
    
    # Average over batches
    n_batches = len(train_loader)
    avg_loss = train_loss / n_batches / 128  # Per sample
    avg_bce = train_bce / n_batches / 128
    avg_kld = train_kld / n_batches / 128
    
    print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f} "
          f"(Recon = {avg_bce:.4f}, KL = {avg_kld:.4f})")

print("\n" + "="*70)
print("VAE Training Complete - Analyzing Results")
print("="*70)

# Test reconstruction
vae.eval()
with torch.no_grad():
    test_data, _ = next(iter(test_loader))
    test_data_flat = test_data.view(-1, 784)
    
    # Encode (using mean, ignoring variance for determinism)
    mu, logvar = vae.encode(test_data_flat)
    
    # Reconstruct
    recon = vae.decode(mu)
    
    # Measure reconstruction error
    recon_error = F.mse_loss(recon, test_data_flat).item()
    print(f"Test reconstruction MSE: {recon_error:.6f}")
    
    # Analyze latent space statistics
    print(f"\nLatent space statistics (should be ~N(0,1) due to KL penalty):")
    print(f"  Mean: {mu.mean(dim=0)[:5].numpy().round(3)} (first 5 dims)")
    print(f"  Std:  {torch.exp(0.5*logvar).mean(dim=0)[:5].numpy().round(3)}")
    print(f"  Overall mean magnitude: {mu.abs().mean().item():.3f}")
    print(f"  Overall std: {torch.exp(0.5*logvar).mean().item():.3f}")

# Generate new samples
print("\n" + "="*70)
print("Generating New Samples from VAE")
print("="*70)

with torch.no_grad():
    # Sample from prior N(0,I)
    num_samples = 64
    samples = vae.sample(num_samples)
    
    print(f"Generated {num_samples} samples by sampling z ~ N(0,I)")
    print(f"Sample shape: {samples.shape}")  # (64, 784)
    
    # Check sample statistics
    print(f"Generated sample mean: {samples.mean():.3f} (should be ~0.5)")
    print(f"Generated sample std: {samples.std():.3f}")
    
    # Reshape for visualization
    samples_img = samples.view(-1, 1, 28, 28)
    print(f"Reshaped for visualization: {samples_img.shape}")

# Demonstrate interpolation
print("\n" + "="*70)
print("Latent Space Interpolation")
print("="*70)

with torch.no_grad():
    # Take two test images
    img1 = test_data_flat[0:1]
    img2 = test_data_flat[7:1]
    
    # Encode to latent means
    mu1, _ = vae.encode(img1)
    mu2, _ = vae.encode(img2)
    
    print("Interpolating between two images in latent space:")
    
    # Interpolate
    n_steps = 9
    for i, t in enumerate(np.linspace(0, 1, n_steps)):
        z_interp = (1-t) * mu1 + t * mu2
        img_interp = vae.decode(z_interp)
        
        if i % 2 == 0:  # Print every other step
            print(f"  t={t:.2f}: Generated interpolation image {i+1}/{n_steps}")
    
    print("\nInterpolation should be smooth due to latent space regularization!")
    print("This is a key advantage of VAE over vanilla autoencoders.")

# Demonstrate β-VAE (varying KL weight)
print("\n" + "="*70)
print("β-VAE: Controlling Disentanglement")
print("="*70)
print("By varying β (weight on KL term), we control tradeoffs:")
print("  β < 1: Prioritize reconstruction (sharper but less disentangled)")
print("  β = 1: Standard VAE (balanced)")
print("  β > 1: Prioritize latent regularity (more disentangled, blurrier)")
print("\nβ-VAE with β=4-10 often learns more interpretable latent dimensions")
print("where each dimension captures one semantic factor (size, rotation, etc.)")
```

## 5. Related Concepts

The relationship between VAEs and standard autoencoders illuminates what the probabilistic framework adds. Both use encoder-decoder architectures and reconstruction losses, but VAEs add: (1) probabilistic encodings (distributions rather than points), (2) the KL divergence regularization term, (3) the ability to sample for generation. These differences stem from VAEs being principled probabilistic models optimizing a lower bound on likelihood, while autoencoders are simply dimensionality reduction with a bottleneck. The probabilistic perspective provides theoretical guarantees: VAEs are approximately maximizing data likelihood, ensuring generated samples should be realistic if training succeeds. Autoencoders have no such guarantee—they minimize reconstruction error, which doesn't directly translate to generating good samples.

VAEs connect deeply to variational inference, a general technique in Bayesian statistics for approximating intractable posterior distributions. The idea is always the same: we have a model $$p(\mathbf{x}, \mathbf{z})$$ but cannot compute $$p(\mathbf{z}|\mathbf{x})$$ exactly, so we approximate it with a simpler distribution $$q(\mathbf{z}|\mathbf{x})$$ from a tractable family (here, factorized Gaussians). We optimize the approximation by maximizing the ELBO, which lower-bounds the quantity we actually care about (log-likelihood). VAEs made variational inference scalable to high-dimensional problems through: (1) using neural networks for $$q$$ and $$p$$, providing enormous flexibility, (2) the reparameterization trick enabling gradient-based optimization, (3) stochastic optimization allowing mini-batch training. Understanding VAEs through the variational inference lens connects them to a rich statistical tradition and motivates extensions like importance-weighted VAEs or hierarchical VAEs.

The connection to information theory provides another perspective. The ELBO can be written:

$$\text{ELBO} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

$$= -H_{q_\phi}(p_\theta(\mathbf{x}|\mathbf{z})) - KL(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

The first term is negative conditional entropy—favoring decoders that confidently reconstruct given latent codes (low uncertainty). The KL term measures information cost of using $$q_\phi$$ instead of the prior $$p$$. This information-theoretic view suggests VAEs trade off between reconstruction fidelity and compression (low information latent codes), a perspective formalized in the β-VAE framework where we explicitly control this tradeoff with $$\beta$$.

VAEs relate to normalizing flows through their treatment of latent variables. Both use latent variables $$\mathbf{z}$$ and learn mappings to data $$\mathbf{x}$$. However, flows use invertible, deterministic mappings with tractable Jacobians, enabling exact likelihood. VAEs use flexible neural networks for encoder/decoder but approximate the likelihood through ELBO. Flows provide exact inference but require constrained architectures. VAEs allow flexible architectures but provide approximate inference. This tradeoff shapes their respective application domains: flows for exact density modeling, VAEs for flexible generation with stable training.

Finally, VAEs connect to representation learning and disentanglement. A disentangled representation has individual latent dimensions corresponding to independent factors of variation (for faces: one dimension for pose, another for lighting, another for identity). VAEs' factorized Gaussian posterior encourages independence between latent dimensions, and β-VAEs with $$\beta > 1$$ further encourage disentanglement by more strongly penalizing KL divergence. Learning disentangled representations is valuable beyond generation: downstream tasks benefit from interpretable, factorized features where we can manipulate specific attributes independently. Understanding how VAE training encourages disentanglement connects to broader questions about what constitutes good representations and how to discover them through unsupervised learning.

## 6. Fundamental Papers

**["Auto-Encoding Variational Bayes" (2014)](https://arxiv.org/abs/1312.6114)**  
*Authors*: Diederik P. Kingma, Max Welling  
This foundational paper introduced VAEs and made variational inference practical for deep learning through the reparameterization trick. Kingma and Welling showed that by expressing sampling as $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$ where $$\boldsymbol{\epsilon} \sim \mathcal{N}(0,I)$$, we can backpropagate through stochastic computation, enabling gradient-based optimization of the ELBO. The paper provided clear mathematical derivations, proposed practical implementation details (using Gaussian encoders/decoders, closed-form KL), and demonstrated results on images. VAEs offered several advantages over existing approaches: principled probabilistic framework (unlike autoencoders), stable training (unlike early GANs), and tractable lower bound on likelihood (unlike implicit models). The work influenced thousands of follow-up papers exploring VAE variants, applications, and theoretical properties. Reading this paper, one appreciates both the mathematical sophistication (connecting neural networks to variational inference) and the practical insight (the reparameterization trick) that made the method work.

**["Tutorial on Variational Autoencoders" (2016)](https://arxiv.org/abs/1606.05908)**  
*Author*: Carl Doersch  
This tutorial paper provided accessible introduction to VAEs for readers without strong background in variational inference. Doersch carefully explained the intuition behind ELBO (why a lower bound is sufficient, what the terms mean), the reparameterization trick (with visual diagrams showing gradient flow), and practical training considerations. The tutorial addressed common confusions (why Gaussian posteriors, what latent space structure means, how to choose hyperparameters) and connected VAEs to related concepts (autoencoders, GANs, Bayesian inference). While not introducing new methods, this tutorial significantly helped VAE adoption by making the framework accessible to practitioners. It exemplifies how clear exposition—explaining not just what equations are but why they make sense—can have major impact on field by lowering barriers to understanding sophisticated techniques.

**["β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)](https://openreview.net/forum?id=Sy2fzU9gl)**  
*Authors*: Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, Alexander Lerchner  
This paper introduced β-VAE, a simple modification where the KL term in the loss is weighted by $$\beta > 1$$ instead of 1. This seemingly minor change has profound effects on the learned latent representations. Higher β more strongly penalizes KL divergence, encouraging the latent dimensions to be independent (factorized), which often leads to disentangled representations where each dimension captures one interpretable factor of variation. The paper demonstrated that β-VAEs learn to separate factors like shape, size, rotation, and color into different latent dimensions, enabling controlled generation by manipulating specific dimensions. The work connected VAEs to the information bottleneck principle and showed that the right amount of compression (through β) can improve representation quality for downstream tasks. β-VAE has become a standard tool for learning disentangled representations and illustrates how hyperparameters (β) can control qualitative properties of learned representations, not just quantitative metrics like reconstruction error.

**["Importance Weighted Autoencoders" (2016)](https://arxiv.org/abs/1509.00519)**  
*Authors*: Yuri Burda, Roger Grosse, Ruslan Salakhutdinov  
This paper improved VAE's ELBO through importance sampling, creating a tighter lower bound on log-likelihood. Standard VAE uses a single sample from $$q_\phi(\mathbf{z}|\mathbf{x})$$ to estimate the ELBO. IWAE uses multiple samples and importance weighting, giving:

$$\mathcal{L}_{\text{IWAE}} = \mathbb{E}\left[\log \frac{1}{K}\sum_{k=1}^K \frac{p_\theta(\mathbf{x}, \mathbf{z}^{(k)})}{q_\phi(\mathbf{z}^{(k)}|\mathbf{x})}\right]$$

This is a tighter lower bound (approaches true log-likelihood as $$K \to \infty$$) and often generates better samples. The paper showed that the quality improvement comes from better training of the generative model $$p_\theta(\mathbf{x}|\mathbf{z})$$, though the inference network $$q_\phi$$ might become less accurate. IWAE demonstrates that even with VAE's solid theoretical foundation, there's room for improvement through better variational approximations. The importance weighting idea has influenced subsequent work on improving variational bounds and shows how classical statistical techniques (importance sampling) can enhance neural approaches.

**["Generating Diverse High-Fidelity Images with VQ-VAE-2" (2019)](https://arxiv.org/abs/1906.00446)**  
*Authors*: Ali Razavi, Aaron van den Oord, Oriol Vinyals  
This paper introduced VQ-VAE-2, achieving state-of-the-art sample quality for VAE-based models by using discrete latent representations and hierarchical priors. Instead of continuous Gaussian latents, VQ-VAE uses vector quantization—the encoder outputs indices into a learned codebook, and the decoder receives the corresponding codebook vectors. This discreteness allows using powerful autoregressive priors over the latent codes, dramatically improving sample quality. The hierarchical structure (separate latent codes for global and local structure) enables generating high-resolution images. While more complex than standard VAEs, VQ-VAE-2 demonstrated that VAE-based models could compete with GANs in sample quality while maintaining VAE's advantages of stable training and latent space structure. The work showed that the VAE framework is flexible enough to accommodate discrete latents, hierarchical structure, and sophisticated priors, pushing VAE performance to new levels.

## Common Pitfalls and Tricks

The most common failure mode in VAE training is posterior collapse, where the encoder learns to ignore the input and output the prior distribution for all inputs: $$q_\phi(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z}) = \mathcal{N}(0,I)$$ regardless of $$\mathbf{x}$$. The KL term becomes zero (good for that term!) but the reconstruction term cannot improve because latent codes contain no information about the input. The decoder learns to generate average images (the mean of the training distribution) regardless of latent code. Symptoms include very low KL divergence (approaching 0) and poor reconstructions. This happens when the decoder is too powerful—it can reconstruct reasonably well without using latent information, so the encoder takes the easy route of outputting the prior to minimize KL.

Solutions include: (1) weakening the decoder (fewer layers/parameters), forcing it to rely on latent codes; (2) KL annealing—start training with β=0 (no KL penalty) and gradually increase to β=1, allowing the encoder to discover useful representations before regularization is applied; (3) free bits—only penalize KL if it's above a threshold, ensuring latent dimensions maintain minimum information content; (4) better optimization—using higher learning rates for the encoder than decoder, giving it advantage in the competition for capacity. Understanding posterior collapse as an optimization pathology rather than fundamental VAE limitation helps implement these solutions appropriately.

Choosing the latent dimension involves a tradeoff between expressiveness and disentanglement. Larger latent dimensions can capture more variation (good for reconstruction) but tend to be less disentangled (dimensions become correlated, harder to interpret). Smaller latent dimensions force more compression and often learn more disentangled representations but may not capture all data variation (poor reconstruction). For MNIST, 10-20 dimensions typically suffice. For CelebA faces, 64-256 dimensions are common. Always validate on both reconstruction quality (quantitative) and latent space interpretability (qualitative).

The choice of β in β-VAE significantly affects outcomes. β=1 is standard VAE, balancing reconstruction and regularization. β>1 (typically 2-10) prioritizes disentanglement, useful when interpretability matters more than perfect reconstruction. β<1 (typically 0.1-0.5) prioritizes reconstruction, useful when generation quality matters more than latent space structure. For representation learning (using VAE features for downstream tasks), β=1-2 often works well. For controllable generation (manipulating specific attributes), β=4-10 provides more disentangled latents. Understanding this knob allows tailoring VAEs to specific application requirements.

A powerful technique for improving sample quality is using more sophisticated decoder distributions than Gaussian or Bernoulli. Mixture of discretized logistics (modeling pixel values as mixture of binned distributions) captures multi-modality better than single Gaussian. Autoregressive decoders (each pixel predicted conditionally on previous pixels) capture dependencies autoencoders' factorized assumptions miss. These more expressive decoders often generate sharper samples while maintaining VAE's stable training. The tradeoff is computational cost—autoregressive decoding is slow. Understanding that decoder choice affects both sample quality and training/generation speed guides appropriate architectural decisions.

When using VAE latent codes for downstream tasks (classification, clustering), a decision is whether to use the mean $$\boldsymbol{\mu}$$ or sample from $$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$$. For deterministic tasks (classification), using mean provides stable features. For tasks requiring uncertainty (active learning, Bayesian inference), sampling reflects the encoder's uncertainty. For most applications, the mean works well and is standard practice, but understanding that the full distribution is available enables more sophisticated uses when appropriate.

## Key Takeaways

Variational Autoencoders combine neural networks with variational inference to create a principled probabilistic framework for generative modeling with latent variables, optimizing the Evidence Lower BOund on log-likelihood as a tractable surrogate for the intractable true likelihood. The encoder learns to map inputs to distributions over latent codes (parameterized as Gaussian with learned mean and variance), while the decoder learns to reconstruct inputs from latent samples, with both trained jointly through the ELBO objective combining reconstruction accuracy and latent distribution regularization. The reparameterization trick—expressing stochastic sampling as deterministic function of parameters and external randomness—enables backpropagation through sampling, making end-to-end training possible with standard gradient descent. The KL divergence between approximate posterior and prior regularizes the latent space to be continuous, complete, and centered around the prior, ensuring samples from the prior decode to realistic outputs and enabling smooth interpolation between examples. β-VAE extends the framework by weighting the KL term, trading off reconstruction quality for latent disentanglement and enabling learned representations where individual dimensions capture interpretable factors of variation. VAEs provide stable training compared to GANs, explicit latent space structure enabling interpolation and manipulation, and a principled probabilistic framework supporting theoretical analysis, though often producing somewhat blurrier samples than adversarially trained models. Understanding VAEs requires appreciating the interplay between deep learning (neural encoders/decoders), probability theory (latent variable models), and optimization (variational bounds), making them both theoretically rich and practically valuable for generation, representation learning, and semi-supervised learning.

Variational autoencoders exemplify how bringing together ideas from different fields—neural networks, variational inference, information theory—can create methods more powerful than the sum of their parts.

