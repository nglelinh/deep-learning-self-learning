---
layout: post
title: 11-01-02-01 Autoregressive Model Implementation
chapter: '11'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter11
---

Before writing a single line of PyTorch for an autoregressive model, picture drawing a 2D point $(x_1, x_2)$ from a ring of eight Gaussian islands. If you try to predict $x_1$ and $x_2$ independently, you get a blurry square grid of 64 phantom islands. The chain rule of probability breaks that curse with surgical elegance: first sample the horizontal coordinate $x_1$ from a multimodal mixture, then ask a neural network—conditioned on the exact $x_1$ you just drew—where the vertical coordinate $x_2$ is allowed to land.

## Background Knowledge Map: Implementing Autoregressive Density Models

- **Pillar 1 — Probability Chain Rule Factorization**: Decomposes any joint density into exact conditionals $p_\theta(x_1, \dots, x_d) = \prod_{i=1}^d p_\theta(x_i \mid x_{<i})$, guaranteeing a normalized density with zero partition-function approximation.
- **Pillar 2 — Mixture Density Parameterization**: Models each conditional $p_\theta(x_i \mid x_{<i})$ as a weighted mixture of logistics or Gaussians so a single input $x_1$ can map to multiple valid $x_2$ modes (e.g., top and bottom of a circle).
- **Pillar 3 — Log-Sum-Exp Numerical Stability**: Evaluates $\log \sum_{k=1}^K \pi_k p_k(x)$ in log-space via `torch.logsumexp` so tiny probabilities never underflow to zero during backpropagation.
- **Pillar 4 — Teacher Forcing vs. Sequential Ancestral Sampling**: Explains why training evaluates all conditionals in parallel using ground-truth prefixes, whereas generation must step sequentially from $x_1$ to $x_d$.

### Step-by-Step Worked Example (Read Slowly): Evaluating a 2-Component Conditional Mixture Density

Suppose at $x_1 = 0$, the ring of Gaussians has two vertical modes at $x_2 = +2$ and $x_2 = -2$. Our autoregressive network outputs mixture weights $\pi = (0.5, 0.5)$, means $\mu = (+2.0, -2.0)$, and equal standard deviations $\sigma = (0.5, 0.5)$. Let us compute the exact log-likelihood of a real observation $x_2 = 2.0$.

1. **Component 1 log-density at $x_2 = 2.0$**:
   $$\log \mathcal{N}(2.0; 2.0, 0.5^2) = -\frac{1}{2}\log(2\pi) - \log(0.5) - \frac{(2.0 - 2.0)^2}{2(0.25)} = -0.9189 + 0.6931 - 0 = -0.2258.$$
2. **Component 2 log-density at $x_2 = 2.0$**:
   $$\log \mathcal{N}(2.0; -2.0, 0.5^2) = -0.9189 + 0.6931 - \frac{16}{0.5} = -0.2258 - 32.0 = -32.2258.$$
3. **Combine via Log-Sum-Exp with $\log \pi_k = \log(0.5) = -0.6931$**:
   $$a_1 = -0.6931 - 0.2258 = -0.9189, \qquad a_2 = -0.6931 - 32.2258 = -32.9189.$$
   $$\log p_\theta(x_2=2.0 \mid x_1=0) = a_1 + \log\big(1 + e^{a_2 - a_1}\big) = -0.9189 + \log(1 + e^{-32}) \approx -0.9189\text{ nats}.$$
4. **Takeaway**: Because $e^{-32} \approx 1.27 \times 10^{-14}$, the distant mode contributes virtually zero gradient, while `logsumexp` keeps the computation completely free of numerical underflow.

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


## Video tham khảo

- **xem trước** [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) — Andrej Karpathy, English. Next-character prediction is the autoregressive model this lesson codes.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, English. The same factorization inside a transformer, once the bigram model is too small.
- [Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Clearly Explained!!!](https://www.youtube.com/watch?v=L8HKweZIOmg) — StatQuest with Josh Starmer, English. Sequence-to-sequence as the conditional version of that product of probabilities.
```
