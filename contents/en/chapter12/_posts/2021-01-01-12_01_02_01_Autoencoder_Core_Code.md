---
layout: post
title: 12-01-02-01 Autoencoder Core Implementation
chapter: '12'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter12
---

When you build a PyTorch autoencoder for MNIST from scratch, the subtle bugs are rarely syntax errors—they are architectural mismatches. Use a ReLU on the final decoder layer when your pixels live in $[0, 1]$ and your loss will stall; feed noisy images to both the input *and* target of a denoising autoencoder and you will teach the network to preserve static instead of removing it. Walking carefully through tensor shapes, activation ranges, and corruption targets turns abstract equations into clean, debuggable code.

## Key ideas: Engineering Autoencoders in PyTorch

- **Pillar 1 — Symmetric Encoder–Decoder Tensor Flow**: Maps flattened $28 \times 28 = 784$-dimensional images down through bottleneck layers ($784 \to 256 \to 64 \to d_z$) and symmetrically back to $784$, balancing parameter count and compression ratio.
- **Pillar 2 — Output Activation & Loss Pairing**: Matches `Sigmoid` outputs in $[0, 1]$ with `BCELoss` or `MSELoss`, versus `Tanh` outputs in $[-1, 1]$ with `MSELoss`, preventing saturation and range clipping.
- **Pillar 3 — Denoising Corruption Pipeline**: Injects Gaussian or masking noise on-the-fly (`x_noisy = torch.clamp(x + noise, 0, 1)`) while computing loss strictly against the *clean* target `x`.
- **Pillar 4 — Optimization & Reconstruction Monitoring**: Tracks per-pixel MSE and visual side-by-side reconstructions across epochs to verify that the bottleneck captures global shape rather than mean-gray averages.

### Worked example: Parameter Count and Compression Ratio of an MNIST MLP Autoencoder

Consider an encoder `784 -> 256 -> 64 -> 16` and symmetric decoder `16 -> 64 -> 256 -> 784` (all linear layers with biases).

1. **Compression ratio at the bottleneck**:
   $$\text{Compression Ratio} = \frac{784}{16} = 49\times \quad (\text{keeping only } 2.04\% \text{ of input dimensionality}).$$
2. **Encoder parameter count**:
   - Layer 1 ($784 \to 256$): $784 \times 256 + 256 = 200{,}704 + 256 = 200{,}960$.
   - Layer 2 ($256 \to 64$): $256 \times 64 + 64 = 16{,}384 + 64 = 16{,}448$.
   - Layer 3 ($64 \to 16$): $64 \times 16 + 16 = 1{,}024 + 16 = 1{,}040$.
   - Total encoder parameters: $200{,}960 + 16{,}448 + 1{,}040 = 218{,}448$.
3. **Decoder parameter count**:
   - $16 \times 64 + 64 = 1{,}088$; $\; 64 \times 256 + 256 = 16{,}640$; $\; 256 \times 784 + 784 = 201{,}488$.
   - Total decoder parameters: $219{,}216$; grand total $= 437{,}664$ parameters ($\approx 1.67\text{ MB}$ in FP32).
4. **Takeaway**: Over $91\%$ of the network's parameters live in the first and last $784 \leftrightarrow 256$ layers—which is why switching to convolutional encoders/decoders dramatically improves parameter efficiency on images.

## 4. Code Snippet

Let's implement autoencoders from scratch with complete training pipeline:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder(nn.Module):
    """
    Standard autoencoder with fully connected layers.
    
    Architecture: Input → Encoder → Latent (bottleneck) → Decoder → Reconstruction
    
    The bottleneck forces compression - input dimensions > latent dimensions.
    The network must learn efficient encoding of data structure.
    """
    
    def __init__(self, input_dim=784, latent_dim=32):
        """
        input_dim: flattened input size (28*28=784 for MNIST)
        latent_dim: bottleneck dimension (compression factor = input_dim/latent_dim)
        
        We'll use symmetric encoder-decoder architecture with progressively
        decreasing then increasing dimensions: 784 → 256 → 128 → 32 → 128 → 256 → 784
        """
        super(Autoencoder, self).__init__()
        
        # Encoder: progressively compress
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # No activation - let latent be unbounded
        )
        
        # Decoder: progressively decompress
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Sigmoid for pixel values in [0,1]
        )
    
    def encode(self, x):
        """Map input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Reconstruct from latent code"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full autoencoder: encode then decode"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z

class DenoisingAutoencoder(nn.Module):
    """
    Denoising autoencoder: trained to reconstruct clean data from corrupted input.
    
    The corruption process forces learning robust features that capture
    data structure rather than memorizing training examples. Results in
    better features for downstream tasks.
    """
    
    def __init__(self, input_dim=784, latent_dim=32, noise_factor=0.3):
        super(DenoisingAutoencoder, self).__init__()
        
        self.noise_factor = noise_factor
        
        # Same architecture as vanilla autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Additional regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def add_noise(self, x, noise_factor=None):
        """
        Corrupt input with noise.
        
        For MNIST, we'll use Gaussian noise and clip to [0,1].
        Other corruption types: masking (zero out pixels),
        salt-and-pepper, or adversarial perturbations.
        """
        if noise_factor is None:
            noise_factor = self.noise_factor
        
        noisy = x + noise_factor * torch.randn_like(x)
        return torch.clamp(noisy, 0., 1.)
    
    def forward(self, x):
        """
        Training: corrupt input, encode corrupted, decode to clean.
        
        Key difference from vanilla: we add noise to input before encoding
        but compute loss against original clean input. This trains the
        network to denoise.
        """
        # Corrupt input
        x_noisy = self.add_noise(x)
        
        # Encode corrupted input
        z = self.encoder(x_noisy)
        
        # Decode (should reconstruct clean input, not noisy input!)
        reconstruction = self.decoder(z)
        
        return reconstruction, z, x_noisy

# Load MNIST for demonstration
print("="*70)
print("Training Autoencoders on MNIST")
print("="*70)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Train vanilla autoencoder
print("\n1. Training Vanilla Autoencoder (latent_dim=32)")
print("-" * 70)

model_ae = Autoencoder(input_dim=784, latent_dim=32)
optimizer_ae = optim.Adam(model_ae.parameters(), lr=0.001)
criterion = nn.MSELoss()

model_ae.train()
for epoch in range(10):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # Flatten images: (batch, 1, 28, 28) → (batch, 784)
        data = data.view(data.size(0), -1)
        
        # Forward pass
        reconstruction, latent = model_ae(data)
        loss = criterion(reconstruction, data)
        
        # Backward pass
        optimizer_ae.zero_grad()
        loss.backward()
        optimizer_ae.step()
        
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

print("\n2. Training Denoising Autoencoder (latent_dim=32, noise=0.3)")
print("-" * 70)

model_dae = DenoisingAutoencoder(input_dim=784, latent_dim=32, noise_factor=0.3)
optimizer_dae = optim.Adam(model_dae.parameters(), lr=0.001)

model_dae.train()
for epoch in range(10):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        
        # Forward pass (adds noise internally)
        reconstruction, latent, noisy = model_dae(data)
        
        # Loss: reconstruct CLEAN data from NOISY input
        loss = criterion(reconstruction, data)
        
        optimizer_dae.zero_grad()
        loss.backward()
        optimizer_dae.step()
        
        train_loss += loss.item()
    
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.6f}")

# Test and visualize
print("\n" + "="*70)
print("Testing Reconstructions")
print("="*70)

model_ae.eval()
model_dae.eval()

with torch.no_grad():
    # Get test batch
    test_data, _ = next(iter(test_loader))
    test_data_flat = test_data.view(test_data.size(0), -1)
    
    # Vanilla autoencoder
    recon_ae, latent_ae = model_ae(test_data_flat)
    
    # Denoising autoencoder (add noise for testing too)
    test_noisy = model_dae.add_noise(test_data_flat)
    recon_dae, latent_dae, _ = model_dae.forward(test_data_flat)
    
    # Compute reconstruction errors
    mse_ae = F.mse_loss(recon_ae, test_data_flat).item()
    mse_dae = F.mse_loss(recon_dae, test_data_flat).item()
    
    print(f"Vanilla AE reconstruction MSE: {mse_ae:.6f}")
    print(f"Denoising AE reconstruction MSE: {mse_dae:.6f}")
    
    # Visualize some reconstructions
    n_display = 8
    print(f"\nDisplaying first {n_display} test images with reconstructions...")
    
    # Reshape for visualization
    originals = test_data[:n_display].cpu().numpy()
    recon_ae_imgs = recon_ae[:n_display].view(-1, 1, 28, 28).cpu().numpy()
    recon_dae_imgs = recon_dae[:n_display].view(-1, 1, 28, 28).cpu().numpy()
    
    # Print shapes (would display in actual notebook)
    print(f"Original shape: {originals.shape}")
    print(f"Reconstructions shape: {recon_ae_imgs.shape}")


## Video tham khảo

- **xem trước** [What are Autoencoders?](https://www.youtube.com/watch?v=qiUEgSCyY5o) — IBM Technology, English. The two-module shape the core code should have.
- [PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) — Patrick Loeber, English. Optimizer, loss, and step, once encode and decode exist.
- [Lecture 13 | Generative Models](https://www.youtube.com/watch?v=5WoItGTWV54) — Stanford University School of Engineering, English. What a healthy reconstruction looks like, so a bug is visible.
```
