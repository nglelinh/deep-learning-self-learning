---
layout: post
title: 14-01-02 GAN Implementation
chapter: '14'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter14
---

## 4. Code Snippet

Let's implement a complete GAN from scratch to understand the training dynamics:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class Generator(nn.Module):
    """
    Generator: maps random noise to fake data.
    
    Architecture follows common pattern for image generation:
    - Start with low spatial resolution but many channels
    - Progressively upsample spatially while reducing channels
    - Final layer outputs image with correct dimensions
    
    For MNIST: noise (100) → (256×7×7) → (128×14×14) → (1×28×28)
    """
    
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Project and reshape noise
        # Linear layer: 100 → 256*7*7, then reshape to (256, 7, 7)
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)
        
        # Upsample through transposed convolutions
        self.deconv = nn.Sequential(
            # 256×7×7 → 128×14×14
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),  # Stabilizes training
            nn.ReLU(),
            
            # 128×14×14 → 1×28×28
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] (we'll normalize real data to match)
        )
    
    def forward(self, z):
        """
        z: (batch, latent_dim) random noise
        Returns: (batch, 1, 28, 28) generated images
        
        The forward pass transforms unstructured noise into structured
        images through learned transformations. Early in training, outputs
        are noise. As training progresses, digit-like structures emerge.
        """
        x = self.fc(z)
        x = x.view(-1, 256, 7, 7)  # Reshape to feature maps
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator: distinguishes real from fake.
    
    Architecture mirrors generator in reverse:
    - Input: (1×28×28) image
    - Conv layers progressively downsample while increasing channels
    - Final: scalar output (probability image is real)
    
    Uses LeakyReLU instead of ReLU to prevent dying units, and no pooling
    (stride for downsampling instead) following DCGAN best practices.
    """
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            # 1×28×28 → 64×14×14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),  # Negative slope 0.2
            
            # 64×14×14 → 128×7×7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Flatten: 128×7×7 → 6272
            nn.Flatten(),
            
            # Final classification
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()  # Probability of being real
        )
    
    def forward(self, x):
        """
        x: (batch, 1, 28, 28) images
        Returns: (batch, 1) probabilities of being real
        
        The discriminator learns hierarchical features for detection:
        - Early layers: edges, textures (distinguish fake textures from real)
        - Middle layers: shapes, patterns (detect anatomically incorrect digits)  
        - Late layers: holistic features (identify subtle statistical differences)
        """
        return self.conv(x)

# Training GAN
print("="*70)
print("Training Generative Adversarial Network on MNIST")
print("="*70)

# Hyperparameters
latent_dim = 100
batch_size = 128
num_epochs = 50
lr = 0.0002
beta1 = 0.5  # Adam beta1 (lower than default 0.9 for GAN stability)

# Data loading (normalize to [-1, 1] to match generator's tanh output)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                          shuffle=True, drop_last=True)

# Initialize networks
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()

# Optimizers (both use Adam with β1=0.5 for stability)
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function (binary cross-entropy)
criterion = nn.BCELoss()

# Labels for real and fake (used in loss computation)
real_label = 1.0
fake_label = 0.0

print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
print(f"\nTraining for {num_epochs} epochs...")
print("This demonstrates the adversarial training dynamics:\n")

# Training loop
G_losses = []
D_losses = []

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size_actual = real_images.size(0)
        
        # ==================== Train Discriminator ====================
        # Discriminator wants to maximize: log D(x) + log(1 - D(G(z)))
        # Equivalently, minimize: -log D(x) - log(1 - D(G(z)))
        
        discriminator.zero_grad()
        
        # Train on real data: maximize log D(x)
        # Loss: -log D(x) (negated because we minimize)
        labels_real = torch.full((batch_size_actual, 1), real_label)
        output_real = discriminator(real_images)
        loss_D_real = criterion(output_real, labels_real)
        
        # Train on fake data: maximize log(1 - D(G(z)))
        # Loss: -log(1 - D(G(z)))
        z = torch.randn(batch_size_actual, latent_dim)
        fake_images = generator(z)
        labels_fake = torch.full((batch_size_actual, 1), fake_label)
        output_fake = discriminator(fake_images.detach())  # Detach! Don't backprop through G
        loss_D_fake = criterion(output_fake, labels_fake)
        
        # Total discriminator loss
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()
        
        # ==================== Train Generator ====================
        # Generator wants to minimize: -log D(G(z))
        # Equivalently, maximize: log D(G(z)) (non-saturating objective)
        
        generator.zero_grad()
        
        # Generate fakes again (no detach this time - we need gradients through G!)
        z = torch.randn(batch_size_actual, latent_dim)
        fake_images = generator(z)
        output_fake_for_G = discriminator(fake_images)
        
        # Generator tries to make discriminator output 1 (real) for its fakes
        labels_real_for_G = torch.full((batch_size_actual, 1), real_label)
        loss_G = criterion(output_fake_for_G, labels_real_for_G)
        
        loss_G.backward()
        optimizer_G.step()
        
        # Track losses
        if i == 0:  # Once per epoch
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}]  "
              f"D_loss: {loss_D.item():.4f}  "
              f"G_loss: {loss_G.item():.4f}  "
              f"D(x): {output_real.mean():.3f}  "
              f"D(G(z)): {output_fake.mean():.3f}")

print("\n" + "="*70)
print("Training Complete! Analyzing Results")
print("="*70)

# Generate samples
generator.eval()
with torch.no_grad():
    # Generate 64 samples
    z_sample = torch.randn(64, latent_dim)
    fake_samples = generator(z_sample)
    
    print(f"\nGenerated {fake_samples.size(0)} fake MNIST digits")
    print(f"Sample shape: {fake_samples.shape}")  # (64, 1, 28, 28)
    
    # Check discriminator's opinion on generated samples
    disc_scores = discriminator(fake_samples)
    print(f"Discriminator scores for generated samples:")
    print(f"  Mean: {disc_scores.mean():.3f} (ideally ~0.5)")
    print(f"  Std:  {disc_scores.std():.3f}")
    
    # If mean is near 0.5, generator is successfully fooling discriminator
    if disc_scores.mean() > 0.4 and disc_scores.mean() < 0.6:
        print("  ✓ Generator successfully fools discriminator!")
    elif disc_scores.mean() < 0.3:
        print("  ✗ Discriminator still easily detects fakes")
    else:
        print("  ~ Generator is somewhat convincing")

# Demonstrate latent space interpolation
print("\n" + "="*70)
print("Latent Space Interpolation in GAN")
print("="*70)

with torch.no_grad():
    # Two random latent codes
    z1 = torch.randn(1, latent_dim)
    z2 = torch.randn(1, latent_dim)
    
    # Interpolate
    n_steps = 7
    print(f"Generating {n_steps} images by interpolating latent codes:\n")
    
    for i, t in enumerate(np.linspace(0, 1, n_steps)):
        z_interp = (1-t) * z1 + t * z2
        img = generator(z_interp)
        
        print(f"  Step {i} (t={t:.2f}): Generated image shape {img.shape}")
    
    print("\nInterpolation should show smooth morphing between different digits.")
    print("Quality of interpolation indicates latent space structure.")

print("\n" + "="*70)
print("GAN Training Insights")
print("="*70)
print("\nKey observations from training:")
print("1. Adversarial dynamics create natural curriculum")
print("2. Balance between D and G is crucial (neither should dominate)")
print("3. Loss values don't directly indicate quality (check samples!)")
print("4. Mode collapse is a constant danger (monitor diversity)")
print("5. Generated samples can be realistic despite imperfect equilibrium")
```

## 5. Related Concepts

The relationship between GANs and variational autoencoders illuminates different approaches to generative modeling. VAEs explicitly model the data distribution through a latent variable model $$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$$, training through maximizing a variational lower bound on likelihood. This probabilistic framework provides theoretical guarantees and enables principled Bayesian inference but requires choosing parametric forms for distributions and often produces blurrier samples due to the reconstruction loss. GANs implicitly model distributions through the generator's learned mapping from noise to data, using adversarial training instead of likelihood. This enables generating sharper, more realistic samples (because the discriminator can learn perceptual similarity rather than pixel-wise reconstruction) but lacks VAE's theoretical guarantees and density estimation capability. Understanding both approaches reveals different tradeoffs: VAEs for theoretical understanding and density modeling, GANs for sample quality and flexibility.

GANs connect to game theory through the minimax formulation. The generator and discriminator play a two-player zero-sum game where one's gain (discriminator correctly identifying fakes) is the other's loss (generator's fakes being detected). Nash equilibrium—where neither player can improve by unilaterally changing strategy—corresponds to the generator matching the data distribution. However, reaching Nash equilibrium in practice is challenging because we're using gradient-based optimization, which makes local moves, in a non-convex game where equilibria might not exist or be unstable. This connection to game theory helps understand why GAN training can be unstable (many games have no pure strategy Nash equilibrium or have multiple equilibria) and motivates algorithms from game theory like unrolled optimization.

The relationship to adversarial examples and robustness provides an interesting perspective. In adversarial examples research, we perturb inputs slightly to fool classifiers. In GANs, we're doing something similar but more ambitious: creating completely synthetic inputs that fool the discriminator. The discriminator trying to resist fooling is analogous to adversarial training for robust classifiers. This connection suggests techniques from adversarial robustness (like certified defenses) might apply to stabilizing GAN training, and conversely, that GAN discriminators might provide insights into what makes classifiers vulnerable to adversarial examples. The mathematical connection is deep: both involve optimizing over input space to maximize or minimize classifier outputs.

GANs' impact on semi-supervised learning demonstrates how generative models can improve discriminative tasks. By adding an auxiliary task to the discriminator—not just real/fake but also classifying real images into categories—we can leverage unlabeled data (used for adversarial training) to improve classification on limited labeled data. The discriminator learns representations through both tasks, with the generative task providing regularization and additional training signal. This semi-supervised GAN framework has been successful in low-data regimes, showing how generative and discriminative learning can be mutually beneficial.

Finally, GANs connect to the broader theme of learning without direct supervision on the target task. We never show the generator example outputs—it learns purely from discriminator feedback. This is analogous to reinforcement learning where agents learn from reward signals rather than supervised examples. Indeed, GANs can be viewed as applying policy gradient methods (from RL) to generative modeling, with the discriminator providing rewards (high scores for good fakes) that guide generator improvement. This connection has led to hybrid approaches combining GAN training with reinforcement learning principles for improved stability and performance.

## 6. Fundamental Papers

**["Generative Adversarial Networks" (2014)](https://arxiv.org/abs/1406.2661)**  
*Authors*: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
This foundational paper introduced the GAN framework and remains one of the most influential papers in modern machine learning. Goodfellow conceived the basic idea—training generator and discriminator adversarially—reportedly in a single evening, though the paper's development involved significant theoretical and empirical work. The paper formalized GANs as a minimax game, proved that at equilibrium the generator learns the data distribution, and demonstrated results on several datasets. What made GANs revolutionary was not just the results but the paradigm shift: generative modeling through competition rather than likelihood maximization or reconstruction. The paper acknowledged training challenges (instability, mode collapse) while showing the approach's potential. Reading it today, one appreciates both the clarity of the core idea and the prescience about challenges that would occupy researchers for years. GANs demonstrated that sometimes the best way to solve a problem isn't to attack it directly (modeling density explicitly) but indirectly (learning to generate through adversarial feedback).

**["Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2016)](https://arxiv.org/abs/1511.06434)**  
*Authors*: Alec Radford, Luke Metz, Soumith Chintala  
The DCGAN paper made GANs practical by identifying architectural guidelines that stabilize training and improve sample quality. The authors systematically explored design choices—convolutional vs fully connected layers, batch normalization placement, activation functions—finding combinations that consistently worked. Their guidelines: use strided convolutions instead of pooling, use batch norm in both networks (except generator output and discriminator input), use ReLU in generator except output (tanh), use LeakyReLU in discriminator. These weren't theoretically motivated but empirically discovered through extensive experimentation, demonstrating that practical progress sometimes comes from systematic engineering rather than mathematical insight. DCGAN showed that GANs could generate high-quality images (64×64 faces) and that the learned latent space had meaningful structure—arithmetic in latent space (vector for "smiling woman" minus "neutral woman" plus "neutral man") produced "smiling man." This demonstrated GANs learn disentangled representations encoding semantic attributes, making them useful beyond generation for representation learning.

**["Improved Techniques for Training GANs" (2016)](https://arxiv.org/abs/1606.03498)**  
*Authors*: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen  
This paper addressed GAN training instabilities through several techniques: feature matching (train generator to match statistics of discriminator's intermediate features rather than fool final output), minibatch discrimination (let discriminator compare examples within a batch to detect lack of diversity), historical averaging (penalize parameters for deviating from historical averages), one-sided label smoothing (use 0.9 instead of 1.0 for real labels to prevent discriminator overconfidence), and virtual batch normalization (normalize using statistics from a reference batch to reduce batch-to-batch variance). Each technique addresses a specific failure mode: feature matching reduces instability, minibatch discrimination combats mode collapse, label smoothing prevents discriminator saturation. The paper also introduced the Inception Score for quantifying sample quality, providing an automated metric (though imperfect) for evaluating GANs. This work established that successful GAN training requires multiple complementary tricks rather than just the basic algorithm, providing a toolkit that has become standard practice.

**["Progressive Growing of GANs for Improved Quality, Stability, and Variation" (2018)](https://arxiv.org/abs/1710.10196)**  
*Authors*: Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen (NVIDIA)  
This paper introduced progressive training: start with low-resolution images (4×4) and progressively add layers to generator and discriminator, increasing resolution (8×8, 16×16, ..., up to 1024×1024). This approach stabilizes training (easier to learn low-resolution distributions first) and enables generating very high-resolution images that were previously impossible. The paper also introduced improved evaluation metrics and training techniques. The generated faces at 1024×1024 resolution were shockingly realistic, demonstrating GANs' capability for high-fidelity generation. Progressive growing has influenced subsequent work (StyleGAN builds on it) and demonstrated that training curriculum—gradually increasing task difficulty—applies not just to data (easy examples first) but to architecture (simple generation first, complex later). The work showed that GAN training instability can be partially addressed through careful training procedures, not just architecture or loss modifications.

**["A Style-Based Generator Architecture for Generative Adversarial Networks" (2019)](https://arxiv.org/abs/1812.04948)**  
*Authors*: Tero Karras, Samuli Laine, Timo Aila (NVIDIA)  
StyleGAN redesigned the generator architecture to enable fine-grained control over generated images. Instead of feeding latent code directly into the generator, StyleGAN maps it through a mapping network to an intermediate latent space $$\mathcal{W}$$, then uses this to control style at different resolution levels through adaptive instance normalization. This enables incredible control: changing coarse styles (pose, face shape) independently of fine styles (hair texture, skin pores). The paper demonstrated unprecedented image quality and introduced tools for analyzing and improving GANs (like perceptual path length metric). StyleGAN generated faces indistinguishable from real photographs, achieving a milestone in generative modeling. The architecture's success showed that generator design matters enormously—not all ways of mapping noise to images are equally good. The disentanglement properties (ability to control attributes independently) made StyleGAN useful for semantic editing and style transfer, expanding GANs from pure generation to controllable synthesis.

## Common Pitfalls and Tricks

Mode collapse is perhaps the most frustrating failure mode in GAN training. The generator discovers it can fool the discriminator by producing only a few types of outputs rather than the full data diversity. For MNIST, this might mean generating only 1s and 7s, ignoring other digits. For faces, generating only certain poses or expressions. Detection requires checking sample diversity, not just quality—generate many samples and verify they span the data distribution. Solutions include minibatch discrimination (let discriminator see multiple samples and detect homogeneity), unrolled optimization (let generator anticipate discriminator's response), or using different loss functions like Wasserstein GAN that are less prone to mode collapse. Understanding that mode collapse stems from the generator finding local optima in the adversarial game helps recognize when it's occurring and motivates these solutions.

Discriminator overpowering the generator early in training is common and destructive. If the discriminator becomes too good too quickly, it assigns probability near 0 to all generator outputs, providing vanishing gradients to the generator which can't learn. This happens when the discriminator is too large relative to the generator, learning rate is too high for discriminator, or real/fake distributions are easily separable initially (generator starts terrible). Solutions: train discriminator less frequently (every $$k$$ generator updates), use lower learning rate for discriminator, add noise to discriminator inputs (blurring the real/fake distinction), or use one-sided label smoothing (real labels = 0.9 instead of 1.0, reducing discriminator overconfidence). Monitoring $$D(\mathbf{x}_{\text{real}})$$ and $$D(G(\mathbf{z}))$$ helps: if real is always near 1 and fake always near 0, discriminator is too strong.

Using batch normalization in the discriminator can cause problems when batch sizes are small because batch statistics become unreliable. With batch size 1, batch norm fails entirely. Solutions include using larger batch sizes (at least 32-64), using layer normalization or instance normalization instead of batch norm, or using virtual batch normalization (normalize using statistics from a fixed reference batch). Understanding that discriminator's normalization affects what features it learns helps debug training issues related to batch size.

Evaluating GAN quality is challenging because we can't compute likelihood. Inception Score measures both quality (samples should be confidently classified) and diversity (should cover all classes) using a pre-trained classifier, but has limitations (doesn't detect memorization, biased toward ImageNet classes). Fréchet Inception Distance (FID) compares statistics of real and generated samples in feature space, providing a better metric but still imperfect. For practical work, visual inspection remains important—generate many samples and manually check quality and diversity. Quantitative metrics complement but don't replace human evaluation.

A powerful technique for stable training is spectral normalization, which constrains discriminator's Lipschitz constant by normalizing weight matrices by their spectral norm (largest singular value). This prevents the discriminator from having arbitrarily large gradients, stabilizing training dynamics. The technique adds minimal computational cost (computing spectral norm via power iteration) while dramatically improving stability. Modern GANs often use spectral normalization in the discriminator as standard practice, showing how theoretical understanding of training dynamics (Lipschitz constraint improves stability) translates to practical techniques.

## Key Takeaways

Generative Adversarial Networks learn to generate realistic data by training two networks adversarially: a generator creating fake samples from random noise and a discriminator distinguishing real from fake. The adversarial objective is formulated as a minimax game where the generator minimizes what the discriminator maximizes, creating competitive dynamics that drive both networks toward higher capability. At equilibrium, the generator's distribution matches the data distribution and the discriminator cannot distinguish real from fake better than random guessing, though reaching this equilibrium in practice is challenging. Training alternates between discriminator updates (using real data and generator's fakes) and generator updates (trying to fool the discriminator), requiring careful balancing to prevent either network from dominating. Mode collapse—the generator producing limited diversity—remains a persistent challenge addressed through architectural choices, modified objectives, and training techniques. GANs excel at generating high-quality, realistic samples (often superior to VAEs) and learning latent spaces with semantic structure enabling interpolation and manipulation. The implicit density modeling approach enables generating complex high-dimensional data without explicit probabilistic formulations, though at the cost of training instability and difficulty in evaluation. Understanding GANs deeply means appreciating both their creative power in generating realistic data and the delicate training dynamics that make them challenging but rewarding to work with in practice.

The GAN framework demonstrates that competition can be a powerful learning signal, a principle that has influenced deep learning far beyond generative modeling.

