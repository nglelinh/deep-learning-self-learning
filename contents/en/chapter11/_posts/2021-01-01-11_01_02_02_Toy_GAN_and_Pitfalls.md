---
layout: post
title: 11-01-02-02 Toy GAN, Papers, and Pitfalls
chapter: '11'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter11
---

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

