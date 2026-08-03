---
layout: post
title: 12-01-01 Autoencoder Theory
chapter: '12'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter12
---

# Autoencoders: Learning Efficient Representations

![Autoencoder Architecture](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Autoencoder_structure.png/600px-Autoencoder_structure.png)
*Hình ảnh: Kiến trúc Autoencoder với encoder và decoder. Nguồn: Wikimedia Commons*

## 1. Concept Overview

Autoencoders represent a fundamentally different paradigm in neural network training compared to the supervised learning we've studied so far. Instead of learning to map inputs to labeled outputs, autoencoders learn to reconstruct their inputs through an information bottleneck. This seemingly circular task—predicting the input from itself—becomes meaningful when we constrain the network to pass information through a lower-dimensional hidden layer called the latent space or code. By forcing the network to compress and then decompress the input, we compel it to learn efficient representations that capture the essential structure of the data while discarding noise and irrelevant details.

The power of autoencoders lies not in the reconstruction itself but in what they learn during the process. The encoder learns to extract the most important features from high-dimensional data and compress them into a compact representation. The decoder learns to generate realistic data from these compressed representations. The latent space that emerges has remarkable properties: nearby points in latent space often correspond to semantically similar inputs, and we can interpolate smoothly between points to generate novel, realistic examples. These properties make autoencoders valuable for dimensionality reduction, denoising, anomaly detection, feature learning for downstream tasks, and as building blocks for more sophisticated generative models.

Understanding autoencoders requires appreciating the interplay between capacity and constraint. If the latent dimension equals or exceeds the input dimension, the network can simply learn the identity function, copying inputs through unchanged—useless for learning meaningful structure. The bottleneck—making the latent dimension smaller than the input—forces the network to make choices about what to preserve. With a 784-dimensional input image compressed to 32 latent dimensions, the network cannot possibly encode every pixel independently. It must discover higher-level features like edges, shapes, and textures that compactly represent the image's essential content. This compression isn't arbitrary but learned from data, adapting to the specific structure present in the training distribution.

The historical significance of autoencoders extends beyond their practical applications. They were among the first successful unsupervised learning methods in deep learning, demonstrating that neural networks could learn meaningful representations without labeled data. This influenced the development of pre-training strategies that enabled training deeper networks in the pre-ReLU era. Modern self-supervised learning and contrastive methods can be seen as descendants of autoencoder ideas—learning representations by predicting or reconstructing parts of the input from other parts. The autoencoder framework also introduced the encoder-decoder architecture pattern that has proven enormously influential, appearing in sequence-to-sequence models, variational autoencoders, and generative adversarial networks.

Yet autoencoders have important limitations that motivate more sophisticated generative models. Standard autoencoders learn to compress and reconstruct training data but don't necessarily learn a good generative model—the latent space might have "holes" where no training examples map, making random sampling produce unrealistic outputs. They don't explicitly model the data distribution, limiting their theoretical guarantees. And the reconstruction loss, while intuitive, might not capture perceptual similarity (two images can be pixel-wise different yet perceptually similar, or pixel-wise similar yet perceptually different). These limitations led to variational autoencoders (which model distributions explicitly), generative adversarial networks (which use adversarial training instead of reconstruction loss), and perceptual losses (which measure similarity in feature space rather than pixel space). Understanding vanilla autoencoders provides the foundation for appreciating these more advanced techniques.

## 2. Mathematical Foundation

The mathematical framework of autoencoders is elegantly simple yet rich in implications. An autoencoder consists of two neural networks composed sequentially: an encoder $$f_\phi$$ parameterized by $$\phi$$ and a decoder $$g_\theta$$ parameterized by $$\theta$$. Given input $$\mathbf{x} \in \mathbb{R}^{d}$$, the encoder produces a latent representation:

$$\mathbf{z} = f_\phi(\mathbf{x}) \in \mathbb{R}^{k}$$

where $$k < d$$ enforces the bottleneck (though we'll discuss cases where this isn't strictly required). The decoder reconstructs from the latent representation:

$$\hat{\mathbf{x}} = g_\theta(\mathbf{z}) = g_\theta(f_\phi(\mathbf{x})) \in \mathbb{R}^{d}$$

The training objective minimizes reconstruction error:

$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

for continuous data (mean squared error), or:

$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = -\sum_{i=1}^{d} [x_i \log(\hat{x}_i) + (1-x_i)\log(1-\hat{x}_i)]$$

for binary data (binary cross-entropy, treating each dimension as independent Bernoulli).

The choice of loss function embodies assumptions about the data and noise model. MSE assumes Gaussian noise: we're modeling $$p(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; g_\theta(\mathbf{z}), \sigma^2 I)$$, and minimizing MSE is equivalent to maximum likelihood under this assumption. Binary cross-entropy assumes Bernoulli noise: each pixel is independently binary with probability $$\hat{x}_i$$. For images with continuous values in [0,1], this is actually modeling each pixel as a probability, which seems odd but works reasonably in practice. More sophisticated approaches use perceptual losses based on feature distances in pre-trained networks, better capturing perceptual similarity.

The bottleneck dimension $$k$$ is the key hyperparameter controlling the compression-fidelity tradeoff. Very small $$k$$ (like 2-3 dimensions) creates extreme compression, forcing the network to capture only the most essential variations in data. This is useful for visualization (we can plot the 2D latent space) but may lose important details. Moderate $$k$$ (32-128 dimensions for image datasets) balances compression and reconstruction quality. Large $$k$$ (approaching input dimension) reduces compression pressure but might not learn interesting structure.

Interestingly, even with $$k \geq d$$ (no dimensional bottleneck), we can force meaningful learning through other constraints. Sparse autoencoders add a sparsity penalty to the latent activations:

$$\mathcal{L}_{\text{sparse}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \sum_{j=1}^{k} KL(\rho \| \hat{\rho}_j)$$

where $$\rho$$ is a target sparsity level (e.g., 0.05) and $$\hat{\rho}_j$$ is the average activation of latent unit $$j$$ over the training set. The KL divergence penalty encourages most latent units to be inactive (near zero) most of the time, forcing different units to specialize in different patterns. This creates a sparse, distributed representation even without dimensional bottleneck.

Denoising autoencoders corrupt the input $$\mathbf{x}$$ with noise to create $$\tilde{\mathbf{x}}$$ but train to reconstruct the original:

$$\mathcal{L} = \|\mathbf{x} - g_\theta(f_\phi(\tilde{\mathbf{x}}))\|^2$$

The corruption process might add Gaussian noise, mask random pixels, or add salt-and-pepper noise. This forces the encoder to learn robust features invariant to the noise type, and the decoder to learn to "fill in" corrupted regions based on uncorrupted context. Denoising autoencoders often learn better features than vanilla autoencoders because the denoising task requires understanding data structure, not just memorizing training examples.

The latent space geometry deserves careful analysis. In a well-trained autoencoder on image data, nearby points in latent space typically correspond to perceptually similar images. We can interpolate linearly between two latent codes $$\mathbf{z}_1$$ and $$\mathbf{z}_2$$:

$$\mathbf{z}_t = (1-t)\mathbf{z}_1 + t\mathbf{z}_2, \quad t \in [0,1]$$

and decode $$g_\theta(\mathbf{z}_t)$$ to generate intermediate images. For well-behaved autoencoders, this produces smooth transitions (morphing one face into another, for example). However, standard autoencoders don't guarantee good interpolation—there might be "holes" in latent space where no training examples map, and interpolating through these holes produces unrealistic reconstructions. Variational autoencoders address this by explicitly regularizing the latent space to be continuous and well-behaved.

## 3. Example / Intuition

To build concrete intuition for how autoencoders learn representations, let's trace through training on MNIST digits. Suppose we compress 28×28=784 pixel images to 32-dimensional latent codes.

Initially, with random weights, the encoder produces meaningless latent codes and the decoder generates random noise as reconstruction. The reconstruction error is enormous—we're trying to match 784 pixel values but getting essentially random outputs. Gradients via backpropagation indicate how to adjust encoder and decoder weights to reduce this error.

As training progresses, the encoder learns to extract increasingly meaningful features. Early on, it might learn that certain pixels tend to be dark (in the background) versus light (in digit strokes), encoding this as latent dimensions representing average brightness in different regions. This primitive encoding already allows better reconstruction than random noise—the decoder learns to generate images with appropriate overall brightness patterns.

With more training, the encoder discovers edge patterns. Certain latent dimensions become active when the digit has vertical strokes (1, 4, 7), others for curves (0, 6, 8, 9), others for horizontal segments (2, 3, 5, 7). The decoder learns to reconstruct digit-like images from these edge indicators. Reconstructions now capture the general shape of digits, though details might be blurry.

Eventually, the 32 latent dimensions self-organize into a meaningful representation space. Dimensions might encode: digit identity (roughly which digit), stroke thickness, slant, size, position in image. This learned representation emerges purely from the reconstruction objective—we never told the network what features to learn, only to compress and reconstruct accurately.

Consider what happens when we encode several "3"s from the training set. Their latent codes cluster together in the 32D latent space because they share structure (similar edges, curves, topology). Different "3"s (thick, thin, slanted) map to slightly different but nearby latent points. Meanwhile, "8"s cluster in a different region of latent space—they share the topological structure (two loops) that "3"s lack. The latent space has self-organized to reflect digit categories and variations within categories, all without any labels.

Now for the interpolation test. Encode a "3" to get $$\mathbf{z}_3$$ and encode an "8" to get $$\mathbf{z}_8$$. Decode intermediate points:

$$\mathbf{z}_{0.0} = \mathbf{z}_3 \to$$ decodes to "3"  
$$\mathbf{z}_{0.25} = 0.75\mathbf{z}_3 + 0.25\mathbf{z}_8 \to$$ decodes to "3 with hint of 8"  
$$\mathbf{z}_{0.5} = 0.5\mathbf{z}_3 + 0.5\mathbf{z}_8 \to$$ decodes to ambiguous digit  
$$\mathbf{z}_{0.75} = 0.25\mathbf{z}_3 + 0.75\mathbf{z}_8 \to$$ decodes to "8 with hint of 3"  
$$\mathbf{z}_{1.0} = \mathbf{z}_8 \to$$ decodes to "8"

If interpolation is smooth, we see gradual morphing. If there are discontinuities, we might get unrealistic outputs at intermediate points. This interpolation quality is a diagnostic for whether the latent space is well-structured.

Denoising autoencoders add an interesting twist. Suppose we corrupt a "7" by randomly zeroing 20% of pixels. The corrupted image is ambiguous—it could be a damaged "7" or possibly a "1". The denoising autoencoder must use context (uncorrupted pixels) to infer the most likely original digit. This requires understanding digit structure, not just memorizing pixel patterns. The encoder learns to extract robust features despite corruption, and the decoder learns to generate complete digits from partial evidence. The learned representations are often more useful for downstream tasks than those from vanilla autoencoders because they're forced to capture semantic structure rather than low-level pixel statistics.

