---
layout: post
title: 01-00-01 What Is Deep Learning
chapter: '01'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter01
---

# Introduction to Deep Learning

## 1. Concept Overview

![Deep Learning Hierarchy](https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/AI-ML-DL.svg/800px-AI-ML-DL.svg.png)
*Hình ảnh: Mối quan hệ giữa AI, Machine Learning và Deep Learning. Nguồn: Wikimedia Commons*

Deep Learning represents one of the most transformative technological advances of the 21st century. At its core, **deep learning** is a subset of machine learning that uses artificial neural networks with multiple layers—hence "deep"—to automatically learn hierarchical representations of data. What makes deep learning revolutionary is not just that it works, but how fundamentally it changes our approach to building intelligent systems.

To truly understand deep learning's significance, we must appreciate what preceded it. Traditional machine learning required human experts to manually engineer features—the relevant patterns or characteristics that algorithms would use to make decisions. For image recognition, this meant designing edge detectors, texture analyzers, and shape descriptors by hand. For speech recognition, it meant crafting phoneme representations and acoustic models based on linguistic theory. This feature engineering was both an art and a science, requiring deep domain expertise and often years of iterative refinement.

Deep learning eliminates this bottleneck through **representation learning**—the ability to automatically discover the representations needed for detection or classification directly from raw data. A deep neural network learns features at multiple levels of abstraction: in computer vision, the first layer might learn to detect edges, the second layer combines edges into simple shapes, the third layer assembles shapes into object parts, and deeper layers recognize complete objects. Critically, the network discovers these hierarchical features on its own, without human guidance beyond providing the training data and the learning objective.

This automatic feature learning has profound implications. It means deep learning can tackle problems where we don't know how to manually engineer good features. It means the same basic architecture—with appropriate modifications—can excel at diverse tasks: recognizing faces, translating languages, generating images, playing games, or folding proteins. It means that as we collect more data and apply more computation, performance continues to improve, rather than plateauing as it often does with carefully hand-tuned classical systems.

### Key Characteristics of Deep Learning

**1. Hierarchical Feature Learning**: Deep networks learn features at multiple levels of abstraction. Low-level layers capture simple patterns (edges, colors, basic phonemes), while higher layers combine these into complex concepts (objects, faces, semantic meanings). This hierarchy mirrors how we believe biological vision and cognition work—building understanding layer by layer from simple to complex.

**2. End-to-End Learning**: Rather than building modular pipelines where each component is optimized separately, deep learning enables end-to-end optimization where the entire system learns jointly. For machine translation, instead of separate modules for parsing, alignment, and generation, a single neural network learns to map source language to target language directly, with all components optimized together toward the final translation quality.

**3. Scalability with Data and Computation**: Traditional machine learning often exhibits diminishing returns—adding more data beyond a certain point provides little benefit. Deep learning's performance continues to improve with more data and more computation, a scaling property that has driven the revolution in large language models and computer vision systems. This scalability is both a strength (enabling superhuman performance on many tasks) and a challenge (requiring massive datasets and computational resources).

**4. Distributed Representations**: Deep networks learn to represent concepts as patterns of activation across many neurons, rather than having dedicated neurons for each concept. This enables generalization: knowledge about "dogs" can inform understanding of "wolves" because they share many representational features. It also provides robustness: if some neurons fail or are dropped out during training, the distributed representation still functions.

### Why Deep Learning Matters

The impact of deep learning extends far beyond academic curiosity. It has fundamentally changed multiple industries and aspects of daily life:

- **Computer Vision**: From barely functional digit recognition in the 1990s to systems that surpass human performance on many visual tasks, recognize thousands of object categories, generate photorealistic images, and enable autonomous vehicles.

- **Natural Language Processing**: From rigid rule-based systems to neural language models that can write essays, answer questions, translate between languages with near-human quality, and engage in coherent dialogue.

- **Healthcare**: From slow, error-prone manual diagnosis to AI systems that detect diseases from medical images with expert-level accuracy, predict patient outcomes, accelerate drug discovery, and personalize treatment plans.

- **Scientific Discovery**: From traditional hypothesis-driven research to AI systems that discover novel materials, predict protein structures (AlphaFold solving a 50-year grand challenge), generate hypotheses from literature, and design experiments.

Perhaps most importantly, deep learning has democratized AI. Open-source frameworks like PyTorch and TensorFlow, pre-trained models available freely, and educational resources have made powerful AI accessible to anyone with a laptop and curiosity. This democratization accelerates innovation as millions of researchers and developers worldwide contribute to advancing the field.

## 2. Mathematical Foundation

At its mathematical core, deep learning is about **function approximation**. Given training data $$\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}$$ where $$\mathbf{x}_i$$ are inputs (images, text, sensor readings) and $$y_i$$ are desired outputs (labels, translations, actions), we want to find a function $$f_\theta$$ parameterized by $$\theta$$ that accurately maps inputs to outputs.

![Neural net = function approximator](/deep-learning-self-learning/img/chapter_img/chapter01/nn_function_approx_title.jpg)
*Figure: The neural “black box” is really fitting a curve $$y = f(x)$$ from data. (Illustration from a video on the nature of neural nets)*

![Overview: network vs target curve](/deep-learning-self-learning/img/chapter_img/chapter01/nn_overview_network_vs_curve.jpg)
*Figure: The same network, given different $$x$$, traces points on a complex curve. (Illustration from a video on the nature of neural nets)*

**Linear → nonlinear intuition.** Linear regression $$f(x)=xw+b$$ only draws straight lines; for wavy data it is **not enough**.

![Linear regression formula](/deep-learning-self-learning/img/chapter_img/chapter01/nn_linear_regression_formula.jpg)
*Figure: A linear neuron is an extension of $$f(x)=xw+b$$. (Illustration from a video on the nature of neural nets)*

![Linear model fails on nonlinear data](/deep-learning-self-learning/img/chapter_img/chapter01/nn_linear_fails_nonlinear.jpg)
*Figure: Linear models are limited — they cannot fit complex curves. (Illustration from a video on the nature of neural nets)*

With nonlinear activations (e.g. sigmoid-like), a network can fit soft shapes:

![Sigmoid-like curve fit](/deep-learning-self-learning/img/chapter_img/chapter01/nn_sigmoid_curve_fit.jpg)
*Figure: An example “soft” function a model can approximate. (Illustration from a video on the nature of neural nets)*

### The Universal Approximation Theorem

A fundamental theoretical result states that a neural network with even a single hidden layer containing sufficiently many neurons can approximate any continuous function to arbitrary accuracy. Mathematically, for any continuous function $$g: \mathbb{R}^n \to \mathbb{R}^m$$ and any $$\epsilon > 0$$, there exists a neural network $$f_\theta$$ such that:

$$\|f_\theta(\mathbf{x}) - g(\mathbf{x})\| < \epsilon \quad \text{for all } \mathbf{x}$$

This is remarkable: neural networks are **universal function approximators**.

![Conclusion: universal function approximator](/deep-learning-self-learning/img/chapter_img/chapter01/nn_conclusion_universal_approx.jpg)
*Figure: Core takeaway — neural nets are universal function approximators. (Illustration from a video on the nature of neural nets)*

However, this theorem has important caveats. It guarantees existence but not learnability—finding the parameters $$\theta$$ through gradient descent is not guaranteed. It requires potentially exponentially many neurons in the hidden layer, which is impractical. And it applies to shallow networks, but doesn't explain why deep networks work better in practice.

### Why Depth Matters

While shallow networks are theoretically sufficient, **deep networks** are exponentially more efficient for many real-world functions. Consider representing a function with $$k$$ levels of composition: $$f = f_k \circ f_{k-1} \circ \cdots \circ f_1$$. A shallow network might need exponentially many neurons to represent this, while a deep network with $$k$$ layers can represent it naturally with polynomial complexity.

The mathematical intuition is that many functions in nature exhibit compositional structure. To recognize a face, we first detect edges, then combine edges into facial features (eyes, nose, mouth), then combine features into a face representation. This hierarchical composition is naturally expressed as successive transformations through layers:

$$\mathbf{h}^{(1)} = \sigma(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{h}^{(2)} = \sigma(\mathbf{W}^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)})$$
$$\vdots$$
$$\mathbf{y} = \mathbf{W}^{(L)}\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}$$

where $$\sigma$$ is a nonlinear activation function (ReLU, sigmoid, tanh), $$\mathbf{W}^{(l)}$$ are weight matrices, and $$\mathbf{b}^{(l)}$$ are bias vectors at layer $$l$$.

### The Learning Objective

Training a neural network means finding parameters $$\theta = \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \ldots, \mathbf{W}^{(L)}, \mathbf{b}^{(L)}\}$$ that minimize a loss function $$\mathcal{L}(\theta)$$ measuring prediction error:

$$\theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$$

For classification, we typically use cross-entropy loss:
$$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_j y_j \log \hat{y}_j$$

For regression, mean squared error:
$$\mathcal{L}(\hat{y}, y) = \frac{1}{2}(y - \hat{y})^2$$

We optimize this via **gradient descent**: iteratively updating parameters in the direction that decreases loss:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

where $$\eta$$ is the learning rate. Computing gradients efficiently through backpropagation—applying the chain rule layer by layer—is what makes training deep networks practical.

### Why It Works: The Bias-Variance Tradeoff

Deep learning's success can be understood through the classical bias-variance tradeoff. High bias (underfitting) means the model can't capture the data's complexity. High variance (overfitting) means the model fits noise rather than true patterns. Deep networks have enormous capacity (low bias) but are surprisingly resistant to overfitting when properly regularized, achieving low variance despite having millions of parameters—often more parameters than training examples!

This seems to violate classical statistical learning theory, which suggests models should be simpler than the data. Recent theoretical work on "double descent" and "implicit regularization" shows that overparameterized networks trained with gradient descent implicitly prefer simpler functions, providing a form of automatic regularization that classical theory didn't account for.

## 3. Example / Intuition

To develop intuition for how deep learning works, let's walk through a concrete example: teaching a network to recognize handwritten digits.

### The Problem: MNIST Digit Recognition

Imagine you have 28×28 pixel grayscale images of handwritten digits (0-9), and you want a system that can correctly identify which digit is in each image. Each image is just a 784-dimensional vector (28×28 = 784 pixels, each with intensity 0-255).

**Traditional Approach**: You might manually design features:
- Count loops (0, 6, 8, 9 have loops; 1, 7 don't)
- Detect vertical/horizontal strokes
- Measure height-to-width ratios
- Identify endpoints and intersections

This requires deep expertise and doesn't generalize well (what about cursive? different fonts?).

**Deep Learning Approach**: Feed the 784 pixel values directly into a neural network:

```
Input (784 pixels) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (64 neurons) → Output (10 classes)
```

The network automatically learns:
- **Layer 1** discovers edge detectors—neurons that activate for vertical lines, horizontal lines, curves at different positions
- **Layer 2** combines edges into stroke patterns—long vertical strokes (for 1, 7), circular shapes (for 0, 6, 8, 9), specific curve combinations
- **Output layer** combines these patterns to recognize complete digits

### How Learning Happens: An Intuitive Example

Initially, weights are random. When shown a "3":
1. **Forward pass**: Network makes a random prediction, say 70% confident it's a "7"
2. **Compute error**: True label is "3", prediction was "7"—big error!
3. **Backward pass (backpropagation)**: 
   - Output layer: "I should have activated neuron 3 more and neuron 7 less"
   - Hidden layers: "Which of my activations contributed to the wrong prediction? Adjust weights to fix this"
4. **Update weights**: Slightly modify all weights to reduce this particular error
5. **Repeat**: After seeing thousands of "3"s with various handwriting styles, the network learns the essential features of "3"-ness

The magic is that this simple process—forward pass, compute error, backpropagate, update—when repeated millions of times, discovers the hierarchical features needed for recognition.

### Why Hierarchical Learning Matters

Consider recognizing a face:
- **Low-level features** (Layer 1): Edge detectors at various orientations, color blobs
- **Mid-level features** (Layer 2-3): Combine edges into simple shapes—curves, corners, textures
- **High-level features** (Layer 4-5): Combine shapes into facial features—eyes (pair of dark circles with highlights), nose (triangular region with shadows), mouth (horizontal dark region, possibly with teeth)
- **Complete concept** (Output): Combine facial features into specific face identities

Each layer learns increasingly abstract representations, naturally capturing the compositional nature of visual recognition. The network discovers that eyes, noses, and mouths are reusable components that appear in all faces, just as strokes and curves are reusable components in all digits.

This hierarchical, distributed representation also explains deep learning's sample efficiency. Once the network learns "edge detector" and "circle detector" neurons from seeing digits, these same neurons help recognize letters, faces, and objects—transfer learning happens naturally through shared low-level features.

<!-- video-references -->

## Video references

Some figures in this lesson are screenshots from the following videos (Machine Learning Thực Chiến). URLs kept for attribution and further viewing:

- [Neural nets as function approximators](https://www.facebook.com/reel/720970114372332)
