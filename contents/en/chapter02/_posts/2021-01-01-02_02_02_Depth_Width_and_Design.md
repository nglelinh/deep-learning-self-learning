---
layout: post
title: 02-02-02 Depth, Width, and Design Patterns
chapter: '02'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter02
---

## Network Depth and Width

### Width

The **width** of a layer refers to the number of neurons it contains.

- **Wider networks**: More neurons per layer
  - Greater capacity to learn complex patterns within a single layer
  - More parameters (can lead to overfitting)
  - More computational cost per layer

### Depth

The **depth** of a network refers to the number of layers.

- **Deeper networks**: More layers
  - Can learn hierarchical representations
  - More expressive (can represent more complex functions)
  - Can be harder to train (vanishing/exploding gradients)
  - The term "deep learning" comes from using deep networks

### The Universal Approximation Theorem

**Theorem**: A feedforward neural network with:
- A single hidden layer
- Finite number of neurons
- Appropriate activation function (e.g., sigmoid, ReLU)

can approximate any continuous function on a compact subset of $$\mathbb{R}^n$$ to arbitrary accuracy.

**Important Notes:**
- This is an **existence theorem**, not a practical guideline
- It doesn't specify how many neurons are needed (could be exponentially many)
- Deeper networks can often approximate functions with far fewer parameters
- Deeper networks tend to learn hierarchical features naturally

**Visualizing depth/width via 1D curve fitting:** same wavy data — low capacity → poor fit; more hidden units + training → strong fit (see Architecture Theory for the full image set).

![Poor fit with low capacity](/deep-learning-self-learning/img/chapter_img/chapter02/nn_shallow_poor_fit.jpg)
*Figure: Too thin/shallow to approximate the target. (Illustration from a video on the nature of neural nets)*

![Good fit with enough capacity](/deep-learning-self-learning/img/chapter_img/chapter02/nn_deeper_good_fit.jpg)
*Figure: Enough depth/width + training → near-perfect function approximation. (Illustration from a video on the nature of neural nets)*

## Common Design Patterns

### Decreasing Width

A common pattern is to gradually decrease the layer width:

```
Input (784) → 512 → 256 → 128 → 64 → Output (10)
```

**Rationale**: Progressively compress information into higher-level abstractions.

### Hourglass/Bottleneck Architecture

Decrease then increase width:

```
Input (784) → 256 → 64 → 256 → Output (784)
```

**Use case**: Autoencoders for dimensionality reduction and reconstruction.

### Uniform Width

Keep all hidden layers the same size:

```
Input (784) → 256 → 256 → 256 → Output (10)
```

**Rationale**: Simplicity and easier hyperparameter tuning.

## Activation Functions Per Layer

Different layers can use different activation functions:

**Typical configuration:**
- **Hidden layers**: ReLU (or variants like Leaky ReLU, ELU)
  - Computational efficiency
  - Mitigates vanishing gradient
  
- **Output layer**: Task-dependent
  - Binary classification: Sigmoid
  - Multi-class classification: Softmax
  - Regression: Linear (identity function)

## Fully Connected vs. Other Architectures

### Fully Connected (Dense) Layers

Every neuron in layer $$l$$ is connected to every neuron in layer $$l-1$$.

**Advantages:**
- Maximum flexibility
- Can learn any pattern (given enough neurons)

**Disadvantages:**
- Many parameters ($$n^{[l]} \times n^{[l-1]}$$)
- No built-in assumption about input structure
- Not efficient for structured data (images, sequences)

### Specialized Architectures

For specific data types, specialized architectures are more efficient:

- **Convolutional layers**: For images (spatial structure)
- **Recurrent layers**: For sequences (temporal structure)
- **Attention mechanisms**: For handling long-range dependencies

We'll cover these in later chapters.

## Network Representation

### Graphical Representation

Networks are often visualized as directed acyclic graphs (DAGs):

```
      Input Layer    Hidden Layer 1   Hidden Layer 2   Output Layer
         (3)             (4)              (4)              (2)
    
    x₁  ○────────────────●──────────────────●──────────────────●  ŷ₁
                        ╱│╲              ╱│╲              ╱│
    x₂  ○──────────────●─●─●────────────●─●─●────────────●─●
                        ╲│╱              ╲│╱              ╲│
    x₃  ○────────────────●──────────────────●──────────────────●  ŷ₂
```

### Matrix Representation

For computational efficiency, we represent operations as matrix multiplications:

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

where:
- $$\mathbf{A}^{[l-1]}$$: activation matrix (each column is one example)
- $$\mathbf{W}^{[l]}$$: weight matrix
- $$\mathbf{b}^{[l]}$$: bias vector (broadcasted across examples)

## Practical Implementation

### Example: Simple Neural Network in Python

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of layer sizes including input and output
        Example: [784, 128, 64, 10] for MNIST
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # He initialization for ReLU networks
            w = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2.0 / layer_sizes[i-1])
            b = np.zeros((layer_sizes[i], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward(self, x):
        """
        x: input of shape (input_size, num_examples)
        Returns: output of shape (output_size, num_examples)
        """
        a = x
        activations = [x]
        zs = []
        
        # Forward through hidden layers
        for i in range(self.num_layers - 2):
            z = self.weights[i] @ a + self.biases[i]
            a = self.relu(z)
            zs.append(z)
            activations.append(a)
        
        # Output layer (softmax)
        z = self.weights[-1] @ a + self.biases[-1]
        a = self.softmax(z)
        zs.append(z)
        activations.append(a)
        
        return a, activations, zs
    
    def predict(self, x):
        """Returns class predictions"""
        output, _, _ = self.forward(x)
        return np.argmax(output, axis=0)

# Example usage
network = NeuralNetwork([784, 128, 64, 10])
x = np.random.randn(784, 5)  # 5 examples
output, _, _ = network.forward(x)
print(f"Output shape: {output.shape}")  # (10, 5)
print(f"Predictions: {network.predict(x)}")
```

## Design Considerations

### Number of Layers

- **1-2 hidden layers**: Simple problems, small datasets
- **3-5 hidden layers**: Moderate complexity
- **5+ hidden layers**: Complex problems, large datasets, "deep" learning

### Number of Neurons Per Layer

Rules of thumb:
- Start with layers of size between input and output size
- Common sizes: 32, 64, 128, 256, 512
- More neurons = more capacity but more overfitting risk
- Use validation performance to guide choices

### Architecture Search

Finding the optimal architecture is often done through:
- **Manual experimentation**: Try different configurations
- **Grid search**: Systematically try combinations
- **Random search**: Often more efficient than grid search
- **Neural Architecture Search (NAS)**: Automated methods (advanced topic)

## Summary

- **Neural networks** consist of layers of neurons organized into input, hidden, and output layers
- **Feedforward networks** (MLPs) are the simplest architecture where information flows in one direction
- **Forward propagation** computes the output by passing inputs through successive layers
- **Network depth** (number of layers) and **width** (neurons per layer) determine capacity
- **Universal Approximation Theorem** shows networks can approximate any function, but doesn't guarantee efficiency
- **Fully connected layers** connect every neuron to every neuron in adjacent layers
- **Proper architecture design** depends on the problem, data, and computational resources

In the next lesson, we'll explore activation functions in more detail and understand their critical role in learning.

<!-- video-references -->

## Video references

Some figures in this lesson are screenshots from the following videos (Machine Learning Thực Chiến). URLs kept for attribution and further viewing:

- [Neural nets as function approximators](https://www.facebook.com/reel/720970114372332)
