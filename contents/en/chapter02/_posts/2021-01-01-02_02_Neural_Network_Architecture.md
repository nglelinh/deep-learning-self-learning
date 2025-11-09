---
layout: post
title: 02-02 Neural Network Architecture
chapter: '02'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter02
lesson_type: required
---

# Neural Network Architecture: From Neurons to Deep Systems

![Neural Network Layers](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/500px-Colored_neural_network.svg.png)
*Hình ảnh: Kiến trúc neural network với input, hidden và output layers. Nguồn: Wikimedia Commons*

## 1. Concept Overview

Neural network architecture is the blueprint that defines how individual neurons are organized, connected, and structured to solve complex problems. While a single neuron can only learn linear decision boundaries (as we saw with the perceptron), the true power of deep learning emerges when we compose many neurons into layers and stack these layers into deep architectures. This compositional structure is not merely a engineering convenience—it reflects a profound insight about how complex intelligence can emerge from simple computational units working in concert.

Understanding architecture is crucial because the way we organize neurons fundamentally determines what a network can learn and how efficiently it learns. A poorly designed architecture might fail to learn even simple patterns, while a well-designed one can discover intricate relationships in data with remarkable efficiency. The architecture embodies our inductive biases—our assumptions about the problem structure—allowing the network to learn more effectively than treating all problems as completely general function approximation tasks.

### The Layered Paradigm

The fundamental organizing principle of neural networks is **layers**—groups of neurons that perform transformations at the same stage of computation. This layered structure naturally implements compositional computation: each layer transforms its input into a new representation, and subsequent layers build on these representations to create increasingly abstract features. When recognizing a face, early layers might detect edges, middle layers combine edges into facial features (eyes, nose, mouth), and deep layers recognize complete identities.

This hierarchical processing mirrors both biological neural systems and the compositional nature of many real-world concepts. A "car" is composed of wheels, windows, and doors; these components are composed of shapes and textures; these are composed of edges and colors. Neural network layers naturally capture this hierarchy through learned transformations, with each layer learning the appropriate level of abstraction for its position in the processing pipeline.

### Why Architecture Matters: The Depth vs Width Tradeoff

A critical insight from both theory and practice is that **depth** (number of layers) and **width** (neurons per layer) have fundamentally different effects on network capacity and learning. The Universal Approximation Theorem tells us that a single hidden layer with sufficiently many neurons can approximate any continuous function. Yet in practice, deep networks with relatively few neurons per layer dramatically outperform shallow wide networks on complex tasks.

This isn't just about parameter efficiency, though deep networks often achieve the same representational power with exponentially fewer parameters than shallow ones. Deep networks learn hierarchical features naturally—you don't need to tell them to detect edges first, then shapes, then objects; this emerges automatically from the training process. They also exhibit better generalization: the intermediate representations learned by deep networks transfer across tasks, enabling powerful techniques like transfer learning and pre-training that shallow networks don't support nearly as well.

Understanding the tradeoff between depth and width, and how architecture choices affect training dynamics, generalization, and computational efficiency, is essential for designing effective neural networks. This lesson provides the foundational understanding of how networks are structured, why these structures work, and how to make informed architectural decisions for your own applications.

## 2. Mathematical Foundation

### Feedforward Neural Networks: Formal Definition

A **feedforward neural network** (also called **Multilayer Perceptron** or **MLP**) is a function $$f: \mathbb{R}^{n_0} \to \mathbb{R}^{n_L}$$ defined by composition of layer transformations. For a network with $$L$$ layers, the function is:

$$f(\mathbf{x}) = f^{[L]} \circ f^{[L-1]} \circ \cdots \circ f^{[1]}(\mathbf{x})$$

where each layer function $$f^{[l]}$$ is an affine transformation followed by an element-wise nonlinearity:

$$f^{[l]}(\mathbf{a}^{[l-1]}) = \sigma^{[l]}(\mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]})$$

Let's carefully unpack each component and understand why this seemingly simple formulation is so powerful.

### Layer-by-Layer Computation

For a network with $$L$$ layers (not counting the input), layer $$l \in \{1, 2, \ldots, L\}$$ computes:

**Pre-activation (linear transformation)**:
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

**Activation (nonlinear transformation)**:
$$\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]})$$

The dimensions are:
- $$\mathbf{a}^{[l-1]} \in \mathbb{R}^{n_{l-1}}$$: input to layer $$l$$ (output from previous layer)
- $$\mathbf{W}^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$$: weight matrix
- $$\mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$$: bias vector
- $$\mathbf{z}^{[l]} \in \mathbb{R}^{n_l}$$: pre-activation values
- $$\mathbf{a}^{[l]} \in \mathbb{R}^{n_l}$$: post-activation values (layer output)
- $$n_l$$: number of neurons in layer $$l$$

### The Role of Each Component

**Weight Matrix $$\mathbf{W}^{[l]}$$**: Each row $$\mathbf{w}_i^{[l]}$$ defines one neuron's linear combination of inputs. The matrix multiplication $$\mathbf{W}^{[l]} \mathbf{a}^{[l-1]}$$ computes all neurons' pre-activations in parallel. The weights are the learnable parameters that adapt during training to capture patterns in data.

**Bias Vector $$\mathbf{b}^{[l]}$$**: Shifts the activation function left or right, allowing neurons to activate even when inputs are near zero. Without bias, a ReLU neuron with all-zero inputs would always output zero, limiting expressiveness. Bias is crucial for learning appropriate thresholds.

**Activation Function $$\sigma^{[l]}$$**: Introduces nonlinearity, enabling the network to learn non-linear decision boundaries. Without activation functions, stacking layers would be pointless—multiple linear transformations compose into a single linear transformation. Common choices include:
- ReLU: $$\sigma(z) = \max(0, z)$$
- Sigmoid: $$\sigma(z) = \frac{1}{1+e^{-z}}$$
- Tanh: $$\sigma(z) = \tanh(z)$$

### Output Layer Design

The output layer's structure depends fundamentally on the task, as it must produce outputs in the appropriate format for the loss function.

**Binary Classification** ($$y \in \{0, 1\}$$):
- Single output neuron with sigmoid activation
- Interpretation: $$\hat{y} = P(y=1|\mathbf{x})$$
- Output: $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0,1)$$
- Loss: Binary cross-entropy $$\mathcal{L} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

**Multi-class Classification** ($$y \in \{1,2,\ldots,K\}$$):
- $$K$$ output neurons with softmax activation
- Interpretation: $$\hat{y}_k = P(y=k|\mathbf{x})$$
- Output: $$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$ where $$\sum_{k=1}^K \hat{y}_k = 1$$
- Loss: Categorical cross-entropy $$\mathcal{L} = -\sum_{k=1}^K y_k \log \hat{y}_k$$

The softmax function has elegant properties: it's differentiable, outputs form a probability distribution, and it "softens" the argmax operation (hence the name), allowing gradient-based learning.

**Regression** ($$y \in \mathbb{R}$$):
- One or more output neurons with linear (identity) activation
- Output: $$\hat{y} = z$$ (no activation function)
- Loss: Mean Squared Error $$\mathcal{L} = \frac{1}{2}(y - \hat{y})^2$$

### Forward Propagation: The Complete Picture

Given input $$\mathbf{x} \in \mathbb{R}^{n_0}$$, forward propagation computes:

$$
\begin{align}
\mathbf{a}^{[0]} &= \mathbf{x} \quad \text{(initialize with input)} \\
\\
\text{For } l &= 1 \text{ to } L: \\
\mathbf{z}^{[l]} &= \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \quad \text{(affine transformation)} \\
\mathbf{a}^{[l]} &= \sigma^{[l]}(\mathbf{z}^{[l]}) \quad \text{(nonlinear activation)} \\
\\
\hat{\mathbf{y}} &= \mathbf{a}^{[L]} \quad \text{(final output)}
\end{align}
$$

This sequential computation builds increasingly complex representations. Each layer learns features at a different level of abstraction, with the composition of layers enabling the network to represent highly complex functions.

### Parameter Count and Complexity

The total number of learnable parameters is:

$$\text{Parameters} = \sum_{l=1}^{L} (n_l \times n_{l-1} + n_l) = \sum_{l=1}^{L} n_l(n_{l-1} + 1)$$

For a concrete example with architecture [784, 128, 64, 10]:
- Layer 1: $$128 \times 784 + 128 = 100,480$$ parameters
- Layer 2: $$64 \times 128 + 64 = 8,256$$ parameters  
- Layer 3: $$10 \times 64 + 10 = 650$$ parameters
- **Total**: $$109,386$$ parameters

This parameter count grows quadratically with layer width but only linearly with depth, explaining why deep narrow networks are often more parameter-efficient than shallow wide ones for similar representational capacity.

### The Universal Approximation Theorem

**Theorem** (Cybenko 1989, Hornik et al. 1989): Let $$\sigma$$ be a non-constant, bounded, monotonically-increasing continuous function (e.g., sigmoid). Then for any continuous function $$g$$ on a compact subset $$K \subset \mathbb{R}^n$$, any $$\epsilon > 0$$, and any probability measure $$\mu$$ on $$K$$, there exists a one-hidden-layer neural network $$f$$ such that:

$$\int_K |f(\mathbf{x}) - g(\mathbf{x})| d\mu(\mathbf{x}) < \epsilon$$

**What This Means**: Neural networks can approximate any continuous function arbitrarily well. This is remarkable—it means neural networks are universal function approximators, capable of representing any relationship we might want to learn.

**Important Caveats**:
1. **Existence ≠ Learnability**: The theorem guarantees a solution exists but doesn't tell us how to find it via gradient descent
2. **Width Requirements**: May need exponentially many neurons (in input dimension or precision $$1/\epsilon$$)
3. **Depth Efficiency**: Deeper networks can often achieve the same approximation with exponentially fewer parameters
4. **No Guidance on Architecture**: Doesn't tell us what activation functions, initializations, or learning rates to use

The theorem explains *why* neural networks work in principle, but practical deep learning success comes from additional insights about depth, architecture, optimization, and regularization that the theorem doesn't address.

## 3. Example / Intuition

To solidify understanding of neural network architecture, let's trace through a concrete example step by step, watching how information transforms as it flows through layers.

### Example: 3-Layer Network for MNIST

Consider a network designed to classify handwritten digits (28×28 grayscale images into 10 classes):

**Architecture**: [784 → 128 → 64 → 10]
- **Input**: 784 pixels (28×28 flattened)
- **Hidden Layer 1**: 128 neurons with ReLU
- **Hidden Layer 2**: 64 neurons with ReLU  
- **Output Layer**: 10 neurons with softmax

### Information Flow: A Detailed Walkthrough

**Step 1: Input (Layer 0)**

We receive a 28×28 image of the digit "3". Flattened into a vector:
$$\mathbf{a}^{[0]} = [0.2, 0.1, 0.0, 0.8, 0.9, \ldots] \in \mathbb{R}^{784}$$

Each value represents a pixel intensity (0=black, 1=white). The network sees this as a point in 784-dimensional space.

**Step 2: First Hidden Layer (Layer 1)**

This layer has 128 neurons, each looking for different patterns:

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{a}^{[0]} + \mathbf{b}^{[1]}$$

Where $$\mathbf{W}^{[1]} \in \mathbb{R}^{128 \times 784}$$ and $$\mathbf{b}^{[1]} \in \mathbb{R}^{128}$$.

Each of the 128 neurons computes:
- Neuron 1 might activate for vertical edges in the top-left
- Neuron 2 might activate for circular curves  
- Neuron 3 might activate for diagonal strokes
- ... and so on

After ReLU activation: $$\mathbf{a}^{[1]} = \max(0, \mathbf{z}^{[1]}) \in \mathbb{R}^{128}$$

Some neurons fire strongly (values close to their maximum), others don't fire at all (zeroed out by ReLU). The network has transformed the raw pixel representation into a feature representation: "this image has strong vertical edges, moderate curves, weak horizontal strokes."

**Step 3: Second Hidden Layer (Layer 2)**

This layer combines the low-level features from Layer 1 into higher-level concepts:

$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$

Where $$\mathbf{W}^{[2]} \in \mathbb{R}^{64 \times 128}$$.

These 64 neurons might recognize:
- Neuron 1: "top loop" (combining curves and top-positioned edges)
- Neuron 2: "bottom loop" (different curve combinations)
- Neuron 3: "vertical stroke" (combining vertical edges)

After ReLU: $$\mathbf{a}^{[2]} = \max(0, \mathbf{z}^{[2]}) \in \mathbb{R}^{64}$$

The representation is now even more abstract: "this image has a top loop and a bottom loop, characteristic of digits like 3, 8, or possibly 0."

**Step 4: Output Layer (Layer 3)**

The final layer makes the classification decision:

$$\mathbf{z}^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]} \in \mathbb{R}^{10}$$

where $$\mathbf{W}^{[3]} \in \mathbb{R}^{10 \times 64}$$.

This gives raw scores (logits) for each digit. To convert to probabilities:

$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{z}^{[3]})$$

Resulting in something like:
$$\hat{\mathbf{y}} = [0.01, 0.02, 0.05, 0.82, 0.03, 0.01, 0.02, 0.02, 0.01, 0.01]$$

The network is 82% confident this is a "3" (index 3), with some probability mass on other digits that share similar features.

### Why This Layered Structure Works

**Hierarchical Feature Learning**: Layer 1 learned edges and curves. Layer 2 combined these into digit parts. Layer 3 combined parts into complete digit predictions. This hierarchy emerges automatically from training—we never explicitly told the network to detect edges first!

**Distributed Representation**: The digit "3" isn't represented by a single neuron but by a pattern of activation across all 64 neurons in Layer 2. This makes the representation robust (losing a few neurons doesn't destroy the concept) and efficient (the same features help recognize multiple digits).

**Dimensionality Reduction**: We started with 784 dimensions and compressed through 128 → 64 → 10. Each reduction forced the network to extract increasingly essential information, discarding noise and irrelevant details while preserving discriminative features.

### Intuition: Why Depth Beats Width

Consider two alternative networks for the same task:

**Shallow-Wide**: [784 → 4096 → 10]  
- Single massive hidden layer with 4096 neurons
- Total parameters: ~3.2M
- Each neuron must learn complete pattern from raw pixels
- No explicit feature hierarchy

**Deep-Narrow**: [784 → 128 → 64 → 32 → 10]
- Four smaller hidden layers
- Total parameters: ~110K (29× fewer!)
- Natural feature hierarchy emerges
- Better generalization, easier to train

The deep network wins because most real-world patterns are compositional. Faces are composed of eyes, noses, mouths (not random pixel patterns). Sentences are composed of phrases, composed of words, composed of letters. Deep networks naturally capture this compositional structure through their layered architecture.

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

