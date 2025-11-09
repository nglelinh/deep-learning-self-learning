---
layout: post
title: 02-04 Forward Propagation
chapter: '02'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter02
lesson_type: required
---

This lesson provides a comprehensive understanding of forward propagation, the process by which neural networks make predictions.

---

## What is Forward Propagation?

**Forward propagation** is the process of computing the output of a neural network given an input. Data "flows forward" through the network from the input layer to the output layer, passing through all hidden layers in sequence.

This is the **inference** or **prediction** phase of a neural network.

## The Forward Pass: Step by Step

Consider a simple 3-layer network:
- **Input layer**: $$n^{[0]} = 3$$ features
- **Hidden layer 1**: $$n^{[1]} = 4$$ neurons with ReLU
- **Hidden layer 2**: $$n^{[2]} = 4$$ neurons with ReLU  
- **Output layer**: $$n^{[3]} = 1$$ neuron with sigmoid (binary classification)

### Layer 0: Input

$$\mathbf{a}^{[0]} = \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$$

### Layer 1: First Hidden Layer

**Linear transformation:**

$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{a}^{[0]} + \mathbf{b}^{[1]}$$

Where:
- $$\mathbf{W}^{[1]} \in \mathbb{R}^{4 \times 3}$$ (4 neurons, 3 inputs each)
- $$\mathbf{b}^{[1]} \in \mathbb{R}^{4}$$ (4 biases)
- $$\mathbf{z}^{[1]} \in \mathbb{R}^{4}$$ (pre-activation values)

**Activation:**

$$\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]}) = \max(0, \mathbf{z}^{[1]})$$

$$\mathbf{a}^{[1]} \in \mathbb{R}^{4}$$ (activations/outputs of layer 1)

### Layer 2: Second Hidden Layer

**Linear transformation:**

$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$$

Where:
- $$\mathbf{W}^{[2]} \in \mathbb{R}^{4 \times 4}$$
- $$\mathbf{b}^{[2]} \in \mathbb{R}^{4}$$

**Activation:**

$$\mathbf{a}^{[2]} = \text{ReLU}(\mathbf{z}^{[2]})$$

### Layer 3: Output Layer

**Linear transformation:**

$$\mathbf{z}^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]}$$

Where:
- $$\mathbf{W}^{[3]} \in \mathbb{R}^{1 \times 4}$$
- $$\mathbf{b}^{[3]} \in \mathbb{R}^{1}$$

**Activation (output):**

$$\hat{y} = \mathbf{a}^{[3]} = \sigma(\mathbf{z}^{[3]}) = \frac{1}{1 + e^{-\mathbf{z}^{[3]}}}$$

This gives us the predicted probability of the positive class.

## Vectorized Forward Propagation

For computational efficiency, we process **multiple examples simultaneously** using **vectorization**.

### Batch Processing

Instead of processing one example at a time, we organize $$m$$ examples into a matrix:

$$\mathbf{X} = \begin{bmatrix} | & | & & | \\ \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)} \\ | & | & & | \end{bmatrix} \in \mathbb{R}^{n^{[0]} \times m}$$

Each column is one training example.

### Vectorized Computation

For layer $$l$$:

$$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

Where:
- $$\mathbf{A}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$$ (each column is activations for one example)
- $$\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$$
- $$\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}$$
- $$\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}$$ (broadcasted across all $$m$$ examples)

### Broadcasting

Python/NumPy automatically broadcasts $$\mathbf{b}^{[l]}$$ across all examples:

$$\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]} \times 1} \rightarrow \mathbb{R}^{n^{[l]} \times m}$$

Each column of the result gets the same bias vector added.

## Concrete Example with Numbers

Let's work through a small example with actual numbers.

### Network Setup

- Input: 2 features ($$n^{[0]} = 2$$)
- Hidden layer: 3 neurons with ReLU ($$n^{[1]} = 3$$)
- Output: 1 neuron with sigmoid ($$n^{[2]} = 1$$)
- Batch size: 2 examples ($$m = 2$$)

### Parameters

$$\mathbf{W}^{[1]} = \begin{bmatrix} 0.5 & -0.3 \\ 0.2 & 0.8 \\ -0.4 & 0.6 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.3 \end{bmatrix}$$

$$\mathbf{W}^{[2]} = \begin{bmatrix} 1.0 & -0.5 & 0.7 \end{bmatrix}, \quad \mathbf{b}^{[2]} = \begin{bmatrix} 0.5 \end{bmatrix}$$

### Input Data

$$\mathbf{X} = \mathbf{A}^{[0]} = \begin{bmatrix} 1.0 & 0.5 \\ 2.0 & 1.5 \end{bmatrix}$$

Example 1: $$\mathbf{x}^{(1)} = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$$, Example 2: $$\mathbf{x}^{(2)} = \begin{bmatrix} 0.5 \\ 1.5 \end{bmatrix}$$

### Forward Pass: Layer 1

**Compute $$\mathbf{Z}^{[1]}$$:**

$$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]} \mathbf{A}^{[0]} + \mathbf{b}^{[1]}$$

$$= \begin{bmatrix} 0.5 & -0.3 \\ 0.2 & 0.8 \\ -0.4 & 0.6 \end{bmatrix} \begin{bmatrix} 1.0 & 0.5 \\ 2.0 & 1.5 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.2 \\ 0.3 \end{bmatrix}$$

**Matrix multiplication:**

First column: $$\begin{bmatrix} 0.5(1.0) + (-0.3)(2.0) \\ 0.2(1.0) + 0.8(2.0) \\ -0.4(1.0) + 0.6(2.0) \end{bmatrix} = \begin{bmatrix} -0.1 \\ 1.8 \\ 0.8 \end{bmatrix}$$

Second column: $$\begin{bmatrix} 0.5(0.5) + (-0.3)(1.5) \\ 0.2(0.5) + 0.8(1.5) \\ -0.4(0.5) + 0.6(1.5) \end{bmatrix} = \begin{bmatrix} -0.2 \\ 1.3 \\ 0.7 \end{bmatrix}$$

After adding bias:

$$\mathbf{Z}^{[1]} = \begin{bmatrix} -0.1+0.1 & -0.2+0.1 \\ 1.8-0.2 & 1.3-0.2 \\ 0.8+0.3 & 0.7+0.3 \end{bmatrix} = \begin{bmatrix} 0.0 & -0.1 \\ 1.6 & 1.1 \\ 1.1 & 1.0 \end{bmatrix}$$

**Apply ReLU activation:**

$$\mathbf{A}^{[1]} = \text{ReLU}(\mathbf{Z}^{[1]}) = \begin{bmatrix} 0.0 & 0.0 \\ 1.6 & 1.1 \\ 1.1 & 1.0 \end{bmatrix}$$

### Forward Pass: Layer 2 (Output)

**Compute $$\mathbf{Z}^{[2]}$$:**

$$\mathbf{Z}^{[2]} = \mathbf{W}^{[2]} \mathbf{A}^{[1]} + \mathbf{b}^{[2]}$$

$$= \begin{bmatrix} 1.0 & -0.5 & 0.7 \end{bmatrix} \begin{bmatrix} 0.0 & 0.0 \\ 1.6 & 1.1 \\ 1.1 & 1.0 \end{bmatrix} + \begin{bmatrix} 0.5 \end{bmatrix}$$

$$= \begin{bmatrix} 1.0(0.0) + (-0.5)(1.6) + 0.7(1.1) + 0.5 & 1.0(0.0) + (-0.5)(1.1) + 0.7(1.0) + 0.5 \end{bmatrix}$$

$$= \begin{bmatrix} 0.47 & 0.65 \end{bmatrix}$$

**Apply sigmoid activation:**

$$\mathbf{A}^{[2]} = \sigma(\mathbf{Z}^{[2]}) = \begin{bmatrix} \frac{1}{1+e^{-0.47}} & \frac{1}{1+e^{-0.65}} \end{bmatrix} \approx \begin{bmatrix} 0.615 & 0.657 \end{bmatrix}$$

### Final Predictions

- Example 1: $$\hat{y}^{(1)} = 0.615$$ (61.5% probability of positive class)
- Example 2: $$\hat{y}^{(2)} = 0.657$$ (65.7% probability of positive class)

## Implementation in Python

### Basic Implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def forward_propagation(X, parameters):
    """
    Arguments:
    X -- input data of shape (n_x, m)
    parameters -- python dictionary containing W1, b1, W2, b2, W3, b3, ...
    
    Returns:
    AL -- last post-activation value (predictions)
    caches -- list of caches containing (A_prev, W, b, Z) for each layer
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers
    
    # Forward through hidden layers (ReLU activation)
    for l in range(1, L):
        A_prev = A
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        
        Z = np.dot(W, A_prev) + b
        A = relu(Z)
        
        cache = (A_prev, W, b, Z)
        caches.append(cache)
    
    # Output layer (sigmoid activation)
    A_prev = A
    W = parameters[f'W{L}']
    b = parameters[f'b{L}']
    
    Z = np.dot(W, A_prev) + b
    AL = sigmoid(Z)
    
    cache = (A_prev, W, b, Z)
    caches.append(cache)
    
    return AL, caches

# Example usage
X = np.array([[1.0, 0.5],
              [2.0, 1.5]])

parameters = {
    'W1': np.array([[0.5, -0.3],
                    [0.2, 0.8],
                    [-0.4, 0.6]]),
    'b1': np.array([[0.1], [-0.2], [0.3]]),
    'W2': np.array([[1.0, -0.5, 0.7]]),
    'b2': np.array([[0.5]])
}

predictions, caches = forward_propagation(X, parameters)
print("Predictions:", predictions)
# Output: Predictions: [[0.615 0.657]]
```

### Object-Oriented Implementation

```python
class NeuralNetwork:
    def __init__(self, layer_dims):
        """
        Arguments:
        layer_dims -- list containing dimensions of each layer
                     Example: [2, 3, 1] means 2 inputs, 3 hidden, 1 output
        """
        self.parameters = self.initialize_parameters(layer_dims)
        self.L = len(layer_dims) - 1
    
    def initialize_parameters(self, layer_dims):
        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)
        
        for l in range(1, L):
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
        return parameters
    
    def forward(self, X):
        """Forward propagation"""
        A = X
        caches = []
        
        # Hidden layers with ReLU
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(W, A_prev) + b
            A = np.maximum(0, Z)  # ReLU
            
            caches.append((A_prev, W, b, Z, A))
        
        # Output layer with sigmoid
        A_prev = A
        W = self.parameters[f'W{self.L}']
        b = self.parameters[f'b{self.L}']
        
        Z = np.dot(W, A_prev) + b
        A = 1 / (1 + np.exp(-Z))  # Sigmoid
        
        caches.append((A_prev, W, b, Z, A))
        
        return A, caches
    
    def predict(self, X):
        """Make predictions (0 or 1)"""
        A, _ = self.forward(X)
        return (A > 0.5).astype(int)

# Usage
nn = NeuralNetwork([2, 3, 1])
X = np.array([[1.0, 0.5], [2.0, 1.5]])
predictions, caches = nn.forward(X)
print("Predictions:", predictions)
```

## Common Issues and Debugging

### 1. Dimension Mismatch

**Problem**: Matrix multiplication fails due to incompatible dimensions.

**Solution**: 
- Check that $$\mathbf{W}^{[l]}$$ has shape $$(n^{[l]}, n^{[l-1]})$$
- Check that $$\mathbf{A}^{[l-1]}$$ has shape $$(n^{[l-1]}, m)$$
- Use print statements or debugger to verify shapes

```python
print(f"Layer {l}:")
print(f"  W shape: {W.shape}")
print(f"  A_prev shape: {A_prev.shape}")
print(f"  b shape: {b.shape}")
print(f"  Z shape: {Z.shape}")
```

### 2. Numerical Instability

**Problem**: Overflow or underflow in exponentials (especially sigmoid/softmax).

**Solution**: Use numerical stability tricks:

```python
# Unstable
def sigmoid_unstable(z):
    return 1 / (1 + np.exp(-z))

# Stable version
def sigmoid_stable(z):
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
```

### 3. Incorrect Broadcasting

**Problem**: Bias not broadcast correctly.

**Solution**: Ensure bias has shape $$(n^{[l]}, 1)$$ not $$(n^{[l]},)$$

```python
# Correct
b = np.zeros((n_l, 1))  # Shape (n_l, 1)

# Incorrect (may cause issues)
b = np.zeros(n_l)  # Shape (n_l,)
```

## Forward Propagation Complexity

### Time Complexity

For a network with $$L$$ layers and $$n$$ neurons per layer:

$$O(L \cdot n^2 \cdot m)$$

where $$m$$ is the batch size.

**Breakdown:**
- Each layer: $$O(n^2 \cdot m)$$ for matrix multiplication $$\mathbf{W}^{[l]} \mathbf{A}^{[l-1]}$$
- $$L$$ layers total

### Space Complexity

$$O(L \cdot n \cdot m)$$

Need to store activations for each layer (needed for backpropagation).

## Summary

- **Forward propagation** computes predictions by passing inputs through the network
- Each layer performs: **linear transformation** → **activation function**
- **Vectorization** allows efficient batch processing of multiple examples
- The process is: $$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}, \quad \mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$
- **Caching** intermediate values is essential for efficient backpropagation
- Proper handling of **dimensions** and **numerical stability** is crucial
- Forward propagation is **computationally efficient** ($$O(L \cdot n^2 \cdot m)$$)

Now that we understand how networks make predictions, we need to learn how to train them. In the next chapter, we'll cover **backpropagation** and **gradient descent**, the algorithms that enable neural networks to learn from data.

