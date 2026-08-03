---
layout: post
title: 02-03-02 Modern Activations and Vanishing Gradients
chapter: '02'
order: 8
owner: Deep Learning Course
lang: en
categories:
- chapter02
---

## Choosing the Right Activation Function

### For Hidden Layers

**Default recommendation: ReLU**
- Start with ReLU for most applications
- Computationally efficient
- Works well in practice

**If dying ReLU is a problem:**
- Try Leaky ReLU or ELU
- Check learning rate and initialization

**For modern/large-scale models:**
- GELU for transformers and NLP
- Swish for image models when performance is critical

**For very deep networks:**
- Consider ELU or SELU
- May need normalization techniques (covered later)

### For Output Layers

**Binary classification:**
- **Sigmoid**: Outputs probability for positive class

**Multi-class classification:**
- **Softmax**: Outputs probability distribution over classes

**Regression:**
- **Linear (identity)**: For unbounded outputs
- **ReLU**: For non-negative outputs (e.g., prices, counts)
- **Sigmoid/tanh**: For bounded outputs

**Multi-label classification:**
- **Sigmoid**: Independent probability for each label

## Practical Implementation

```python
import numpy as np

class Activations:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(z):
        s = Activations.sigmoid(z)
        return s * (1 - s)
    
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z)**2
    
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)
    
    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        return np.where(z > 0, 1, alpha)
    
    @staticmethod
    def elu(z, alpha=1.0):
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))
    
    @staticmethod
    def elu_derivative(z, alpha=1.0):
        return np.where(z > 0, 1, Activations.elu(z, alpha) + alpha)
    
    @staticmethod
    def softmax(z):
        # Numerical stability: subtract max
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    @staticmethod
    def swish(z):
        return z * Activations.sigmoid(z)
    
    @staticmethod
    def gelu(z):
        # Approximation
        return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

# Example usage
z = np.array([-2, -1, 0, 1, 2])
print("ReLU:", Activations.relu(z))
print("Leaky ReLU:", Activations.leaky_relu(z))
print("Sigmoid:", Activations.sigmoid(z))
print("Tanh:", Activations.tanh(z))
```

## The Vanishing Gradient Problem

### Why It Matters

During backpropagation, gradients are multiplied through layers:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \cdot \frac{\partial \mathbf{a}^{[L]}}{\partial \mathbf{z}^{[L]}} \cdot \ldots \cdot \frac{\partial \mathbf{z}^{[2]}}{\partial \mathbf{a}^{[1]}} \cdot \frac{\partial \mathbf{a}^{[1]}}{\partial \mathbf{z}^{[1]}} \cdot \frac{\partial \mathbf{z}^{[1]}}{\partial \mathbf{W}^{[1]}}$$

### Problem with Sigmoid/Tanh

- Maximum derivative: $$\sigma'(z) = 0.25$$ (sigmoid), $$\tanh'(z) = 1$$ (tanh at $$z=0$$)
- Typical derivative: Much smaller ($$< 0.25$$ for sigmoid)
- After many layers: $$0.25^{10} \approx 9.5 \times 10^{-7}$$ (extremely small!)

**Result**: Gradients vanish, early layers learn very slowly or not at all.

### ReLU to the Rescue

- Derivative is 1 for positive inputs (no vanishing)
- Gradient flows unchanged through active ReLU units
- Enables training of much deeper networks

### The Dying ReLU Problem

- If $$z < 0$$ always, gradient is 0, no learning
- Can happen with:
  - Poor initialization
  - High learning rates
  - Unlucky updates

**Solutions:**
- Use Leaky ReLU, ELU, or other variants
- Proper initialization (He initialization for ReLU)
- Reasonable learning rates
- Batch normalization (covered later)

## Summary

- **Activation functions** introduce nonlinearity, enabling networks to learn complex patterns
- **ReLU** is the default choice for hidden layers in modern deep learning
- **Sigmoid** is used for binary classification outputs
- **Softmax** is used for multi-class classification outputs
- **Advanced activations** (ELU, Swish, GELU) can provide performance improvements
- **Vanishing gradients** are a major issue with sigmoid/tanh in deep networks
- **ReLU** alleviates vanishing gradients but introduces the dying ReLU problem
- Choice of activation function significantly impacts training and performance

In the next lesson, we'll explore forward propagation in detail with concrete examples.

