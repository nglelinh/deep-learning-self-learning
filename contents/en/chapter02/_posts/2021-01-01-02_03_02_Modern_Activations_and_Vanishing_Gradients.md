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

> *Open the source code of GPT, Llama, or Gemma and you will not find a single ReLU in the Transformer blocks—instead you will see GELU, SiLU (Swish), and SwiGLU. Modern activations combine ReLU's non-saturating positive ray with smooth curvature near zero to stabilize deep optimization.*

## Key ideas: Modern Activations & Gradient Preservation

1. **GELU & SiLU/Swish (Smooth Stochastic Gating)**: Weight each input smoothly by a probability gate—$$\text{SiLU}(z) = z\sigma(z)$$ and $$\text{GELU}(z) = z\Phi(z)$$—giving smooth derivatives everywhere and slight non-monotonicity for negative $$z$$.
2. **Gated Linear Units (SwiGLU)**: Splits the hidden projection into two pathways, $$\text{SwiGLU}(\mathbf{x}) = \text{SiLU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)$$, providing multiplicative feature interaction that consistently lowers LLM perplexity per FLOP.
3. **Output-Layer Matching Principle**: Pairs the final activation strictly with the task's probability model—linear for regression, sigmoid for binary/multi-label, and numerically stabilized log-softmax for multi-class classification.
4. **Vanishing Gradient Mitigation Stack**: Combines non-saturating activations with variance-preserving initialization (He/Xavier), normalization (LayerNorm/RMSNorm), and residual skip connections.

### Worked example: Computing SiLU (Swish) Value, Derivative, and Numerically Stable Softmax

1. **SiLU Forward & Backward at $$z = 0$$ and $$z = -1$$** ($$\text{SiLU}(z) = z\sigma(z)$$):
   By the product rule:
   $$\text{SiLU}'(z) = \sigma(z) + z\,\sigma(z)(1 - \sigma(z)) = \sigma(z)\big(1 + z(1 - \sigma(z))\big)$$
   - At $$z = 0$$ ($$\sigma(0) = 0.5$$):
     $$\text{SiLU}(0) = 0, \qquad \text{SiLU}'(0) = 0.5(1 + 0) = 0.500$$
   - At $$z = -1$$ ($$\sigma(-1) \approx 0.2689$$):
     $$\text{SiLU}(-1) = -0.2689, \qquad \text{SiLU}'(-1) = 0.2689\big(1 - 1(0.7311)\big) = 0.2689(0.2689) \approx +0.0723$$
   Unlike ReLU (whose derivative is identically $$0$$ at $$z = -1$$), SiLU remains smoothly differentiable with a gentle nonzero gradient.
2. **Why Shift-Invariant Softmax Prevents Overflow**:
   Suppose logits are $$\mathbf{z} = [1000, 1001, 1000]$$. Direct `exp(1000)` overflows `float32` (`inf / inf = NaN`).
   Subtract $$\max(\mathbf{z}) = 1001$$ first: $$\tilde{\mathbf{z}} = [-1, 0, -1]$$.
   $$e^{-1} \approx 0.3679, \quad e^0 = 1.0000 \implies Z = 1.7358$$
   $$\text{softmax}(\mathbf{z}) = \left[\frac{0.3679}{1.7358}, \frac{1.0000}{1.7358}, \frac{0.3679}{1.7358}\right] \approx [0.2119, 0.5761, 0.2119]$$
   Exact mathematical equality, zero floating-point overflow!

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


## Video tham khảo

- **xem trước** [The GELU, SiLU and SwiGLU activation functions, clearly explained!!!](https://www.youtube.com/watch?v=2FaI2Fen1mQ) — StatQuest with Josh Starmer, English. GELU, SiLU, and SwiGLU are the modern functions this lesson adds after ReLU.
- [Vanishing/Exploding Gradients (C2W1L10)](https://www.youtube.com/watch?v=qhXZsFVxGKo) — DeepLearningAI, English. Vanishing and exploding gradients are the failure mode the lesson uses to justify those choices.
- [Neural Networks Pt. 3: ReLU In Action!!!](https://www.youtube.com/watch?v=68BZ5f7P94E) — StatQuest with Josh Starmer, English. ReLU is the baseline those newer functions are compared with.
