---
layout: post
title: 02-03 Activation Functions in Detail
chapter: '02'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter02
lesson_type: required
---

This lesson provides an in-depth exploration of activation functions, their properties, and how to choose the right one for your neural network.

---

## Why Activation Functions Matter

**Without activation functions**, neural networks would be limited to learning only linear transformations. No matter how many layers you stack, a composition of linear functions is still linear:

$$f(g(h(\mathbf{x}))) = \mathbf{A}_1(\mathbf{A}_2(\mathbf{A}_3 \mathbf{x})) = (\mathbf{A}_1 \mathbf{A}_2 \mathbf{A}_3)\mathbf{x} = \mathbf{A}\mathbf{x}$$

**Activation functions introduce nonlinearity**, enabling networks to learn complex patterns and approximate arbitrary functions.

## Desirable Properties of Activation Functions

An ideal activation function should have:

1. **Nonlinearity**: Enable learning of complex patterns
2. **Differentiability**: Enable gradient-based deep-learning
3. **Monotonicity**: Preserve ordering (helpful for deep-learning)
4. **Computational efficiency**: Fast to compute forward and backward
5. **Bounded or unbounded appropriately**: Depending on the task
6. **Zero-centered**: Help with gradient flow (for hidden layers)
7. **Avoid saturation**: Prevent vanishing gradients

No single activation function satisfies all properties perfectly, so the choice depends on the specific use case.

## Common Activation Functions

### 1. Sigmoid (Logistic Function)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Derivative:**

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Properties:**
- Range: $$(0, 1)$$
- Smooth, differentiable everywhere
- Monotonically increasing
- Saturates at both ends

**Advantages:**
- Clear probabilistic interpretation
- Smooth gradient
- Historically popular

**Disadvantages:**
- **Vanishing gradient problem**: Gradients near 0 for $$|z| > 4$$
- **Not zero-centered**: Outputs always positive
- **Computationally expensive**: Requires exponential calculation

**Use cases:**
- Output layer for binary classification
- Gate activations in LSTMs
- Generally avoided in hidden layers of deep networks

### 2. Hyperbolic Tangent (tanh)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1} = 2\sigma(2z) - 1$$

**Derivative:**

$$\tanh'(z) = 1 - \tanh^2(z)$$

**Properties:**
- Range: $$(-1, 1)$$
- Zero-centered (improvement over sigmoid)
- Saturates at both ends

**Advantages:**
- Zero-centered (better gradient flow)
- Stronger gradients than sigmoid (derivative range: $$(0, 1]$$)

**Disadvantages:**
- Still suffers from vanishing gradients
- Computationally expensive

**Use cases:**
- Hidden layers (better than sigmoid but worse than ReLU)
- RNN/LSTM cells
- When zero-centered outputs are beneficial

### 3. Rectified Linear Unit (ReLU)

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Derivative:**

$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \\ \text{undefined} & \text{if } z = 0 \end{cases}$$

(In practice, we define $$\text{ReLU}'(0) = 0$$ or $$0.5$$)

**Properties:**
- Range: $$[0, \infty)$$
- Not saturating for positive values
- Sparse activation (many neurons output 0)

**Advantages:**
- **Computational efficiency**: Just thresholding at zero
- **Alleviates vanishing gradient**: Gradient is 1 for positive inputs
- **Sparse representations**: Natural sparsity
- **Empirically successful**: Works very well in practice

**Disadvantages:**
- **Not zero-centered**: All outputs are non-negative
- **Dying ReLU problem**: Neurons can become inactive forever
  - If $$z < 0$$ always, gradient is always 0, no learning occurs
  - Can happen with high learning rates or poor initialization

**Use cases:**
- **Default choice** for hidden layers in deep networks
- CNNs, ResNets, most modern architectures

### 4. Leaky ReLU

$$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

where $$\alpha$$ is a small constant (typically 0.01)

**Derivative:**

$$\text{LeakyReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z \leq 0 \end{cases}$$

**Advantages:**
- **Prevents dying ReLU**: Small gradient even for negative inputs
- Computationally efficient
- All benefits of ReLU

**Disadvantages:**
- Introduces hyperparameter $$\alpha$$
- Not always better than ReLU in practice

**Variants:**
- **Parametric ReLU (PReLU)**: $$\alpha$$ is learned during training
- **Randomized Leaky ReLU (RReLU)**: $$\alpha$$ is randomly sampled during training

### 5. Exponential Linear Unit (ELU)

$$\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

where $$\alpha > 0$$ (typically $$\alpha = 1$$)

**Derivative:**

$$\text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha e^z = \text{ELU}(z) + \alpha & \text{if } z \leq 0 \end{cases}$$

**Properties:**
- Smooth everywhere
- Negative values push mean activation closer to zero

**Advantages:**
- **Closer to zero-centered**: Negative outputs possible
- **No dying ReLU problem**: Gradients exist for all inputs
- **Smooth**: Better deep-learning landscape
- Often leads to faster learning and better performance

**Disadvantages:**
- Computationally more expensive (exponential)
- Introduces hyperparameter $$\alpha$$

**Use cases:**
- Alternative to ReLU when extra computation is acceptable
- Tasks where zero-centered activations help

### 6. Scaled Exponential Linear Unit (SELU)

$$\text{SELU}(z) = \lambda \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$

where $$\lambda \approx 1.0507$$ and $$\alpha \approx 1.6733$$

**Properties:**
- **Self-normalizing**: Under certain conditions, activations automatically converge to zero mean and unit variance
- Requires specific initialization (LeCun normal)
- Requires specific architecture (fully connected layers)

**Advantages:**
- Can enable very deep networks without batch normalization
- Theoretical guarantees about convergence

**Disadvantages:**
- Strict requirements on network architecture
- Not widely adopted
- Doesn't work well with dropout or convolutional layers

### 7. Swish (SiLU - Sigmoid Linear Unit)

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

**Properties:**
- Smooth, non-monotonic
- Unbounded above, bounded below
- Self-gated (input modulated by sigmoid of itself)

**Advantages:**
- **Better performance**: Empirically shown to outperform ReLU in some tasks
- Smooth gradients
- Non-monotonicity can be beneficial

**Disadvantages:**
- Computationally more expensive than ReLU
- Requires careful tuning

**Use cases:**
- Modern architectures (EfficientNet uses Swish)
- When computational cost is not critical

### 8. GELU (Gaussian Error Linear Unit)

$$\text{GELU}(z) = z \cdot \Phi(z)$$

where $$\Phi(z)$$ is the cumulative distribution function of the standard normal distribution.

**Approximation:**

$$\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right)\right)$$

**Properties:**
- Smooth, non-monotonic
- Stochastic regularizer interpretation

**Advantages:**
- **State-of-the-art performance**: Used in BERT, GPT models
- Smooth everywhere
- Captures aspects of dropout and zoneout

**Disadvantages:**
- Computationally expensive
- Harder to interpret

**Use cases:**
- **Transformer models**: BERT, GPT-2, GPT-3
- NLP tasks
- Modern large-scale models

### 9. Softmax (Output Layer)

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

**Properties:**
- Converts logits to probability distribution
- Output range: $$(0, 1)$$ with $$\sum_i p_i = 1$$

**Derivative (for class $$i$$ with respect to $$z_j$$):**

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \begin{cases} \text{softmax}(\mathbf{z})_i(1 - \text{softmax}(\mathbf{z})_i) & \text{if } i = j \\ -\text{softmax}(\mathbf{z})_i \cdot \text{softmax}(\mathbf{z})_j & \text{if } i \neq j \end{cases}$$

**Use cases:**
- **Multi-class classification** output layer
- Attention mechanisms
- Any scenario requiring probability distribution over classes

### 10. Softplus

$$\text{softplus}(z) = \ln(1 + e^z)$$

**Properties:**
- Smooth approximation of ReLU
- Always positive
- Asymptotically approaches ReLU for large $$z$$

**Derivative:**

$$\text{softplus}'(z) = \frac{e^z}{1 + e^z} = \sigma(z)$$

**Use cases:**
- Occasionally used in hidden layers
- Generative models (ensuring positive outputs)

## Comparison Summary

| Activation | Range | Zero-Centered | Vanishing Gradient | Dying Units | Computational Cost | Common Use |
|------------|-------|---------------|-------------------|-------------|-------------------|------------|
| Sigmoid | (0,1) | No | Yes | No | High | Output (binary) |
| tanh | (-1,1) | Yes | Yes | No | High | Hidden (old), RNN |
| ReLU | [0,∞) | No | No (for z>0) | Yes | **Low** | **Hidden (default)** |
| Leaky ReLU | (-∞,∞) | No | No | No | **Low** | Hidden |
| ELU | (-α,∞) | ~Yes | No | No | Medium | Hidden |
| SELU | (-λα,∞) | Yes (self-norm) | No | No | Medium | Hidden (specific) |
| Swish | (-∞,∞) | No | No | No | Medium | Hidden (modern) |
| GELU | (-∞,∞) | No | No | No | High | **Transformers** |
| Softmax | (0,1) | N/A | Yes | No | High | **Output (multi-class)** |

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

