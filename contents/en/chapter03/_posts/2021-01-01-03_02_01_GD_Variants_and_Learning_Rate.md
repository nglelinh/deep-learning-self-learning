---
layout: post
title: 03-02-01 Gradient Descent Variants and Learning Rate
chapter: '03'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

This lesson introduces gradient descent, the fundamental deep-learning algorithm used to train neural networks.

---

## The Deep Learning Problem

Training a neural network is an **deep-learning problem**: Find parameters $$\theta = \{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \ldots, \mathbf{W}^{[L]}, \mathbf{b}^{[L]}\}$$ that minimize the cost function:

$$\theta^* = \arg\min_{\theta} J(\theta)$$

where $$J(\theta)$$ is the average loss over all training examples.

## Gradient Descent: The Core Idea

![Gradient Descent Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/600px-Gradient_descent.svg.png)
*Hình ảnh: Minh họa Gradient Descent trên hàm mất mát. Nguồn: Wikimedia Commons*

**Gradient descent** is an iterative deep-learning algorithm that moves parameters in the direction that decreases the cost function most rapidly.

### The Gradient

The **gradient** $$\nabla_{\theta} J(\theta)$$ is a vector of partial derivatives:

$$\nabla_{\theta} J = \begin{bmatrix} \frac{\partial J}{\partial \theta_1} \\ \frac{\partial J}{\partial \theta_2} \\ \vdots \\ \frac{\partial J}{\partial \theta_n} \end{bmatrix}$$

**Key property**: The gradient points in the direction of **steepest ascent**. Therefore, the negative gradient points in the direction of **steepest descent**.

### Update Rule

The gradient descent update rule is:

$$\theta := \theta - \eta \nabla_{\theta} J(\theta)$$

where:
- $$\eta$$ is the **learning rate** (a positive scalar hyperparameter)
- $$:=$$ denotes assignment/update
- $$\nabla_{\theta} J(\theta)$$ is the gradient of the cost function

**For each parameter in a neural network:**

$$\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \eta \frac{\partial J}{\partial \mathbf{W}^{[l]}}$$

$$\mathbf{b}^{[l]} := \mathbf{b}^{[l]} - \eta \frac{\partial J}{\partial \mathbf{b}^{[l]}}$$

### Geometric Intuition

Imagine you're on a mountainside and want to reach the valley (minimum):
1. Check the slope around you (compute gradient)
2. Take a step downhill (in the direction of negative gradient)
3. Repeat until you reach the bottom (convergence)

The learning rate $$\eta$$ determines the **step size**.

## Variants of Gradient Descent

### 1. Batch Gradient Descent (Vanilla GD)

Uses **all training examples** to compute the gradient at each step.

**Algorithm:**
```
Repeat until convergence:
    1. Compute gradient using all m examples:
       ∇J(θ) = (1/m) Σᵢ₌₁ᵐ ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
    
    2. Update parameters:
       θ := θ - η ∇J(θ)
```

**Advantages:**
- Guaranteed to converge to global minimum (for convex functions)
- Stable convergence
- Can use theoretical convergence guarantees

**Disadvantages:**
- **Very slow** for large datasets (must process all data before one update)
- Requires entire dataset in memory
- Can get stuck in local minima (for non-convex functions)

### 2. Stochastic Gradient Descent (SGD)

Uses **one random training example** at a time to compute the gradient.

**Algorithm:**
```
Repeat until convergence:
    1. Randomly shuffle training data
    
    2. For each example i:
        a. Compute gradient using only example i:
           ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
        
        b. Update parameters:
           θ := θ - η ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
```

**Advantages:**
- **Much faster** updates (can start learning immediately)
- Can escape local minima due to noisy updates
- Online learning possible (process data streams)
- Memory efficient

**Disadvantages:**
- **Noisy gradient estimates** → erratic convergence path
- Never truly "converges" (oscillates around minimum)
- Harder to parallelize

### 3. Mini-Batch Gradient Descent (Most Common)

Uses a **small batch of examples** (typically 32, 64, 128, or 256) to compute the gradient.

**Algorithm:**
```
Repeat until convergence:
    1. Randomly shuffle training data
    
    2. Divide data into mini-batches of size B
    
    3. For each mini-batch:
        a. Compute gradient using the batch:
           ∇J_batch(θ) = (1/B) Σᵢ∈batch ∇L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
        
        b. Update parameters:
           θ := θ - η ∇J_batch(θ)
```

**Advantages:**
- **Best of both worlds**: Fast updates + stable convergence
- **Highly parallelizable**: Can utilize GPU/TPU efficiently
- Reduced variance in gradient estimates
- Memory efficient (process batches, not entire dataset)

**Disadvantages:**
- Introduces batch size as a hyperparameter
- Still has some noise (less than SGD)

### Comparison Table

| Variant | Examples per Update | Speed | Stability | Memory | Parallelization |
|---------|-------------------|-------|-----------|---------|-----------------|
| Batch GD | All (m) | Slow | High | High | Difficult |
| SGD | 1 | Fast | Low | Low | Difficult |
| Mini-batch GD | Batch size (B) | **Fast** | **Medium** | **Low** | **Easy** |

**Recommendation**: Use **mini-batch gradient descent** with batch size 32-256.

## The Learning Rate

The learning rate $$\eta$$ is one of the most important hyperparameters.

### Effect of Learning Rate

#### Too Small ($$\eta$$ too low)
- Very slow convergence
- May take too long to train
- Can get stuck in plateaus

#### Too Large ($$\eta$$ too high)
- Unstable training
- May overshoot minimum
- Loss may diverge (increase)

#### Just Right
- Smooth, steady decrease in loss
- Reasonable training time
- Converges to good solution

### Typical Values

- **Good starting points**: 0.001, 0.01, 0.1
- **Deep networks**: Often 0.001 - 0.01
- **Shallow networks**: Can use higher rates (0.01 - 0.1)

### Learning Rate Schedules

Instead of a fixed learning rate, use a **schedule** that changes $$\eta$$ during training:

#### 1. Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

where:
- $$\eta_0$$ is initial learning rate
- $$\gamma \in (0, 1)$$ is decay factor (e.g., 0.5)
- $$k$$ is step interval (e.g., every 10 epochs)

**Example**: Start at 0.1, multiply by 0.5 every 10 epochs

#### 2. Exponential Decay

$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

where $$\lambda$$ is decay constant.

#### 3. 1/t Decay

$$\eta_t = \frac{\eta_0}{1 + \lambda t}$$

#### 4. Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

where $$T$$ is the total number of iterations.

**Warm restarts**: Periodically reset learning rate to initial value.

#### 5. Learning Rate Warm-up

Start with very small learning rate and gradually increase to target value:

$$\eta_t = \eta_0 \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)$$

**Use case**: Large batch training, transformers

