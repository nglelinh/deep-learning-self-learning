---
layout: post
title: 03-02 Gradient Descent
chapter: '03'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter03
lesson_type: required
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

## Implementation

### Basic Gradient Descent

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Basic gradient descent for linear regression
    
    X: (m, n) input matrix
    y: (m, 1) target vector
    """
    m, n = X.shape
    theta = np.zeros((n, 1))  # Initialize parameters
    cost_history = []
    
    for i in range(num_iterations):
        # Forward pass: predictions
        y_pred = X @ theta
        
        # Compute cost
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)
        
        # Compute gradient
        gradient = (1 / m) * X.T @ (y_pred - y)
        
        # Update parameters
        theta = theta - learning_rate * gradient
        
        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return theta, cost_history

# Example usage
X = np.random.randn(100, 3)  # 100 examples, 3 features
y = np.random.randn(100, 1)  # 100 targets

theta_optimal, costs = gradient_descent(X, y, learning_rate=0.1, num_iterations=1000)
```

### Mini-Batch Gradient Descent

```python
def mini_batch_gradient_descent(X, y, learning_rate=0.01, batch_size=32, num_epochs=10):
    """
    Mini-batch gradient descent
    
    X: (m, n) input matrix
    y: (m, 1) target vector
    batch_size: size of each mini-batch
    num_epochs: number of complete passes through the dataset
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    
    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            # Get batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            y_pred = X_batch @ theta
            
            # Compute gradient on batch
            batch_size_actual = X_batch.shape[0]
            gradient = (1 / batch_size_actual) * X_batch.T @ (y_pred - y_batch)
            
            # Update parameters
            theta = theta - learning_rate * gradient
        
        # Compute cost on full dataset (for monitoring)
        y_pred_full = X @ theta
        cost = (1 / (2 * m)) * np.sum((y_pred_full - y) ** 2)
        cost_history.append(cost)
        
        print(f"Epoch {epoch + 1}/{num_epochs}: Cost = {cost:.4f}")
    
    return theta, cost_history

# Example usage
theta_optimal, costs = mini_batch_gradient_descent(
    X, y, 
    learning_rate=0.1, 
    batch_size=32, 
    num_epochs=50
)
```

### With Learning Rate Schedule

```python
class LearningRateSchedule:
    def __init__(self, initial_lr, schedule_type='step', **kwargs):
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.kwargs = kwargs
    
    def get_lr(self, iteration):
        if self.schedule_type == 'step':
            decay_rate = self.kwargs.get('decay_rate', 0.5)
            decay_steps = self.kwargs.get('decay_steps', 1000)
            return self.initial_lr * (decay_rate ** (iteration // decay_steps))
        
        elif self.schedule_type == 'exponential':
            decay_rate = self.kwargs.get('decay_rate', 0.95)
            return self.initial_lr * np.exp(-decay_rate * iteration)
        
        elif self.schedule_type == 'inverse':
            decay_rate = self.kwargs.get('decay_rate', 0.01)
            return self.initial_lr / (1 + decay_rate * iteration)
        
        else:
            return self.initial_lr

def gradient_descent_with_schedule(X, y, initial_lr=0.01, num_iterations=1000, schedule_type='step'):
    """Gradient descent with learning rate schedule"""
    m, n = X.shape
    theta = np.zeros((n, 1))
    
    lr_schedule = LearningRateSchedule(initial_lr, schedule_type, decay_rate=0.5, decay_steps=200)
    
    for i in range(num_iterations):
        # Get current learning rate
        lr = lr_schedule.get_lr(i)
        
        # Forward and gradient computation
        y_pred = X @ theta
        gradient = (1 / m) * X.T @ (y_pred - y)
        
        # Update with current learning rate
        theta = theta - lr * gradient
        
        if i % 100 == 0:
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            print(f"Iteration {i}: LR = {lr:.6f}, Cost = {cost:.4f}")
    
    return theta

# Example usage
theta = gradient_descent_with_schedule(X, y, initial_lr=0.1, num_iterations=1000)
```

## Convergence Criteria

How do we know when to stop training?

### 1. Maximum Iterations

Stop after a fixed number of iterations/epochs.

```python
if iteration >= max_iterations:
    break
```

### 2. Cost Threshold

Stop when cost drops below a threshold.

```python
if cost < threshold:
    break
```

### 3. Gradient Magnitude

Stop when gradient is very small (near stationary point).

```python
if np.linalg.norm(gradient) < epsilon:
    break
```

### 4. Cost Change

Stop when cost stops decreasing significantly.

```python
if abs(cost - previous_cost) < epsilon:
    break
```

### 5. Validation Loss (Most Common in Deep Learning)

Stop when validation loss stops improving (early stopping).

```python
if validation_loss > best_validation_loss:
    patience_counter += 1
    if patience_counter >= patience:
        break
else:
    best_validation_loss = validation_loss
    patience_counter = 0
```

## Challenges with Gradient Descent

### 1. Local Minima

Non-convex functions (like neural networks) have multiple local minima.

**Solutions:**
- Random initialization (try different starting points)
- Momentum (covered in advanced optimizers)
- Simulated annealing

### 2. Saddle Points

Points where gradient is zero but not a minimum.

**Solutions:**
- Momentum and adaptive learning rates help escape
- Second-order methods (Newton's method)

### 3. Plateaus

Flat regions where gradient is very small.

**Solutions:**
- Patience (wait longer)
- Learning rate schedules
- Adaptive optimizers (Adam, RMSprop)

### 4. Vanishing/Exploding Gradients

Gradients become too small or too large in deep networks.

**Solutions:**
- Proper initialization (Xavier, He)
- Batch normalization
- Residual connections
- Gradient clipping (for exploding gradients)

## Summary

- **Gradient descent** is the fundamental deep-learning algorithm for neural networks
- **Update rule**: $$\theta := \theta - \eta \nabla_{\theta} J(\theta)$$
- **Mini-batch gradient descent** is the most commonly used variant
- **Learning rate** $$\eta$$ is crucial: too small → slow, too large → unstable
- **Learning rate schedules** can improve convergence
- **Convergence criteria** help determine when to stop training
- **Challenges** include local minima, saddle points, and gradient issues

In the next lesson, we'll explore **backpropagation**, the algorithm that efficiently computes gradients in neural networks.

