---
layout: post
title: 03-02-02 Gradient Descent Implementation and Challenges
chapter: '03'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

> *Writing a 10-line gradient descent loop in NumPy is easy; keeping it from oscillating in narrow ravines, stalling on high-dimensional saddle points, or diverging from unscaled features is where real optimization engineering begins.*

## Key ideas: Practical Obstacles in Gradient Descent

1. **Ill-Conditioned Ravines (Anisotropic Curvature)**: When one parameter axis is $$100\times$$ steeper than another, vanilla GD bounces violently across the narrow canyon while barely moving along the valley floor.
2. **Saddle Points vs. Local Minima**: In high-dimensional parameter spaces ($$d \gg 10^6$$), nearly all critical points with $$\nabla L = \mathbf{0}$$ are saddle points (some positive, some negative Hessian eigenvalues), where deterministic GD can stall indefinitely.
3. **Feature Scaling & Condition Number**: Feeding unnormalized features (e.g., age $$[0, 100]$$ next to income $$[0, 10^6]$$) stretches loss contours by $$10^4\times$$, crippling vanilla GD until features are standardized.
4. **Vectorized Implementation & Numerical Checks**: Replacing Python loops with matrix ops and verifying analytic gradients against finite-difference numerical gradients eliminates silent bugs.

### Worked example: Watching Vanilla GD Zig-Zag in an Ill-Conditioned Valley

Minimize $$L(w_1, w_2) = 0.5 w_1^2 + 5 w_2^2$$ (Hessian eigenvalues $$\lambda_1 = 1, \lambda_2 = 10$$, condition number $$\kappa = 10$$), starting at $$(w_1, w_2) = (10, 1)$$ with learning rate $$\eta = 0.18$$ (just under the stability limit $$2/\lambda_2 = 0.20$$).

1. **Compute Gradients**:
   $$\frac{\partial L}{\partial w_1} = w_1, \qquad \frac{\partial L}{\partial w_2} = 10 w_2$$
2. **Trace Step 1 ($$t = 0 \to 1$$)**:
   - $$w_1^{(1)} = 10 - 0.18(10) = 8.20$$ (smooth progress toward $$0$$).
   - $$w_2^{(1)} = 1 - 0.18(10 \times 1) = 1 - 1.8 = -0.80$$ (overshot zero to the opposite wall!).
3. **Trace Step 2 ($$t = 1 \to 2$$)**:
   - $$w_1^{(2)} = 8.20 - 0.18(8.20) = 6.724$$.
   - $$w_2^{(2)} = -0.80 - 0.18(-8.0) = -0.80 + 1.44 = +0.64$$ (bounced back across zero!).
4. **Diagnosis**:
   Even though $$\eta = 0.18$$ is barely safe for $$w_2$$ (decay factor $$-0.8$$ per step), $$w_1$$ only shrinks by $$0.82$$ per step. Increasing $$\eta$$ to speed up $$w_1$$ immediately blows up $$w_2$$! Rescaling features or adding momentum resolves this exact bottleneck.

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


## Video tham khảo

- **xem trước** [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, English. A convergence check you can code: stop when the step gets small.
- [Stochastic Gradient Descent, Clearly Explained!!!](https://www.youtube.com/watch?v=vMh0zPT0tLI) — StatQuest with Josh Starmer, English. Noise in the stochastic gradient is the main challenge the implementation section has to live with.
- [PyTorch Tutorial 06 - Training Pipeline: Model, Loss, and Optimizer](https://www.youtube.com/watch?v=VVDHU_TWwUg) — Patrick Loeber, English. The training loop that actually applies the update, which is the code half of this lesson.
