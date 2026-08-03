---
layout: post
title: 10-01-02-01 Optimizer Core Implementation
chapter: '10'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter10
---

## 4. Code Snippet

Let's implement optimizers from scratch to understand their mechanics:

```python
import numpy as np
import matplotlib.pyplot as plt

class SGDMomentum:
    """
    Stochastic Gradient Descent with Momentum.
    
    Maintains exponentially weighted average of gradients (velocity)
    and uses this for updates instead of raw gradients. Accelerates
    in consistent directions, dampens oscillations.
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        """
        params: list of parameter arrays to optimize
        lr: learning rate
        momentum: coefficient for velocity (β in equations)
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        
        # Initialize velocities to zero
        # Each parameter gets its own velocity of same shape
        self.velocities = [np.zeros_like(p) for p in params]
    
    def step(self, grads):
        """
        Update parameters using momentum.
        
        grads: list of gradients (same structure as params)
        
        The velocity update v_t = β*v_{t-1} + g_t creates exponential
        weighting: recent gradients contribute fully, older gradients
        contribute with weight β^k. Typical β=0.9 means we effectively
        average over ~10 recent gradients.
        """
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update velocity: exponential moving average of gradients
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            
            # Update parameter using velocity
            # Note: some formulations use (1-β)*g instead of g
            # We follow PyTorch convention
            param -= self.lr * self.velocities[i]

class RMSprop:
    """
    RMSprop: Root Mean Square Propagation.
    
    Adapts learning rate per parameter based on exponential moving
    average of squared gradients. Parameters with consistently large
    gradients get smaller effective learning rate.
    """
    
    def __init__(self, params, lr=0.001, beta=0.9, epsilon=1e-8):
        """
        beta: decay rate for gradient square average
        epsilon: small constant for numerical stability
        """
        self.params = params
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        
        # Initialize squared gradient averages
        self.sq_grads = [np.zeros_like(p) for p in params]
    
    def step(self, grads):
        """
        Update using adaptive learning rates.
        
        The division by √E[g²] means parameters with large typical gradients
        get smaller updates (to prevent instability), while parameters with
        small typical gradients get larger updates (to make progress).
        """
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update squared gradient moving average
            # E[g²]_t = β*E[g²]_{t-1} + (1-β)*g²_t
            self.sq_grads[i] = (self.beta * self.sq_grads[i] + 
                               (1 - self.beta) * grad**2)
            
            # Adaptive learning rate: lr / √E[g²]
            # Adding epsilon prevents division by zero
            adapted_lr = self.lr / (np.sqrt(self.sq_grads[i]) + self.epsilon)
            
            # Update parameter
            param -= adapted_lr * grad

class Adam:
    """
    Adam: Adaptive Moment Estimation.
    
    Combines momentum (first moment) and RMSprop (second moment).
    Includes bias correction for proper behavior early in training.
    The de facto standard optimizer for many deep learning tasks.
    """
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        beta1: decay rate for first moment (momentum)
        beta2: decay rate for second moment (RMSprop)
        
        Default values work well across many tasks - Adam's strength
        is robustness to hyperparameter choices.
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moments
        self.m = [np.zeros_like(p) for p in params]  # First moment
        self.v = [np.zeros_like(p) for p in params]  # Second moment
        
        self.t = 0  # Time step (for bias correction)
    
    def step(self, grads):
        """
        Adam update with bias correction.
        
        The bias correction is crucial early in training when m_t and v_t
        are biased toward zero. Without correction, early updates are too
        small, slowing initial training. The correction factor 1/(1-β^t)
        grows as t increases, then approaches 1.
        """
        self.t += 1  # Increment timestep
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update biased first moment estimate (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate (RMSprop)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Compute bias-corrected moments
            # These corrections are largest early (when t is small)
            # and approach 1 as t → ∞
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameter
            # Combines momentum direction (m_hat) with adaptive scaling (√v_hat)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class AdamW:
    """
    AdamW: Adam with decoupled weight decay.
    
    Separates L2 regularization from gradient-based optimization.
    Better generalization than Adam, especially for Transformers.
    """
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, grads):
        """
        Adam update with decoupled weight decay.
        
        Key difference from Adam: weight decay is applied directly to
        parameters (θ ← θ - λθ) rather than being added to gradients.
        This ensures regularization strength is independent of adaptive
        learning rate scaling.
        """
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # Update moments (same as Adam)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update with decoupled weight decay
            # Weight decay happens outside adaptive scaling
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + 
                               self.weight_decay * param)

