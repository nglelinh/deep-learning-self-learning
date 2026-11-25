---
layout: post
title: 03-04-01 Initialization, Preprocessing, Batches, and Splits
chapter: '03'
order: 12
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

> *Initialize every weight in a neural network to zero and something eerie happens: every neuron in a hidden layer computes the exact same activation, receives the exact same gradient, and updates to the exact same weight forever—turning a 1,000-neuron layer into a single expensive neuron. Breaking symmetry at the right variance scale is step zero of deep learning.*

## Key ideas: Initialization, Preprocessing, and Batching

1. **Symmetry Breaking (Why Zero Init Fails)**: Identical weights force all neurons in a layer to evolve identically across epochs; random initialization breaks symmetry so neurons specialize into distinct feature detectors.
2. **Xavier (Glorot) vs. He (Kaiming) Variance Scaling**: Xavier sets $$\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$ for linear/tanh regimes; He sets $$\text{Var}(W) = \frac{2}{n_{\text{in}}}$$ to compensate for ReLU zeroing out half the activation energy.
3. **Input Standardization (Zero Mean, Unit Variance)**: Centers and rescales features so loss contours become roughly spherical, preventing large-scale features from dominating gradient updates.
4. **Mini-Batch Shuffling & Stratified Splits**: Randomizing batch order every epoch eliminates cyclical gradient bias, while stratified splitting preserves rare-class ratios across train, validation, and test sets.

### Worked example: Comparing He vs. Xavier Signal Variance Across a 10-Layer ReLU Network

Suppose a 10-layer ReLU MLP has constant width $$n = 512$$ per layer and unit-variance input $$\text{Var}(a^{[0]}) = 1.0$$. For a ReLU layer with symmetric zero-mean weights, forward variance evolves as:
$$\text{Var}(a^{[l]}) = \frac{1}{2}\,n\,\text{Var}(W)\,\text{Var}(a^{[l-1]})$$
(The factor $$\frac{1}{2}$$ appears because ReLU zeros out the negative half of a symmetric zero-mean pre-activation!)

1. **Using Xavier Initialization ($$\text{Var}(W) = \frac{1}{n} = \frac{1}{512}$$)**:
   Each layer multiplies activation variance by $$\frac{1}{2} \cdot 512 \cdot \frac{1}{512} = 0.5$$.
   After $$L = 10$$ layers:
   $$\text{Var}(a^{[10]}) = (0.5)^{10} \times 1.0 = \frac{1}{1024} \approx 0.000977$$
   The signal loses **99.9%** of its variance before even reaching the output layer!
2. **Using He (Kaiming) Initialization ($$\text{Var}(W) = \frac{2}{n} = \frac{2}{512}$$)**:
   Each layer multiplies activation variance by $$\frac{1}{2} \cdot 512 \cdot \frac{2}{512} = 1.0$$.
   After $$L = 10$$ layers:
   $$\text{Var}(a^{[10]}) = (1.0)^{10} \times 1.0 = 1.000$$
   Variance stays locked at $$1.0$$ across all 10 (or 100!) layers—the exact mathematical reason He initialization unlocked deep ReLU networks.

This lesson covers practical techniques and best practices for training neural networks effectively.

---

## Weight Initialization

Proper initialization is crucial for successful training. Poor initialization can lead to vanishing/exploding gradients or slow convergence.

### Bad Initialization Methods

#### 1. All Zeros

```python
W = np.zeros((n_l, n_l_prev))
```

**Problem**: All neurons compute the same output and receive the same gradient → No learning!

#### 2. All Same Values

```python
W = np.ones((n_l, n_l_prev)) * 0.5
```

**Problem**: Same as zeros - breaks symmetry.

### Good Initialization Methods

#### 1. Random Small Values

```python
W = np.random.randn(n_l, n_l_prev) * 0.01
```

**Pros**: Breaks symmetry
**Cons**: May be too small for deep networks

#### 2. Xavier/Glorot Initialization

For **sigmoid** or **tanh** activations:

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{1}{n^{[l-1]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(1 / n_l_prev)
```

or

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]} + n^{[l]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(2 / (n_l_prev + n_l))
```

**Rationale**: Maintains variance of activations across layers.

#### 3. He Initialization

For **ReLU** activations (most common):

$$\mathbf{W}^{[l]} \sim \mathcal{N}\left(0, \frac{2}{n^{[l-1]}}\right)$$

```python
W = np.random.randn(n_l, n_l_prev) * np.sqrt(2 / n_l_prev)
```

**Why factor of 2?** ReLU zeros out half the neurons on average.

### Bias Initialization

Biases can typically be initialized to zero:

```python
b = np.zeros((n_l, 1))
```

### Complete Initialization Function

```python
def initialize_parameters(layer_dims, initialization_method='he'):
    """
    Initialize network parameters
    
    layer_dims: list of layer sizes [n_x, n_h1, ..., n_y]
    initialization_method: 'zeros', 'random', 'xavier', 'he'
    """
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        if initialization_method == 'zeros':
            parameters[f'W{l}'] = np.zeros((layer_dims[l], layer_dims[l-1]))
        
        elif initialization_method == 'random':
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        
        elif initialization_method == 'xavier':
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1 / layer_dims[l-1])
        
        elif initialization_method == 'he':
            parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    return parameters
```

## Data Preprocessing

### 1. Feature Scaling

Normalize input features to similar ranges.

#### Standardization (Z-score Normalization)

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

```python
def standardize(X):
    """Standardize features to mean=0, std=1"""
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    X_norm = (X - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero
    return X_norm, mean, std
```

#### Min-Max Normalization

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

```python
def min_max_normalize(X):
    """Scale features to [0, 1]"""
    x_min = np.min(X, axis=0, keepdims=True)
    x_max = np.max(X, axis=0, keepdims=True)
    X_norm = (X - x_min) / (x_max - x_min + 1e-8)
    return X_norm, x_min, x_max
```

**When to use what:**
- **Standardization**: When features are normally distributed or have outliers
- **Min-Max**: When you need features in a specific range (e.g., [0, 1])

### 2. Data Shuffling

Shuffle training data before each epoch to prevent learning order-dependent patterns.

```python
def shuffle_data(X, Y):
    """Shuffle training data"""
    m = X.shape[1]
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    return X_shuffled, Y_shuffled
```

## Batch Processing

### Creating Mini-Batches

```python
def create_mini_batches(X, Y, batch_size):
    """
    Create list of mini-batches
    
    X: (n_x, m)
    Y: (n_y, m)
    batch_size: size of each mini-batch
    
    Returns: list of (X_batch, Y_batch) tuples
    """
    m = X.shape[1]
    mini_batches = []
    
    # Shuffle data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    
    # Partition
    num_complete_batches = m // batch_size
    
    for k in range(num_complete_batches):
        X_batch = X_shuffled[:, k * batch_size:(k + 1) * batch_size]
        Y_batch = Y_shuffled[:, k * batch_size:(k + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))
    
    # Handle remaining examples (if m is not divisible by batch_size)
    if m % batch_size != 0:
        X_batch = X_shuffled[:, num_complete_batches * batch_size:]
        Y_batch = Y_shuffled[:, num_complete_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))
    
    return mini_batches
```

## Train/Validation/Test Split

### Why Three Sets?

- **Training set**: Learn parameters
- **Validation set**: Tune hyperparameters, monitor overfitting
- **Test set**: Final evaluation (use only once!)

### Typical Splits

**Small dataset (< 10,000 examples):**
- Train: 60%, Val: 20%, Test: 20%

**Medium dataset (10,000 - 1,000,000):**
- Train: 80%, Val: 10%, Test: 10%

**Large dataset (> 1,000,000):**
- Train: 98%, Val: 1%, Test: 1%

### Implementation

```python
def train_val_test_split(X, Y, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into train, validation, and test sets
    """
    m = X.shape[1]
    
    # Shuffle first
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    
    # Calculate split indices
    train_end = int(train_ratio * m)
    val_end = train_end + int(val_ratio * m)
    
    # Split
    X_train = X_shuffled[:, :train_end]
    Y_train = Y_shuffled[:, :train_end]
    
    X_val = X_shuffled[:, train_end:val_end]
    Y_val = Y_shuffled[:, train_end:val_end]
    
    X_test = X_shuffled[:, val_end:]
    Y_test = Y_shuffled[:, val_end:]
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
```


## Video tham khảo

- **xem trước** [L11.6 Xavier Glorot and Kaiming He Initialization](https://www.youtube.com/watch?v=ScWTYHQra5E) — Sebastian Raschka, English. Xavier and He initialization are the weight-scale rules this lesson states.
- [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M) — Stanford University School of Engineering, English. The lecture context for those rules, plus preprocessing and batching.
- [Train, Test, & Validation Sets explained](https://www.youtube.com/watch?v=Zi-0rlM4RDs) — deeplizard, English. Why the lesson splits data into train, validation, and test before any epoch runs.
