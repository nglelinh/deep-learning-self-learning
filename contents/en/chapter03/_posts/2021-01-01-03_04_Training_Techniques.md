---
layout: post
title: 03-04 Practical Training Techniques
chapter: '03'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter03
lesson_type: required
---

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

## Monitoring Training

### Metrics to Track

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease; if it increases, overfitting!
3. **Training Accuracy**: Should increase
4. **Validation Accuracy**: Should increase; gap with training accuracy indicates overfitting

### Visualization

```python
import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Early Stopping

Stop training when validation loss stops improving.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        patience: number of epochs to wait before stopping
        min_delta: minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_parameters = None
    
    def __call__(self, val_loss, parameters):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_parameters = parameters.copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_parameters = parameters.copy()
            self.counter = 0
        
        return self.early_stop

# Usage in training loop
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    # Training...
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    
    if early_stopping(val_loss, parameters):
        print("Early stopping triggered!")
        parameters = early_stopping.best_parameters
        break
```

## Gradient Clipping

Prevent exploding gradients by clipping gradient magnitudes.

### Clip by Value

```python
def clip_gradients_by_value(gradients, max_value=5.0):
    """Clip gradients to [-max_value, max_value]"""
    clipped_gradients = {}
    for key in gradients.keys():
        clipped_gradients[key] = np.clip(gradients[key], -max_value, max_value)
    return clipped_gradients
```

### Clip by Norm

```python
def clip_gradients_by_norm(gradients, max_norm=5.0):
    """Clip gradients by global norm"""
    # Compute global norm
    total_norm = 0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        clipped_gradients = {}
        for key, grad in gradients.items():
            clipped_gradients[key] = grad * clip_coef
        return clipped_gradients
    else:
        return gradients
```

## Complete Training Loop

```python
def train_model(X_train, Y_train, X_val, Y_val, layer_dims, 
                learning_rate=0.01, batch_size=32, num_epochs=100,
                initialization='he', early_stopping_patience=10):
    """
    Complete training function with all best practices
    """
    # Initialize
    parameters = initialize_parameters(layer_dims, initialization)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # History
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Create mini-batches
        mini_batches = create_mini_batches(X_train, Y_train, batch_size)
        epoch_loss = 0
        
        # Process each mini-batch
        for X_batch, Y_batch in mini_batches:
            # Forward propagation
            AL, caches = forward_propagation(X_batch, parameters)
            
            # Compute cost
            batch_cost = compute_cost(AL, Y_batch)
            epoch_loss += batch_cost
            
            # Backward propagation
            gradients = backward_propagation(AL, Y_batch, caches)
            
            # Gradient clipping (optional)
            gradients = clip_gradients_by_norm(gradients, max_norm=5.0)
            
            # Update parameters
            parameters = update_parameters(parameters, gradients, learning_rate)
        
        # Average loss over all batches
        epoch_loss /= len(mini_batches)
        train_losses.append(epoch_loss)
        
        # Compute training accuracy
        train_predictions = predict(X_train, parameters)
        train_acc = np.mean(train_predictions == Y_train)
        train_accs.append(train_acc)
        
        # Validation
        val_predictions, val_caches = forward_propagation(X_val, parameters)
        val_loss = compute_cost(val_predictions, Y_val)
        val_losses.append(val_loss)
        
        val_pred_labels = (val_predictions > 0.5).astype(int)
        val_acc = np.mean(val_pred_labels == Y_val)
        val_accs.append(val_acc)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs}: "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_loss, parameters):
            print(f"Early stopping at epoch {epoch}")
            parameters = early_stopping.best_parameters
            break
    
    # Plot history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    return parameters, (train_losses, val_losses, train_accs, val_accs)
```

## Hyperparameter Tuning

### Key Hyperparameters

1. **Learning rate** (most important)
2. **Batch size**
3. **Number of layers**
4. **Number of neurons per layer**
5. **Activation functions**
6. **Initialization method**

### Tuning Strategies

#### 1. Manual Search

Try different values based on intuition and experience.

#### 2. Grid Search

Try all combinations of a predefined set of values.

```python
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]

best_val_acc = 0
best_params = None

for lr in learning_rates:
    for bs in batch_sizes:
        model, history = train_model(X_train, Y_train, X_val, Y_val, 
                                     layer_dims, learning_rate=lr, batch_size=bs)
        val_acc = history[3][-1]  # Last validation accuracy
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (lr, bs)

print(f"Best: LR={best_params[0]}, BS={best_params[1]}, Val Acc={best_val_acc:.4f}")
```

#### 3. Random Search

Often more efficient than grid search.

```python
import random

num_trials = 20
best_val_acc = 0

for trial in range(num_trials):
    lr = 10 ** random.uniform(-4, -1)  # Log-uniform between 0.0001 and 0.1
    bs = random.choice([16, 32, 64, 128])
    
    model, history = train_model(X_train, Y_train, X_val, Y_val,
                                 layer_dims, learning_rate=lr, batch_size=bs)
    val_acc = history[3][-1]
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"New best: LR={lr:.6f}, BS={bs}, Val Acc={val_acc:.4f}")
```

## Summary

- **Proper initialization** (He for ReLU) prevents training issues
- **Data preprocessing** (standardization/normalization) improves convergence
- **Mini-batch processing** balances speed and stability
- **Train/Val/Test split** enables proper evaluation
- **Monitoring** training/validation metrics detects overfitting
- **Early stopping** prevents overfitting and saves time
- **Gradient clipping** prevents exploding gradients
- **Hyperparameter tuning** is essential for optimal performance

With these techniques, you're equipped to train neural networks effectively. The next chapter covers Convolutional Neural Networks for computer vision tasks!

