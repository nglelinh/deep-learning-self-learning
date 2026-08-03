---
layout: post
title: 03-04-02 Monitoring, Early Stopping, and the Training Loop
chapter: '03'
order: 13
owner: Deep Learning Course
lang: en
categories:
- chapter03
---

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

