---
layout: post
title: 01-00-02 First MNIST Walkthrough
chapter: '01'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter01
---

## 4. Code Snippet

Let's implement a simple deep learning example to make concepts concrete. We'll build a neural network to classify MNIST digits using PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple deep neural network
class SimpleDeepNet(nn.Module):
    """
    A 3-layer neural network for MNIST digit classification.
    
    Architecture:
    - Input: 784 dimensions (28x28 flattened image)
    - Hidden Layer 1: 128 neurons with ReLU activation
    - Hidden Layer 2: 64 neurons with ReLU activation  
    - Output Layer: 10 neurons (one per digit class)
    """
    def __init__(self):
        super(SimpleDeepNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)    # Second hidden layer
        self.fc3 = nn.Linear(64, 10)     # Output layer
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Flatten the 28x28 images to 784-dimensional vectors
        x = x.view(-1, 784)
        
        # Layer 1: Learn low-level features
        x = self.relu(self.fc1(x))
        
        # Layer 2: Learn mid-level feature combinations
        x = self.relu(self.fc2(x))
        
        # Output layer: Classify into 10 digit classes
        x = self.fc3(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean/std
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleDeepNet()
criterion = nn.CrossEntropyLoss()  # Combines softmax + negative log likelihood
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()  # Set model to training mode
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass: compute predictions
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Backpropagation
        
        # Update weights
        optimizer.step()
        
        # Track accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} Training: Avg Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%')

# Evaluation loop
def test(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test: Avg Loss={test_loss:.4f}, Accuracy={accuracy:.2f}%\n')
    return accuracy

# Train for multiple epochs
print("Starting training...")
for epoch in range(1, 6):  # Train for 5 epochs
    train(model, train_loader, optimizer, criterion, epoch)
    test_accuracy = test(model, test_loader, criterion)

print(f"Final test accuracy: {test_accuracy:.2f}%")
print("Training complete! The network learned to recognize digits through:")
print("1. Forward propagation (making predictions)")
print("2. Loss computation (measuring errors)")
print("3. Backpropagation (computing gradients)")
print("4. Weight updates (learning from mistakes)")
```

### Understanding the Code

This simple example demonstrates core deep learning principles:

**1. Architecture Design**: Three layers transform 784-dimensional input to 10-dimensional output through learned representations.

**2. Automatic Feature Learning**: We never told the network what features to look for—it discovers useful representations automatically.

**3. The Training Loop**: The standard pattern of forward pass → compute loss → backpropagate → update weights that underlies all deep learning.

**4. Nonlinearity is Crucial**: ReLU activation functions between layers enable learning complex, nonlinear functions. Without them, multiple layers would collapse to a single linear transformation.

**5. Scalability**: This same code structure, with appropriate modifications, works for much larger datasets and more complex tasks—computer vision, natural language processing, etc.

After just 5 epochs (5 passes through 60,000 training images), this simple network typically achieves ~97% accuracy—demonstrating deep learning's power to learn from data.

## 5. Related Concepts

Understanding deep learning requires seeing how it connects to broader machine learning and AI concepts:

### Supervised vs Unsupervised vs Reinforcement Learning

**Supervised Learning** (what we've primarily discussed) learns from labeled examples: input-output pairs like (image, label) or (sentence, translation). The network learns to map inputs to correct outputs.

**Unsupervised Learning** discovers structure in data without labels. Autoencoders learn compressed representations. Clustering groups similar examples. Generative models learn data distributions to create new samples. These techniques are crucial when labels are expensive or unavailable.

**Reinforcement Learning** learns from interaction: an agent takes actions in an environment, receives rewards, and learns policies to maximize cumulative reward. This enables learning behaviors (game playing, robotics) where we can't provide explicit correct actions for every situation, only feedback on outcomes.

Deep learning has transformed all three paradigms, but the principles differ significantly. This course focuses primarily on supervised learning initially, with later chapters covering unsupervised and reinforcement learning.

### Classical Machine Learning vs Deep Learning

Traditional machine learning (SVMs, decision trees, logistic regression) typically requires:
- Hand-engineered features
- Explicit model assumptions (linearity, independence)
- Works well with moderate data (hundreds to thousands of examples)
- More interpretable (feature importance, decision boundaries)

Deep learning:
- Learns features automatically end-to-end
- Fewer assumptions about data structure
- Requires large datasets (thousands to millions of examples)
- Less interpretable but more powerful for complex patterns

Neither approach is universally superior—classical ML can be better for small datasets, tabular data, or when interpretability is critical. Deep learning excels with large datasets, high-dimensional inputs (images, text), and complex patterns.

### Transfer Learning and Pre-training

One of deep learning's most powerful techniques is **transfer learning**: training a network on one task (e.g., ImageNet classification) then adapting it to related tasks (medical image analysis, wildlife detection). The network's learned representations—edge detectors, texture patterns, shape recognizers—transfer across domains.

**Pre-training** on large general datasets, then **fine-tuning** on specific tasks, has become standard practice. GPT, BERT, and other large language models are pre-trained on massive text corpora, then specialized for particular applications through fine-tuning with much less task-specific data. This dramatically reduces data requirements for new tasks.

### The Role of Architecture vs Data vs Compute

Deep learning's success derives from three factors working together:

**Architecture innovations** (CNNs, Transformers, ResNets) enable learning certain patterns efficiently. The right architecture provides appropriate inductive biases for the problem structure.

**Data scale** provides the raw material for learning. More diverse, high-quality data enables networks to learn more robust, generalizable representations.

**Computational scale** makes training large networks on large datasets practical. GPUs parallelize the matrix operations neural networks depend on, reducing training time from months to hours.

Modern deep learning progress comes from advances in all three: better architectures (Transformers), larger datasets (web-scale text and images), and more compute (GPU clusters, TPUs). No single factor alone explains the field's success.

## 6. Fundamental Papers

Understanding deep learning's historical development through key papers provides context for current practice and future directions.

**["A Logical Calculus of Ideas Immanent in Nervous Activity" (1943)](https://link.springer.com/article/10.1007/BF02478259)**  
*Authors*: Warren McCulloch and Walter Pitts  
This foundational paper introduced the mathematical model of artificial neurons, showing that networks of simple threshold units could compute any logical function. While vastly simplified compared to biological neurons, this work established the theoretical basis for neural computation and inspired subsequent research in both neuroscience and artificial intelligence.

**["Learning representations by back-propagating errors" (1986)](https://www.nature.com/articles/323533a0)**  
*Authors*: David Rumelhart, Geoffrey Hinton, Ronald Williams  
Backpropagation wasn't invented here (it was discovered independently multiple times), but this paper brought it to widespread attention and demonstrated its power for training multi-layer networks. By showing how to efficiently compute gradients through composition of functions via the chain rule, backpropagation made deep learning practical. This paper ended the first AI winter by proving that neural networks could learn complex functions.

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
LeNet-5, introduced in this paper, demonstrated that convolutional neural networks could achieve excellent performance on real-world tasks (check reading, digit recognition). More importantly, it established design principles—local connectivity, weight sharing, pooling—that remain central to modern computer vision. This paper showed that deep learning could move from toy problems to practical applications.

**["ImageNet Classification with Deep Convolutional Neural Networks" (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)**  
*Authors*: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton  
AlexNet's crushing victory in the 2012 ImageNet competition (15.3% error vs 26.2% for second place) sparked the modern deep learning revolution. By combining deeper architectures, ReLU activations, dropout regularization, and GPU training, it demonstrated that neural networks could scale to large, complex datasets. This success convinced the broader computer vision community to adopt deep learning.

**["Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)**  
*Authors*: Ashish Vaswani et al. (Google)  
The Transformer architecture introduced here has become the foundation of modern NLP and increasingly other domains. By replacing recurrence with attention mechanisms, Transformers enable full parallelization during training and better capture long-range dependencies. This paper's influence extends far beyond its original machine translation application—BERT, GPT, and most recent large language models build on this architecture.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Authors*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
ResNet introduced skip connections that enabled training networks hundreds of layers deep by providing direct gradient pathways. Beyond winning ImageNet 2015, this work fundamentally changed how we think about deep architectures—depth is crucial, but networks need architectural innovations (skip connections, careful normalization) to train effectively. ResNet's principles appear in most modern deep architectures.

