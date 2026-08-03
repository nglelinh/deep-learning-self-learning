---
layout: post
title: 04-01-03-02 Convolution PyTorch and CNN Blocks
chapter: '04'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionExplained(nn.Module):
    """
    A convolutional layer with detailed explanations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create the convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
    def forward(self, x):
        """
        Forward pass with shape tracking.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        """
        batch_size, C_in, H_in, W_in = x.shape
        
        # Compute output dimensions
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Apply convolution
        out = self.conv(x)
        
        # Verify shapes
        assert out.shape == (batch_size, self.out_channels, H_out, W_out)
        
        return out
    
    def count_parameters(self):
        """Count and explain parameters."""
        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        weight_params = self.out_channels * self.in_channels * self.kernel_size**2
        bias_params = self.out_channels if self.conv.bias is not None else 0
        
        total = weight_params + bias_params
        
        print(f"Parameter breakdown:")
        print(f"  Weights: {self.out_channels} filters × "
              f"({self.kernel_size}×{self.kernel_size}×{self.in_channels}) = {weight_params}")
        print(f"  Biases: {bias_params}")
        print(f"  Total: {total}")
        
        return total


# ============== DEMONSTRATIONS ==============

print("=" * 60)
print("PyTorch Convolution Examples")
print("=" * 60)

# Example 1: Basic convolution
print("\n--- Example 1: RGB Image to 64 Feature Maps ---")
conv_layer = ConvolutionExplained(
    in_channels=3,      # RGB input
    out_channels=64,    # 64 learned filters
    kernel_size=3,      # 3×3 filters
    stride=1,
    padding=1           # "Same" padding
)

# Input: batch of 8 RGB images, 32×32 pixels
x = torch.randn(8, 3, 32, 32)
output = conv_layer(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
conv_layer.count_parameters()


# Example 2: Downsampling with strided convolution
print("\n--- Example 2: Strided Convolution for Downsampling ---")
conv_downsample = ConvolutionExplained(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=2,           # Stride 2 halves spatial dimensions
    padding=1
)

x2 = torch.randn(8, 64, 32, 32)
output2 = conv_downsample(x2)

print(f"Input shape:  {x2.shape}")
print(f"Output shape: {output2.shape}")  # 32×32 → 16×16
conv_downsample.count_parameters()


# Example 3: 1×1 Convolution for channel manipulation
print("\n--- Example 3: 1×1 Convolution (Channel Reduction) ---")
conv_1x1 = ConvolutionExplained(
    in_channels=512,
    out_channels=64,    # Reduce channels dramatically
    kernel_size=1,
    stride=1,
    padding=0
)

x3 = torch.randn(8, 512, 16, 16)
output3 = conv_1x1(x3)

print(f"Input shape:  {x3.shape}")
print(f"Output shape: {output3.shape}")  # Same spatial, fewer channels
conv_1x1.count_parameters()


# Example 4: Visualizing learned filters
print("\n--- Example 4: Inspecting Filter Weights ---")
# Access the learned weights
weights = conv_layer.conv.weight.data
print(f"Weight tensor shape: {weights.shape}")
print(f"  = {weights.shape[0]} filters of size "
      f"{weights.shape[2]}×{weights.shape[3]}×{weights.shape[1]}")

# First filter visualization (would show in matplotlib)
first_filter = weights[0]  # Shape: (3, 3, 3) = (in_channels, H, W)
print(f"First filter shape: {first_filter.shape}")
print(f"Filter values (channel 0):\n{first_filter[0]}")
```

### Complete CNN Block with Best Practices

```python
class ConvBlock(nn.Module):
    """
    Standard convolutional block: Conv → BatchNorm → ReLU
    
    This pattern is used in virtually all modern CNNs.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1):
        super().__init__()
        
        # Convolution (no bias when using BatchNorm)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        
        # Batch normalization stabilizes training
        self.bn = nn.BatchNorm2d(out_channels)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection (from ResNet).
    
    Key insight: Learn residual F(x) = H(x) - x instead of H(x) directly.
    Output: y = F(x) + x
    """
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x  # Save input for skip connection
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # Skip connection!
        out = self.relu(out)
        
        return out


class SimpleCNN(nn.Module):
    """
    A simple but effective CNN for image classification.
    
    Architecture: 
    Input → ConvBlocks (with pooling) → Global Average Pool → Classifier
    """
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 32×32×3 → 16×16×32
            ConvBlock(input_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 16×16×32 → 8×8×64
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 8×8×64 → 4×4×128
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
        )
        
        # Global average pooling: 4×4×128 → 128
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classify
        x = self.classifier(x)
        
        return x


# ============== TRAINING EXAMPLE ==============

def train_step(model, images, labels, optimizer, criterion):
    """Single training step with forward and backward pass."""
    model.train()
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).float().mean()
    
    return loss.item(), accuracy.item()


# Create model and training components
model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Simulate one training step
dummy_images = torch.randn(32, 3, 32, 32)  # Batch of 32 CIFAR-like images
dummy_labels = torch.randint(0, 10, (32,))

loss, acc = train_step(model, dummy_images, dummy_labels, optimizer, criterion)
print(f"  Training step - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
```

---
