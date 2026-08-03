---
layout: post
title: 04-02-02-02 VGG, ResNet, and Architecture Evolution
chapter: '04'
order: 13
owner: Deep Learning Course
lang: en
categories:
- chapter04
---

### VGGNet (2014) — Simplicity and Depth

VGG by Simonyan and Zisserman showed that **depth matters** and **simple is better**.

**Key insight**: Use only 3×3 convolutions stacked deeply.

**VGG-16 Architecture**:
```
Input (224×224×3)

Block 1: Conv3-64, Conv3-64, MaxPool → 112×112×64
Block 2: Conv3-128, Conv3-128, MaxPool → 56×56×128
Block 3: Conv3-256, Conv3-256, Conv3-256, MaxPool → 28×28×256
Block 4: Conv3-512, Conv3-512, Conv3-512, MaxPool → 14×14×512
Block 5: Conv3-512, Conv3-512, Conv3-512, MaxPool → 7×7×512

Flatten → FC 4096 → FC 4096 → FC 1000
```

**Why 3×3 is optimal**:
- Two 3×3 convs = one 5×5 receptive field, but:
  - Fewer parameters: 2×(3×3) = 18 vs 5×5 = 25
  - More non-linearity: 2 ReLUs vs 1
- Three 3×3 convs = one 7×7 receptive field

**~138 million parameters** (mostly in FC layers)

```python
class VGG16(nn.Module):
    """VGG-16 architecture"""
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### ResNet (2015) — The Skip Connection Revolution

ResNet by He et al. solved the **degradation problem**: deeper networks had *higher* training error than shallow ones. This was counterintuitive—more capacity should help, not hurt.

**The problem**: In very deep networks, gradients either vanish or explode, making optimization difficult. Even with careful initialization and batch normalization, networks deeper than ~20 layers trained poorly.

**The solution**: **Residual connections** (skip connections)

Instead of learning $$H(x)$$ directly, learn the residual $$F(x) = H(x) - x$$, then:

$$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

**Why it works**:
1. **Identity mapping is easy**: If identity is optimal, just set $$F(x) = 0$$
2. **Gradient highway**: Gradients flow directly through skip connections
3. **Ensemble effect**: ResNet behaves like an ensemble of shallower networks

**Basic Residual Block**:
```
         ┌────────────────────────────────────┐
         │               Identity             │
         │                                    │
    x ───┼──→ Conv 3×3 → BN → ReLU           │
         │        ↓                           │
         │   Conv 3×3 → BN                    │
         │        ↓                           │
         └───────⊕───→ ReLU → output
              (add)
```

**Bottleneck Block** (for deeper networks like ResNet-50+):
```
         ┌──────────────────────────────────┐
         │             Identity             │
         │                                  │
    x ───┼──→ Conv 1×1 → BN → ReLU (reduce)│
         │        ↓                         │
         │   Conv 3×3 → BN → ReLU           │
         │        ↓                         │
         │   Conv 1×1 → BN (expand)         │
         │        ↓                         │
         └───────⊕───→ ReLU → output
```

```python
class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # For matching dimensions
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Handle dimension mismatch
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # The key: skip connection!
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3×3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1×1 expand
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet implementation"""
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# Create standard ResNet variants
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# ============== MODEL COMPARISON ==============
print("=" * 60)
print("CNN Architecture Comparison")
print("=" * 60)

models = {
    'LeNet-5': LeNet5(10),
    'VGG-16': VGG16(1000),
    'ResNet-18': resnet18(1000),
    'ResNet-50': resnet50(1000),
}

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name:12s}: {params:>15,} parameters")
```

### Architecture Evolution Summary

| Model | Year | Depth | Params | Top-5 Error | Key Innovation |
|-------|------|-------|--------|-------------|----------------|
| LeNet-5 | 1998 | 5 | 60K | - | First successful CNN |
| AlexNet | 2012 | 8 | 60M | 15.3% | ReLU, Dropout, GPU |
| VGG-16 | 2014 | 16 | 138M | 7.3% | 3×3 only, depth |
| GoogLeNet | 2014 | 22 | 6.8M | 6.7% | Inception modules |
| ResNet-50 | 2015 | 50 | 25M | 3.6% | Skip connections |
| ResNet-152 | 2015 | 152 | 60M | 3.0% | Very deep |
| DenseNet | 2017 | 121 | 8M | 4.2% | Dense connections |
| EfficientNet | 2019 | - | 5.3M | 2.9% | Neural architecture search |

---

## 5. Fundamental Papers

**["Gradient-Based Learning Applied to Document Recognition" (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)**  
*Authors*: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner  
Introduced LeNet-5, establishing the conv-pool-conv-pool pattern that became the blueprint for CNNs. Demonstrated end-to-end gradient-based learning on raw pixels, proving that hand-crafted features were unnecessary for image recognition.

**["Network In Network" (2014)](https://arxiv.org/abs/1312.4400)**  
*Authors*: Min Lin, Qiang Chen, Shuicheng Yan  
Introduced 1×1 convolutions for cross-channel pooling and Global Average Pooling (GAP) to replace fully connected layers. These innovations dramatically reduced parameters and became standard in all subsequent architectures.

**["Very Deep Convolutional Networks for Large-Scale Image Recognition" (2015)](https://arxiv.org/abs/1409.1556)**  
*Authors*: Karen Simonyan, Andrew Zisserman  
VGGNet proved that depth with small 3×3 filters outperforms shallow networks with large filters. The simple, repeating architecture became a reference design and is still widely used for transfer learning.

**["Deep Residual Learning for Image Recognition" (2016)](https://arxiv.org/abs/1512.03385)**  
*Authors*: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
ResNet's skip connections solved the degradation problem, enabling training of networks with 100+ layers. Won ImageNet 2015 with 3.57% top-5 error (surpassing human performance). The residual learning framework is now used in transformers, diffusion models, and nearly all deep architectures.

**["Densely Connected Convolutional Networks" (2017)](https://arxiv.org/abs/1608.06993)**  
*Authors*: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger  
DenseNet connects each layer to every other layer, enabling feature reuse and strong gradient flow. Achieved state-of-the-art accuracy with fewer parameters by maximizing information flow through the network.

**["EfficientNet: Rethinking Model Scaling" (2019)](https://arxiv.org/abs/1905.11946)**  
*Authors*: Mingxing Tan, Quoc V. Le  
Used neural architecture search to find efficient baseline architectures, then proposed compound scaling (depth × width × resolution) for systematic scaling. Achieved state-of-the-art accuracy with dramatically fewer parameters and FLOPs.

---

## 6. Key Takeaways

1. **Pooling reduces dimensions** while preserving important features. Max pooling works best for classification; average pooling for other tasks.

2. **Global Average Pooling** replaces heavy FC layers, reducing parameters by ~40× while improving generalization.

3. **Architecture evolution**: LeNet (simple) → AlexNet (scale) → VGG (depth) → ResNet (skip connections) → EfficientNet (efficiency)

4. **3×3 convolutions are optimal**: Stack multiple small filters instead of using large ones.

5. **Skip connections are essential** for training deep networks. They appear in ResNet, DenseNet, U-Net, Transformers, and diffusion models.

6. **Modern best practices**:
   - Conv → BatchNorm → ReLU
   - Skip connections every 2-3 layers
   - Global Average Pooling instead of Flatten + FC
   - Strided convolution for learnable downsampling

7. **Pre-trained models** (transfer learning) are the standard starting point for most vision tasks. ImageNet pre-training transfers remarkably well to other domains.

**Next chapter**: Recurrent Neural Networks for sequential data!
