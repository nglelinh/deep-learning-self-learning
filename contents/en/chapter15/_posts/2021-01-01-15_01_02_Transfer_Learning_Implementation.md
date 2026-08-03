---
layout: post
title: 15-01-02 Transfer Learning Implementation
chapter: '15'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter15
---

## 4. Code Snippet

Complete transfer learning implementation:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

print("="*70)
print("Transfer Learning: Fine-tuning Pre-trained ResNet for Custom Dataset")
print("="*70)

# Load pre-trained ResNet18 (smaller than ResNet50 for demonstration)
print("\n1. Loading Pre-trained Model")
print("-" * 70)

# weights='IMAGENET1K_V1' loads ImageNet pre-trained weights
model_pretrained = models.resnet18(weights='IMAGENET1K_V1')

print(f"Loaded ResNet18 pre-trained on ImageNet")
print(f"Original output layer: {model_pretrained.fc}")
print(f"  (1000 classes for ImageNet)")

# Examine what pre-trained model has learned
print(f"\nPre-trained features:")
print(f"  Layer 1 filters: {model_pretrained.conv1.weight.shape}")  # (64, 3, 7, 7)
print(f"  These are edge/texture detectors learned from ImageNet")

# 2. Adapt for new task
print("\n2. Adapting for Custom Task (10 Classes)")
print("-" * 70)

# Replace final fully-connected layer for our task
# Everything else keeps pre-trained weights
num_classes = 10  # Our custom task has 10 classes
num_features = model_pretrained.fc.in_features  # Get input size of fc layer

print(f"Original FC input features: {num_features}")
print(f"Replacing final layer for {num_classes} classes...")

model_pretrained.fc = nn.Linear(num_features, num_classes)

print(f"New model output layer: {model_pretrained.fc}")
print("  (10 classes for our custom task)")

# Create dummy dataset (in practice, use your real data)
# We'll simulate with CIFAR-10 as our "custom" task
print("\n3. Preparing Custom Dataset")
print("-" * 70)

transform_train = transforms.Compose([
    transforms.Resize(224),  # ResNet expects 224×224 (CIFAR is 32×32)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# Simulate limited data scenario: use only 1000 training images
full_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)

# Take subset to simulate limited data
limited_size = 1000
remaining = len(full_dataset) - limited_size
train_dataset, _ = random_split(full_dataset, [limited_size, remaining])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_train)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training with only {limited_size} images (simulating limited data)")
print(f"Test set: {len(test_dataset)} images")

# 3. Training Strategy: Feature Extraction vs Fine-Tuning
print("\n4. Strategy A: Feature Extraction (Freeze Pre-trained Layers)")
print("-" * 70)

# Create copy of model for feature extraction
model_features = models.resnet18(weights='IMAGENET1K_V1')
model_features.fc = nn.Linear(num_features, num_classes)

# Freeze all layers except final FC
for param in model_features.parameters():
    param.requires_grad = False

# Unfreeze final layer
for param in model_features.fc.parameters():
    param.requires_grad = True

trainable_params_fe = sum(p.numel() for p in model_features.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_features.parameters())

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params_fe:,} ({trainable_params_fe/total_params*100:.1f}%)")
print("Only training the final classification layer!")

# Optimizer for feature extraction (only fc layer parameters)
optimizer_fe = optim.Adam(model_features.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train
model_features.train()
print("\nTraining feature extraction model (5 epochs)...")

for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer_fe.zero_grad()
        
        # Forward (conv layers frozen, only fc trains)
        outputs = model_features(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer_fe.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    train_acc = 100. * correct / total
    print(f"  Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}, "
          f"Train Acc = {train_acc:.2f}%")

# Evaluate
model_features.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model_features(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

test_acc_fe = 100. * correct / total
print(f"\nFeature Extraction Test Accuracy: {test_acc_fe:.2f}%")

# 4. Strategy B: Fine-Tuning
print("\n5. Strategy B: Fine-Tuning (Update All Layers)")
print("-" * 70)

model_finetune = models.resnet18(weights='IMAGENET1K_V1')
model_finetune.fc = nn.Linear(num_features, num_classes)

# All parameters trainable
trainable_params_ft = sum(p.numel() for p in model_finetune.parameters())
print(f"Trainable parameters: {trainable_params_ft:,} (100%)")

# Use differential learning rates
# Lower LR for pre-trained layers, higher for new layer
optimizer_ft = optim.Adam([
    {'params': model_finetune.layer1.parameters(), 'lr': 0.0001},
    {'params': model_finetune.layer2.parameters(), 'lr': 0.0001},
    {'params': model_finetune.layer3.parameters(), 'lr': 0.0002},
    {'params': model_finetune.layer4.parameters(), 'lr': 0.0005},
    {'params': model_finetune.fc.parameters(), 'lr': 0.001}  # Highest for new layer
], lr=0.0001)  # Default for any params not specified

print("Using layer-wise differential learning rates:")
print("  Early layers: 0.0001 (barely change)")
print("  Middle layers: 0.0002")
print("  Deep layers: 0.0005")
print("  New FC layer: 0.001 (change most)")

# Train
model_finetune.train()
print("\nTraining fine-tuning model (5 epochs)...")

for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer_ft.zero_grad()
        
        outputs = model_finetune(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer_ft.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    train_acc = 100. * correct / total
    print(f"  Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}, "
          f"Train Acc = {train_acc:.2f}%")

# Evaluate
model_finetune.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model_finetune(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

test_acc_ft = 100. * correct / total
print(f"\nFine-Tuning Test Accuracy: {test_acc_ft:.2f}%")

# Compare results
print("\n" + "="*70)
print("Transfer Learning Results Comparison")
print("="*70)
print(f"Feature Extraction: {test_acc_fe:.2f}% test accuracy")
print(f"Fine-Tuning:        {test_acc_ft:.2f}% test accuracy")
print(f"\nBoth dramatically outperform training from scratch (~60% on 1000 images)")
print("Transfer learning enabled competitive performance with limited data!")
```

Demonstrate feature visualization:

```python
print("\n" + "="*70)
print("Analyzing Transferred Features")
print("="*70)

# Extract features for analysis
def extract_features(model, dataloader, layer_name='layer4'):
    """
    Extract features from a specific layer.
    
    This shows what representations the model uses for classification.
    Pre-trained features should be meaningful even for custom task.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # Register hook to capture layer outputs
    features_hook = []
    def hook_fn(module, input, output):
        features_hook.append(output.detach())
    
    # Get the layer
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            _ = model(inputs)
            features_list.append(features_hook[-1])
            labels_list.append(labels)
            features_hook.clear()
    
    handle.remove()
    
    # Concatenate all batches
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return features, labels

# Extract layer4 features (last conv layer before FC)
features_train, labels_train = extract_features(
    model_finetune, 
    DataLoader(train_dataset, batch_size=32), 
    'layer4'
)

print(f"Extracted features from layer4 (last conv layer)")
print(f"Feature shape: {features_train.shape}")  # (1000, 512, 7, 7)

# Global average pool to get 512-dim vectors
features_pooled = features_train.mean(dim=[2,3])  # (1000, 512)

print(f"After global average pooling: {features_pooled.shape}")
print(f"\nThese 512-dimensional features encode:")
print("  - Low-level: edges, textures (from ImageNet)")
print("  - Mid-level: object parts, shapes (from ImageNet)")
print("  - High-level: bird-specific patterns (adapted during fine-tuning)")
print("\nThe pre-trained features provide strong starting point!")
```

## 5. Related Concepts

Transfer learning connects to multi-task learning, where we train a single model on multiple related tasks simultaneously rather than sequentially. Multi-task learning uses shared representations (early/middle layers) while maintaining task-specific heads (final layers), similar to transfer learning's architecture but with joint training. The shared representation learns features useful across tasks, providing implicit regularization (a feature must help multiple tasks to be retained) that often improves generalization compared to single-task training. Understanding this connection helps appreciate that transfer learning and multi-task learning address similar problems—leveraging shared structure across related tasks—through different training procedures.

The relationship to meta-learning (learning to learn) is more subtle but important. Meta-learning aims to learn an initialization or learning algorithm that enables fast adaptation to new tasks with minimal data. Model-Agnostic Meta-Learning (MAML), for instance, learns an initialization that's a few gradient steps away from good performance on any task from a distribution. Transfer learning can be viewed as a simple form of meta-learning where the "meta-training" is pre-training on the source task and "adaptation" is fine-tuning on the target. More sophisticated meta-learning approaches extend this idea to learn better adaptations or to handle more diverse task distributions.

Transfer learning's success in NLP through pre-trained language models exemplifies domain-specific evolution of the paradigm. Word2Vec and GloVe provided pre-trained word embeddings, transferring lexical knowledge. ELMo provided pre-trained contextual representations. BERT revolutionized the field by pre-training entire Transformer models on massive text corpora through masked language modeling, then fine-tuning for specific tasks. GPT took this further with models so large that fine-tuning isn't always necessary—few-shot learning through prompting can adapt the model without any parameter updates. This progression from transferring embeddings to transferring complete models to avoiding fine-tuning entirely shows how transfer learning evolved as models scaled.

The connection to curriculum learning provides another perspective. Transfer learning can be viewed as a two-stage curriculum: first learn general features (easier task with abundant data), then learn task-specific features (harder task with limited data). This staged approach mirrors how humans learn—general education before specialization—and often works better than jumping directly to the hardest problem. Understanding this connection suggests we might use multi-stage transfer: pre-train on general data (ImageNet), intermediate training on domain-specific data (medical images broadly), then fine-tune on specific task (lung cancer detection). Such staged transfer has proven effective in specialized domains.

Finally, transfer learning connects to the broader question of sample efficiency in machine learning. Deep learning's data hunger—requiring millions of examples—limits applications where data is expensive (medical imaging, rare events) or impossible to collect at scale (private data, unique scenarios). Transfer learning dramatically improves sample efficiency by amortizing the cost of learning general features across many downstream tasks. Understanding transfer learning's sample efficiency provides insights into what makes learning difficult (learning general features requires lots of data) versus easier (learning task-specific mappings given good features needs less data), informing when to expect transfer to help most dramatically.

## 6. Fundamental Papers

**["How transferable are features in deep neural networks?" (2014)](https://arxiv.org/abs/1411.1792)**  
*Authors*: Jason Yosinski, Jeff Clune, Yoshua Bengio, Hector Lipson  
This paper systematically investigated transferability of neural network features across tasks, providing empirical evidence and theoretical understanding of when transfer works. The authors trained networks on ImageNet variants, freezing different numbers of layers when transferring to new tasks, measuring performance degradation versus full fine-tuning. Key findings: early layers learn general features that transfer almost universally; middle layers are more task-specific but still broadly useful; final layers are highly task-specific and benefit most from adaptation. The paper also showed that co-adapted features (features that work well together) can be disrupted by freezing some while training others, suggesting fine-tuning all layers often works better than freezing many. This work established the empirical foundations for transfer learning best practices and demonstrated that feature transferability isn't universal but depends on layer depth and task similarity.

**["Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks" (2014)](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)**  
*Authors*: Maxime Oquab, Leon Bottou, Ivan Laptev, Josef Sivic  
This paper demonstrated that CNN features pre-trained on ImageNet transfer effectively to diverse visual recognition tasks including object detection, scene classification, and fine-grained recognition. The authors showed that simply using pre-trained conv layers as feature extractors and training a classifier on these features achieved strong performance across tasks, outperforming hand-engineered features. The work established transfer learning as practical standard practice in computer vision, showing the features learned on one large dataset (ImageNet) generalize to many other vision tasks. The paper's experimental methodology—systematic evaluation across multiple tasks with controlled comparisons—set standards for demonstrating transfer effectiveness and influenced the widespread adoption of pre-trained models in computer vision.

**["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)](https://arxiv.org/abs/1810.04805)**  
*Authors*: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
While primarily introducing BERT, this paper revolutionized transfer learning in NLP by showing that pre-training Transformers on massive text through masked language modeling, then fine-tuning on specific tasks, achieved state-of-the-art results across eleven diverse NLP tasks including question answering, natural language inference, and named entity recognition. BERT demonstrated the power of transfer learning at scale: a single pre-trained model could be adapted to vastly different tasks with minimal architectural changes (just adding simple task-specific heads). The pre-train-then-fine-tune paradigm became dominant in NLP, showing that transfer learning isn't vision-specific but a general principle applicable across modalities. The paper's impact extended beyond BERT itself to establishing large-scale unsupervised pre-training as a standard first step in NLP model development.

**["A Survey on Transfer Learning" (2010)](https://ieeexplore.ieee.org/document/5288526)**  
*Authors*: Sinno Jialin Pan, Qiang Yang  
This comprehensive survey paper organized and taxonomized transfer learning approaches across machine learning, not just deep learning. Pan and Yang defined: inductive transfer (labeled target data), transductive transfer (no labeled target data), and unsupervised transfer. They analyzed when transfer works (source and target share marginal or conditional distributions) versus fails (large domain shift), providing theoretical frameworks for understanding transferability. While pre-dating the deep learning era's dramatic transfer learning successes, the theoretical foundations remain relevant: understanding transfer as leveraging shared structure between source and target, analyzing domain shift quantitatively, and recognizing that negative transfer (where using source data hurts target performance) can occur when domains are too dissimilar. The survey connects deep transfer learning to broader machine learning traditions, providing theoretical context for why and when transfer is effective.

**["Rethinking ImageNet Pre-training" (2019)](https://arxiv.org/abs/1811.08883)**  
*Authors*: Kaiming He, Ross Girshick, Piotr Dollár  
This paper challenged the conventional wisdom that ImageNet pre-training is always beneficial, showing that for tasks with sufficient data (tens of thousands of images), training from scratch can match or exceed fine-tuning pre-trained models, given enough training time. The key insight was that pre-training's advantage is primarily in faster convergence and better performance with limited data, not in reaching fundamentally better solutions. With abundant task-specific data and proper regularization, random initialization can work well, though requiring much longer training. The paper refined understanding of when transfer helps most: in low-data regimes (hundreds to thousands of examples), pre-training provides massive advantages; in high-data regimes (hundreds of thousands+), advantages diminish. This nuanced view helps practitioners make informed decisions about whether to use pre-trained models or train from scratch based on data availability and computational budget.

## Common Pitfalls and Tricks

The most common mistake is using pre-trained models without matching preprocessing to the pre-training protocol. If a model was pre-trained on ImageNet with specific normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] for RGB channels), fine-tuning or feature extraction must use identical normalization. Mismatched preprocessing causes the model to receive inputs from a different distribution than it was trained on, degrading performance dramatically. Always check the pre-training protocol (input size, normalization statistics, preprocessing steps) and replicate it exactly for transfer.

Using too high a learning rate when fine-tuning destroys pre-trained features before they can adapt to the new task. The pre-trained weights represent useful features; large updates can push them far from this useful region into random territory. A good rule: use 10-100× smaller learning rate for fine-tuning than for training from scratch. If normal training uses lr=0.1, fine-tuning should use lr=0.001-0.01. Even better: use differential learning rates with earlier layers getting smaller rates (they're more general, should change less) and later layers getting larger rates (more task-specific, should adapt more).

Forgetting to set the model to eval mode when extracting features is a subtle bug that causes inconsistent results. If the model contains batch normalization or dropout layers and remains in train mode during feature extraction, these layers behave differently on different calls (batch norm uses batch statistics, dropout drops random units), causing the same input to produce different features. Always call `model.eval()` and use `torch.no_grad()` when extracting features or making predictions.

When fine-tuning NLP models like BERT, a common issue is catastrophic forgetting on short sequences. BERT was pre-trained on sequences of 512 tokens. Fine-tuning on a task with short sequences (tweets, SMS messages of 20-50 tokens) can degrade the model's ability to handle long sequences. If you need to preserve this capability, include long-sequence examples during fine-tuning or use a mixture of source and target data (partial fine-tuning).

A powerful trick for better transfer is using cyclical learning rates during fine-tuning. Start with a low learning rate, gradually increase to a moderate peak, then decrease again. This allows gentle adaptation initially (not destroying pre-trained features), more aggressive updates at the peak (finding task-specific features), and refinement finally (fine-tuning the adapted features). Combined with gradual unfreezing (start by training only the head, then unfreeze top layers, then middle layers), this provides smooth adaptation from pre-trained to task-specific features.

For tasks very different from the pre-training domain, partial fine-tuning often works better than full fine-tuning. Freeze early layers (most general, least likely to need adaptation), fine-tune middle and late layers. This preserves universal low-level features while adapting task-specific higher-level features. The choice of where to freeze involves experimentation but follows the principle: freeze what transfers well, adapt what needs task-specific learning.

## Key Takeaways

Transfer learning leverages pre-trained models to achieve strong performance on target tasks with limited data by transferring learned features from related source tasks with abundant data. The hierarchical nature of deep network features—general low-level features in early layers, task-specific high-level features in late layers—enables selective transfer where we keep useful general features and adapt task-specific components. Feature extraction freezes pre-trained weights and trains only a new task-specific head, working well with very limited data (hundreds of examples) and minimal computational cost. Fine-tuning adapts all or most layers with small learning rates, typically achieving better performance with moderate data (thousands of examples) by specializing pre-trained features to the target domain. Layer-wise differential learning rates control adaptation, with early layers changing minimally (preserving general features) and late layers changing more (learning task-specific features), preventing catastrophic forgetting while enabling effective specialization. Transfer learning's effectiveness depends on source-target similarity—more similar domains enable better transfer—and the quality of pre-training—better source task performance generally improves transfer. Modern practice in computer vision starts with ImageNet pre-training, in NLP with BERT/GPT pre-training, and in speech with Wav2Vec pre-training, making transfer learning a standard first step rather than advanced technique. Understanding transfer learning deeply means recognizing it as amortizing the cost of learning general representations across many tasks, democratizing deep learning by making high performance achievable without massive task-specific datasets, and embodying the principle that good representations learned on one task often help on related tasks—a form of knowledge reuse fundamental to efficient learning.

Transfer learning exemplifies how deep learning matured from requiring massive datasets for every task to enabling strong performance with limited task-specific data through strategic reuse of learned knowledge.

