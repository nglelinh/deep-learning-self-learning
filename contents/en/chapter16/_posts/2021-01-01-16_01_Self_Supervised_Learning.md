---
layout: post
title: 16-01 Self-Supervised Learning
chapter: '16'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter16
lesson_type: required
---

# Self-Supervised Learning: Learning from Data Itself

## 1. Concept Overview

Self-supervised learning represents a paradigm shift in how we leverage unlabeled data, creating supervision signals automatically from the data itself rather than requiring expensive human annotations. The key insight is that data contains inherent structure and relationships that can serve as learning signals: images have spatial structure allowing us to predict one part from others, text has sequential structure enabling prediction of masked words from context, videos have temporal coherence making frame ordering predictable. By formulating prediction tasks that exploit these structures, we can train neural networks on massive unlabeled datasets, learning representations that transfer effectively to downstream supervised tasks.

The distinction between self-supervised and unsupervised learning is subtle but important. Traditional unsupervised learning (clustering, PCA) discovers structure without using it for prediction. Self-supervised learning formulates supervised prediction tasks using automatically generated labels from data structure, essentially converting unsupervised data into supervised learning problems through clever task design. Masked language modeling in BERT—predicting masked words from context—is supervised learning where labels (the original words) come from the data itself rather than human annotators.

Modern self-supervised learning has achieved remarkable success, particularly in NLP where pre-training on massive text through self-supervised objectives (masked language modeling, next sentence prediction) followed by fine-tuning on supervised tasks has become standard. BERT, GPT, and similar models learn rich language understanding from unlabeled text that transfers to diverse tasks. In computer vision, contrastive learning methods like SimCLR and MoCo learn visual representations by distinguishing augmented versions of the same image from different images, achieving representations competitive with supervised pre-training.

Understanding self-supervised learning deeply requires appreciating the pretext tasks—the automatically supervised objectives used during pre-training—and what representations they encourage. Good pretext tasks should be: (1) solvable from data alone without labels, (2) require understanding semantic structure to solve effectively, (3) produce representations useful for downstream tasks. The art is designing tasks where solving them necessitates learning general useful features rather than exploiting data shortcuts.

## 2. Mathematical Foundation

### Contrastive Learning

Contrastive methods learn by distinguishing similar (positive) pairs from dissimilar (negative) pairs. Given anchor $$\mathbf{x}$$, positive $$\mathbf{x}^+$$ (augmented version of same image), and negatives $$\{\mathbf{x}_i^-\}$$ (different images):

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}^+))/\tau)}{\exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}^+))/\tau) + \sum_i \exp(\text{sim}(f(\mathbf{x}), f(\mathbf{x}_i^-))/\tau)}$$

where $$f$$ is encoder network, $$\text{sim}$$ is similarity (typically cosine), $$\tau$$ is temperature. This NT-Xent (normalized temperature-scaled cross-entropy) loss encourages representations where augmentations of same image are close while different images are far.

### Masked Prediction

BERT masks 15% of tokens and predicts them from context:

$$\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{\mathbf{x}}\left[\sum_{i \in \mathcal{M}} \log p(x_i | \mathbf{x}_{\backslash \mathcal{M}})\right]$$

where $$\mathcal{M}$$ is set of masked positions. The model must use bidirectional context to predict masked words, learning deep language understanding.

## 3. Example / Intuition

Consider learning visual representations from unlabeled images. Take a photo of a cat. Create two augmented versions: one with random cropping and color jittering, another with different cropping and rotation. These are positive pairs—different views of the same cat, should have similar representations.

Sample photos of dogs, cars, trees as negatives. Contrastive learning pushes cat augmentations together in representation space while pushing cat apart from dogs/cars/trees. After training on millions of images, representations cluster by semantic content: all cats nearby, all dogs nearby, cats and dogs closer to each other (both animals) than to cars.

These learned representations transfer to classification—even though we never used labels during pre-training, the encoder learned to extract features distinguishing objects, which directly helps supervised classification with limited labels.

## 4. Code Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """Simplified SimCLR for contrastive learning"""
    
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        
        # Encoder (e.g., ResNet)
        self.encoder = base_encoder
        
        # Projection head
        # Maps encoder output to space where contrastive loss is computed
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)  # L2 normalize

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    NT-Xent loss for contrastive learning
    
    z1, z2: (batch, dim) representations of augmented pairs
    """
    batch_size = z1.size(0)
    
    # Concatenate augmentations
    z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.T) / temperature  # (2*batch, 2*batch)
    
    # Mask out self-similarity
    mask = ~torch.eye(2*batch_size, dtype=torch.bool)
    sim_matrix = sim_matrix[mask].view(2*batch_size, -1)
    
    # Positive pairs: (i, i+batch) and (i+batch, i)
    pos_sim = torch.cat([
        sim_matrix[range(batch_size), range(batch_size, 2*batch_size)],
        sim_matrix[range(batch_size, 2*batch_size), range(batch_size)]
    ])
    
    # NT-Xent loss
    loss = -pos_sim + torch.log(sim_matrix.exp().sum(dim=1))
    
    return loss.mean()

print("Self-supervised learning enables learning from unlabeled data!")
```

## 5. Related Concepts

Self-supervised learning connects to transfer learning as pre-training strategy. Instead of ImageNet supervised pre-training, use self-supervised pre-training on unlabeled images, often learning representations that transfer even better.

## 6. Fundamental Papers

**["Momentum Contrast for Unsupervised Visual Representation Learning" (2020)](https://arxiv.org/abs/1911.05722)**  
*Authors*: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick  
MoCo maintained large dictionary of negative examples through momentum encoder, enabling effective contrastive learning. Achieved representations competitive with supervised pre-training.

**["A Simple Framework for Contrastive Learning of Visual Representations" (2020)](https://arxiv.org/abs/2002.05709)**  
*Authors*: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton  
SimCLR showed that simple contrastive learning with strong augmentations and large batches achieves excellent representations, outperforming many supervised pre-training approaches.

**["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)](https://arxiv.org/abs/1810.04805)**  
BERT's masked language modeling is self-supervised learning, creating supervision from text itself.

## Key Takeaways

Self-supervised learning creates supervision automatically from data structure, enabling learning from massive unlabeled datasets through pretext tasks like masked prediction or contrastive learning. Contrastive methods learn by distinguishing augmented views of same example from different examples, learning invariances to augmentations while capturing semantic content. Masked prediction tasks like BERT's MLM learn to predict missing parts from context, requiring deep understanding of data structure. Self-supervised pre-training often matches or exceeds supervised pre-training for transfer learning, demonstrating that task-agnostic learning from unlabeled data produces versatile representations.

