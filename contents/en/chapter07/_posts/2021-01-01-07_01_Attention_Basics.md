---
layout: post
title: 07-01 Attention Mechanism Fundamentals
chapter: '07'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter07
lesson_type: required
---

## Motivation: The Bottleneck Problem

In seq2seq models (encoder-decoder), the encoder compresses entire input into a fixed-size vector:

```
Input: "The cat sat on the mat" → Encoder → [fixed vector] → Decoder → Output
```

**Problem**: Fixed vector is a bottleneck for long sequences!

**Solution**: **Attention** - Let decoder "look at" all encoder outputs, focusing on relevant parts.

## Attention Intuition

When translating "The cat sat on the mat" to French:
- Translating "chat" → Focus on "cat"
- Translating "assis" → Focus on "sat"
- Translating "tapis" → Focus on "mat"

**Attention computes how much to focus on each input position.**

## Bahdanau Attention (Additive Attention)

**Components**:
- Encoder outputs: $$\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T$$
- Decoder state: $$\mathbf{s}_t$$

**Steps**:

**1. Compute Alignment Scores** (how much attention to pay):

$$e_{t,i} = v_a^T \tanh(\mathbf{W}_a \mathbf{s}_t + \mathbf{U}_a \mathbf{h}_i)$$

**2. Compute Attention Weights** (normalize scores):

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}$$

Note: $$\sum_{i=1}^{T} \alpha_{t,i} = 1$$ (probability distribution)

**3. Compute Context Vector** (weighted sum):

$$\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i} \mathbf{h}_i$$

**4. Combine with Decoder State**:

$$\tilde{\mathbf{s}}_t = \tanh(\mathbf{W}_c [\mathbf{c}_t; \mathbf{s}_t])$$

## Luong Attention (Multiplicative Attention)

**Simpler scoring functions**:

### Dot Product:
$$e_{t,i} = \mathbf{s}_t^T \mathbf{h}_i$$

### General (with weight matrix):
$$e_{t,i} = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_i$$

### Concatenation:
$$e_{t,i} = \mathbf{v}_a^T \tanh(\mathbf{W}_a [\mathbf{s}_t; \mathbf{h}_i])$$

**More efficient** than Bahdanau for dot product version.

## Self-Attention

**Key innovation**: Attention within the same sequence!

**For each position**, compute attention over all positions (including itself).

**Example**: Understanding "it" in "The animal didn't cross the street because it was too tired"
- "it" attends to "animal" (high attention)

### Scaled Dot-Product Attention

Given:
- **Query** $$\mathbf{Q}$$: What I'm looking for
- **Key** $$\mathbf{K}$$: What I have to offer
- **Value** $$\mathbf{V}$$: The actual content

**Formula**:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

where $$d_k$$ is dimension of keys (for numerical stability).

**Steps**:
1. Compute similarity: $$\mathbf{Q}\mathbf{K}^T$$ (query-key dot products)
2. Scale: divide by $$\sqrt{d_k}$$
3. Normalize: softmax to get attention weights
4. Weight values: multiply by $$\mathbf{V}$$

### Implementation

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: queries (seq_len_q, d_k)
    K: keys (seq_len_k, d_k)
    V: values (seq_len_v, d_v)  where seq_len_k = seq_len_v
    mask: optional mask for padding/future positions
    """
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply mask if provided (set masked positions to -inf)
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # Softmax to get attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Weight the values
    output = attention_weights @ V
    
    return output, attention_weights

# Example
seq_len = 4
d_k = 64
d_v = 64

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # (4, 64)
print(f"Attention weights shape: {weights.shape}")  # (4, 4)
print(f"Attention weights sum: {weights.sum(axis=1)}")  # [1, 1, 1, 1]
```

## Multi-Head Attention

**Idea**: Multiple attention "heads" learn different aspects of relationships.

**Example**:
- Head 1: Syntactic relationships
- Head 2: Semantic relationships
- Head 3: Positional relationships

**Formula**:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O$$

where

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

**Parameters**:
- $$\mathbf{W}_i^Q, \mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}$$
- $$\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_v}$$
- $$\mathbf{W}^O \in \mathbb{R}^{hd_v \times d_{model}}$$

Typically: $$d_k = d_v = d_{model} / h$$

### Implementation

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x, batch_size):
        """Split last dimension into (num_heads, d_k)"""
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # Linear projections
        Q = Q @ self.W_q  # (batch, seq_len, d_model)
        K = K @ self.W_k
        V = V @ self.W_v
        
        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention for each head
        attention_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3)  # (batch, seq_len, num_heads, d_k)
        attention_output = attention_output.reshape(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = attention_output @ self.W_o
        
        return output

# Example
mha = MultiHeadAttention(d_model=512, num_heads=8)
batch_size = 2
seq_len = 10
X = np.random.randn(batch_size, seq_len, 512)

output = mha.forward(X, X, X)  # Self-attention
print(f"Output shape: {output.shape}")  # (2, 10, 512)
```

## Types of Attention

### 1. Self-Attention
Q, K, V all from same source

**Use**: Understanding relationships within a sequence

### 2. Cross-Attention (Encoder-Decoder Attention)
Q from decoder, K and V from encoder

**Use**: Machine translation, image captioning

### 3. Masked Attention
Prevent attending to future positions

**Use**: Autoregressive generation (language modeling)

```python
def create_causal_mask(seq_len):
    """Create mask that prevents attending to future positions"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask  # 1 where we should mask (future), 0 where we can attend

mask = create_causal_mask(4)
# [[0, 1, 1, 1],
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]
```

## Attention Visualization

Attention weights can be visualized as heatmaps:

```python
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, source_words, target_words):
    """
    attention_weights: (target_len, source_len)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    
    plt.xticks(range(len(source_words)), source_words, rotation=45)
    plt.yticks(range(len(target_words)), target_words)
    
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.show()
```

## Advantages of Attention

1. **No fixed-length bottleneck**: Can attend to entire input
2. **Long-range dependencies**: Direct connections between any positions
3. **Interpretability**: Can visualize what model focuses on
4. **Parallelization**: Unlike RNNs, can compute all attentions in parallel
5. **Performance**: State-of-the-art results on many tasks

## Summary

- **Attention** allows models to focus on relevant parts of input
- **Scaled dot-product attention**: Core mechanism using Q, K, V
- **Self-attention**: Attention within same sequence
- **Multi-head attention**: Multiple attention heads for different aspects
- **Types**: Self-attention, cross-attention, masked attention
- **Benefits**: Better long-range dependencies, parallelizable, interpretable
- **Foundation** for Transformers (next chapter)

Attention is the key breakthrough that led to modern NLP models like BERT and GPT!

