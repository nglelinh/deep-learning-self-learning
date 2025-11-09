---
layout: post
title: 07-01 Attention Mechanisms
chapter: '07'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter07
lesson_type: required
---

# Attention Mechanisms: Learning Where to Focus

![Attention Mechanism Visualization](https://raw.githubusercontent.com/tensorflow/nmt/master/nmt/g3doc/img/attention_mechanism.jpg)
*Hình ảnh: Minh họa cơ chế Attention trong dịch máy. Nguồn: TensorFlow NMT*

## 1. Concept Overview

Attention mechanisms represent one of the most transformative ideas in modern deep learning, fundamentally changing how neural networks process sequential and structured data. The core insight is deceptively simple yet profound: not all parts of the input are equally relevant for making a particular prediction, and the network should learn to focus on the most relevant parts dynamically based on context. This idea, first introduced to address limitations in sequence-to-sequence models for machine translation, has become so central to deep learning that it forms the foundation of Transformers, the dominant architecture in natural language processing and increasingly in computer vision and other domains.

To understand why attention emerged and why it matters so profoundly, we must first appreciate the bottleneck problem it solves. In early encoder-decoder architectures for tasks like machine translation, the encoder processes the entire source sentence into a single fixed-size vector, which the decoder then uses to generate the translation. This fixed-size vector—regardless of whether the source is five words or fifty—must encode all information about the source that might be relevant for generating the target. This is an extreme information bottleneck. Moreover, it violates an intuitive principle of translation: when generating each target word, we should focus primarily on the corresponding source words, not treat all source words equally.

The attention mechanism solves this by allowing the decoder to directly access all encoder hidden states, not just the final one, and to compute a weighted average of these states where the weights reflect relevance to the current decoding step. When translating "The black cat" to French, when generating "noir" (black), the attention mechanism learns to focus heavily on "black" in the source, largely ignoring "the" and "cat." This dynamic, learned focusing ability eliminates the fixed-size bottleneck and provides interpretability—we can visualize attention weights to see what the model is focusing on, making the translation process more transparent.

The mathematical elegance of attention lies in its generality. At its core, attention is a mechanism for computing weighted averages where the weights are determined by relevance or similarity, typically measured through learned functions. This abstraction applies far beyond machine translation. In image captioning, attention can focus on different image regions when generating different caption words. In reading comprehension, attention can focus on relevant passages when answering questions. In self-attention (used in Transformers), positions in a sequence can attend to each other to build contextualized representations. The same mathematical framework—queries, keys, values, and similarity-based weighting—works across all these domains.

What makes attention particularly powerful is that it's differentiable and can be trained end-to-end with backpropagation. The network learns what to attend to purely from the training objective, without explicit supervision about which source words correspond to which target words in translation, or which image regions correspond to which caption words. This learned attention often discovers alignments and relationships that match human intuitions, providing both performance gains and interpretability. The attention weights—visualizable as heatmaps showing which inputs the model focused on for each output—give us unprecedented insight into neural network decision-making.

The evolution from simple attention in encoder-decoder models to self-attention in Transformers represents a conceptual leap. In encoder-decoder attention, the decoder (generating output) attends to the encoder (representing input)—a one-way relationship from source to target. Self-attention allows elements within the same sequence to attend to each other bidirectionally. Every word in a sentence can look at every other word to build its representation. This enables capturing complex linguistic relationships like coreference ("The cat" and "it" referring to the same entity), syntactic dependencies, and semantic relationships, all learned automatically from data. Self-attention's power comes from enabling direct communication between all positions in a sequence, creating $$O(1)$$ path lengths for information flow compared to $$O(n)$$ in RNNs where information must traverse the sequence linearly.

## 2. Mathematical Foundation

To understand attention deeply, we must build up from first principles, starting with the simplest formulation and progressing to the sophisticated multi-head self-attention used in Transformers. The core idea is to compute outputs as weighted combinations of values, where weights are determined by the relevance of each value to the current query.

### The General Attention Framework

Suppose we have a query $$\mathbf{q}$$ representing what we're looking for, a set of keys $$\{\mathbf{k}_1, \ldots, \mathbf{k}_n\}$$ representing what each input offers, and corresponding values $$\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$$ representing the actual content. The attention mechanism computes:

1. **Similarity scores**: $$e_i = \text{score}(\mathbf{q}, \mathbf{k}_i)$$ measuring relevance
2. **Attention weights**: $$\alpha_i = \frac{\exp(e_i)}{\sum_{j=1}^n \exp(e_j)}$$ (softmax normalization)
3. **Context vector**: $$\mathbf{c} = \sum_{i=1}^n \alpha_i \mathbf{v}_i$$ (weighted sum)

The choice of similarity function $$\text{score}(\mathbf{q}, \mathbf{k}_i)$$ gives rise to different attention variants. Let's examine the most important ones and understand why each design choice matters.

### Additive (Bahdanau) Attention

The first widely successful attention mechanism, introduced for neural machine translation, uses:

$$\text{score}(\mathbf{q}, \mathbf{k}_i) = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{q} + \mathbf{U}_a \mathbf{k}_i)$$

This additive formulation has several properties worth examining. The query and key are first projected through learned weight matrices $$\mathbf{W}_a$$ and $$\mathbf{U}_a$$, allowing the model to learn what aspects of the query and key are relevant for computing similarity. These projections happen in the same space (both result in vectors of the same dimension), which are then added and passed through $$\tanh$$ nonlinearity. The $$\tanh$$ serves dual purposes: it provides nonlinearity (allowing the similarity function to capture complex relationships) and it bounds the pre-softmax scores (preventing extremely large values that would cause softmax saturation).

The final projection through $$\mathbf{v}_a^T$$ (a learned vector) combines the features from the tanh layer into a single scalar score. This vector $$\mathbf{v}_a$$ can be seen as learning which feature combinations indicate high relevance. The entire similarity function is learned from data through backpropagation—initially random, it adapts to discover what constitutes relevance for the specific task.

### Multiplicative (Luong) Attention

A simpler formulation that often works as well or better:

$$\text{score}(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^T \mathbf{W}_a \mathbf{k}_i$$

or even more simply (dot-product attention):

$$\text{score}(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^T \mathbf{k}_i$$

The dot product measures similarity through alignment—high dot product means the query and key point in similar directions in the representation space. This is computationally efficient (just vector dot products) and works well when queries and keys are already in the same semantic space. The learned matrix $$\mathbf{W}_a$$ in the general form allows transforming the key before comparison, giving more flexibility.

The dot product formulation becomes particularly elegant when we consider batch processing. With query matrix $$\mathbf{Q} \in \mathbb{R}^{m \times d}$$ ($$m$$ queries) and key matrix $$\mathbf{K} \in \mathbb{R}^{n \times d}$$ ($$n$$ keys):

$$\mathbf{E} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{m \times n}$$

This single matrix multiplication computes all $$m \times n$$ pairwise similarities simultaneously, perfectly suited for GPU parallelization. This efficiency is one reason dot-product attention became standard in Transformers.

### Scaled Dot-Product Attention

The Transformer uses dot-product attention with a crucial modification—scaling:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

where $$d_k$$ is the dimension of keys (and queries). Why this scaling? Consider what happens as $$d_k$$ grows. If query and key components are independent random variables with mean 0 and variance 1, their dot product has variance $$d_k$$. For $$d_k = 512$$ (common in Transformers), unnormalized dot products have standard deviation $$\sqrt{512} \approx 22.6$$. After softmax, such large magnitudes cause one probability to dominate (approaching 1) with others near 0, creating sharp attention distributions where gradients vanish.

Dividing by $$\sqrt{d_k}$$ normalizes the variance back to 1, regardless of dimension. This keeps dot products in a range where softmax has meaningful gradients, allowing the model to learn nuanced attention distributions—attending to multiple positions with varying weights rather than hard selection of a single position. This seemingly minor scaling factor was crucial to making Transformer attention work for large model dimensions.

### Self-Attention: The Key Innovation

Self-attention applies attention within a single sequence, allowing elements to attend to each other. For a sequence $$\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_n]$$, we create queries, keys, and values through learned projections:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

Each position becomes a query asking "what context is relevant to me?" Its query is compared against all positions' keys (including its own), producing attention weights that determine how much to incorporate from each position's value. The output for position $$i$$ is:

$$\mathbf{o}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j, \quad \text{where} \quad \alpha_{ij} = \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k})}{\sum_{j'=1}^n \exp(\mathbf{q}_i^T \mathbf{k}_{j'} / \sqrt{d_k})}$$

This creates an $$n \times n$$ matrix of interactions between all position pairs. Position $$i$$ can directly attend to position $$j$$ regardless of how far apart they are in the sequence. This direct connectivity is what enables capturing long-range dependencies without the vanishing gradient issues of RNNs—information flows directly from position $$j$$ to position $$i$$ in one step, not through $$|i-j|$$ sequential transformations.

### Multi-Head Attention: Multiple Perspectives

A single attention mechanism can learn one type of relationship, but natural language (and many other structured domains) involves multiple simultaneous relationship types. Multi-head attention addresses this by computing attention multiple times in parallel with different learned projections:

$$\text{head}_h = \text{Attention}(\mathbf{X}\mathbf{W}_h^Q, \mathbf{X}\mathbf{W}_h^K, \mathbf{X}\mathbf{W}_h^V)$$

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}^O$$

Each head has its own projection matrices $$\mathbf{W}_h^Q, \mathbf{W}_h^K, \mathbf{W}_h^V$$, allowing it to learn different attention patterns. Empirically, different heads specialize: one might learn to attend to syntactically related words, another to semantically similar words, another to simply nearby words. The network discovers these specializations automatically through backpropagation—we don't specify what each head should do, we merely provide the capacity for specialization through separate parameters.

The typical configuration uses 8 heads with $$d_k = d_v = d_{\text{model}}/8$$, meaning each head operates in a lower-dimensional space (64 dimensions if $$d_{\text{model}} = 512$$). This design choice maintains the same total computational cost as single-head full-dimension attention while providing representational advantages of multiple perspectives. The final concatenation and projection through $$\mathbf{W}^O$$ integrates information from all heads, allowing them to collaborate rather than operating independently.

### Masking in Attention

For many applications, we need to prevent attention to certain positions. Two types of masking are crucial:

**Padding mask**: When sequences in a batch have different lengths, we pad shorter sequences. The mask prevents attending to padding tokens:

$$\text{mask}_{ij} = \begin{cases} 1 & \text{if position } j \text{ is real content} \\ 0 & \text{if position } j \text{ is padding} \end{cases}$$

**Causal (look-ahead) mask**: For autoregressive generation, position $$i$$ cannot attend to positions $$j > i$$ (future positions):

$$\text{mask}_{ij} = \begin{cases} 1 & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}$$

These masks are applied before softmax by setting masked positions to $$-\infty$$:

$$\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)$$

where $$\mathbf{M}_{ij} = 0$$ if allowed, $$-\infty$$ if masked. Since $$\exp(-\infty) = 0$$, masked positions receive zero attention weight after softmax.

## 3. Example / Intuition

To build genuine intuition for how attention works, let's trace through the mechanics of translating the sentence "I love deep learning" to French "J'aime l'apprentissage profond" using encoder-decoder attention.

The encoder processes the English sentence, producing hidden states $$\mathbf{h}_1^{enc}, \ldots, \mathbf{h}_4^{enc}$$ for the four words. These hidden states contain rich representations of each word in context—"deep" is represented not in isolation but as modifying "learning."

Now the decoder begins generating the French translation. At the first step, it must generate "J'" (I). The decoder has its own hidden state $$\mathbf{s}_1$$ representing "about to generate the first word." This becomes the query: "What source information is relevant for generating the first word?" The encoder hidden states serve as keys and values.

The attention mechanism computes similarities between the query $$\mathbf{s}_1$$ and each encoder hidden state:

$$e_{1,1} = \mathbf{s}_1^T \mathbf{W}_a \mathbf{h}_1^{enc}$$ (similarity to "I")
$$e_{1,2} = \mathbf{s}_1^T \mathbf{W}_a \mathbf{h}_2^{enc}$$ (similarity to "love")
$$e_{1,3} = \mathbf{s}_1^T \mathbf{W}_a \mathbf{h}_3^{enc}$$ (similarity to "deep")
$$e_{1,4} = \mathbf{s}_1^T \mathbf{W}_a \mathbf{h}_4^{enc}$$ (similarity to "learning")

Suppose these scores are $$[2.5, 0.8, 0.3, 0.2]$$. After softmax normalization, we get attention weights approximately $$[0.77, 0.14, 0.05, 0.04]$$. The model has learned to focus primarily on "I" (0.77) when generating "J'", which aligns perfectly with the translation.

The context vector is then:

$$\mathbf{c}_1 = 0.77 \cdot \mathbf{h}_1^{enc} + 0.14 \cdot \mathbf{h}_2^{enc} + 0.05 \cdot \mathbf{h}_3^{enc} + 0.04 \cdot \mathbf{h}_4^{enc}$$

This context is heavily influenced by the representation of "I" but includes minor contributions from other words, capturing that even when translating "I", other context matters (formal vs informal, sentence structure, etc.). The decoder then combines this context with its own state to generate "J'".

Moving to the third French word "l'apprentissage" (learning), the decoder state $$\mathbf{s}_3$$ now asks: "What's relevant for generating this word?" The attention mechanism might produce weights $$[0.03, 0.05, 0.12, 0.80]$$, focusing primarily on "learning" (0.80) with some attention to "deep" (0.12) since they form a compound in English. The context vector:

$$\mathbf{c}_3 = 0.03 \cdot \mathbf{h}_1^{enc} + 0.05 \cdot \mathbf{h}_2^{enc} + 0.12 \cdot \mathbf{h}_3^{enc} + 0.80 \cdot \mathbf{h}_4^{enc}$$

This adaptive focus on different source parts for different target words is attention's power—it solves the alignment problem in translation without explicit alignment annotations.

Now consider self-attention within the English sentence itself. When building a representation for "learning," we compute its similarity to all words including itself:

- "learning" ↔ "I": Low similarity (different parts of speech, distant semantically)
- "learning" ↔ "love": Medium (syntactically related—"love" governs "learning")
- "learning" ↔ "deep": High (forms compound noun phrase "deep learning")
- "learning" ↔ "learning": High (self-similarity)

After softmax, suppose we get weights $$[0.08, 0.17, 0.40, 0.35]$$. The contextualized representation becomes:

$$\mathbf{o}_{\text{learning}} = 0.08 \mathbf{v}_I + 0.17 \mathbf{v}_{\text{love}} + 0.40 \mathbf{v}_{\text{deep}} + 0.35 \mathbf{v}_{\text{learning}}$$

This representation now encodes not just "learning" in isolation but "deep learning" as a concept, incorporating information from its modifier. This is how self-attention builds contextualized representations—every word's representation becomes a function of its relationships to all other words.

The beauty of multi-head attention is that different heads can capture different aspects simultaneously. Head 1 might focus on syntactic dependencies (subject-verb, adjective-noun). Head 2 might focus on semantic relationships (coreference, similarity). Head 3 might use positional patterns (attending to adjacent words). Each head uses different projection matrices $$\mathbf{W}_h^Q, \mathbf{W}_h^K, \mathbf{W}_h^V$$, allowing it to implement a different similarity function and thus discover different relationships.

Mathematically, with $$H$$ heads and $$d_k = d_v = d_{\text{model}}/H$$:

$$\text{head}_h = \text{Attention}(\mathbf{X}\mathbf{W}_h^Q, \mathbf{X}\mathbf{W}_h^K, \mathbf{X}\mathbf{W}_h^V)$$

where $$\mathbf{W}_h^Q, \mathbf{W}_h^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$$ and $$\mathbf{W}_h^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$. Each head produces output of dimension $$d_v$$, and concatenating $$H$$ heads gives dimension $$H \cdot d_v = d_{\text{model}}$$, which is then projected:

$$\text{MultiHead}(\mathbf{X}) = [\text{head}_1; \ldots; \text{head}_H] \mathbf{W}^O$$

where $$\mathbf{W}^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$. This final projection integrates information from all heads, allowing them to collaborate. Without it, heads would be completely independent, potentially learning redundant patterns.

## 4. Code Snippet

Let's implement attention mechanisms from scratch to understand every detail:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention:
    """
    Additive (Bahdanau) attention for encoder-decoder models.
    
    This was the first successful attention mechanism for neural machine
    translation. While more complex than dot-product attention, it's
    instructive for understanding the general attention framework.
    """
    
    def __init__(self, hidden_dim):
        """
        hidden_dim: dimension of encoder/decoder hidden states
        
        We'll learn to project both encoder states and decoder state
        into a common space where we measure similarity.
        """
        self.hidden_dim = hidden_dim
        
        # Projection matrices
        # Wa projects encoder hidden states
        # Ua projects decoder state (query)
        self.Wa = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Ua = np.random.randn(hidden_dim, hidden_dim) * 0.01
        
        # va projects combined features to scalar score
        self.va = np.random.randn(hidden_dim, 1) * 0.01
    
    def compute_attention(self, decoder_state, encoder_states):
        """
        Compute attention weights and context vector.
        
        decoder_state: (hidden_dim, 1) - current decoder state (query)
        encoder_states: list of (hidden_dim, 1) - all encoder states (keys/values)
        
        Returns:
            context: weighted average of encoder states
            attention_weights: what we attended to (for visualization)
        """
        scores = []
        
        # Compute score for each encoder state
        for h_enc in encoder_states:
            # Additive scoring: v^T tanh(Wa*h_enc + Ua*s_dec)
            # This is more flexible than dot product but more expensive
            projected_enc = self.Wa @ h_enc
            projected_dec = self.Ua @ decoder_state
            
            # Add and pass through tanh
            combined = np.tanh(projected_enc + projected_dec)
            
            # Project to scalar score
            score = (self.va.T @ combined).item()
            scores.append(score)
        
        # Softmax to get attention weights
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Compute weighted average (context vector)
        context = np.zeros_like(encoder_states[0])
        for weight, h_enc in zip(attention_weights, encoder_states):
            context += weight * h_enc
        
        return context, attention_weights

# Demonstrate attention on simple translation example
print("="*70)
print("Bahdanau Attention Example: Neural Machine Translation")
print("="*70)

# Simulate encoder states for "I love deep learning" (4 words)
# In practice, these come from running encoder RNN
np.random.seed(42)
encoder_states = [np.random.randn(8, 1) for _ in range(4)]
source_words = ["I", "love", "deep", "learning"]

# Create attention mechanism
attention = BahdanauAttention(hidden_dim=8)

# Simulate decoder states when generating "J'aime l'apprentissage profond"
target_words = ["J'", "aime", "l'apprentissage", "profond"]

print("Attention weights when generating each French word:\n")
print(f"{'Target':<20} | {'Attention to source words':<50}")
print("-" * 72)

for target_word in target_words:
    # Simulate decoder state for this target word
    decoder_state = np.random.randn(8, 1)
    
    # Compute attention
    context, weights = attention.compute_attention(decoder_state, encoder_states)
    
    # Display attention distribution
    weight_str = " ".join([f"{src}:{w:.2f}" for src, w in zip(source_words, weights)])
    print(f"{target_word:<20} | {weight_str}")

print("\nThe model learns to focus on relevant source words for each target word!")
print("In a trained model, these alignments would be much sharper.\n")
```

Now implement the Transformer's scaled dot-product attention:

```python
class ScaledDotProductAttention:
    """
    Scaled dot-product attention as used in Transformers.
    
    This is simpler and more efficient than additive attention,
    while being equally or more effective. The scaling by √d_k
    is critical for training stability.
    """
    
    def __init__(self, d_k):
        self.d_k = d_k
        self.scale = np.sqrt(d_k)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q: queries (seq_len_q, d_k)
        K: keys (seq_len_k, d_k)
        V: values (seq_len_v, d_v) where seq_len_k == seq_len_v
        mask: optional (seq_len_q, seq_len_k), 1 where allowed, 0 where masked
        
        Returns:
            output: (seq_len_q, d_v)
            attention_weights: (seq_len_q, seq_len_k)
        """
        # Compute attention scores: Q·K^T / √d_k
        # This is a matrix of all pairwise similarities
        scores = Q @ K.T / self.scale  # (seq_len_q, seq_len_k)
        
        # Apply mask if provided
        if mask is not None:
            # Set masked positions to very negative (will be ~0 after softmax)
            scores = np.where(mask == 1, scores, -1e9)
        
        # Softmax along key dimension (each query gets probability distribution over keys)
        attention_weights = self._softmax(scores, axis=-1)
        
        # Weighted sum of values
        output = attention_weights @ V  # (seq_len_q, d_v)
        
        return output, attention_weights
    
    def _softmax(self, x, axis=-1):
        """Numerically stable softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Demonstrate self-attention
print("="*70)
print("Self-Attention Example: Building Contextualized Representations")
print("="*70)

# Simulated word embeddings for "The cat sat"
# In practice, these come from an embedding layer
words = ["The", "cat", "sat"]
d_model = 16
embeddings = np.random.randn(len(words), d_model)

print("Creating contextualized representations using self-attention...\n")

# Create Q, K, V through linear projections (simplified: use same embeddings)
# In Transformers, these would be learned projections
d_k = d_v = d_model
Q = embeddings  # Each word queries for relevant context
K = embeddings  # Each word offers its representation as key
V = embeddings  # Each word provides its content as value

# Apply attention
attention_mechanism = ScaledDotProductAttention(d_k)
contextualized, attn_weights = attention_mechanism.forward(Q, K, V)

print("Attention weights (each row shows what that word attended to):")
print(f"{'Word':<6} | {'The':<8} {'cat':<8} {'sat':<8}")
print("-" * 35)
for i, word in enumerate(words):
    weights_str = " ".join([f"{w:.3f}" for w in attn_weights[i]])
    print(f"{word:<6} | {weights_str}")

print("\nContextualized representations incorporate information from attended words.")
print("'cat' representation now includes context from 'The' and 'sat'.")
```

Complete PyTorch multi-head attention implementation:

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as used in Transformers.
    
    This implementation shows all details: splitting into heads,
    computing attention for each head in parallel, and recombining.
    Modern frameworks optimize this heavily, but understanding the
    mechanics is crucial.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V (for all heads combined)
        # Why combined? GPU efficiency - single matrix multiply vs H separate ones
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection to integrate heads
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scale for dot-product attention
        self.scale = np.sqrt(self.d_k)
    
    def split_heads(self, x, batch_size):
        """
        Split last dimension into (num_heads, d_k).
        
        Input:  (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        
        This reshaping allows each head to operate independently
        on its d_k dimensional subspace.
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
    
    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq_len, d_model)
        mask: optional (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
        
        For self-attention: query = key = value = input sequence
        For encoder-decoder: query = decoder, key = value = encoder output
        """
        batch_size = query.size(0)
        
        # Linear projections in batch
        # Each word gets projected to Q, K, V spaces
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        # Each head works with d_k dimensions
        Q = self.split_heads(Q, batch_size)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention for all heads in parallel
        # Q·K^T gives (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        # Each query position gets probability distribution over key positions
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, d_k)
        
        # Recombine heads
        # Transpose and reshape to (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attn_weights

# Demonstrate multi-head attention
print("="*70)
print("Multi-Head Self-Attention Example")
print("="*70)

batch_size = 2
seq_len = 6
d_model = 64
num_heads = 8

# Create multi-head attention
mha = MultiHeadAttention(d_model, num_heads)

# Random input (simulating embedded sequence)
x = torch.randn(batch_size, seq_len, d_model)

# Self-attention: query = key = value = x
output, attention_weights = mha(x, x, x)

print(f"Input shape: {x.shape}")  # (2, 6, 64)
print(f"Output shape: {output.shape}")  # (2, 6, 64) - same as input
print(f"Attention weights shape: {attention_weights.shape}")  # (2, 8, 6, 6)
print(f"\nAttention weights: (batch, num_heads, seq_len_q, seq_len_k)")
print(f"  - 2 batches")
print(f"  - 8 heads (each learns different attention pattern)")
print(f"  - 6×6 attention matrix (each query attends to all keys)")

# Visualize attention pattern for first batch, first head
print(f"\nAttention pattern (batch 0, head 0):")
attn_matrix = attention_weights[0, 0].detach().numpy()
print("Each row shows what that position attends to:")
print(attn_matrix.round(3))
print(f"\nEach row sums to 1.0: {attn_matrix.sum(axis=1)}")

# Demonstrate masked attention (causal mask for autoregressive)
print("\n" + "="*70)
print("Masked Self-Attention (Causal Mask for Language Modeling)")
print("="*70)

def create_causal_mask(seq_len):
    """
    Create lower triangular mask: position i can only attend to positions ≤ i
    
    This prevents information leakage from future tokens during training
    of autoregressive models like GPT.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1 where allowed, 0 where masked

causal_mask = create_causal_mask(seq_len)
print("Causal mask (1 = allowed, 0 = masked future):")
print(causal_mask.numpy().astype(int))

# Apply masked attention
output_masked, attn_masked = mha(x, x, x, mask=causal_mask.unsqueeze(0).unsqueeze(0))

print(f"\nMasked attention weights (batch 0, head 0):")
attn_masked_matrix = attn_masked[0, 0].detach().numpy()
print(attn_masked_matrix.round(3))
print("\nNotice: Future positions (upper triangle) have zero attention weight!")
print("Each position only attends to current and previous positions.")
```

Complete example showing attention in sequence-to-sequence:

```python
class Seq2SeqWithAttention(nn.Module):
    """
    Sequence-to-sequence model with attention mechanism.
    
    Demonstrates encoder-decoder attention (decoder attending to encoder)
    which is different from self-attention. This was the original
    use case for attention mechanisms.
    """
    
    def __init__(self, input_vocab_size, output_vocab_size, 
                 embedding_dim=64, hidden_dim=128):
        super().__init__()
        
        # Encoder: embedding + LSTM
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Decoder: embedding + LSTM + attention + output
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, 
                                    batch_first=True)
        
        # Attention mechanism (decoder queries encoder)
        self.attention = MultiHeadAttention(hidden_dim, num_heads=4)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_vocab_size)
    
    def encode(self, src):
        """
        Encode source sequence.
        src: (batch, src_len) token indices
        
        Returns all encoder hidden states for attention
        """
        embedded = self.encoder_embedding(src)
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(embedded)
        
        return encoder_outputs, (h_n, c_n)
    
    def decode_step(self, tgt_token, decoder_state, encoder_outputs):
        """
        One decoder step with attention.
        
        tgt_token: (batch, 1) current target token
        decoder_state: (h, c) decoder LSTM state
        encoder_outputs: (batch, src_len, hidden_dim) to attend to
        """
        # Embed target token
        embedded = self.decoder_embedding(tgt_token)  # (batch, 1, embedding_dim)
        
        # Compute attention over encoder outputs
        # Query: current decoder state (use h from LSTM state)
        # Keys/Values: encoder outputs
        query = decoder_state[0].transpose(0, 1)  # (batch, 1, hidden_dim)
        context, attn_weights = self.attention(query, encoder_outputs, 
                                               encoder_outputs)
        
        # Combine embedded input with context from attention
        lstm_input = torch.cat([embedded, context], dim=-1)
        
        # Decoder LSTM step
        output, new_state = self.decoder_lstm(lstm_input, decoder_state)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits, new_state, attn_weights
    
    def forward(self, src, tgt):
        """
        Full forward pass for training.
        Teacher forcing: use true target tokens as decoder inputs
        """
        # Encode source
        encoder_outputs, encoder_state = self.encode(src)
        
        # Initialize decoder state with encoder's final state
        decoder_state = encoder_state
        
        # Decode target sequence (teacher forcing)
        batch_size, tgt_len = tgt.size()
        vocab_size = self.output_proj.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size)
        
        for t in range(tgt_len):
            # Use true target token at this step (teacher forcing)
            tgt_token = tgt[:, t:t+1]
            
            # Decoder step with attention
            logits, decoder_state, attn = self.decode_step(
                tgt_token, decoder_state, encoder_outputs)
            
            outputs[:, t:t+1, :] = logits
        
        return outputs

# Example usage
print("\n" + "="*70)
print("Sequence-to-Sequence with Attention")
print("="*70)

src_vocab = 100
tgt_vocab = 120
model = Seq2SeqWithAttention(src_vocab, tgt_vocab)

# Simulate translation: source [12, 34, 56, 78] → target [23, 45, 67]
src = torch.randint(0, src_vocab, (2, 4))  # Batch of 2, length 4
tgt = torch.randint(0, tgt_vocab, (2, 3))  # Batch of 2, length 3

output = model(src, tgt)
print(f"Source shape: {src.shape}")  # (2, 4)
print(f"Target shape: {tgt.shape}")  # (2, 3)
print(f"Output shape: {output.shape}")  # (2, 3, 120)
print("\nModel uses attention to focus on relevant source words for each target word!")
```

## 5. Related Concepts

The relationship between attention mechanisms and human cognitive attention provides useful but imperfect analogies. When humans read, we don't process all words equally—we focus on content-bearing words, skim over function words, and our eye movements reflect this selective attention. When listening, we focus on the speaker while filtering out background noise. Attention in neural networks captures this principle of selective processing, though the mechanism is quite different from biological attention. The neural attention is soft (weights sum to 1 rather than hard selection) and learned (discovered through backpropagation rather than evolved through biology). These differences matter: soft attention allows gradient flow through all paths (essential for learning), while hard attention would require reinforcement learning to train.

The connection between attention and memory systems is profound. In computer architecture, attention is analogous to content-addressable memory: we query memory based on content similarity rather than fixed addresses. The keys serve as memory addresses, values as memory content, and queries specify what we're looking for. This parallel extends to database systems where queries select relevant records based on key matching. Understanding attention through this lens helps clarify why the query-key-value decomposition is natural: it mirrors how we retrieve information from any indexed collection.

Attention mechanisms have interesting relationships to traditional machine learning techniques. The attention weights, computed through softmax over similarities, resemble kernel methods in classical machine learning where we compute weighted combinations based on similarity kernels. The difference is that attention learns the similarity function (through the query and key projections) rather than using a fixed kernel like RBF. This learned similarity is more flexible, adapting to task-specific notions of relevance. Understanding this connection helps appreciate attention as part of a longer tradition of similarity-based learning, not an isolated invention.

The evolution from basic attention to self-attention to multi-head self-attention shows progressive generalization. Basic encoder-decoder attention allows decoder to query encoder—a one-directional relationship. Self-attention allows all positions to query each other—any element can attend to any other. Multi-head self-attention computes multiple independent attention patterns—different heads can specialize in different relationship types. This progression from specific to general made attention increasingly powerful and versatile, ultimately enabling its use as the sole mechanism for sequence processing in Transformers.

Attention's relationship to convolutional operations provides another perspective. Standard convolution uses fixed, learned kernels applied uniformly across the input. Attention can be viewed as dynamic, input-dependent convolution where the kernel (attention weights) changes based on content. A 1×1 convolution in CNNs is nearly equivalent to attention with query equal to keys (all positions attend equally), while attention with learned queries allows focus to vary by position and context. This connection helps understand why Vision Transformers work—attention generalizes convolution's ability to process spatial structure while adding dynamic, context-dependent weighting.

Finally, attention connects to the broader theme of routing information in neural networks. Skip connections in ResNets route information around layers. Gating in LSTMs routes information through or around the cell state update. Attention routes information from source positions to target positions with learned weights. This routing perspective suggests attention is part of a general pattern: neural networks need mechanisms to selectively pass information through different paths based on content, and learned gating (whether through attention weights, LSTM gates, or other mechanisms) is the standard solution. Understanding this pattern helps recognize when attention-like mechanisms might be useful in novel architectures.

## 6. Fundamental Papers

**["Neural Machine Translation by Jointly Learning to Align and Translate" (2015)](https://arxiv.org/abs/1409.0473)**  
*Authors*: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio  
This is THE paper that introduced attention mechanisms to neural networks and transformed sequence-to-sequence learning. Bahdanau and colleagues identified the bottleneck problem in encoder-decoder models—compressing the entire source sentence into a single fixed-size vector—and proposed attention as the solution. Their key insight was to let the decoder access all encoder hidden states and learn to weight them based on relevance to the current decoding step. The paper demonstrated dramatic improvements in machine translation, particularly for longer sentences where the bottleneck was most severe. What makes this paper historically significant is not just the performance gains but the general principle it established: neural networks can learn to selectively focus on relevant information through differentiable mechanisms. This principle has been applied far beyond translation to attention mechanisms in image captioning, reading comprehension, speech recognition, and ultimately to self-attention in Transformers. The attention visualization showing alignment between source and target words provided unprecedented interpretability, demonstrating that neural networks could discover linguistic correspondences without explicit supervision.

**["Effective Approaches to Attention-based Neural Machine Translation" (2015)](https://arxiv.org/abs/1508.04025)**  
*Authors*: Minh-Thang Luong, Hieu Pham, Christopher D. Manning  
Introduced shortly after Bahdanau attention, this paper systematically explored attention mechanism design choices and proposed simpler alternatives. Luong attention uses dot-product similarity ($$\mathbf{q}^T \mathbf{k}$$) or general multiplicative ($$\mathbf{q}^T \mathbf{W} \mathbf{k}$$) instead of Bahdanau's additive formulation, showing these simpler mechanisms often perform better while being more computationally efficient. The paper also distinguished between global attention (attending to all source positions) and local attention (attending to a window around an aligned position), providing options for different computation-accuracy tradeoffs. The careful empirical comparison methodology established best practices for evaluating attention variants. Luong's dot-product attention, particularly the scaled version, became the foundation for Transformer attention, showing how systematic exploration of architectural choices leads to better designs. The paper also demonstrated that attention could be applied at different granularities (word-level, character-level) and in different configurations (input-feeding, where attention is fed back into the decoder), expanding understanding of attention's versatility.

**["Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" (2015)](https://arxiv.org/abs/1502.03044)**  
*Authors*: Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio  
This paper extended attention from NLP to computer vision, using attention to focus on different image regions when generating different caption words. When generating "red" in "A red car is parked," the model learns to attend to the car's color; when generating "parked," it attends to the scene context. The paper introduced both soft attention (differentiable weighted average, trainable with backpropagation) and hard attention (stochastic selection of single position, requiring reinforcement learning). It demonstrated that attention mechanisms are not domain-specific but represent a general principle applicable wherever selective focus is beneficial. The visualizations showing attention maps overlaid on images provided compelling evidence that the model was learning meaningful correspondences between visual content and language. This work inspired attention applications across modalities and contributed to the eventual development of vision-and-language models like CLIP and multimodal Transformers.

**["Attention is All You Need" (2017)](https://arxiv.org/abs/1706.03762)**  
*Authors*: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
While primarily introducing the Transformer, this paper's treatment of attention mechanisms themselves was transformative. The authors showed that self-attention—attention within a single sequence—could replace recurrence entirely, not just augment it. The scaled dot-product attention formulation with the $$1/\sqrt{d_k}$$ scaling became standard. Multi-head attention, allowing multiple attention patterns to be learned in parallel, addressed the limitation of single-head attention's inability to capture multiple relationship types simultaneously. The paper demonstrated that attention's computational properties (fully parallelizable, constant path length between any positions) could be advantages rather than just supplements to recurrent processing. The success of Transformers established attention not as a useful addition to RNNs but as a standalone mechanism sufficient for sequence processing, fundamentally changing how the field approaches sequential data.

**["Self-Attention with Relative Position Representations" (2018)](https://arxiv.org/abs/1803.02155)**  
*Authors*: Peter Shaw, Jakob Uszkoreit, Ashish Vaswani  
This paper addressed a limitation of standard Transformer attention: the reliance on absolute positional encodings added before attention. Shaw and colleagues proposed incorporating relative position information directly into the attention mechanism itself, making attention weights depend not just on content but on the distance between positions. This allows the model to learn patterns like "attend to the word 2 positions before" more naturally than with absolute positions. The paper showed improved performance on translation and other tasks, and relative position encodings have been adopted in many Transformer variants (T5, Transformer-XL). The work illustrates how even after a major architectural innovation (Transformers), refinements addressing subtle limitations continue to improve performance. It also demonstrates the principle that inductive biases (like position matters) should be incorporated where they're most relevant (in the attention mechanism itself) rather than through separate mechanisms (positional encodings), when possible.

## Common Pitfalls and Tricks

A fundamental mistake when implementing attention is forgetting to apply softmax to the attention scores, using raw similarity scores as weights instead. Without softmax normalization, the weighted sum $$\sum_i e_i \mathbf{v}_i$$ grows with the number of keys (since we're summing more terms), making the context vector's magnitude dependent on sequence length. Softmax ensures weights sum to 1, creating a weighted average rather than weighted sum, so context magnitude is independent of sequence length. This normalization is not optional—it's essential for stable training and meaningful interpretation of attention weights as probabilities.

Another common error is applying masking incorrectly. When using masks to prevent attention to certain positions (padding or future positions), we must set masked positions to $$-\infty$$ (or a very large negative number like -1e9) before softmax, not to 0 after softmax. Setting post-softmax weights to 0 doesn't affect the gradients properly during backpropagation because the softmax computation graph still includes the masked positions. Setting pre-softmax scores to $$-\infty$$ ensures both that the position receives zero weight (since $$\exp(-\infty) = 0$$) and that gradients flow correctly during backpropagation.

The scaling in scaled dot-product attention is frequently omitted in naive implementations, causing subtle training issues. Without dividing by $$\sqrt{d_k}$$, dot products have variance proportional to $$d_k$$. For large dimensions (512, 1024), this pushes values into softmax saturation regions where one position gets nearly all attention weight and gradients vanish. The model fails to learn distributed attention patterns and may not train at all. The $$\sqrt{d_k}$$ scaling is not a minor detail but essential for stable training with high-dimensional representations.

When implementing multi-head attention, a common bug is not properly splitting and recombining dimensions. The reshaping operations—splitting $$d_{\text{model}}$$ into $$(\text{num\_heads}, d_k)$$, computing attention, then concatenating back to $$d_{\text{model}}$$—must preserve the batch and sequence dimensions while shuffling the feature dimension. Getting the transpose and reshape operations in the wrong order produces tensors of correct shape but with scrambled data. Always verify with small examples that information flows correctly: a simple test is checking that with query = key = value and no mask, the output equals the input (identity attention).

A powerful technique for understanding what attention has learned is visualizing attention weights as heatmaps. For encoder-decoder attention in translation, plot source words on one axis, target words on the other, with heatmap intensity showing attention weight. This reveals learned alignments—which source words the model focuses on when generating each target word. For self-attention, the $$n \times n$$ matrix shows which positions attend to which other positions. Patterns that emerge—attention to nearby words, attention from pronouns to their antecedents, attention across syntactic dependencies—provide insight into what linguistic structure the model has discovered. This interpretability is one of attention's major advantages over black-box RNN hidden states.

For computational efficiency with very long sequences, consider sparse attention patterns. Full attention requires $$O(n^2)$$ memory and computation where $$n$$ is sequence length. For sequences of thousands of tokens (documents, long-form text), this becomes prohibitive. Sparse attention restricts each position to attend to only a subset of positions (e.g., local window plus strided positions), reducing complexity to $$O(n\sqrt{n})$$ or even $$O(n)$$ while maintaining ability to capture long-range dependencies through multiple layers. Understanding when full attention is necessary versus when sparse patterns suffice helps choose appropriate architectures for different sequence length regimes.

## Key Takeaways

Attention mechanisms enable neural networks to selectively focus on relevant parts of the input when making predictions, solving the fixed-size bottleneck problem of encoder-decoder models and providing interpretability through visualizable attention weights. The core mathematical framework—computing queries, keys, and values, measuring similarity between queries and keys, using softmax to convert similarities to weights, and computing weighted averages of values—is general enough to apply across domains and tasks. Scaled dot-product attention combines simplicity, efficiency, and effectiveness, scaling similarity scores by $$\sqrt{d_k}$$ to maintain reasonable magnitudes regardless of dimensionality. Multi-head attention allows learning multiple attention patterns simultaneously, with different heads discovering different types of relationships (syntactic, semantic, positional) through their separate learned projections. Self-attention applies attention within sequences rather than between encoder and decoder, enabling each position to build representations incorporating information from all other positions with constant path length for information flow. Masking controls which positions can be attended to, enabling both handling of variable-length sequences (padding masks) and autoregressive generation (causal masks). The transition from attention as an augmentation to RNNs to attention as the sole mechanism in Transformers demonstrates how an idea can evolve from useful addition to foundational principle, ultimately transforming an entire field by enabling architectures that are simultaneously more powerful, more interpretable, and more efficient to train than their predecessors.

Attention is not just a technical mechanism but a paradigm shift in how we think about neural network architectures: from fixed processing pipelines to dynamic, learned routing of information based on relevance and context.

