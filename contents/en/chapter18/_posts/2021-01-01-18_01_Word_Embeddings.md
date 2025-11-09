---
layout: post
title: 18-01 Word Embeddings and Language Representation
chapter: '18'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter18
lesson_type: required
---

# Word Embeddings: Representing Language in Vector Space

![Word2Vec Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Word2vec.png/800px-Word2vec.png)
*Hình ảnh: Minh họa Word2Vec embeddings trong không gian vector. Nguồn: Wikimedia Commons*

## 1. Concept Overview

Word embeddings represent one of the most fundamental innovations in natural language processing, transforming how we represent and process text in machine learning systems. The core idea is elegantly simple yet profoundly impactful: represent each word as a dense vector of real numbers (typically 100-300 dimensions) such that semantically similar words have similar vectors. This continuous vector representation replaces older sparse representations like one-hot encoding (vectors with thousands of dimensions, all zeros except one) with compact, meaningful embeddings that capture semantic relationships through geometric properties in vector space.

Understanding why embeddings revolutionized NLP requires appreciating the limitations of discrete word representations. Traditional approaches treated words as atomic symbols—"king," "queen," and "car" were equally distant from each other, sharing no structure. One-hot encoding represents a 50,000-word vocabulary with 50,000-dimensional vectors that are all orthogonal, providing no notion of similarity. This makes learning difficult because the model cannot generalize from seeing "king lives in palace" to understanding "queen lives in palace"—it must learn facts about "queen" independently despite semantic similarity to "king."

Word embeddings solve this by learning continuous representations where similar words cluster together. The distance between "king" and "queen" vectors is small (they're both royalty), while "king" and "car" are distant (semantically unrelated). More remarkably, embedding spaces exhibit analogical relationships through vector arithmetic: the vector from "man" to "woman" is similar to the vector from "king" to "queen," capturing the gender relationship. We can solve analogies through simple vector math: king - man + woman ≈ queen. This emergent structure wasn't explicitly programmed but arose from training on text, demonstrating that embeddings capture deep semantic regularities.

The training objective for learning embeddings is elegantly formulated through the distributional hypothesis from linguistics: words appearing in similar contexts have similar meanings. This simple principle enables unsupervised learning from massive text corpora. Models like Word2Vec and GloVe learn embeddings by predicting context words from target words or vice versa, or by factorizing co-occurrence statistics. The resulting vectors encode lexical semantics, syntactic patterns, and even some world knowledge, all discovered purely from word co-occurrence patterns in text without any labeled data.

The impact of word embeddings on NLP cannot be overstated. They provided the foundation for the deep learning revolution in language processing, enabling neural networks to leverage vast unlabeled text for learning representations that then transfer to downstream tasks. Pre-trained embeddings like Word2Vec and GloVe became standard components in virtually all NLP systems from 2013-2018. While modern contextual embeddings from BERT and GPT have largely superseded static word embeddings for many tasks, understanding static embeddings remains crucial for appreciating how representation learning in NLP evolved and for applications where their simplicity and efficiency remain advantages.

## 2. Mathematical Foundation

Word embeddings map discrete symbols (words) to continuous vectors in a way that captures semantic similarity. Formally, we have a vocabulary $$V$$ of size $$|V|$$ and learn an embedding matrix $$\mathbf{E} \in \mathbb{R}^{d \times |V|}$$ where column $$\mathbf{e}_w \in \mathbb{R}^d$$ is the embedding for word $$w$$. Typical embedding dimension $$d = 100\text{-}300$$, much smaller than vocabulary size (10,000-100,000).

### Word2Vec: Skip-gram Model

The skip-gram model predicts context words given a target word, based on the distributional hypothesis. For a corpus with words $$w_1, w_2, \ldots, w_T$$, the objective is:

$$\max_\theta \frac{1}{T}\sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t; \theta)$$

where $$c$$ is context window size (typically 5), and $$\theta$$ includes the embedding matrix and output weights. The conditional probability uses softmax:

$$p(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^T \mathbf{v}_{w_I})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}_w^T \mathbf{v}_{w_I})}$$

where $$\mathbf{v}_{w_I}$$ is the input embedding of word $$w_I$$ and $$\mathbf{v}_{w_O}$$ is the output embedding of $$w_O$$. Computing this softmax requires summing over the entire vocabulary (expensive!), motivating approximations.

**Negative sampling** approximates the softmax by sampling a few negative examples instead of summing over all words:

$$\log \sigma(\mathbf{v}_{w_O}^T \mathbf{v}_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_i}^T \mathbf{v}_{w_I})]$$

where $$\sigma$$ is sigmoid, $$k$$ is number of negative samples (typically 5-20), and $$P_n(w)$$ is noise distribution (often unigram raised to 3/4 power to oversample rare words). This transforms the multi-class problem into $$k+1$$ binary classifications, tractable even for large vocabularies.

The remarkable property of Word2Vec embeddings is that semantic relationships are encoded as linear translations in vector space:

$$\mathbf{e}_{\text{queen}} \approx \mathbf{e}_{\text{king}} - \mathbf{e}_{\text{man}} + \mathbf{e}_{\text{woman}}$$

$$\mathbf{e}_{\text{Paris}} \approx \mathbf{e}_{\text{France}} - \mathbf{e}_{\text{Germany}} + \mathbf{e}_{\text{Berlin}}$$

These analogies weren't explicitly trained but emerge from the distributional hypothesis: "king" and "queen" appear in similar contexts (royal, throne, palace), as do "king" and "man" (gendered contexts), creating vector geometry that reflects these semantic patterns.

### GloVe: Global Vectors

GloVe takes a different approach, directly factorizing word co-occurrence statistics. Let $$X_{ij}$$ be the number of times word $$j$$ appears in word $$i$$'s context. GloVe minimizes:

$$J = \sum_{i,j=1}^{|V|} f(X_{ij})(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

where $$\mathbf{w}_i$$ and $$\tilde{\mathbf{w}}_j$$ are word and context embeddings, $$b_i, \tilde{b}_j$$ are biases, and $$f(X_{ij})$$ is a weighting function:

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

This weights frequent co-occurrences less heavily (they're already well-represented) and caps influence of very frequent pairs. GloVe combines the benefits of global matrix factorization methods (leveraging entire corpus statistics) with local context window methods (Word2Vec), often producing embeddings competitive or superior to Word2Vec.

## 3. Example / Intuition

Imagine learning embeddings for a small vocabulary: {cat, dog, car, truck, animal, vehicle}. Initially, vectors are random. As we process text:

"The cat is an animal" → "cat" and "animal" co-occur  
"The dog is an animal" → "dog" and "animal" co-occur  
"The car is a vehicle" → "car" and "vehicle" co-occur  
"The truck is a vehicle" → "truck" and "vehicle" co-occur

The model adjusts vectors so:
- "cat" and "dog" become close (both appear with "animal")
- "car" and "truck" become close (both appear with "vehicle")
- "cat" and "car" stay distant (appear in different contexts)

After seeing enough text, the 2D embedding space might organize as:

```
     animal
        ↑
    dog • cat
        |
    ----+---- 
        |
  truck • car
        ↓
     vehicle
```

Semantic categories (animals vs vehicles) cluster, and within categories, similar items are nearby. We can compute:

"cat" - "animal" ≈ "dog" - "animal" (both point from category to instance)  
"car" + "vehicle" ≈ "truck" (category + similarity gives similar item)

This geometric structure enables generalization: if the model learns facts about "cat," it can transfer to "dog" through their vector similarity.

## 4. Code Snippet

Complete Word2Vec implementation:

```python
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

class Word2VecSkipGram(nn.Module):
    """
    Skip-gram Word2Vec with negative sampling.
    
    Learns word embeddings by predicting context words from target words.
    Uses negative sampling to make training efficient.
    """
    
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        
        # Input and output embeddings
        # Input: embeddings used when word is target
        # Output: embeddings used when word is context
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with small random values
        self.input_embeddings.weight.data.uniform_(-0.5/embedding_dim, 
                                                   0.5/embedding_dim)
        self.output_embeddings.weight.data.zero_()
    
    def forward(self, target_words, context_words, negative_words):
        """
        target_words: (batch,) target word indices
        context_words: (batch,) context word indices (positive examples)
        negative_words: (batch, k) negative samples
        
        Returns negative log-likelihood
        """
        # Get embeddings
        target_embeds = self.input_embeddings(target_words)  # (batch, emb_dim)
        context_embeds = self.output_embeddings(context_words)  # (batch, emb_dim)
        neg_embeds = self.output_embeddings(negative_words)  # (batch, k, emb_dim)
        
        # Positive scores (target-context similarity)
        pos_scores = (target_embeds * context_embeds).sum(dim=1)  # (batch,)
        pos_loss = -torch.log(torch.sigmoid(pos_scores)).mean()
        
        # Negative scores (target-negative dissimilarity)
        neg_scores = torch.bmm(neg_embeds, target_embeds.unsqueeze(2)).squeeze()  # (batch, k)
        neg_loss = -torch.log(torch.sigmoid(-neg_scores)).sum(dim=1).mean()
        
        return pos_loss + neg_loss

# Prepare training data
print("="*70)
print("Training Word2Vec Embeddings")
print("="*70)

# Simple corpus for demonstration
corpus = """
the cat sat on the mat .
the dog sat on the rug .
the cat and the dog are animals .
the car is a vehicle .
the truck is a vehicle .
cats and dogs are pets .
cars and trucks are vehicles .
""".lower().split()

# Build vocabulary
vocab = list(set(corpus))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Corpus: {len(corpus)} words")
print(f"Vocabulary: {vocab_size} unique words")
print(f"Sample vocab: {vocab[:10]}")

# Generate training pairs
def generate_training_data(corpus, word_to_idx, window_size=2):
    """Generate (target, context) pairs"""
    pairs = []
    for i, word in enumerate(corpus):
        target_idx = word_to_idx[word]
        
        # Get context (words within window)
        context_start = max(0, i - window_size)
        context_end = min(len(corpus), i + window_size + 1)
        
        for j in range(context_start, context_end):
            if j != i:  # Don't pair with self
                context_idx = word_to_idx[corpus[j]]
                pairs.append((target_idx, context_idx))
    
    return pairs

pairs = generate_training_data(corpus, word_to_idx, window_size=2)
print(f"\nGenerated {len(pairs)} training pairs")
print(f"Sample pairs: {pairs[:5]}")

# Train Word2Vec
model = Word2VecSkipGram(vocab_size, embedding_dim=10)  # Small dim for demo
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nTraining embeddings...")

for epoch in range(500):
    epoch_loss = 0
    
    for target_idx, context_idx in pairs:
        # Sample negatives
        neg_indices = np.random.choice(vocab_size, size=5, replace=False)
        
        # To tensors
        target = torch.LongTensor([target_idx])
        context = torch.LongTensor([context_idx])
        negatives = torch.LongTensor(neg_indices).unsqueeze(0)
        
        # Forward
        loss = model(target, context, negatives)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}: Loss = {epoch_loss/len(pairs):.4f}")

# Analyze learned embeddings
print("\n" + "="*70)
print("Analyzing Learned Embeddings")
print("="*70)

model.eval()
embeddings = model.input_embeddings.weight.data.numpy()

# Find nearest neighbors
def find_nearest(word, embeddings, word_to_idx, idx_to_word, k=3):
    """Find k nearest words to query word"""
    word_idx = word_to_idx[word]
    word_vec = embeddings[word_idx]
    
    # Compute cosine similarities
    similarities = embeddings @ word_vec / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_vec) + 1e-10
    )
    
    # Get top k (excluding self)
    top_k = similarities.argsort()[-k-1:-1][::-1]
    
    return [(idx_to_word[i], similarities[i]) for i in top_k]

# Test semantic similarity
test_words = ['cat', 'dog', 'car']
for word in test_words:
    if word in word_to_idx:
        neighbors = find_nearest(word, embeddings, word_to_idx, idx_to_word)
        print(f"\nNearest to '{word}': {neighbors}")

print("\nEmbeddings learned semantic relationships from co-occurrence patterns!")
print("Similar words (cat/dog, car/truck) have similar embeddings.")
```

## 5. Related Concepts

Word embeddings connect to distributional semantics, the linguistic theory that word meaning is determined by context. The computational implementation—learning vectors such that words in similar contexts have similar representations—directly operationalizes this theory. Understanding this connection helps appreciate why embeddings work: they're not arbitrary feature engineering but implementations of fundamental linguistic principles.

Embeddings relate to dimensionality reduction techniques like PCA or autoencoders. We're compressing high-dimensional one-hot vectors (vocab size) to low-dimensional dense vectors (embedding size) while preserving semantic information. The learned compression discovers that semantic relationships can be captured in far fewer dimensions than explicit symbol identity, revealing the intrinsic dimensionality of word semantics is much lower than vocabulary size.

The evolution from static embeddings (Word2Vec, GloVe) to contextual embeddings (ELMo, BERT) reflects increasing sophistication. Static embeddings assign one vector per word type, so "bank" (financial) and "bank" (river) have identical representations despite different meanings. Contextual embeddings produce different vectors based on context, resolving polysemy. This evolution shows the field progressing from learning word-level representations to modeling language's context-dependent nature.

## 6. Fundamental Papers

**["Efficient Estimation of Word Representations in Vector Space" (2013)](https://arxiv.org/abs/1301.3781)**  
*Authors*: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean  
Word2Vec introduced efficient methods (skip-gram and CBOW) for learning word embeddings at scale. The negative sampling training procedure enabled processing billions of words, making embeddings practical for large vocabularies. The paper demonstrated remarkable semantic properties—analogies solved through vector arithmetic—showing embeddings capture sophisticated language patterns. Word2Vec's simplicity, efficiency, and quality made it widely adopted, establishing embeddings as fundamental to NLP.

**["GloVe: Global Vectors for Word Representation" (2014)](https://aclanthology.org/D14-1162/)**  
*Authors*: Jeffrey Pennington, Richard Socher, Christopher Manning  
GloVe combined global matrix factorization with local context, factorizing word co-occurrence matrices to learn embeddings. The method achieved competitive or superior performance to Word2Vec while providing intuitive interpretation through co-occurrence statistics. GloVe demonstrated that different training objectives could produce similar high-quality embeddings, suggesting the representation itself matters more than specific training procedure.

**["Deep contextualized word representations" (2018)](https://arxiv.org/abs/1802.05365)**  
*Authors*: Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer  
ELMo introduced contextual embeddings from deep bidirectional LSTMs, representing each word differently based on sentence context. This addressed static embeddings' inability to handle polysemy and context-dependent meaning. ELMo showed that deep language models learn different types of information at different layers (syntax in lower, semantics in higher), and combining layers improves downstream tasks. ELMo represented transition from static to contextual representations, paving way for BERT and Transformers.

## Common Pitfalls and Tricks

Using pre-trained embeddings without proper vocabulary alignment causes out-of-vocabulary issues. If your task vocabulary contains words not in the pre-trained embeddings, you need strategies: use subword embeddings (BPE, WordPiece), initialize missing words from similar words (if embeddings for "coronavirus" missing, average "virus" and "corona"), or fine-tune embeddings on domain-specific text.

## Key Takeaways

Word embeddings represent words as dense continuous vectors where semantic similarity corresponds to geometric proximity, enabling neural networks to generalize across semantically related words through shared vector representations. Skip-gram Word2Vec predicts context from targets using negative sampling for efficiency, while GloVe factorizes co-occurrence matrices, both learning from unlabeled text through distributional hypothesis. The resulting embeddings exhibit remarkable properties including analogical reasoning through vector arithmetic (king - man + woman ≈ queen) and semantic clustering (synonyms have similar vectors), all emerging from co-occurrence patterns without explicit supervision. Pre-trained embeddings like Word2Vec and GloVe transfer to downstream tasks, providing semantic representations that improve performance across NLP applications from sentiment analysis to machine translation. Modern contextual embeddings from BERT provide context-dependent representations addressing polysemy, though static embeddings remain useful for efficiency and interpretability. Understanding word embeddings provides foundation for all representation learning in NLP, demonstrating how neural networks can discover semantic structure purely from text patterns.

