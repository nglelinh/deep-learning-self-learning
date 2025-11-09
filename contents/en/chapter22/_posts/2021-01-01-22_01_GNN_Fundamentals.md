---
layout: post
title: 22-01 Graph Neural Networks
chapter: '22'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter22
lesson_type: required
---

# Graph Neural Networks: Deep Learning on Graphs

![Graph Neural Network](https://distill.pub/2021/gnn-intro/graph_neural_network.png)
*Hình ảnh: Kiến trúc Graph Neural Network với message passing. Nguồn: Distill.pub*

## 1. Concept Overview

Graph Neural Networks extend deep learning to graph-structured data, enabling neural networks to process networks of entities and relationships that pervade real-world data: social networks connecting people, molecular graphs connecting atoms, knowledge graphs linking concepts, citation networks relating papers, and recommendation systems connecting users and items. Unlike images (regular 2D grids) or sequences (1D chains), graphs have arbitrary structure—variable numbers of neighbors, no spatial ordering, complex connectivity patterns—requiring specialized neural architectures that can leverage this structure while remaining differentiable and trainable via backpropagation.

The fundamental challenge graphs present is permutation invariance: a graph's representation shouldn't depend on how we order its nodes. If nodes are numbered 1,2,3 versus 3,2,1, the graph is identical, so representations should be too. This rules out naive approaches like feeding adjacency matrices to standard neural networks (which treat different orderings as different inputs). GNNs solve this through message passing: nodes iteratively aggregate information from neighbors through learned transformations, with aggregation functions (sum, mean, max) that are permutation-invariant. After several iterations, each node's representation incorporates information from its neighborhood, with larger neighborhoods accessible through more iterations.

Understanding GNNs requires appreciating that different graph learning tasks require different outputs. Node classification predicts labels for nodes (categorizing users in social networks). Link prediction predicts missing or future edges (recommending friendships or products). Graph classification predicts labels for entire graphs (classifying molecules as active/inactive for drugs). Each task uses the same message passing framework but differs in how node representations are aggregated and what predictions are made.

The applications of GNNs span diverse domains. In chemistry, molecular graphs (atoms as nodes, bonds as edges) are processed by GNNs to predict properties like solubility or toxicity. In social networks, GNNs detect communities, predict friendships, or identify influential users. In recommender systems, bipartite graphs (users and items) are processed to predict preferences. In protein structure prediction, GNNs model amino acid interactions. In traffic prediction, road networks inform forecasting. This versatility demonstrates that graph structure is ubiquitous, and GNNs provide a general framework for learning from it.

## 2. Mathematical Foundation

A graph $$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$ consists of nodes $$\mathcal{V} = \{v_1, \ldots, v_n\}$$ and edges $$\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$$. Nodes have features $$\mathbf{x}_i \in \mathbb{R}^d$$, and edges may have features $$\mathbf{e}_{ij}$$. The adjacency matrix $$\mathbf{A} \in \{0,1\}^{n \times n}$$ encodes structure: $$A_{ij} = 1$$ if edge $$(v_i, v_j) \in \mathcal{E}$$, else 0.

### Message Passing Framework

GNNs update node representations through iterative message passing. At layer $$k$$, node $$i$$'s representation $$\mathbf{h}_i^{(k)}$$ is updated based on neighbors:

$$\mathbf{h}_i^{(k)} = \text{UPDATE}^{(k)}\left(\mathbf{h}_i^{(k-1)}, \text{AGGREGATE}^{(k)}\left(\{\mathbf{h}_j^{(k-1)} : j \in \mathcal{N}(i)\}\right)\right)$$

where $$\mathcal{N}(i)$$ are neighbors of $$i$$. AGGREGATE combines neighbor features (must be permutation-invariant), UPDATE combines node's own features with aggregated neighbor information.

### Graph Convolutional Networks (GCN)

GCN uses spectral graph theory, defining convolution through graph Laplacian. The practical form:

$$\mathbf{H}^{(k+1)} = \sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(k)}\mathbf{W}^{(k)})$$

where $$\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$$ (adjacency plus self-loops), $$\tilde{\mathbf{D}}$$ is degree matrix, $$\mathbf{H}^{(k)}$$ are node representations at layer $$k$$, $$\mathbf{W}^{(k)}$$ are learned weights.

This performs weighted aggregation of neighbors followed by linear transformation and nonlinearity, with weights inversely proportional to node degrees (high-degree nodes contribute less per neighbor).

### GraphSAGE

GraphSAGE samples fixed-size neighborhoods and uses learned aggregation:

$$\mathbf{h}_{\mathcal{N}(i)}^{(k)} = \text{AGGREGATE}(\{\mathbf{h}_j^{(k-1)} : j \in \mathcal{N}(i)\})$$

$$\mathbf{h}_i^{(k)} = \sigma(\mathbf{W}^{(k)} \cdot [\mathbf{h}_i^{(k-1)}, \mathbf{h}_{\mathcal{N}(i)}^{(k)}])$$

AGGREGATE can be mean, max, or LSTM over neighborhood. This enables mini-batch training and handles variable-size neighborhoods.

## 3. Example / Intuition

Consider a social network: nodes are people, edges are friendships, node features include age, location, interests. We want to predict which users will like a new product (node classification).

A 2-layer GNN works as follows. Initially, each user's representation $$\mathbf{h}_i^{(0)}$$ is their raw features. After layer 1, $$\mathbf{h}_i^{(1)}$$ incorporates information from immediate friends (1-hop neighbors). User A's representation now includes "my friends are ages 25-30, mostly in urban areas, interested in tech"—aggregated neighbor information. After layer 2, $$\mathbf{h}_i^{(2)}$$ incorporates friends-of-friends (2-hop neighborhood). User A's representation captures broader social context.

For prediction, the network learns that users whose neighborhoods have certain patterns (many tech-interested friends, urban clustering) are likely to like tech products. The GNN provides features capturing both individual attributes and social context for classification.

## 4. Code Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        """
        x: (num_nodes, in_features)
        adj: (num_nodes, num_nodes) adjacency matrix
        """
        # Add self-loops
        adj_hat = adj + torch.eye(adj.size(0))
        
        # Normalize
        deg = adj_hat.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = torch.diag(deg_inv_sqrt)
        adj_normalized = norm @ adj_hat @ norm
        
        # Apply convolution
        support = x @ self.weight
        output = adj_normalized @ support
        
        return output

class GCN(nn.Module):
    """2-layer Graph Convolutional Network"""
    
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.gc1 = GCNLayer(num_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, num_classes)
    
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# Example
num_nodes = 100
num_features = 16
num_classes = 7

gcn = GCN(num_features, hidden_dim=32, num_classes=num_classes)

# Random graph
x = torch.randn(num_nodes, num_features)
adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
adj = (adj + adj.T) / 2  # Symmetric

output = gcn(x, adj)
print(f"GCN output: {output.shape}")  # (100, 7)
print("Each node gets class predictions using graph structure!")
```

## 5. Related Concepts

GNNs connect to spectral graph theory through their mathematical foundations. Graph Laplacians and eigendecompositions provide theoretical basis for defining convolution on graphs, generalizing CNNs' convolution from regular grids to arbitrary graphs.

## 6. Fundamental Papers

**["Semi-Supervised Classification with Graph Convolutional Networks" (2017)](https://arxiv.org/abs/1609.02907)**  
*Authors*: Thomas Kipf, Max Welling  
Introduced GCN, establishing message passing on graphs as effective deep learning approach. Demonstrated semi-supervised node classification using graph structure plus limited labels.

**["Inductive Representation Learning on Large Graphs" (2017)](https://arxiv.org/abs/1706.02216)**  
*Authors*: William Hamilton, Rex Ying, Jure Leskovec  
GraphSAGE enabled learning on large graphs through neighborhood sampling, allowing mini-batch training. Extended GNNs from transductive (fixed graph) to inductive (generalizing to new nodes/graphs).

**["Graph Attention Networks" (2018)](https://arxiv.org/abs/1710.10903)**  
*Authors*: Petar Veličković, Guilermo Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio  
GAT used attention mechanisms to weight neighbor contributions, learning importance of different neighbors. More flexible than fixed aggregation.

## Common Pitfalls and Tricks

Oversmoothing occurs with many GNN layers—node representations become indistinguishable as they aggregate from increasingly large neighborhoods. Use skip connections, batch normalization, or careful depth selection (2-3 layers often sufficient).

## Key Takeaways

Graph Neural Networks process graph-structured data through message passing, iteratively updating node representations by aggregating information from neighbors through learned, permutation-invariant functions. GCNs use normalized adjacency matrices for spectral graph convolution. GraphSAGE samples neighborhoods for scalability. Graph attention networks learn neighbor importance through attention. Applications span social networks, molecules, knowledge graphs, and recommendation systems. GNNs enable deep learning on irregular, relational data where CNNs and RNNs don't apply, opening graph-structured domains to modern AI.

