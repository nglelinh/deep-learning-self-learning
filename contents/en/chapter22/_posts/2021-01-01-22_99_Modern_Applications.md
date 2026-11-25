---
layout: post
title: 22-99 Modern Applications and Updates (2022–2026)
chapter: '22'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter22
lesson_type: optional
---

Standard message-passing GNNs are nearsighted: a 2-layer GCN can only see 2 hops away, and trying to route long-range signals through a narrow graph bottleneck causes *over-squashing*. Between 2022 and 2026, geometric deep learning broke that barrier by fusing global Transformer attention with graph structural encodings (GraphGPS) and 3D equivariant diffusion (AlphaFold 3)—predicting joint structures of proteins, DNA, RNA, and small-molecule ligands at atomic precision.

## Key ideas: Modern Geometric & Graph Deep Learning (2022–2026)

- **Pillar 1 — Graph Transformers & Structural Encodings**: Allows every node to attend globally to every other node while injecting graph topology via Laplacian eigenvectors, random-walk landing probabilities, or shortest-path attention biases (Graphormer, GraphGPS).
- **Pillar 2 — 3D Equivariance & AlphaFold 3**: Combines Pairformer relational attention with diffusion directly over 3D atomic coordinates, respecting $\text{SE}(3)$ rotations and translations without brittle handcrafted torsion angles.
- **Pillar 3 — Industrial Heterogeneous Graphs**: Scales GNNs to billion-edge user–item–creator recommendation graphs (PinSage, PyG) using temporal neighbor sampling and multi-relation message passing.
- **Pillar 4 — Hybrid Local + Global Layers**: Pairs a local MPNN (inductive bias for immediate bonds/edges) with global self-attention (long-range communication) in parallel within every block.

### Worked example: Computing Shortest-Path Attention Bias in a Graph Transformer

Consider a 4-node linear molecule $1 - 2 - 3 - 4$. Suppose raw dot-product attention logits from node 1 to all 4 nodes are equal: $q_1^\top k_j / \sqrt{d} = [1.0, 1.0, 1.0, 1.0]$. A Graphormer layer adds a learned shortest-path bias $b(\text{spd}(1, j))$ where $b(0) = 0.5,\; b(1) = 0.2,\; b(2) = -0.3,\; b(3) = -0.8$.

1. **Compute biased attention logits $z_{1j} = 1.0 + b(\text{spd}(1, j))$**:
   $$z_1 = [1.5,\; 1.2,\; 0.7,\; 0.2].$$
2. **Apply Softmax**:
   $$e^{1.5} = 4.482, \quad e^{1.2} = 3.320, \quad e^{0.7} = 2.014, \quad e^{0.2} = 1.221 \implies Z = 11.037.$$
   $$\alpha_1 = [40.6\%,\; 30.1\%,\; 18.2\%,\; 11.1\%].$$
3. **Takeaway**: Unlike a 1-hop GCN (which puts $0\%$ weight on nodes 3 and 4), the Graph Transformer lets node 1 communicate directly with node 4 ($11.1\%$ weight) in a single layer while still honoring graph distance.

# Optional: GNNs after message passing — graph Transformers and AlphaFold 3

> This lesson is **optional**. It does **not** replace message passing, permutation invariance, or the GCN-style update. It covers 2022–2025 models that mix **attention on graphs** with classical MPNN layers, plus the structure-prediction products that made GNNs famous outside recsys.

The MPNN step

$$\mathbf{h}_i^{(t+1)} = \mathrm{UPD}\Big(\mathbf{h}_i^{(t)},\; \mathrm{AGG}_{j\in\mathcal{N}(i)} \mathrm{MSG}(\mathbf{h}_i^{(t)},\mathbf{h}_j^{(t)},e_{ij})\Big)$$

is still the right first algorithm. Many SOTA models now let **every node attend to many others** (Graphormer, GraphGPS) when the graph is small enough (molecules, proteins).

## 1. Graph Transformers

[Ying et al., 2021](https://arxiv.org/abs/2106.05234) (Graphormer) and [Rampášek et al., 2022](https://arxiv.org/abs/2205.12454) (GraphGPS) add Laplacian / spatial encodings so a Transformer respects graph distance. Use them on molecules (hundreds of nodes), not on a 10M-node social graph — there you still want neighbor sampling (GraphSAGE) as in the theory notes.

## 2. Concrete applications

### AlphaFold 3

[Abramson et al., 2024](https://www.nature.com/articles/s41586-024-07487-w) (AlphaFold 3) predict biomolecular complexes with a **diffusion** module on atom coordinates plus a pair/token trunk. The pairwise trunk is the intellectual descendant of attention-on-graphs from AlphaFold 2. Application: drug discovery and structural biology, not a new GCN formula.

### Recsys and knowledge graphs

Production recommenders still run **bipartite GNNs** or two-tower embeddings (PinSage-class). 2023–2025 papers add LLM features on nodes; the aggregator stays a GNN.

### Software

- [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) 2.x — `TransformerConv`, GraphGPS examples.
- [dmlc/dgl](https://github.com/dmlc/dgl).
- [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) (inference code, 2024+).

## 3. Citations (2022–2026)

- [Recipe for a General, Powerful, Scalable Graph Transformer — GraphGPS (Rampášek et al., 2022)](https://arxiv.org/abs/2205.12454).
- [Do Transformers Really Perform Badly for Graph Representation? — Graphormer (Ying et al., 2021)](https://arxiv.org/abs/2106.05234).
- [Accurate structure prediction of biomolecular interactions with AlphaFold 3 (Abramson et al., 2024)](https://www.nature.com/articles/s41586-024-07487-w).

## 4. How this complements the core notes

Write a message-passing layer first. This lesson only adds **graph Transformers** and **AlphaFold 3** as the applications that cite this chapter’s ideas in 2024–2026.

## Video tham khảo

- **xem trước** [Graph Neural Networks Explained: A Clear Guide to GNN Basics & Models](https://www.youtube.com/watch?v=eGoszzMkGfU) — IBM Technology, English. Message passing is what a graph transformer still does, with a learned attention mask.
- [Message Passing: How Graph Neural Networks Actually Work](https://www.youtube.com/watch?v=1DmNdMUbDkg) — The Synthetic Mind, English. A single round, so the graph-transformer block has a baseline.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, English. Self-attention is the mixer that block uses once neighbors are not the only tokens it may read.
