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
