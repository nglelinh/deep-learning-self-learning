---
layout: post
title: 07-02-02 Attention Implementation
chapter: '07'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter07
---

# 07-02-02 Attention Implementation

Mathematics tells you *what* attention computes. Implementation tells you whether you actually built that object—shapes, masks, stability, and APIs included.

## Lessons in this section

1. **Core Implementation**  
   Bahdanau attention from scratch, scaled self-attention, translation-style demos, and a clear NumPy → PyTorch path.

2. **Multi-Head, Papers, and Pitfalls**  
   Multi-head modules, causal masking, a minimal seq2seq-with-attention sketch, canonical papers, and a debugging checklist used in real training runs.

## Implementation checklist (keep this open while coding)

| Check | Why it matters |
|-------|----------------|
| Softmax over the **key** axis | Weights must form a distribution over memory slots |
| Scale by $$\sqrt{d_k}$$ | Prevents saturated attention at large width |
| Mask **before** softmax with large negatives | Correct zeros + correct gradients |
| Causal mask for autoregressive decoders | Stops future leakage |
| Padding mask in batched training | Stops attending to `PAD` |
| Multi-head reshape: `(B,L,H,d)` ↔ `(B,H,L,d)` | Wrong transpose silently scrambles features |

Start with **Core Implementation**, then continue to multi-head and pitfalls.
