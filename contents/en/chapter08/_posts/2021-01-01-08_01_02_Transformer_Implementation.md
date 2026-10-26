---
layout: post
title: 08-01-02 Transformer Implementation
chapter: '08'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter08
---

# 08-01-02 Transformer Implementation

Theory defines the maps. Implementation must respect **shapes, masks, residual order, and numerical stability**. These lessons build a small but honest Transformer stack.

## Lessons

1. **Core Implementation**  
   Scaled attention, multi-head, positional encoding, FFN, encoder layer/stack—from scratch in PyTorch.

2. **Demos, Papers, and Pitfalls**  
   Forward smoke tests, attention inspection, a tiny sequence task, literature, and a production-oriented debugging checklist.

## Module map

```text
Token ids
  → Embedding + PositionalEncoding
  → N × EncoderLayer
        MultiHeadSelfAttention (+ residual/norm)
        PositionwiseFFN        (+ residual/norm)
  → (optional) DecoderLayer stack with causal + cross-attn
  → Linear vocabulary head
```

## Checklist before you scale up

| Item | Test |
|------|------|
| Causal mask | Future weights ≈ 0 |
| Padding mask | Pad keys get weight 0 |
| Residual path | Ablation: removing residual should hurt depth |
| PE added once at input (classic) | Not every layer unless you chose that design |
| `batch_first` consistency | All modules agree on `(B,L,d)` vs `(L,B,d)` |
| Dropout only in train mode | `model.eval()` for inference |

Start with **Core Implementation**.

## Video tham khảo

- **xem trước** [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, English. The module map (attention, MLP, block, language-model head) is built on camera.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, English. The equations those modules must match.
- [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://www.youtube.com/watch?v=zxQyTK8quyY) — StatQuest with Josh Starmer, English. A second labeling of the block if a shape error is hard to localize.
