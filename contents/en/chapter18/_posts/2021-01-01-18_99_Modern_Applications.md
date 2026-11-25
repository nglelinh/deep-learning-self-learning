---
layout: post
title: 18-99 Modern Applications and Updates (2022–2026)
chapter: '18'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter18
lesson_type: optional
---

Today's NLP pipelines rarely train static Word2Vec embeddings from scratch, yet dense vector geometry is more central than ever: Retrieval-Augmented Generation (RAG) embeds entire documents into dense vectors to ground LLM answers in private facts, while structured JSON extraction and BPE tokenization determine whether an applied system stays accurate, fast, and within context budgets.

## Key ideas: Applied Modern NLP (2022–2026)

- **Pillar 1 — Dense Bi-Encoder Retrieval & RAG**: Encodes queries and document chunks into normalized vectors via contrastive embedding models, retrieves top-$k$ passages via Approximate Nearest Neighbors (HNSW/FAISS), and injects them into an LLM prompt.
- **Pillar 2 — Hybrid Search & Cross-Encoder Reranking**: Combines lexical BM25 (exact IDs, rare jargon) with dense semantic search via Reciprocal Rank Fusion (RRF), then reranks top candidates with a cross-encoder.
- **Pillar 3 — Structured Extraction & Constrained Decoding**: Uses JSON schemas and grammar-constrained sampling so generative LLMs reliably emit typed records for downstream databases.
- **Pillar 4 — Applied Tokenization Economics**: Tracks token-to-word ratios across languages and code, directly controlling API latency, context-window utilization, and billing cost.

### Worked example: Reciprocal Rank Fusion (RRF) for Hybrid RAG Search

Suppose a user queries *"Error E402 timeout"*. Lexical BM25 returns ranks `{Doc A: 1, Doc B: 3, Doc C: 2}` while dense vector search returns ranks `{Doc A: 4, Doc B: 1, Doc C: 2}`. Using the standard RRF formula $\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$ with $k = 60$:

1. **Compute RRF score for each document**:
   - **Doc A**: $\frac{1}{61} + \frac{1}{64} = 0.01639 + 0.01563 = 0.03202$.
   - **Doc B**: $\frac{1}{63} + \frac{1}{61} = 0.01587 + 0.01639 = 0.03226$.
   - **Doc C**: $\frac{1}{62} + \frac{1}{62} = 0.01613 + 0.01613 = 0.03226$.
2. **Compare with pure dense search**:
   Pure dense search ranked `Doc A` 4th because `"E402"` is an opaque error code, whereas RRF promotes `Doc A` right alongside `Doc B` and `Doc C` without needing to calibrate incompatible BM25 and cosine score scales!
3. **Takeaway**: RRF's rank-based formula makes hybrid retrieval robust, scale-free, and dramatically better at queries mixing exact codes with natural language.

# Optional: NLP after static embeddings — LLMs, RAG, and tokenizers

> This lesson is **optional**. It does **not** replace Word2Vec / GloVe geometry. The existing lesson already points at contextual BERT vectors; this note covers the **application stack** that took over NLP products in 2022–2026.

Static embeddings are still the right first model of “nearby = similar.” Almost no new product *stops* there: they use a **decoder LM** plus retrieval, tools, or structured extraction.

## 1. From one vector per type to a generated answer

A 2024–2026 NLP service is usually:

1. Tokenizer (BPE / Unigram: [tiktoken](https://github.com/openai/tiktoken), SentencePiece).
2. Instruction-tuned Transformer (Chapter 08 optional).
3. Optional **RAG**: embed the query, fetch chunks, condition the LM.

The embedding table $$E\in\mathbb{R}^{V\times d}$$ from this chapter is still layer 0 of that LM. What you ship is the rest of the stack.

## 2. Concrete applications

### Retrieval-augmented generation

[Lewis et al., 2020](https://arxiv.org/abs/2005.11401) (RAG; ubiquitous after 2022) and later Dense Passage / [E5](https://arxiv.org/abs/2212.03533) / [GTE](https://arxiv.org/abs/2308.03281) embedders turn “search + generate” into the default enterprise NLP pattern. Evaluate with groundedness, not only BLEU.

### Structured extraction and classification

Instead of a custom BiLSTM-CRF, teams prompt or LoRA-tune an LM to emit JSON (NER, sentiment, ticket routing). Keep a small encoder (BERT/DeBERTa) when latency and calibration beat a 7B decoder.

### Tokenization as an applied skill

Subword vocabularies decide cost and multilingual quality. [Sennrich et al., BPE](https://arxiv.org/abs/1508.07909) is old; **using** tiktoken / Llama tokenizers on messy DS text is the 2022–2026 skill (off-by-one context limits, undocumented special tokens).

## 3. Widely used software

- [huggingface/transformers](https://github.com/huggingface/transformers) + [datasets](https://github.com/huggingface/datasets).
- [langchain](https://github.com/langchain-ai/langchain) / [llama_index](https://github.com/run-llama/llama_index) — RAG glue (use carefully; the algorithm is still embed–retrieve–generate).
- Sentence-Transformers / [intfloat/e5](https://huggingface.co/intfloat) — retrieval embeddings.

## 4. Citations (2022–2026)

- [Text Embeddings by Weakly-Supervised Contrastive Pre-training — E5 (Wang et al., 2022)](https://arxiv.org/abs/2212.03533).
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) — the RAG template every 2023–2026 stack reimplements.
- [Llama 2 (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288) — instruction-tuned open LM used as the generator.

## 5. How this complements the core notes

Finish the Word2Vec arithmetic and the “static vs contextual” note first. This lesson only adds **RAG, instruction-tuned generators, and modern tokenizers** — not a new skip-gram derivation.

## Video tham khảo

- **xem trước** [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs) — 3Blue1Brown, English. A static word vector is no longer the whole model; this short video is the generated-answer stack the note describes.
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown, English. Contextual embeddings inside a transformer, the replacement for one vector per word type.
- [\[1hr Talk\] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Andrej Karpathy, English. What you do with those contextual states once the model can answer.
