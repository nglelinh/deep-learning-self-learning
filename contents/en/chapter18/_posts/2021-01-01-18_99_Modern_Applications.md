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
