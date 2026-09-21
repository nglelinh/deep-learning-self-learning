---
layout: post
title: 26-06 Interviews Practice
chapter: '26'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter26
lesson_type: optional
---

# Optional: LLM interview prompts

> This lesson is **optional**. It does **not** replace 26-01–26-05. The prompts are course-written. They are **not** taken from Kashani & Ivry, *Deep Learning Interviews*, or from any blog.

Use them closed-book after you can sketch next-token loss, ICL vs SFT, the SFT → preference pipeline, KV-cache, and the three distillation families. For the course-wide practice map, stay on **01-98**.

## P1. What is actually being trained?

You have two minutes. Define an LLM as used in this chapter in one sentence, then name the loss of pretraining and the thing that loss does *not* guarantee.

**Hint.** Decoder-only, next-token, not truth.

**Discussion.** A causal Transformer LM trained with token-level cross-entropy on a corpus. The loss rewards corpus likelihood, not factuality, not instruction-following, and not a small memory footprint. Those come from later adaptations (26-02–26-04).

## P2. Prompt vs fine-tune vs retrieve

A teammate wants “the model to always answer in the company’s invoice JSON schema.” Give one prompting solution, one weight-update solution, and one reason retrieval might still be required.

**Hint.** Template / ICL; SFT or LoRA; facts vs format.

**Discussion.** A fixed schema belongs in the prompt or in SFT/LoRA so the format is stable. Retrieval (18-99) is for *which* invoice terms or SKUs exist this week — alignment and prompting will not invent a product catalog.

## P3. Why not always distill logits?

Explain two reasons a team that only has a hosted chat API would choose **response distillation** over full token-distribution KD, even if they like the 2015 loss on paper.

**Hint.** Access and $$V$$.

**Discussion.** The API typically does not expose a $$V$$-way distribution each step. Even if it did, storing or matching that vector on long sequences is expensive; sampled text is what you can legally *and* mechanically collect (and only if the terms allow training on it).

## P4. Temperature as a teaching device

A teacher is 99.9% on the correct ImageNet class. Why raise $$T$$ before the KL, and what goes wrong if you set $$T$$ huge and drop the hard-label term entirely?

**Hint.** Dark knowledge vs uniform soup.

**Discussion.** High $$T$$ lifts the interesting off-diagonal mass (dark knowledge). If $$T\to\infty$$, the target approaches uniform and the student learns “every class is equally fine,” which is not a useful teacher. The hard CE term (or a moderate $$T$$) keeps the mode anchored. Same intuition applies to a peaked token distribution, which is why LM logit KD often uses top-$$k$$ instead of a huge $$T$$ alone.

## P5. Serving before students

Name two changes you would try **before** training an 8B student from a 70B teacher you own, and one situation where you would skip those and distill anyway.

**Hint.** 26-04 then 26-05.

**Discussion.** Quantize (4-bit / GGUF), cut unused context, batch, or speculate. Distill anyway when the *architecture* must shrink (edge memory, a cheaper SKU, many LoRA specialists on a small base) or when you explicitly want a new $$\theta$$ that imitates behavior rather than the same checkpoint with fewer bits.

## P6. Rights, not headlines

In four sentences, contrast legitimate distillation with unauthorized harvesting **without** naming a lawsuit or a 2026 news item.

**Hint.** Who owns the teacher; what the API is for.

**Discussion.** If you own the teacher or its license invites synthetic-data / in-family students, distillation is a compression and product-tier tool. If you only have a customer API, the text is a service output; many terms forbid using it as a training corpus for a competing model. The tension is structural: useful APIs leak capability through the same tokens they sell. Evaluation of any public allegation belongs outside this course.

## How to use these

Speak the answer out loud in 90 seconds, then check the discussion. If you need RL math, open **21-99**, not this page. If you need classical KD formulas, open **23-01** and **26-05**.
