---
layout: post
title: 26-02 Prompting, In-Context Learning, and Instruction Following
chapter: '26'
order: 3
owner: Deep Learning Course
lang: en
categories:
- chapter26
---

How can a frozen neural network—whose weights $\theta$ do not change by a single bit—learn a brand-new classification task just by reading four examples inside its prompt? In-Context Learning (ICL) works because self-attention conditions every next-token distribution $p_\theta(y \mid x_1, y_1, \dots, x_k, y_k, x_{\text{query}})$ on the entire prefix: the prompt acts as a temporary soft program that selects and composes latent skills already crystallized during pretraining.

## Key ideas: Mechanics of Prompting & In-Context Learning

- **Pillar 1 — Prefix Conditioning ($p_\theta(x_t \mid x_{<t})$)**: Frames every system prompt, few-shot example, and retrieved document strictly as conditioning context steering the same frozen next-token distribution.
- **Pillar 2 — Implicit Bayesian Task Selection & Induction Heads**: Explains how few-shot examples narrow posterior uncertainty over *which* latent task concept is active, assisted by induction-head circuits copying patterns across context.
- **Pillar 3 — Base Pretrained vs. Instruction-Tuned Behavior**: Distinguishes a raw base model (which completes *"Translate to French: hello"* with *"Translate to Spanish: ..."*) from an SFT/chat-tuned model trained on role-delimited assistant turns.
- **Pillar 4 — Prompt Sensitivity & Structured Verification**: Addresses order sensitivity, label-space bias, and why Chain-of-Thought (CoT) unlocks extra serial computation steps per token before committing to an answer.

### Worked example: How Few-Shot Context Shifts Next-Token Probabilities via Bayes' Rule

Suppose a pretrained LM implicitly models two equally likely latent tasks ($P(T_{\text{syn}}) = P(T_{\text{ant}}) = 0.5$): Task $T_{\text{syn}}$ outputs synonyms (predicts `"hot -> warm"` with prob $0.8$, `"hot -> cold"` with prob $0.1$) and Task $T_{\text{ant}}$ outputs antonyms (predicts `"hot -> cold"` with prob $0.8$, `"hot -> warm"` with prob $0.1$).

1. **Zero-shot prediction for `"big -> ?"`** (before seeing any examples):
   Both tasks have posterior $0.5$, so `"large"` and `"small"` tie at $0.45$ each—the model is ambiguous!
2. **Now prepend 2 few-shot antonym examples**: `"hot -> cold\nfast -> slow\n"` (each has likelihood $0.8$ under $T_{\text{ant}}$ and $0.1$ under $T_{\text{syn}}$):
   - Unnormalized posterior for $T_{\text{ant}}$: $0.5 \times 0.8^2 = 0.320$.
   - Unnormalized posterior for $T_{\text{syn}}$: $0.5 \times 0.1^2 = 0.005$.
   - Posterior $P(T_{\text{ant}} \mid \text{prompt}) = \frac{0.320}{0.325} = 98.46\%$!
3. **Resulting predictive probability for `"big -> small"`**:
   $$0.9846(0.8) + 0.0154(0.1) = 78.9\% \text{ vs. } 11.1\% \text{ for } \text{"large"}.$$
4. **Takeaway**: Just two clean examples collapse task ambiguity from $50/50$ to $98.5\%$ without a single gradient update—illustrating why example quality and format consistency dominate prompt engineering.

# Prompting, In-Context Learning, and Instruction Following

After pretraining, the weights already implement a next-token machine. **Prompting** is the cheapest way to steer that machine: you change the prefix $$x_{<t}$$, not $$\theta$$. **Fine-tuning** changes $$\theta$$. Most LLM products use both, in that order.

This lesson separates three interfaces that interviews and blogs often mash together: zero-shot prompting, in-context (few-shot) learning, and instruction-following after supervised adaptation.

## 1. The prompt is part of the sequence

Because $$p_\theta(x_t \mid x_{<t})$$ conditions on *everything* in the window, a system prompt, retrieved documents, tool results, and the user’s question are just more tokens. There is no separate “prompt input” in the architecture. Formatting conventions (chat templates, XML-ish tags, JSON schemas) matter because they are **stable prefixes** the model saw, or was later trained to expect.

A useful habit: when a model “ignores the instruction,” first check whether the instruction is still inside the context window and whether the chat template is the one the checkpoint was adapted with. That is a data/format bug more often than an architecture bug.

## 2. Zero-shot vs in-context learning

**Zero-shot.** You describe the task in language (“Translate to Vietnamese:”) and decode. The model generalizes from pretraining patterns that look like that request.

**In-context learning (ICL) / few-shot.** You put a handful of input–output exemplars in the prefix, then the new input. [Brown et al., 2020](https://arxiv.org/abs/2005.14165) (GPT-3) showed that at sufficient scale this can look like task learning *without a gradient step*. The mechanism is still next-token prediction: the exemplars make the desired continuation more likely.

ICL is **not** the same as fine-tuning:

| | In-context learning | Classical fine-tuning |
| --- | --- | --- |
| What changes | Tokens in the window | Parameters $$\theta$$ (or adapters) |
| Persistence | Dies when the prompt is gone | Survives across requests |
| Cost | Extra prefill compute / cache | Training compute, then cheaper prompts |
| Data | A few exemplars you can edit live | A dataset and an optimizer |

When to prefer ICL: the task is new, labeled data are scarce, or you must change behavior per request (different customers, different schemas). When to prefer fine-tuning: the behavior must be stable, the prompt would become huge, or you need a smaller/faster specialist (often after distillation — **26-05**).

**Chain-of-thought** ([Wei et al., 2022](https://arxiv.org/abs/2201.11903)) is a prompting pattern: ask the model to emit intermediate steps before the answer. It does not add a new module; it reallocates decode tokens to a longer, more checkable computation. Later “reasoning” models *train* that habit (Chapter 25-99). Do not confuse a prompt trick with a new architecture.

## 3. Instruction-following is usually not raw pretraining

A base LM completes text. If the prefix is a blog post, it continues the blog post. Users want a **helpful assistant**: follow the request, refuse when appropriate, use a consistent tone.

The standard first adaptation is **supervised fine-tuning (SFT)** on (instruction, response) pairs — demonstrations written or filtered by humans, or generated by a stronger model and then edited. The loss is still next-token cross-entropy, but the *data distribution* is now “answer this request,” not “continue the web.”

After SFT, prompting still matters (tools, RAG, style), but the model is much more likely to treat the user text as a *task* rather than as a document to extend. Preference optimization (next lesson) then ranks *which* valid answers are preferred.

Classical transfer learning (Chapter 15) updates a backbone for a fixed head and label set. Instruction SFT updates the same generative head on many tasks written in language. Same optimizer family; different interface.

## 4. A small prompting checklist (self-study)

Write these down and try them on any open instruct model:

1. State the **role** and the **output contract** (plain text, JSON keys, language).
2. Put **non-negotiable constraints** before long retrieved context (they can be pushed out of $$L$$ if you append them last).
3. For ICL, keep exemplars **format-identical** to the query you will ask.
4. If you need sources, retrieve first (Chapter 18-99); do not ask the model to invent citations.
5. If you need lasting behavior, plan SFT / adapters — do not grow a 8k-token prompt forever.

## Key takeaways

- Prompting edits the conditioning prefix; fine-tuning edits weights.
- ICL is few-shot *in the window*, not a hidden inner-loop optimizer you get for free.
- Instruction-following is mostly a **data** change (SFT) on top of the same next-token loss.
- Alignment preferences are a further change — lesson **26-03** — not a substitute for a clear prompt.
- Decode-time sampling (temperature, top-$$k$$ / top-$$p$$) is a serving knob, not a new prompt type — **26-04**.

## Video tham khảo

- **xem trước** [What Is In-Context Learning in Deep Learning?](https://www.youtube.com/watch?v=As9a15poQHs) — Deep Learning with Yacine, English. In-context learning is conditioning on examples inside the prompt, which is this lesson.
- [\[1hr Talk\] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — Andrej Karpathy, English. A prompt is more tokens in the same next-token model; that is the lesson's first section.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy, English. You can see a prompt as nothing but the prefix the sampler continues.
