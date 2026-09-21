---
layout: post
title: 26-05 Model Distillation for LLMs
chapter: '26'
order: 6
owner: Deep Learning Course
lang: en
categories:
- chapter26
---

# Model Distillation for LLMs

Chapter 23 introduced **knowledge distillation (KD)** as a compression tool: a small student matches a large teacher. That picture still holds. What changes for LLMs is the *output space* (a huge token vocabulary, one step at a time) and the *interface* you have to the teacher (full weights vs text only).

This lesson refreshes classical KD, explains why the 2015 recipe is awkward for autoregressive LMs, then presents three modern families in original teaching language. It is **not** a reprint of any blog or paper.

## 1. Classical KD refresh (the Chapter 23 core)

[Hinton, Vinyals, and Dean, 2015](https://arxiv.org/abs/1503.02531), *Distilling the Knowledge in a Neural Network*, trained a student on the teacher’s **soft** class distribution, not only on one-hot labels.

**Why soft targets help.** A hard label says “this image is class 7.” A trained teacher might say “7 with 0.80, 3 with 0.15, 8 with 0.04, …” Those off-diagonal masses encode *which mistakes are reasonable* — similarity structure Hinton et al. called **dark knowledge**. The student sees a richer gradient than “right vs wrong.”

**Temperature.** At $$T = 1$$ a confident teacher is nearly one-hot, so the extra signal vanishes. Softmax with temperature

$$p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

flattens the distribution as $$T > 1$$ grows. The student matches $$p_{\mathrm{teacher}}^{(T)}$$ (typically with a KL divergence) and often still matches the hard label. A common blend is

$$\mathcal{L} = \alpha\, T^{2}\, \mathrm{KL}\big(p_{\mathrm{teacher}}^{(T)} \,\|\, p_{\mathrm{student}}^{(T)}\big) + (1-\alpha)\,\mathrm{CE}(y, z_{\mathrm{student}}).$$

The $$T^{2}$$ factor keeps gradient scale from collapsing as you raise $$T$$ (see the 2015 note and the Chapter 23 snippet). $$\alpha$$ trades “imitate the teacher” against “hit the dataset label.”

That is the right first model for **fixed-$$K$$ classification** (ImageNet, speech senones, a 10-way head). Review **23-01** if this paragraph is new.

## 2. Why classical logit KD is awkward for huge-vocabulary LMs

A decoder LM is a classifier *at every position*, with $$K = V$$ in the tens or hundreds of thousands, and the “example” is a *sequence* of such decisions.

Practical friction:

- **Cost.** Storing or streaming a full teacher distribution per token is $$O(V)$$ extra memory and I/O. For long sequences that dominates the student’s own logits.
- **Sparsity.** After a small $$T$$, almost all of the $$V$$ mass sits on a handful of tokens. You are paying to match near-zeros unless you truncate to a top-$$k$$ or a sample.
- **Exposure.** Training on teacher logits usually needs a **white-box** (or at least logit-exposing) teacher. Public chat APIs typically return *text*, maybe a few logprobs, not a $$V$$-vector per step.
- **Sequence, not i.i.d. images.** Error compounds along the rollout. Matching a teacher *token distribution given the teacher’s prefix* is not the same as matching the teacher’s *completed answers*.

So teams still use logit KD **in house** when they own both models (and often restrict it to top-$$k$$ or selected positions). They do not treat the 2015 formula as a drop-in for “make a cheap clone of a closed API model.”

## 3. Three modern families (original grouping)

Names vary across papers. Group by **what the student is asked to match** and **what access you need**.

### 3.1 Synthetic-data / response distillation (text in, text out)

The teacher generates completions — answers, step-by-step rationales, code, rewritten documents. You store `(prompt, teacher_text)` and train the student with ordinary next-token SFT on that text (sometimes mixed with human data).

This is the dominant **black-box** pattern: you only need samples from the teacher, not $$z$$ or hidden states. Instruction-following clones of the early 2020s (research projects that fine-tuned small LMs on outputs of a stronger chat model) are this family. Later work asks the teacher for *rationales* and trains the student to produce the reasoning tokens as well as the answer, so the student gets a longer, more structured target than a short final string.

**What transfers.** Surface behavior and, if the prompts cover the task, some of the teacher’s *decisions*. **What does not automatically transfer.** Calibration of the full $$p(\cdot\mid x_{<t})$$, internal features, or capabilities the prompt set never elicited.

Because the student only sees sampled strings, two teacher samples for the same prompt can disagree. Dataset design (temperature, filtering, mixing real labels) matters as much as the loss.

### 3.2 Feature / hidden-state distillation (white-box)

The student matches intermediate activations: hidden states, attention maps, or a projection of the teacher’s residual stream. This needs **aligned depths or learned adapters** and access to teacher internals. It is the LM analogue of hint losses used in vision KD.

Use it when you control the teacher checkpoint and want the student to share *representational geometry*, not only final strings. It is a poor fit for an API you cannot instrument.

### 3.3 Logit / token-distribution distillation (white-box)

Apply the 2015 idea at each position: match $$p_{\mathrm{teacher}}(\cdot \mid x_{<t})$$ and $$p_{\mathrm{student}}(\cdot \mid x_{<t})$$, usually with temperature and often with a reverse-KL or top-$$k$$ variant so the student does not waste capacity on the tail. [Sanh et al., 2019](https://arxiv.org/abs/1910.01108) (DistilBERT) is the classic *encoder* example; generative LM papers explore related losses under names such as MiniLLM ([Gu et al., 2024](https://arxiv.org/abs/2306.08543)).

This is the most faithful *distributional* copy, and the most demanding: same (or mapped) tokenizer, stored or on-the-fly teacher logits, and usually the same prefixing policy.

| Family | Student target | Teacher access | Typical use |
| --- | --- | --- | --- |
| Response / synthetic data | Teacher *text* (and optional rationales) | Samples only | Open teachers, product SFT sets, most public recipes |
| Feature / hidden state | Layer activations | White-box | In-house, related architectures |
| Logit / token distribution | Softmax over $$V$$ | White-box (or full logprobs) | In-house, same tokenizer |

Many production pipelines **mix** them: generate a synthetic SFT set, then add a cheap logit term on a subset of tokens if both models run locally.

## 4. Legitimate in-house use vs unauthorized harvesting

Distillation is ordinary engineering when you **have the right to train on the teacher’s outputs or weights**:

- shrinking your own frontier model into a faster tier (mobile, batch, on-prem),
- following an **open license** that lists distillation or synthetic data as an intended use (several large open checkpoints have been released with that story),
- research on public models whose terms allow it.

The structural tension is not a mystery about any one company. **A model that is useful through an API emits text that is also a training signal.** Providers therefore put **terms of service** and technical controls (rate limits, abuse detection) around using outputs to build a competing model. That is a contract and product-integrity issue: capability leaks through the same channel you opened for customers.

This course does **not** recap news-cycle allegations, lawsuit claims, or unverified incident counts. Those change quickly and are easy to get wrong. If you need the industry debate as extra reading, use a tutorial that frames the *engineering* first — for example Machine Learning Mastery’s [A Gentle Introduction to Model Distillation](https://machinelearningmastery.com/a-gentle-introduction-to-model-distillation/) (Chugani, 2026) — and treat any controversy section as **reporting to verify**, not as a primary source.

**Self-study rule.** If you do not own the teacher and the terms forbid training on outputs, do not build a distillation set from that API. Use an openly licensed teacher, your own model, or public datasets.

## 5. Tiny code sketches

### 5.1 Temperature-scaled KD for a *fixed* classifier

This is the Chapter 23 setting, written so the $$T^{2}$$ and KL direction are explicit. It is **not** a full LM trainer.

```python
import torch
import torch.nn.functional as F

def classification_kd_loss(student_logits, teacher_logits, labels,
                           temperature=4.0, alpha=0.7):
    """Soft-target KD for a shared, small class set.

    student_logits, teacher_logits: (batch, num_classes)
    labels: (batch,) integer class ids
    """
    t = temperature
    hard = F.cross_entropy(student_logits, labels)
    log_p_s = F.log_softmax(student_logits / t, dim=-1)
    p_t = F.softmax(teacher_logits / t, dim=-1)
    # KL(teacher || student); T^2 restores gradient scale as T grows
    soft = F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)
    return alpha * soft + (1.0 - alpha) * hard
```

To imagine the LM analogue, think of `num_classes = vocab_size` and a loop over time, with `ignore_index` on padding — and then remember why people switch to top-$$k$$ or to text SFT instead.

### 5.2 Response distillation *is* a fine-tune dataset

No new loss is required. You build rows the student will see as ordinary instruction data:

```python
# Each row is teacher-generated text the student should assign high likelihood to.
sft_rows = [
    {
        "prompt": "Give a one-paragraph intuition for dropout.",
        "completion": "Dropout randomly masks units at train time so ...",
    },
    {
        "prompt": "Now show a tiny numeric example.",
        "completion": "Suppose a hidden layer has four units and p=0.5 ...",
    },
]
# Train with next-token CE on `completion` (and the chat template), same as 26-01 / 26-02.
# Optional: keep a human-written subset so the student does not only copy teacher quirks.
```

If the teacher also wrote a rationale, store it *inside* `completion` (or as a second field you concatenate). The student is still doing SFT; the novelty is **who wrote the targets**.

## 6. Further reading

- Hinton, G., Vinyals, O., and Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). Historical source for soft targets, temperature, and dark knowledge.
- Machine Learning Mastery — [A Gentle Introduction to Model Distillation](https://machinelearningmastery.com/a-gentle-introduction-to-model-distillation/) (Chugani, 2026). Extra tutorial framing of classical vs LLM-era families; **cited as further reading, not copied here**.
- This course: **23-01 Model Compression** (pruning, quantization, classical KD) and **23-99** (GPTQ / AWQ / speculative decoding — complementary, not a substitute for a student). Serving-time draft-and-verify is also in **26-04**; FLOP / memory accounting before you pick a student size is in **26-07**.
- Optional technical pointers (not assigned): DistilBERT ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)); rationale-style distillation such as Distilling Step-by-Step ([Hsieh et al., 2023](https://arxiv.org/abs/2305.02301)); MiniLLM ([Gu et al., 2024](https://arxiv.org/abs/2306.08543)).

## Key takeaways

- Classical KD = match a teacher’s *soft class distribution*; temperature reveals dark knowledge.
- Autoregressive LMs make full-logit matching expensive and often inaccessible.
- Modern practice clusters into **response (synthetic)**, **feature**, and **logit** distillation.
- In-house / licensed distillation is standard efficiency work; training on a forbidden API is a terms-and-leakage problem, not a new algorithm.
- A temperature-scaled KL is the right *picture*; a teacher-written SFT file is what most LLM students actually train on.
