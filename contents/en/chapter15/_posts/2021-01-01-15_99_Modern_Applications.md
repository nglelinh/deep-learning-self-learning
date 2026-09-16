---
layout: post
title: 15-99 Modern Applications and Updates (2022–2026)
chapter: '15'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter15
lesson_type: optional
---

# Optional: transfer learning as PEFT — LoRA, QLoRA, and adapters

> This lesson is **optional**. It does **not** replace feature-extraction vs full fine-tune theory. The core notes still start from ResNet/BERT heads; this lesson is the 2022–2026 default for **billion-parameter** models.

Full fine-tuning updates every $$\theta$$. Parameter-efficient fine-tuning (PEFT) freezes $$\theta$$ and learns a small $\Delta$:

$$W' = W + \frac{\alpha}{r} BA, \quad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r\ll \min(d,k).$$

That is [LoRA (Hu et al., 2021/2022)](https://arxiv.org/abs/2106.09685). It is still *transfer learning*: the pretrained features move, but only along a low-rank subspace.

## 1. QLoRA and quantized transfer

[Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) keep the base weights in 4-bit (NF4) and train LoRA in 16-bit. A 65B model becomes fine-tunable on a single 48 GB GPU. [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) plus [Hugging Face PEFT](https://github.com/huggingface/peft) made this the applied-DS default.

## 2. Concrete applications

### Instruction and domain adapters

Instead of a new BERT for every client, teams ship **one base + many LoRA adapters** (legal, medical, SQL). Serving stacks (vLLM, SGLang) can swap adapters per request.

### Vision and multimodal PEFT

LoRA on U-Net / DiT attention (DreamBooth-LoRA, SDXL LoRA) is the consumer “train my style on 10 photos” path. Same math as language LoRA; different tensors.

### When *not* to LoRA

Tiny CNNs, from-scratch tabular nets, and cases where the whole representation must move (heavy domain shift with abundant labels) still want full fine-tune or a new head — the theory lesson’s feature-extractor vs fine-tune split.

## 3. Widely used software

- [huggingface/peft](https://github.com/huggingface/peft) — LoRA, AdaLoRA, IA3, prefix-tuning.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) + [QLoRA](https://github.com/artidoro/qlora).
- [unsloth](https://github.com/unslothai/unsloth) — faster LoRA trainers used in 2024–2026 tutorials.

## 4. Citations (2022–2026)

- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — low-rank adapters; ubiquitous after 2022.
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) — 4-bit base + LoRA.
- [The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691) — prompt tuning as the other PEFT pole.

## 5. How this complements the core notes

Do the ResNet/BERT fine-tune homework first. This lesson only adds the **low-rank / quantized** transfer recipe that replaced “unfreeze the last three layers” for LLMs.
