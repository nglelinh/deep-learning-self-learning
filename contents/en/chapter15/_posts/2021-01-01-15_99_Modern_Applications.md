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

When a foundation model has 70 billion parameters, classical full fine-tuning requires over a terabyte of GPU memory—putting adaptation out of reach for almost every team. Parameter-Efficient Fine-Tuning (PEFT) changed the economics of transfer learning overnight: by freezing the giant base weight matrix $W_0$ and learning a tiny low-rank update $\Delta W = B A$ (LoRA) on top of 4-bit quantized base weights (QLoRA), a single workstation GPU can specialize a frontier model while storing a 50 MB adapter instead of a 140 GB checkpoint.

## Background Knowledge Map: Modern Transfer Learning via PEFT (2022–2026)

- **Pillar 1 — Low-Rank Adaptation (LoRA)**: Parameterizes weight updates as $W = W_0 + \frac{\alpha}{r} B A$ with $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$ ($r \ll \min(d_{\text{in}}, d_{\text{out}})$), adding zero inference latency once merged into $W_0$.
- **Pillar 2 — Quantized Transfer (QLoRA & NF4)**: Stores frozen $W_0$ in 4-bit NormalFloat with double quantization and dequantizes on-the-fly into BF16 during forward/backward passes, slashing memory by $\sim 4\times$.
- **Pillar 3 — Multimodal & Multi-Tenant Adapters**: Enables serving dozens of domain-specific LoRA adapters dynamically on top of a single shared frozen base model in GPU memory.
- **Pillar 4 — Limits of Low-Rank Updates**: Clarifies when full fine-tuning or continual pretraining still beats LoRA (e.g., injecting massive new factual knowledge or learning an entirely new language/modality).

### Step-by-Step Worked Example (Read Slowly): LoRA Parameter Reduction & Rank-2 Matrix Update

Consider a Transformer projection matrix $W_0 \in \mathbb{R}^{4096 \times 4096}$ adapted with LoRA rank $r = 16$ and scaling $\alpha = 32$.

1. **Parameter count comparison**:
   - Full matrix $W_0$: $4096 \times 4096 = 16{,}777{,}216$ parameters.
   - LoRA matrices $A \in \mathbb{R}^{16 \times 4096}$ and $B \in \mathbb{R}^{4096 \times 16}$: $2 \times 16 \times 4096 = 131{,}072$ parameters.
   - Reduction factor: $\frac{16{,}777{,}216}{131{,}072} = 128\times$ fewer trainable parameters ($0.78\%$).
2. **Concrete $2 \times 2$ rank-1 LoRA forward pass** ($r = 1, \alpha = 2 \implies \alpha/r = 2$):
   Let $W_0 = \begin{bmatrix}1 & 0\\ 0 & 1\end{bmatrix}$, $B = \begin{bmatrix}0.5\\ -0.5\end{bmatrix}$, $A = \begin{bmatrix}1.0 & 2.0\end{bmatrix}$, and input $x = \begin{bmatrix}2\\ 1\end{bmatrix}$.
   - Base path: $W_0 x = [2,\; 1]^\top$.
   - LoRA path: $A x = 1(2) + 2(1) = 4$; $\; B(Ax) = [2,\; -2]^\top$; $\; \frac{\alpha}{r} B A x = 2[2,\; -2]^\top = [4,\; -4]^\top$.
   - Combined output: $y = [2, 1]^\top + [4, -4]^\top = [6,\; -3]^\top$.
3. **Takeaway**: Because $B$ is initialized to $0$ and $A$ to Gaussian noise, $\Delta W = 0$ at step 0—preserving the pretrained model's exact behavior at initialization while cutting trainable parameters by $128\times$.

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

## Video tham khảo

- **xem trước** [QLoRA: Efficient Finetuning of Quantized LLMs | Tim Dettmers](https://www.youtube.com/watch?v=fQirE9N5q_Y) — AMLD Intelligence Summit, English. QLoRA is quantized low-rank fine-tuning, the transfer method this note adds. The talk is Tim Dettmers at AMLD.
- [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, English. The low-precision arithmetic QLoRA depends on.
- [Fine-tuning a Neural Network explained](https://www.youtube.com/watch?v=5T-iXNNiwIs) — deeplizard, English. Ordinary fine-tuning, so the quantized variant is a change of how, not of why.
