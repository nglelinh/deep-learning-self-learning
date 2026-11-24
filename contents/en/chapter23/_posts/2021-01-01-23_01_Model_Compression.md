---
layout: post
title: 23-01 Model Compression and Efficiency
chapter: '23'
order: 2
owner: Deep Learning Course
lang: en
categories:
- chapter23
---

Over-parameterized neural networks are remarkably redundant: studies on the Lottery Ticket Hypothesis and post-training quantization show you can zero out 70% of weights or round 32-bit floats down to 8-bit integers and lose less than half a percent of accuracy. Understanding the exact arithmetic of affine quantization, temperature-scaled KL distillation, and depthwise-separable FLOPs lets you engineer compression budgets quantitatively instead of guessing.

## Background Knowledge Map: Mathematical Mechanics of Model Compression

- **Pillar 1 — Affine & Symmetric Quantization Arithmetic**: Maps floating-point ranges $[\alpha, \beta]$ to $b$-bit integers via scale $s = \frac{\beta - \alpha}{2^b - 1}$ and zero-point $z$, with Quantization-Aware Training (QAT) using Straight-Through Estimators (STE) to differentiate through `round()`.
- **Pillar 2 — Temperature-Scaled Knowledge Distillation**: Softens teacher and student logits via $p_i^{(T)} = \operatorname{softmax}(z_i / T)$ and scales KL loss by $T^2$ so gradient magnitudes stay invariant to $T$.
- **Pillar 3 — Structured vs. Unstructured Pruning**: Balances fine-grained weight-magnitude pruning against structured filter/neuron removal that immediately shrinks dense GEMM matrix dimensions.
- **Pillar 4 — Depthwise Separable Convolutions**: Factors a $K \times K \times C_{\text{in}} \times C_{\text{out}}$ convolution into a $K \times K$ spatial pass plus a $1 \times 1$ pointwise mix, cutting MACs by roughly $K^2 \approx 8\times$–$9\times$.

### Step-by-Step Worked Example (Read Slowly): Symmetric INT8 Quantization & Depthwise-Separable FLOP Ratio

**Part A — Symmetric INT8 Quantization** ($q \in [-127, 127]$) of weight vector $w = [-0.635,\; 0.120,\; 1.270]$:
1. Find max absolute value: $\alpha = \max |w| = 1.270$.
2. Compute scale: $s = \frac{1.270}{127} = 0.010$.
3. Quantize $q = \operatorname{round}(w / s)$:
   $$q = [\operatorname{round}(-63.5),\; \operatorname{round}(12.0),\; \operatorname{round}(127.0)] = [-64,\; 12,\; 127].$$
4. Dequantize $\hat{w} = s \cdot q$:
   $$\hat{w} = [-0.640,\; 0.120,\; 1.270], \quad \text{max error } |w - \hat{w}| = 0.005 \text{ (exactly } s/2\text{!)}.$$

**Part B — Depthwise Separable Parameter Reduction** for $K = 3, C_{\text{in}} = 64, C_{\text{out}} = 128$:
- Standard Conv2d: $3 \times 3 \times 64 \times 128 = 73{,}728$ weights.
- Depthwise ($3 \times 3 \times 64$) + Pointwise ($1 \times 1 \times 64 \times 128$): $576 + 8{,}192 = 8{,}768$ weights ($8.41\times$ fewer!).
5. **Takeaway**: Both quantization error ($s/2$) and separable-conv savings ($\frac{1}{C_{\text{out}}} + \frac{1}{K^2}$) can be calculated on the back of an envelope before touching a GPU.

# Efficient Deep Learning: Compression and Acceleration

## 1. Concept Overview

As deep learning models have grown to billions of parameters, deploying them on resource-constrained devices (mobile phones, embedded systems, edge devices) or serving them at scale (millions of queries) has become challenging. Model compression techniques reduce model size, memory footprint, and computational requirements while maintaining accuracy, enabling deployment in scenarios where full models are impractical. These techniques—pruning, quantization, knowledge distillation, and efficient architectures—represent crucial engineering innovations making deep learning accessible beyond cloud servers with powerful GPUs.

Pruning removes unnecessary parameters or connections, exploiting the observation that many neural network weights contribute minimally to outputs. Studies show networks remain effective after removing 50-90% of weights, suggesting significant redundancy. Structured pruning removes entire filters or layers, providing hardware-friendly speedups. Unstructured pruning removes individual weights, achieving higher compression but requiring specialized hardware for acceleration.

Quantization reduces numerical precision, representing weights and activations with fewer bits (8-bit integers instead of 32-bit floats), reducing memory by 4× and enabling faster integer arithmetic on many processors. Post-training quantization applies to trained models without retraining. Quantization-aware training includes quantization in the training loop, allowing the network to adapt to reduced precision.

Knowledge distillation transfers knowledge from large "teacher" networks to small "student" networks by training students to match teacher predictions (soft targets) rather than just hard labels. Students learn from teacher's uncertainty and similarities between classes, often achieving better performance than training on labels alone despite being much smaller. **For LLM-era distillation** — why full-vocab logit matching is awkward, synthetic-data / response distillation, and legitimate vs unauthorized use — see **Chapter 26**, lesson **26-05**. This lesson stays with the classical, fixed-output-space picture.

Efficient architectures like MobileNet and EfficientNet are designed for efficiency from the start through depthwise separable convolutions, neural architecture search, and careful scaling. These achieve competitive accuracy with fraction of computation/parameters compared to standard architectures.

## 2. Mathematical Foundation

### Pruning

Define importance score for parameter $$w$$:

$$I(w) = |\frac{\partial \mathcal{L}}{\partial w}|$$ (magnitude pruning)

or

$$I(w) = |w \cdot \frac{\partial \mathcal{L}}{\partial w}|$$ (Taylor expansion approximation)

Remove weights with $$I(w) < \tau$$ (threshold). After pruning, fine-tune remaining weights.

### Quantization

Map continuous values to discrete levels. For 8-bit quantization:

$$w_{\text{quant}} = \text{round}\left(\frac{w - w_{\min}}{w_{\max} - w_{\min}} \cdot 255\right)$$

Dequantize for computation:

$$w \approx w_{\min} + w_{\text{quant}} \cdot \frac{w_{\max} - w_{\min}}{255}$$

### Knowledge Distillation

Student network trained on soft targets from teacher:

$$\mathcal{L} = \alpha \mathcal{L}_{\text{hard}}(y, \hat{y}_{\text{student}}) + (1-\alpha) \mathcal{L}_{\text{soft}}(\hat{y}_{\text{teacher}}, \hat{y}_{\text{student}})$$

Soft targets use temperature $$T$$:

$$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Higher $$T$$ creates softer distributions revealing teacher's uncertainty.

## 3. Example / Intuition

Imagine compressing a ResNet-50 (25M parameters, 4GB memory). Pruning might find that 60% of weights have magnitude < 0.001 and contribute negligibly. Removing these yields 10M parameters, 1.6GB—60% reduction with minimal accuracy loss.

Quantization to INT8 (8-bit integers) from FP32 (32-bit floats) gives 4× memory reduction: 10M parameters now only 10MB vs 40MB. Combined with pruning: 2.5GB → 1GB → 250MB—10× total compression.

Knowledge distillation transfers this compressed model's knowledge to a MobileNet (4M parameters). The MobileNet learns from ResNet's soft predictions, achieving 75% accuracy versus 70% training on labels alone—the extra 5% comes from learning what ResNet thinks about hard examples.

Final: 4M parameters, 40MB INT8 model achieving 75% accuracy versus original 25M parameters, 4GB FP32 achieving 78% accuracy. 100× smaller, 10× faster inference, only 3% accuracy loss—enabling mobile deployment!

## 4. Code Snippet

```python
import torch
import torch.nn as nn

# Pruning example
def magnitude_prune(model, sparsity=0.5):
    """Remove smallest magnitude weights"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight.data.abs()
            threshold = weight.quantile(sparsity)
            mask = weight > threshold
            module.weight.data *= mask.float()

# Quantization
def quantize_tensor(tensor, num_bits=8):
    """Quantize tensor to num_bits"""
    qmin = 0
    qmax = 2**num_bits - 1
    
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    
    q = torch.round((tensor - min_val) / scale).clamp(qmin, qmax)
    
    return q, scale, min_val

# Knowledge distillation
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    """Combine hard and soft targets"""
    hard_loss = F.cross_entropy(student_logits, labels)
    
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * T * T
    
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

## 5. Related Concepts

Model compression connects to neural architecture search (NAS), which discovers efficient architectures automatically. NAS explores architecture space, evaluating candidates by accuracy-efficiency tradeoffs.

Compression relates to lottery ticket hypothesis: random networks contain subnetworks that, when trained, match full network performance. Finding these "winning tickets" provides alternative to post-training pruning.

## 6. Fundamental Papers

**["Learning both Weights and Connections for Efficient Neural Networks" (2015)](https://arxiv.org/abs/1506.02626)**  
*Authors*: Song Han, Jeff Pool, John Tran, William Dally  
Introduced magnitude-based pruning achieving 9-13× compression on AlexNet and VGGNet with minimal accuracy loss.

**["Distilling the Knowledge in a Neural Network" (2015)](https://arxiv.org/abs/1503.02531)**  
*Authors*: Geoffrey Hinton, Oriol Vinyals, Jeff Dean  
Knowledge distillation paper showing student networks learn better from teacher predictions than from labels alone.

**["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (2017)](https://arxiv.org/abs/1704.04861)**  
*Authors*: Andrew Howard et al.  
MobileNets use depthwise separable convolutions, reducing computation 8-9× versus standard convolutions while maintaining accuracy.

## Key Takeaways

Model compression makes deep learning practical on resource-constrained devices through pruning (removing unnecessary parameters), quantization (reducing numerical precision), knowledge distillation (transferring large model knowledge to small models), and efficient architectures (designed for efficiency from scratch). These techniques enable 10-100× compression with minimal accuracy loss, democratizing deep learning deployment beyond cloud servers to edge devices, enabling real-time inference, and reducing environmental impact through lower computational requirements.


## Video tham khảo

- **xem trước** [Lecture 15 | Efficient Methods and Hardware for Deep Learning](https://www.youtube.com/watch?v=eZdOkDtYMoo) — Stanford University School of Engineering, English. Pruning, quantization, and the hardware constraints together.
- [How LLMs survive in low precision | Quantization Fundamentals](https://www.youtube.com/watch?v=qoQJq5UwV1c) — Julia Turc, English. Low-precision weights, the quantization section's subject.
- [2.3 Knowledge Distillation | Geoffery Hinton | Soft Labels | Softmax with Temperature](https://www.youtube.com/watch?v=g1r4Esohw-g) — AI_Snippets, English. Hinton's soft labels and temperature, the classical distillation the lesson stops at before Chapter 26.
