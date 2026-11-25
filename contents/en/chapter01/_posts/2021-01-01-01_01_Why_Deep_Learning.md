---
layout: post
title: 01-01 Why Deep Learning?
chapter: '01'
order: 5
owner: Deep Learning Course
lang: en
categories:
- chapter01
---

> *Every new deep learning engineer faces a critical architectural question on their very first production project: "Should I build a deep neural network here, or will XGBoost beat it in ten minutes?" Knowing why deep learning wins—and where it loses—is what separates an engineer from a hype follower.*

## Key ideas: The Case for (and Against) Deep Learning

1. **End-to-End Representation Discovery**: Eliminates fragile manual feature pipelines on unstructured modalities (pixels, waveforms, raw text, graphs) where human-engineered descriptors plateau early.
2. **Data & Compute Scaling Laws**: Classical ML error curves flatten out past moderate dataset sizes, whereas deep models continue improving predictably as data, parameters, and compute scale together.
3. **Inductive Bias & Tabular Boundaries**: On small-to-medium structured tabular datasets with heterogeneous features, gradient-boosted decision trees (XGBoost, LightGBM) still often match or beat neural nets at a fraction of the cost.
4. **Ecosystem & Transfer Learning Leverage**: Pretrained foundation models let small teams inherit billions of tokens or images of prior representation learning via fine-tuning rather than training from scratch.

### Worked example: Quantifying Scaling Break-Even Between Classical ML and Deep Learning

Suppose on a document-classification benchmark, a TF-IDF + Logistic Regression baseline achieves error rate $$E_{\text{ML}}(N) = 0.08 + \frac{2.0}{\sqrt{N}}$$ (floor of 8% due to fixed bag-of-words features), while a fine-tuned Transformer achieves $$E_{\text{DL}}(N) = 0.02 + \frac{8.0}{\sqrt{N}}$$ (lower 2% floor, higher sample complexity). Find the dataset size $$N^*$$ where deep learning overtakes classical ML.

1. **Test at Small Data ($$N = 400$$, so $$\sqrt{N} = 20$$)**:
   - Classical ML: $$E_{\text{ML}}(400) = 0.08 + \frac{2.0}{20} = 0.08 + 0.10 = 18.0\%$$ error.
   - Deep Learning: $$E_{\text{DL}}(400) = 0.02 + \frac{8.0}{20} = 0.02 + 0.40 = 42.0\%$$ error.
   At $$N = 400$$, classical ML crushes deep learning!
2. **Solve for the Exact Crossover Point $$E_{\text{DL}}(N^*) = E_{\text{ML}}(N^*)$$**:
   $$0.02 + \frac{8.0}{\sqrt{N^*}} = 0.08 + \frac{2.0}{\sqrt{N^*}} \implies \frac{6.0}{\sqrt{N^*}} = 0.06 \implies \sqrt{N^*} = 100 \implies N^* = 10{,}000$$
3. **Verify at Large Data ($$N = 40{,}000$$, so $$\sqrt{N} = 200$$)**:
   - Classical ML: $$0.08 + \frac{2.0}{200} = 9.0\%$$ error.
   - Deep Learning: $$0.02 + \frac{8.0}{200} = 6.0\%$$ error.
   This exact trade-off between **approximation floor** (bias) and **sample complexity** (variance) governs every real-world model selection decision.

Why deep learning, and when to use it:

1. **Limitations of Traditional ML and What Makes DL Different**
2. **When to Use Deep Learning and the Ecosystem** — success stories, data/compute, workflow, tools, books, trends

## Video tham khảo

- **xem trước** [Machine Learning vs Deep Learning](https://www.youtube.com/watch?v=q6kJ71tEYqM) — IBM Technology, English. It states what deep learning adds beyond classical machine learning, which is this block's question.
- [MIT Introduction to Deep Learning (2024) | 6.S191](https://www.youtube.com/watch?v=ErnWZxJovaM) — Alexander Amini, English. The 'why deep learning' segment of the MIT lecture lists the scaling and data reasons the lessons expand.
- [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk) — 3Blue1Brown, English. Shows the hierarchical features that make depth useful on images, the success case this block keeps citing.
