---
layout: post
title: 01-01-02 When to Use Deep Learning and the Ecosystem
chapter: '01'
order: 7
owner: Deep Learning Course
lang: en
categories:
- chapter01
---

## When to Use Deep Learning

### Deep Learning Excels When:

✅ **Large amounts of data** available  
✅ **Complex patterns** to learn  
✅ **High-dimensional inputs** (images, text, audio)  
✅ **End-to-end learning** desired  
✅ **Sufficient computational resources**  
✅ **Non-linear relationships** in data

### Traditional ML May Be Better When:

⚠️ Small datasets (< 1000 samples)  
⚠️ Need interpretability (medical diagnosis decisions)  
⚠️ Limited computational resources  
⚠️ Simple, well-understood problems  
⚠️ Fast training required  
⚠️ Linear relationships suffice

## Success Stories

### Computer Vision: ImageNet (2012)

**AlexNet** achieved 15.3% error rate (vs 26% for traditional methods)
- First deep learning victory in computer vision
- Sparked the deep learning revolution
- 8-layer CNN with 60M parameters

### Natural Language: Machine Translation

**Google Neural Machine Translation (2016)**
- Reduced translation errors by 60%
- Learned to translate better than phrase-based systems
- Enabled near-human quality translation

### Games: AlphaGo (2016)

- Defeated world champion Lee Sedol 4-1
- Combined deep learning with Monte Carlo tree search
- Mastered Go, considered much harder than chess
- Demonstrated creative, intuitive play

### Healthcare: Medical Imaging

**Skin Cancer Detection (2017)**
- Deep learning matched dermatologist performance
- Analyzed dermoscopic images
- Potential to democratize expert-level diagnosis

### Speech: Voice Assistants

- Near-human accuracy in speech recognition
- Enables Siri, Alexa, Google Assistant
- Works across accents and languages

### Recommendation systems: short-video feeds (TikTok-style)

![Recommendation algorithms and neural nets](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_tiktok_title_nn.jpg)
*Figure: Large-scale content recommendation powered by ML/DL. (Illustration from a recommendation-system explainer video)*

An everyday ML/DL application: the feed automatically picks videos/products for each person. A typical stack includes:

1. **Collaborative filtering** — users who engage with item A often like item B → recommend B (without “understanding” content).

![Collaborative filtering: view A → suggest B](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_collaborative_filtering.jpg)
*Figure: After someone views A, the system suggests related items (B, …). (Illustration from a recommendation-system explainer video)*

2. **Content-based recommendation** — match item features (tags, style, price, …) to find “similar” items.

![Content-based: item A vs B](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_content_based_similar.jpg)
*Figure: Finding similar items by attributes (illustrated with product icons). (Illustration from a recommendation-system explainer video)*

![Item feature tags](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_item_feature_tags.jpg)
*Figure: Items represented as sets of tags/features. (Illustration from a recommendation-system explainer video)*

3. **Cold start** — new users/items with no history → CF is weak; fall back to content, global popularity, or short surveys.

![Cold start: no history yet](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_cold_start.jpg)
*Figure: Empty state with no history — the cold-start problem. (Illustration from a recommendation-system explainer video)*

4. **Ranking & engagement signals** — likes, watch time, skips… pick candidates that “fit” in milliseconds.

![Candidates and engagement signals](/deep-learning-self-learning/img/chapter_img/chapter01/recsys_engagement_candidates.jpg)
*Figure: From many candidate videos, the system picks ones matching viewing behavior. (Illustration from a recommendation-system explainer video)*

This is deep learning at billions of users: large data, realtime feedback, multi-objective tradeoffs (retention, diversity, safety). Embeddings / two-tower models appear in Chapter 18; ε-greedy exploration in Chapter 20.

## The Data Advantage

### Why More Data Helps

Traditional ML:
```
Performance
    |     _______________
    |    /
    |   /
    |  /
    |_/________________
         Amount of Data
```

Deep Learning:
```
Performance
    |              /
    |            /
    |          /
    |        /
    |      /
    |    /
    |__/________________
         Amount of Data
```

### The Scaling Law

Empirical observation: Deep learning performance often follows:

$$\text{Error} \propto \frac{1}{(\text{Data size})^\alpha}$$

where $$\alpha \approx 0.5$$ for many tasks.

**Implication**: 
- 4× more data → 2× error reduction
- 100× more data → 10× error reduction

## Computational Requirements

### GPU Revolution

Deep learning became practical due to GPUs:

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Matrix Multiply (1000×1000) | 100ms | 5ms | 20× |
| Conv2D Layer | 1000ms | 10ms | 100× |
| Full Model Training | Days/Weeks | Hours/Days | 10-100× |

### Modern Infrastructure

- **Cloud Computing**: AWS, Google Cloud, Azure
- **Specialized Hardware**: TPUs, Neural Processing Units
- **Distributed Training**: Multi-GPU, multi-machine
- **Mixed Precision**: FP16 training for speed

## The Deep Learning Workflow

### Typical Process

1. **Data Collection**
   - Gather large dataset
   - Ensure quality and diversity

2. **Data Preparation**
   - Clean and preprocess
   - Split into train/val/test
   - Augment if needed

3. **Model Design**
   - Choose architecture
   - Define layers and connections
   - Set hyperparameters

4. **Training**
   - Initialize parameters
   - Forward pass → loss → backward pass
   - Update weights
   - Monitor metrics

5. **Evaluation**
   - Test on held-out data
   - Analyze errors
   - Visualize predictions

6. **Iteration**
   - Improve data
   - Tune architecture
   - Adjust hyperparameters

7. **Deployment**
   - Optimize for inference
   - Deploy to production
   - Monitor performance

## Tools and Frameworks

### Deep Learning Frameworks

**PyTorch**
- Dynamic computation graphs
- Pythonic interface
- Popular in research

**TensorFlow/Keras**
- Production-ready
- Extensive ecosystem
- Easy deployment

**JAX**
- Functional approach
- Fast and flexible
- Growing adoption

### Supporting Libraries

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Preprocessing, metrics

## Recommended Books for This Course

To deepen your understanding, we recommend these essential deep learning books:

### 1. **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **The definitive deep learning textbook**
- Comprehensive coverage of theory and mathematics
- Written by pioneers in the field
- Free online: [deeplearningbook.org](http://www.deeplearningbook.org/)
- **Use for**: Mathematical foundations, theoretical depth

### 2. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** by Aurélien Géron
- **Best practical implementation guide**
- Step-by-step code examples
- End-to-end ML projects
- Covers Scikit-Learn, TensorFlow, and Keras
- **Use for**: Implementation, practical projects

### 3. **Understanding Deep Learning** by Simon J.D. Prince
- **Modern, accessible introduction**
- Clear explanations with excellent visualizations
- Covers recent architectures (Transformers, etc.)
- Intuitive approach to complex concepts
- **Use for**: Building intuition, visual understanding

### 4. **MIT Deep Learning Book**
- **Rigorous academic treatment**
- Strong theoretical foundations
- Research-oriented perspective
- Mathematical proofs and derivations
- **Use for**: Academic depth, research preparation

### 5. **Deep Learning Interviews** by Shlomo Kashani and Amir Ivry
- **Free solved interview / exam problems**
- PDF: [arXiv:2201.00650](https://arxiv.org/abs/2201.00650) — GitHub: [interviews.ai](https://github.com/BoltzmannEntropy/interviews.ai)
- This course maps book *themes* (not reprinted questions) onto optional `*-98` companions
- **Use for**: Interview drill after the matching theory lesson; start at **01-98**

### How to Use These Books

**Beginner Path**:
1. Start with "Understanding Deep Learning" for intuition
2. Follow this course for structured learning
3. Reference "Hands-On ML" for implementation
4. Dive into "Deep Learning" for theory

**Intermediate Path**:
1. Use this course as primary guide
2. Reference "Hands-On ML" for practical tips
3. Read "Deep Learning" chapters for depth
4. Consult "Understanding DL" for clarifications

**Advanced Path**:
1. Use this course for comprehensive coverage
2. Study "Deep Learning" for rigorous theory
3. Implement with "Hands-On ML" techniques
4. Reference MIT book for research details

## Current Trends (2024-2025)

### Large Language Models (LLMs)
- GPT-4, Claude, Gemini
- Billions to trillions of parameters
- Emergent capabilities at scale

### Multimodal Models
- CLIP: Vision + Language
- GPT-4V: Text + Images
- Unified understanding across modalities

### Efficient AI
- Model compression
- Quantization
- Neural architecture search
- Edge deployment

### Foundation Models
- Pre-train on massive data
- Fine-tune for specific tasks
- Transfer learning at scale

## Challenges Ahead

### Technical Challenges
- Sample efficiency (learning from less data)
- Robustness (handling distribution shift)
- Interpretability (understanding decisions)
- Computational cost (training and inference)

### Societal Challenges
- Bias and fairness
- Privacy concerns
- Environmental impact
- Job displacement
- Misinformation (deepfakes)

## Summary

Deep learning succeeds because it:
- ✅ Learns features automatically
- ✅ Scales with data and compute
- ✅ Handles high-dimensional inputs
- ✅ Achieves state-of-the-art results
- ✅ Enables end-to-end learning

Best used when:
- Large datasets available
- Complex patterns exist
- Computational resources sufficient
- High performance required

The field continues to evolve rapidly with new architectures, techniques, and applications emerging constantly.

**Next**: We'll dive into the fundamentals of neural networks and understand how they actually work!

<!-- video-references -->

## Video references

Some figures in this lesson are screenshots from the following videos (Machine Learning Thực Chiến). URLs kept for attribution and further viewing:

- [Recommendation algorithms (TikTok-style)](https://www.facebook.com/reel/1444915507374636)
