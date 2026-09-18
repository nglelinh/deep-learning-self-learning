---
layout: post
title: 01-98 Deep Learning Interviews practice track
chapter: '01'
order: 9
owner: Deep Learning Course
lang: en
categories:
- chapter01
lesson_type: optional
---

# Optional: Deep Learning Interviews practice track

> This lesson is **optional**. It does **not** replace any theory chapter. It is a map and a study plan for using one free interview book as extra drill — after you finish the matching course notes.

## What the book is

**Deep Learning Interviews** (second edition) by **Shlomo Kashani** and **Amir Ivry** is a free volume of fully solved interview- and exam-style problems across core AI themes. It is written for graduate students and job seekers who need to speak clearly under time pressure, and it is also useful as a research-foundation checklist.

This course does **not** reprint the book. Open the official sources yourself:

- PDF / abstract: [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)
- Companion repository: [github.com/BoltzmannEntropy/interviews.ai](https://github.com/BoltzmannEntropy/interviews.ai)
- Author site: [interviews.ai](http://www.interviews.ai/)

Download the PDF from arXiv (or the authors' links). Respect the authors' terms on those pages. Do not paste long excerpts into notes, issues, or shared answer keys.

## How to use it with this course

Pick one mode and stay honest about timing:

1. **Interview drill.** After a chapter, attempt the original prompts in that chapter's optional companion *out loud*, then open the book for a harder, fully solved set on the same *theme*.
2. **Class quiz.** Instructors can assign 2–3 companion prompts as a short oral or written quiz. Use the book only as the instructor key / extra homework, not as a copy-paste worksheet.
3. **Research foundation.** Before reading a paper that leans on information theory, autodiff, or uncertainty, skim the matching book *topic area* (not a dumped chapter) and the course theory lesson.

Work closed-book first. Then check the course hint. Only then open the PDF for additional solved Q&A.

## Study plan: book themes → course chapters 00–25

The book is organized by interview themes, not by this course's chapter numbers. Use the table as a routing sheet. Theme names below are **high-level labels** only — they are not a table of contents reprint.

| Course chapter | Course topic | Book themes to study (in the PDF) | Optional companion |
| --- | --- | --- | --- |
| 00 | Mathematical prerequisites | Information theory (entropy, KL, mutual information); calculus / autodiff intuition | **00-98** |
| 01 | Introduction and course guide | How the book is meant to be used; this hub | this lesson |
| 02 | Neural network fundamentals | Perceptrons, activations, and the “expanded deep learning” architecture basics | **02-98** |
| 03 | Training, losses, backpropagation | Algorithmic differentiation; Hessian intuition; logistic / classification | **03-98** |
| 04 | CNNs | Convolution, CNN architectures, early feature-extraction ideas | **04-98** |
| 05–08 | RNN, LSTM, attention, Transformers | No Volume I home — stay with course notes; Volume II is planned around NLP / sequences | hub only |
| 09 | Regularization | Overfitting, validation, capacity control | **09-98** |
| 10 | Optimizers | Adaptive methods (Adam and relatives), hyperparameters, training dynamics | **10-98** |
| 11–14 | Generative models (incl. VAE, GAN) | No Volume I home — Volume II plan mentions GAN / VAE; use course chapters | hub only |
| 15 | Transfer learning | CNN feature extraction and frozen vs fine-tuned backbones | **15-98** |
| 16–23 | SSL, CV apps, NLP, speech, RL, GNN, efficiency | No Volume I home. Volume II plan mentions detection, segmentation, NLP, RL — use the matching course chapter | hub only |
| 24 | Interpretability | Related discussion only; Bayesian *uncertainty* lives in 25 | hub only |
| 25 | Advanced topics | Bayesian deep learning and probabilistic thinking | **25-98** |

### Themes with no course chapter of their own

Keep these on the hub. Do **not** invent a new required chapter.

**Ensemble methods (bagging / boosting / stacking).** The book treats neural-net ensembles as an interview topic. This course has no dedicated ensemble chapter. If you interview on that theme:

- Bagging: average independently trained models to cut variance.
- Boosting: add models that correct residual mistakes (more bias-reduction than bagging).
- Stacking: a second model learns how to combine base predictors.

**Discussion (original, not from the book).** When would you rather bag ten independently trained CNNs than train one model ten times larger? What breaks if the base models are highly correlated?

**Volume II (planned).** The authors outline a later volume on advanced CNNs, detection, segmentation, NLP, GANs, VAEs, and reinforcement learning. Until that volume is your assigned text, use this course's Chapters 11–21 for those topics and treat the Volume I PDF as the solved-practice source for the table above.

## Original warm-up (hub only)

These three prompts are course-written. They are not taken from the book.

### H1. What is the interviewer actually testing?

You are asked “why cross-entropy and not MSE for 10-way classification?” in two minutes. Name the *modeling* reason and the *optimization* reason, then stop.

**Hint.** Think likelihood of a categorical distribution, and the shape of the gradient when the prediction is confidently wrong.

**Discussion.** Modeling: a softmax output is a discrete distribution; cross-entropy is the negative log-likelihood of that model. Optimization: MSE on probabilities often gives weak gradients when the predicted class is already peaked on the wrong label; NLL still pushes the logit of the true class up. Open the PDF later for more classification / loss interview items — do not copy them here.

### H2. Closed-book vs open-book

You may use the PDF in a take-home exam but not in a 45-minute onsite. How should your weekly practice change?

**Hint.** Onsite interviews reward short derivations you can write on a whiteboard (sigmoid derivative, conv output shape, Adam update in one line).

**Discussion.** Drill the companion prompts on a timer. Use the book for depth and for checking language. If you can only solve a topic with the PDF open, that topic is not interview-ready.

### H3. Ensemble without an ensemble chapter

A hiring loop asks you to reduce variance of a well-tuned ResNet without changing the architecture. Give two methods that are *not* “collect more labels.”

**Hint.** One method changes how you use existing models; one changes how you use existing data.

**Discussion.** Snapshot / bagged fine-tunes, test-time augmentation, and dropout-as-ensemble at test time all reduce variance. A larger model is a capacity change, not a variance-reduction ensemble. Full solved ensemble problems belong in the book, not in this repo.

## Companion index

After the matching theory lesson, open:

- **00-98** — information theory and calculus / autodiff
- **02-98** — neurons, activations, MLP structure
- **03-98** — autodiff, Hessian intuition, logistic classification
- **04-98** — CNN geometry and architectures
- **09-98** — overfitting and regularizers
- **10-98** — Adam-family optimizers
- **15-98** — feature extraction and transfer
- **25-98** — Bayesian / uncertainty themes

Each companion has 3–6 **original** prompts with short hints. They do not rewrite theory.

## Attribution and license reminder

Kashani, S., and Ivry, A. *Deep Learning Interviews*. arXiv:2201.00650. GitHub: [BoltzmannEntropy/interviews.ai](https://github.com/BoltzmannEntropy/interviews.ai).

The solved question bank is the authors' work. This course only cites the title, authors, arXiv id, and theme names, then adds original practice of its own. **Students download the book from arXiv** (or the official GitHub / interviews.ai links). Do not vendor a local copy of the PDF in this repository and do not paste substantial book text into pull requests.
