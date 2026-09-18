---
layout: post
title: 25-98 Interviews Practice (Bayesian and uncertainty)
chapter: '25'
order: 4
owner: Deep Learning Course
lang: en
categories:
- chapter25
lesson_type: optional
---

# Optional: interview practice — Bayesian deep learning and uncertainty

> This lesson is **optional**. It does **not** rewrite the chapter’s survey of future directions. After that survey (and the probability notes in Chapter 00), use this drill, then open *Deep Learning Interviews* (Kashani & Ivry, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650)) for solved items on probabilistic programming and Bayesian deep learning.

**Book themes for this chapter (study in the PDF, not here):** Bayesian reasoning, priors / posteriors at a conceptual level, and uncertainty in deep models. There is no dedicated Bayesian chapter in this course; Chapter 25 is the advanced-topics home.

Hub: **01-98 Deep Learning Interviews practice track**.

## Practice prompts (original)

### Q1. Two uncertainties

A medical model outputs $$p(\text{disease}=1\mid\mathbf{x})=0.51$$. Give one story in which that number is *aleatoric* and one in which it is *epistemic*. What extra experiment would distinguish them?

**Hint.** Aleatoric: noise that remains with infinite data. Epistemic: shrinks if you collect more relevant labels.

**Discussion.** Aleatoric: the image is genuinely ambiguous (motion blur, two overlapping findings); more training data of the same kind will not push the probability to 0 or 1. Epistemic: this scanner / hospital is new and the weights are under-determined; more in-domain labels or a better prior should concentrate the posterior. Repeating the forward pass with MC dropout or deep ensembles (below) changes little in the first story and a lot in the second.

### Q2. L2 as a Gaussian prior

Show that maximizing $$ p(\mathbf{w}\mid\mathcal{D}) \propto p(\mathcal{D}\mid\mathbf{w})\,p(\mathbf{w}) $$ with $$p(\mathbf{w})=\mathcal{N}(\mathbf{0},\lambda^{-1}I)$$ and a Gaussian observation model is (up to constants) the same as minimizing MSE plus $$\frac{\lambda}{2}\|\mathbf{w}\|_2^2$$.

**Hint.** Take $$-\log$$ of the posterior; the prior contributes a quadratic.

**Discussion.** $$-\log p(\mathbf{w}) = \frac{\lambda}{2}\|\mathbf{w}\|_2^2 + \mathrm{const}$$. Under a Gaussian likelihood, $$-\log p(\mathcal{D}\mid\mathbf{w})$$ is MSE (or a scaled square loss). So MAP with that prior *is* L2-regularized regression. This is the bridge from Chapter 00 / 09 to “Bayesian language.” It does **not** by itself give you error bars — MAP is still a point.

### Q3. Why a point-estimate net looks overconfident

A softmax trained with CE on a clean set assigns $$0.99$$ to the wrong class on an OOD input. In Bayesian language, what did the training procedure throw away?

**Hint.** You optimized $$\mathbf{w}_{\mathrm{MAP}}$$ (or a SGD point) and then treated $$p(y\mid\mathbf{x},\mathbf{w}_{\mathrm{MAP}})$$ as $$p(y\mid\mathbf{x},\mathcal{D})$$.

**Discussion.** The predictive distribution should integrate $$p(y\mid\mathbf{x},\mathbf{w})$$ against $$p(\mathbf{w}\mid\mathcal{D})$$. A single $$\mathbf{w}$$ ignores disagreement among weight settings that all fit the training set. On OOD $$\mathbf{x}$$, that disagreement is often large, so the mixture is closer to uniform than any one sharp softmax. Temperature scaling can fix *in-domain* calibration without being Bayesian; it will not systematically fix OOD.

### Q4. MC dropout as a cheap posterior sketch

You run the same dropout network $$T=20$$ times at test time and average the softmaxes. What Bayesian object is this approximating *informally*, and name one thing it is not?

**Hint.** Each mask is a different function; the average is a mixture.

**Discussion.** Informally it is a mixture predictive $$\frac1T\sum_t p(y\mid\mathbf{x},\mathbf{w}\odot m_t)$$, sometimes motivated as variational inference over dropout masks. It is **not** a guarantee that the mixture matches the true posterior, and it is not free: you pay $$T$$ forwards. Deep ensembles (several independently trained nets) are a competing sketch; they live on the hub’s ensemble note, not in a new chapter.

### Q5. A prior you can defend

You must put a prior on a bias parameter in a well-specified logistic model (Chapter 03). Why is $$\mathcal{N}(0,10^2)$$ often more defensible than a point-mass at $$0$$, and when would you *not* want a huge variance?

**Hint.** A Dirac at 0 is “I already know the intercept.”

**Discussion.** A wide Gaussian says “the intercept is unknown but not insane.” A hard zero forces the decision boundary through a particular log-odds when features are zero — a strong scientific claim. Huge variance is a problem if the feature coding is arbitrary (unstandardized inputs): the prior then depends on units. Hierarchical / weakly informative priors belong in the book’s Bayesian chapters; here the course-level point is that “I used the default” is still a prior.

## Attribution

Kashani, S., and Ivry, A. *Deep Learning Interviews*, [arXiv:2201.00650](https://arxiv.org/abs/2201.00650). Download the PDF from arXiv for the full solved Q&A. This page is original course practice, not a reprint.
