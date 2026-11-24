---
layout: post
title: 00-03 Real Analysis And Set Theory
chapter: '00'
order: 11
owner: GitHub Copilot
lang: en
categories:
- chapter00
---

> *Why can we even be sure that a loss function achieves a minimum on a bounded weight ball, or that a sequence of iterates produced by an optimizer converges to a limit inside our feasible set? Real analysis and set theory supply the precision behind every "exists," "converges," and "generalizes" statement in machine learning theory.*

## Background Knowledge Map: Foundations of Sets, Spaces, and Convergence

1. **Set Theory & Mappings (Formalizing Data and Models)**: Treats datasets as finite sets or multisets, classifiers as functions $$f: \mathcal{X} \to \mathcal{Y}$$, and hypothesis classes $$\mathcal{H}$$ as families of functions over which learning searches.
2. **Metric Spaces & Neighborhoods (Quantifying Closeness)**: Generalizes Euclidean distance to arbitrary metrics ($$L_1, L_2, L_\infty$$, Wasserstein), defining $$\varepsilon$$-balls used in adversarial robustness and nearest-neighbor retrieval.
3. **Open, Closed, and Compact Sets (Existence of Optima)**: Establishes the Extreme Value Theorem—every continuous loss on a compact (closed and bounded in $$\mathbb{R}^d$$) parameter set attains its global minimum.
4. **Sequences & Completeness (Convergence Guarantees)**: Explains why Cauchy sequences converge in $$\mathbb{R}^d$$, allowing optimization proofs to guarantee that vanishing step sizes settle at a concrete limit point.

### Step-by-Step Worked Example (Read Slowly): Why Closedness Matters for Finding a Minimum

Compare minimizing the continuous loss $$L(w) = w^2 + 1$$ over two feasible sets: the open interval $$U = (1, 3)$$ versus the closed, compact interval $$K = [1, 3]$$.

1. **Evaluate the Infimum (Greatest Lower Bound)**:
   On both $$U$$ and $$K$$, $$L(w) = w^2 + 1$$ is strictly increasing for $$w > 0$$. As $$w \to 1^+$$, $$L(w) \to 1^2 + 1 = 2$$. Thus:
   $$\inf_{w \in U} L(w) = \inf_{w \in K} L(w) = 2$$
2. **Test Attainment on the Open Set $$U = (1, 3)$$**:
   Take the sequence $$w_n = 1 + \frac{1}{n} \in U$$:
   - $$w_1 = 2.0 \implies L(2.0) = 5.0$$
   - $$w_{10} = 1.1 \implies L(1.1) = 2.21$$
   - $$w_{100} = 1.01 \implies L(1.01) = 2.0201$$
   The sequence converges to $$w^* = 1$$, yet $$1 \notin (1, 3)$$. No point inside $$U$$ achieves $$L(w) = 2$$—the minimum does not exist in $$U$$!
3. **Test Attainment on the Compact Set $$K = [1, 3]$$**:
   Because $$K$$ is closed (it contains its boundary point $$1$$) and bounded, the Extreme Value Theorem guarantees a minimizer exists in $$K$$, and indeed $$w^* = 1 \in K$$ attains $$L(1) = 2$$. This is why constrained optimization (like projected gradient descent onto an $$L_2$$ ball $$\|\mathbf{w}\|_2 \leq R$$) always uses **closed** balls.

This lesson covers essential concepts from real analysis and set theory needed for deep-learning, organized into two main sections for comprehensive understanding.

## Video tham khảo

- **xem trước** [Intro to Sets | Examples, Notation & Properties](https://www.youtube.com/watch?v=B1v2-nGXNzs) — Dr. Trefor Bazett, English. Sets, membership, and functions are the language of the set-theory lesson.
- [Intro to Open Sets (with Examples) | Real Analysis](https://www.youtube.com/watch?v=pnWgj8jjs3w) — Wrath of Math, English. Open sets are the first topology object the second lesson needs.
- [Limits, L'Hôpital's rule, and epsilon delta definitions | Chapter 7, Essence of calculus](https://www.youtube.com/watch?v=kfF40MiS7zA) — 3Blue1Brown, English. Epsilon-delta continuity is the analysis half this overview points at.
