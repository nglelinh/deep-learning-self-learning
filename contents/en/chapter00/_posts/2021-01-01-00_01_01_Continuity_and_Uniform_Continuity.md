---
layout: post
title: 00-01-01 Continuity and Uniform Continuity
chapter: '00'
order: 3
owner: GitHub Copilot
lang: en
categories:
- chapter00
---

> *Why can a single imperceptible pixel perturbation fool a high-accuracy image classifier into calling a panda a gibbon, or cause a GAN discriminator to explode? The answer lies not just in whether a network is continuous, but in how tightly its worst-case sensitivity—its Lipschitz constant—is controlled across the entire input space.*

## Key ideas: Continuity, Uniformity, and Lipschitz Control

1. **Pointwise Continuity (Local Predictability)**: Guarantees that around a specific input $$c$$, sufficiently small perturbations stay close in output space, though the required tolerance $$\delta(\varepsilon, c)$$ may shrink arbitrarily at other points.
2. **Uniform Continuity (Global Consistency)**: Strengthens continuity by demanding a single $$\delta(\varepsilon)$$ that works simultaneously everywhere on the domain, preventing arbitrarily steep cliffs from forming as you move across the space.
3. **Lipschitz Continuity (Quantitative Slope Bound)**: Caps the maximum amplification ratio $$\lvert f(x) - f(y)\rvert / \lvert x - y\rvert \leq L$$, which directly governs adversarial robustness, Wasserstein GAN constraints, and safe gradient descent step sizes ($$\eta < 2/L$$).
4. **Compactness & Heine–Cantor Bridge**: Explains why continuous functions on closed, bounded domains (like normalized image hypercubes $$[0, 1]^d$$) are automatically uniformly continuous, giving theoretical safety on bounded feature spaces.

### Worked example: Computing Exact Lipschitz Bounds vs. Failure of Uniform Continuity

Compare two functions on the open interval $$A = (0, 1]$$: $$f(x) = x^2$$ and $$g(x) = \frac{1}{x}$$.

1. **Checking Lipschitz Continuity of $$f(x) = x^2$$ on $$(0, 1]$$**:
   For any $$x, y \in (0, 1]$$, factor the difference:
   $$\lvert f(x) - f(y)\rvert = \lvert x^2 - y^2\rvert = \lvert x + y\rvert \cdot \lvert x - y\rvert$$
   Since $$x \leq 1$$ and $$y \leq 1$$, we have $$\lvert x + y\rvert \leq 2$$. Thus:
   $$\lvert f(x) - f(y)\rvert \leq 2\lvert x - y\rvert$$
   So $$f$$ is $$2$$-Lipschitz on $$(0, 1]$$, and hence uniformly continuous with $$\delta = \varepsilon / 2$$.
2. **Concrete Numerical Verification for $$f$$**:
   Take $$\varepsilon = 0.1$$, so $$\delta = 0.05$$. Pick $$x = 0.99, y = 0.96$$ (distance $$0.03 < 0.05$$):
   $$\lvert 0.99^2 - 0.96^2\rvert = \lvert 0.9801 - 0.9216\rvert = 0.0585 < 0.1$$
3. **Why $$g(x) = 1/x$$ Fails Uniform Continuity on $$(0, 1]$$**:
   Fix $$\varepsilon = 1$$. For any candidate $$\delta > 0$$, choose an integer $$n$$ large enough that $$\frac{1}{n} < \delta$$, and set $$x = \frac{1}{n}, y = \frac{1}{2n}$$.
   - Input distance: $$\lvert x - y\rvert = \frac{1}{2n} < \delta$$.
   - Output jump: $$\lvert g(x) - g(y)\rvert = \lvert n - 2n\rvert = n \geq 1 = \varepsilon$$.
   For example, with $$n = 100$$: $$x = 0.01, y = 0.005$$ are only $$0.005$$ apart, yet $$\lvert g(0.01) - g(0.005)\rvert = \lvert 100 - 200\rvert = 100$$. No single $$\delta$$ can tame this blow-up near $$0$$—the exact analytical pathology behind exploding gradients near singularities.

This lesson introduces the fundamental concepts of continuity and uniform continuity, which are essential for understanding the behavior of functions in deep-learning.

---

## Continuity and Uniform Continuity

**Continuity** and **Uniform Continuity** are fundamental concepts that describe the behavior of functions, particularly concerning their "smoothness" or "predictability." While closely related, they capture distinct properties, with uniform continuity being a stronger condition than mere continuity.

### Definition of Continuity

A function $$f: A \to \mathbb{R}$$ is said to be **continuous at a point** $$c \in A$$ if, for every positive real number $$\varepsilon > 0$$, there exists a positive real number $$\delta > 0$$ such that for all $$x \in A$$, if 

$$\lvert x - c \rvert < \delta$$

then 

$$\lvert f(x) - f(c) \rvert < \varepsilon$$ 

Intuitively, it means that for any desired level of precision $$\varepsilon$$ in the output $$f(x)$$, we can find a sufficiently small interval around $$c$$ (of width $$2\delta$$) such that all $$x$$ values within this interval map to $$f(x)$$ values within an $$\varepsilon$$-interval around $$f(c)$$. The crucial aspect here is that the choice of $$\delta$$ generally depends not only on $$\varepsilon$$ but also on the specific point $$c$$.

A function is **continuous on a set** $$A$$ if it is continuous at every point $$c \in A$$.

**Examples of continuous functions** include all polynomials (e.g., $$f(x) = x^2 + 3x - 1$$), trigonometric functions like $$\sin(x)$$ and $$\cos(x)$$, and exponential functions $$e^x$$ on their respective domains. 

### Definition of Uniform Continuity

**Uniform Continuity**, on the other hand, imposes a more stringent condition. A function $$f: A \to \mathbb{R}$$ is said to be **uniformly continuous on a set** $$A$$ if, for every positive real number $$\varepsilon > 0$$, there exists a positive real number $$\delta > 0$$ such that for all $$x, y \in A$$, if 

$$\lvert x - y \rvert < \delta$$

then 

$$\lvert f(x) - f(y) \rvert < \varepsilon$$

The **key distinction** from point-wise continuity lies in the order of quantifiers: for **uniform** continuity, the $$\delta$$ depends *only* on $$\varepsilon$$ and is independent of the specific points $$x$$ and $$y$$ in the domain.

### Definition of Lipschitz Continuity

**Lipschitz Continuity** provides an even more specific and quantitative notion of continuity. A function $$f: A \to \mathbb{R}$$ is said to be **Lipschitz continuous** (or **L-Lipschitz**) on a set $$A$$ if there exists a constant $$L \geq 0$$ such that for all $$x, y \in A$$:

$$\lvert f(x) - f(y) \rvert \leq L \lvert x - y \rvert$$

The smallest such constant $$L$$ is called the **Lipschitz constant** or **Lipschitz modulus** of $$f$$.

**Key Properties of Lipschitz Continuity:**

1. **Bounded Rate of Change**: The Lipschitz condition ensures that the function cannot change faster than a linear rate determined by $$L$$.

2. **Uniform Continuity**: Every Lipschitz continuous function is uniformly continuous (choose $$\delta = \varepsilon/L$$ for $$L > 0$$).

3. **Almost Everywhere Differentiability**: Lipschitz continuous functions are differentiable almost everywhere, and where the derivative exists, $$\lvert f'(x) \rvert \leq L$$.

**Examples:**
- $$f(x) = \lvert x \rvert$$ is 1-Lipschitz on $$\mathbb{R}$$
- $$f(x) = \sin(x)$$ is 1-Lipschitz on $$\mathbb{R}$$ (since $$\lvert \cos(x) \rvert \leq 1$$)
- $$f(x) = x^2$$ is not Lipschitz on $$\mathbb{R}$$ but is Lipschitz on any bounded interval

### Key Differences and Hierarchy

The three types of continuity form a hierarchy of increasingly strong conditions:

**Continuity ⊆ Uniform Continuity ⊆ Lipschitz Continuity**

1. **Point-wise vs Global**: 
   - **Continuity**: Local property (checked at each point)
   - **Uniform Continuity**: Global property of the entire function
   - **Lipschitz Continuity**: Global property with quantitative bounds

2. **Choice of $$\delta$$**: 
   - **Continuity**: $$\delta$$ can depend on both $$\varepsilon$$ and the specific point $$c$$
   - **Uniform Continuity**: $$\delta$$ depends only on $$\varepsilon$$, working for all points simultaneously
   - **Lipschitz Continuity**: $$\delta = \varepsilon/L$$ provides explicit relationship

3. **Rate of Change Control**:
   - **Continuity**: No control over rate of change
   - **Uniform Continuity**: Ensures bounded variation over small intervals
   - **Lipschitz Continuity**: Provides explicit linear bound on rate of change

4. **Strength Relationships**: 
   - Every Lipschitz continuous function is uniformly continuous
   - Every uniformly continuous function is continuous
   - The converses are not generally true

### Detailed Examples and Comparisons

**Example 1: $$f(x) = x^2$$**
- **On $$\mathbb{R}$$**: Continuous but not uniformly continuous (rate of change $$\lvert f'(x) \rvert = 2\lvert x \rvert$$ is unbounded)
- **On $$[0,1]$$**: Continuous, uniformly continuous, and Lipschitz with $$L = 2$$

**Example 2: $$f(x) = \sin(x)$$**
- **On $$\mathbb{R}$$**: Continuous, uniformly continuous, and 1-Lipschitz (since $$\lvert \cos(x) \rvert \leq 1$$)

**Example 3: $$f(x) = \lvert x \rvert$$**
- **On $$\mathbb{R}$$**: Continuous, uniformly continuous, and 1-Lipschitz
- **Note**: Not differentiable at $$x = 0$$, but still Lipschitz

**Example 4: $$f(x) = \sqrt{x}$$**
- **On $$[0,1]$$**: Continuous and uniformly continuous, but not Lipschitz (derivative unbounded near $$x = 0$$)
- **On $$[a,1]$$ for $$a > 0$$**: Lipschitz with $$L = 1/(2\sqrt{a})$$

## Video tham khảo

- **xem trước** [Continuity at a point | Limits and continuity | AP Calculus AB | Khan Academy](https://www.youtube.com/watch?v=3KMqU5j7irw) — Khan Academy, English. Continuity at a single point is the definition the lesson starts from.
- [Limits, L'Hôpital's rule, and epsilon delta definitions | Chapter 7, Essence of calculus](https://www.youtube.com/watch?v=kfF40MiS7zA) — 3Blue1Brown, English. The epsilon-delta language in the second half is the same language used to upgrade pointwise continuity to uniform continuity.
- [Real Analysis | Intro to uniform continuity.](https://www.youtube.com/watch?v=gioaYgOQ3AQ) — Michael Penn, English. This is the uniform-continuity strengthening itself: one delta that works on a whole set, which is the lesson's second definition.
