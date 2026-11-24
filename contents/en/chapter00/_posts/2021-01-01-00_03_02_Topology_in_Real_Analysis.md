---
layout: post
title: 00-03-02 Topology in Real Analysis
chapter: '00'
order: 13
owner: GitHub Copilot
lang: en
categories:
- chapter00
---

> *When researchers talk about "adversarial $$\varepsilon$$-balls," "manifold hypotheses," or "flat versus sharp minima," they are speaking the language of point-set topology. Topology strips away coordinates to ask the questions optimization truly cares about: What is a neighborhood? Where is the boundary? And can a sequence escape?*

## Background Knowledge Map: Metric Topology for Optimization & Generalization

1. **Metric Spaces & Open Balls (Neighborhood Geometry)**: Defines the $$\varepsilon$$-ball $$B_\varepsilon(\mathbf{x}) = \{\mathbf{y} : d(\mathbf{x}, \mathbf{y}) < \varepsilon\}$$, which directly formalizes adversarial threat models ($$L_2$$ and $$L_\infty$$ perturbation balls) and local minima.
2. **Open vs. Closed Sets & Boundary Points**: Distinguishes interior points (where small gradient steps stay feasible) from boundary points (where constraints like weight clipping or probability simplices activate).
3. **Compactness (Heine–Borel Theorem)**: In $$\mathbb{R}^d$$, a set is compact if and only if it is closed and bounded—guaranteeing that every continuous loss achieves a global minimum and every bounded sequence has a convergent subsequence.
4. **Connectedness & Path-Connectedness (Mode Connectivity)**: Explains whether two distinct neural network minima can be joined by a continuous low-loss curve, a cornerstone of modern loss-landscape research.

### Step-by-Step Worked Example (Read Slowly): Comparing $$L_2$$ vs. $$L_\infty$$ Open Balls in Adversarial Perturbations

Suppose an input image patch has two pixel intensities $$\mathbf{x} = (0.50, 0.50)$$ and an attacker can perturb it to $$\mathbf{y} = (0.58, 0.56)$$ with perturbation $$\boldsymbol{\delta} = \mathbf{y} - \mathbf{x} = (0.08, 0.06)$$. Test whether $$\mathbf{y}$$ lies inside the open ball $$B_{0.1}(\mathbf{x})$$ under the $$L_2$$ and $$L_\infty$$ metrics.

1. **Compute the $$L_2$$ (Euclidean) Distance**:
   $$d_2(\mathbf{x}, \mathbf{y}) = \sqrt{(0.08)^2 + (0.06)^2} = \sqrt{0.0064 + 0.0036} = \sqrt{0.0100} = 0.10$$
   Because an **open** ball requires strict inequality $$d_2(\mathbf{x}, \mathbf{y}) < 0.10$$, $$\mathbf{y}$$ is **not** in the open $$L_2$$ ball—it lies right on the boundary sphere $$\partial B_{0.1}^{(2)}(\mathbf{x})$$!
2. **Compute the $$L_\infty$$ (Chebyshev / Max-Pixel) Distance**:
   $$d_\infty(\mathbf{x}, \mathbf{y}) = \max(\lvert 0.08\rvert, \lvert 0.06\rvert) = 0.08 < 0.10$$
   So $$\mathbf{y}$$ **is** strictly in the interior of the open $$L_\infty$$ ball $$B_{0.1}^{(\infty)}(\mathbf{x})$$, with safety margin $$0.10 - 0.08 = 0.02$$ before hitting the boundary.
3. **Geometric Takeaway**:
   The $$L_\infty$$ ball is a square of side $$0.20$$ that encloses the $$L_2$$ circle of radius $$0.10$$; in $$d$$ dimensions, a corner of the $$L_\infty$$ ball has $$L_2$$ distance $$\varepsilon\sqrt{d}$$, explaining why tiny per-pixel $$L_\infty$$ attacks accumulate huge $$L_2$$ energy on high-resolution images.

This lesson covers essential topological concepts from real analysis that are crucial for understanding the structure of feasible regions, continuity, and the existence of optimal solutions in deep-learning problems.

---

## Introduction to Topology

Topology studies the properties of space that are preserved under continuous deformations. In deep-learning, topological concepts help us understand the structure of feasible regions and the behavior of functions, particularly regarding the existence and characterization of optimal solutions.

### Metric Spaces and Distance

Before discussing topology, we need the concept of distance. In $$\mathbb{R}^n$$, the standard **Euclidean distance** between points $$\mathbf{x}$$ and $$\mathbf{y}$$ is:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

### Open Balls and Neighborhoods

An **open ball** centered at $$\mathbf{x}_0$$ with radius $$\epsilon > 0$$ is:

$$B(\mathbf{x}_0, \epsilon) = \{\mathbf{y} \in \mathbb{R}^n : d(\mathbf{x}_0, \mathbf{y}) < \epsilon\}$$

This represents all points within distance $$\epsilon$$ from $$\mathbf{x}_0$$.

**Examples:**
- In $$\mathbb{R}$$: $$B(0, 1) = (-1, 1)$$ (open interval)
- In $$\mathbb{R}^2$$: $$B(\mathbf{0}, 1) = \{(x, y) : x^2 + y^2 < 1\}$$ (open unit disk)

---

## Open Sets

An **open set** is characterized by the property that it **contains none of its boundary points**.

### Formal Definition

A set $$S$$ in $$\mathbb{R}^n$$ is **open** if for every point $$\mathbf{x} \in S$$, there exists a positive real number $$\epsilon > 0$$ such that the open ball $$B(\mathbf{x}, \epsilon)$$ is entirely contained within $$S$$:

$$\forall \mathbf{x} \in S, \exists \epsilon > 0 : B(\mathbf{x}, \epsilon) \subseteq S$$

### Intuitive Understanding

An open set has the property that if you're inside it, you can move a small distance in any direction and still remain inside the set. There's always some "wiggle room" around every point.

### Examples of Open Sets

**In $$\mathbb{R}$$:**
- $$(0, 1) = \{x : 0 < x < 1\}$$
- $$(-\infty, 5) = \{x : x < 5\}$$
- $$\mathbb{R}$$ itself

**In $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 < 1\}$$ (open unit disk)
- $$\{(x, y) : x > 0, y > 0\}$$ (first quadrant, excluding axes)
- $$\mathbb{R}^2$$ itself

**In $$\mathbb{R}^n$$:**
- Any open ball $$B(\mathbf{x}_0, r)$$
- $$\mathbb{R}^n$$ itself
- $$\emptyset$$ (empty set - vacuously open)

### Properties of Open Sets

1. The union of any collection of open sets is open
2. The intersection of finitely many open sets is open
3. $$\mathbb{R}^n$$ and $$\emptyset$$ are both open

---

## Closed Sets

A **closed set** is defined as a set that contains all of its boundary points. Equivalently, a set $$S$$ is closed if its complement $$\mathbb{R}^n \setminus S$$ is an **open set**.

### Formal Definition

A set $$S$$ is **closed** if it contains all its limit points. That is, if a sequence of points $$(x_n)$$ from $$S$$ converges to a point $$\mathbf{x}$$, then $$\mathbf{x}$$ must also be in $$S$$:

$$\text{If } \mathbf{x}_n \in S \text{ for all } n \text{ and } \lim_{n \to \infty} \mathbf{x}_n = \mathbf{x}, \text{ then } \mathbf{x} \in S$$

### Examples of Closed Sets

**In $$\mathbb{R}$$:**
- $$[0, 1] = \{x : 0 \leq x \leq 1\}$$
- $$[a, \infty) = \{x : x \geq a\}$$
- $$\{0\}$$ (single point)
- $$\mathbb{Z}$$ (integers)

**In $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 \leq 1\}$$ (closed unit disk)
- $$\{(x, y) : x \geq 0, y \geq 0\}$$ (first quadrant, including axes)
- $$\{(0, 0)\}$$ (single point)

**In $$\mathbb{R}^n$$:**
- Any closed ball $$\overline{B}(\mathbf{x}_0, r) = \{\mathbf{x} : d(\mathbf{x}, \mathbf{x}_0) \leq r\}$$
- $$\mathbb{R}^n$$ itself
- $$\emptyset$$ (empty set)
- Any finite set

### Properties of Closed Sets

1. The intersection of any collection of closed sets is closed
2. The union of finitely many closed sets is closed
3. $$\mathbb{R}^n$$ and $$\emptyset$$ are both closed

### Important Note

Sets can be:
- **Open but not closed:** $$(0, 1)$$
- **Closed but not open:** $$[0, 1]$$
- **Both open and closed:** $$\mathbb{R}^n$$, $$\emptyset$$
- **Neither open nor closed:** $$[0, 1)$$, $$(0, 1]$$

---

## Boundary, Interior, and Closure

### Boundary

The **boundary** of a set $$S$$, denoted $$\partial S$$, consists of points that are "on the edge" of the set. A point $$\mathbf{x}$$ is a **boundary point** of $$S$$ if every open ball centered at $$\mathbf{x}$$ intersects both $$S$$ and its complement $$S^c$$:

$$\partial S = \{\mathbf{x} : \forall \epsilon > 0, B(\mathbf{x}, \epsilon) \cap S \neq \emptyset \text{ and } B(\mathbf{x}, \epsilon) \cap S^c \neq \emptyset\}$$

### Interior

The **interior** of a set $$S$$, denoted $$S^\circ$$ or $$\text{int}(S)$$, includes all points strictly "inside" the set, excluding the boundary:

$$S^\circ = \{\mathbf{x} \in S : \exists \epsilon > 0, B(\mathbf{x}, \epsilon) \subseteq S\}$$

### Closure

The **closure** of a set $$S$$, denoted $$\overline{S}$$ or $$\text{cl}(S)$$, is the smallest closed set containing $$S$$:

$$\overline{S} = S \cup \partial S$$

### Example Analysis

For the interval $$S = [0, 1)$$ in $$\mathbb{R}$$:
- **Interior:** $$S^\circ = (0, 1)$$
- **Boundary:** $$\partial S = \{0, 1\}$$
- **Closure:** $$\overline{S} = [0, 1]$$

For the open disk $$S = \{(x, y) : x^2 + y^2 < 1\}$$ in $$\mathbb{R}^2$$:
- **Interior:** $$S^\circ = S$$ (the set is already open)
- **Boundary:** $$\partial S = \{(x, y) : x^2 + y^2 = 1\}$$ (unit circle)
- **Closure:** $$\overline{S} = \{(x, y) : x^2 + y^2 \leq 1\}$$ (closed unit disk)

---

## Compact Sets

A **compact set** is one of the most important concepts in deep-learning theory.

### Definition in Euclidean Spaces

**Heine-Borel Theorem:** In Euclidean spaces ($$\mathbb{R}^n$$), a set is compact if and only if it is both **closed and bounded**.

- **Bounded:** A set $$S$$ is bounded if it can be contained within some sufficiently large open ball: $$\exists M > 0, \mathbf{x}_0$$ such that $$S \subseteq B(\mathbf{x}_0, M)$$
- **Closed:** As defined above

### Examples of Compact Sets

**In $$\mathbb{R}$$:**
- $$[a, b]$$ (any closed, bounded interval)
- $$\{0\}$$ (single point)
- Any finite set

**In $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 \leq 1\}$$ (closed unit disk)
- $$[0, 1] \times [0, 1]$$ (unit square)
- Any finite set of points

**In $$\mathbb{R}^n$$:**
- Any closed ball $$\overline{B}(\mathbf{x}_0, r)$$
- Any closed, bounded rectangle $$[a_1, b_1] \times [a_2, b_2] \times \cdots \times [a_n, b_n]$$

### Non-Compact Sets

- $$(0, 1)$$ (bounded but not closed)
- $$[0, \infty)$$ (closed but not bounded)
- $$\mathbb{R}^n$$ (not bounded)
- $$\{1, 1/2, 1/3, 1/4, \ldots\}$$ (bounded but not closed, since 0 is a limit point not in the set)

---

## Continuity of Functions

### Point-wise Continuity

A function $$f: A \to \mathbb{R}$$ is **continuous at a point** $$\mathbf{c} \in A$$ if for every $$\varepsilon > 0$$, there exists $$\delta > 0$$ such that for all $$\mathbf{x} \in A$$:

$$\|\mathbf{x} - \mathbf{c}\| < \delta \implies |f(\mathbf{x}) - f(\mathbf{c})| < \varepsilon$$

**Intuitive meaning:** Small changes in input lead to small changes in output.

### Global Continuity

$$f$$ is **continuous on $$A$$** if it's continuous at every point in $$A$$.

### Sequential Characterization

$$f$$ is continuous at $$\mathbf{c}$$ if and only if for every sequence $$(\mathbf{x}_n)$$ in $$A$$ converging to $$\mathbf{c}$$:

$$\lim_{n \to \infty} f(\mathbf{x}_n) = f(\mathbf{c})$$

---

## Important Theorems for Deep Learning

### Extreme Value Theorem

**If $$f$$ is continuous on a compact set $$K$$, then $$f$$ attains its maximum and minimum on $$K$$.**

This is fundamental for deep-learning: it guarantees that continuous objective functions have optimal solutions on compact feasible regions.

**Proof idea:** Compactness ensures that the supremum and infimum of $$f$$ on $$K$$ are actually achieved at points in $$K$$.

### Intermediate Value Theorem

**If $$f$$ is continuous on $$[a, b]$$ and $$y$$ is between $$f(a)$$ and $$f(b)$$, then there exists $$c \in [a, b]$$ such that $$f(c) = y$$.**

This helps establish the existence of solutions to equations $$f(x) = 0$$.

### Bolzano-Weierstrass Theorem

**Every bounded sequence in $$\mathbb{R}^n$$ has a convergent subsequence.**

This is crucial for proving convergence of deep-learning algorithms.

### Weierstrass Approximation Theorem

**Every continuous function on a closed interval can be uniformly approximated by polynomials.**

This justifies using polynomial approximations in deep-learning algorithms.

---

## Applications in Deep Learning

### 1. Existence of Solutions

**Compact feasible sets guarantee optimal solutions exist:**
- If the feasible region $$S$$ is compact and the objective function $$f$$ is continuous, then the deep-learning problem $$\min_{\mathbf{x} \in S} f(\mathbf{x})$$ has a solution.

### 2. Constraint Qualification

Understanding topological properties of constraint sets:
- **Regular points:** Points where constraint gradients are linearly independent
- **Interior point methods:** Require the feasible region to have non-empty interior

### 3. Convergence Analysis

Analyzing whether deep-learning algorithms converge:
- **Closed sets:** Ensure limit points of convergent sequences remain feasible
- **Compactness:** Guarantees convergent subsequences exist

### 4. Local vs Global Optima

Using neighborhoods to define optimality:
- **Local minimum:** $$f(\mathbf{x}^*) \leq f(\mathbf{x})$$ for all $$\mathbf{x}$$ in some neighborhood of $$\mathbf{x}^*$$
- **Global minimum:** $$f(\mathbf{x}^*) \leq f(\mathbf{x})$$ for all $$\mathbf{x}$$ in the feasible region

### 5. Feasible Region Analysis

Determining properties of constraint sets:
- **Linear constraints:** Define closed sets (half-spaces)
- **Nonlinear constraints:** May create sets that are neither open nor closed
- **Compact feasible regions:** Guarantee existence of optimal solutions

### Example: Portfolio Deep Learning

Consider minimizing portfolio risk subject to constraints:

$$\begin{align}
\min_{\mathbf{w}} \quad & \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \\
\text{s.t.} \quad & \mathbf{1}^T \mathbf{w} = 1 \\
& \mathbf{w} \geq \mathbf{0}
\end{align}$$

The feasible region $$S = \{\mathbf{w} : \mathbf{1}^T \mathbf{w} = 1, \mathbf{w} \geq \mathbf{0}\}$$ is:
- **Closed:** It's the intersection of closed sets
- **Bounded:** The constraint $$\mathbf{1}^T \mathbf{w} = 1$$ with $$\mathbf{w} \geq \mathbf{0}$$ bounds the feasible region
- **Compact:** Being closed and bounded in $$\mathbb{R}^n$$

Since the objective function $$\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$ is continuous and $$S$$ is compact, the Extreme Value Theorem guarantees that an optimal portfolio exists.

Understanding topology and real analysis provides the rigorous foundation needed to prove that deep-learning problems have solutions and that algorithms will find them. These concepts are essential for both theoretical analysis and practical algorithm design.

## Video tham khảo

- **xem trước** [Intro to Open Sets (with Examples) | Real Analysis](https://www.youtube.com/watch?v=pnWgj8jjs3w) — Wrath of Math, English. An open set is defined by a ball around every point, which is this lesson's starting definition.
- [Identifying Open, Closed, and Compact Sets | Real Analysis Exercises](https://www.youtube.com/watch?v=0_mottmadEU) — Wrath of Math, English. Works examples that separate open, closed, and compact sets, including boundary and closure.
- [Open Covers, Finite Subcovers, and Compact Sets | Real Analysis](https://www.youtube.com/watch?v=dAz9ax-bjzU) — Wrath of Math, English. Compactness via open covers is the theorem-level definition the lesson uses for extreme-value arguments.
