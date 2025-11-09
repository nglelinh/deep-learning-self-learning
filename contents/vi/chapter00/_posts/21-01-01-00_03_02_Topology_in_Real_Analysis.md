---
layout: post
title: 00-03-02 Tôpô trong Giải tích thực
chapter: '00'
order: 13
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

Bài học này bao gồm các khái niệm tôpô cần thiết từ giải tích thực, rất quan trọng để hiểu cấu trúc của các vùng khả thi, tính liên tục và sự tồn tại của nghiệm tối ưu trong các bài toán tối ưu hóa.

---

## Giới thiệu về Tôpô

Tôpô nghiên cứu các tính chất của không gian được bảo toàn dưới các biến dạng liên tục. Trong tối ưu hóa, các khái niệm tôpô giúp chúng ta hiểu cấu trúc của các vùng khả thi và hành vi của các hàm số, đặc biệt liên quan đến sự tồn tại và đặc trưng của nghiệm tối ưu.

### Không gian metric và Khoảng cách

Trước khi thảo luận về tôpô, chúng ta cần khái niệm khoảng cách. Trong $$\mathbb{R}^n$$, **khoảng cách Euclid** chuẩn giữa các điểm $$\mathbf{x}$$ và $$\mathbf{y}$$ là:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

### Hình cầu mở và Lân cận

Một **hình cầu mở** có tâm tại $$\mathbf{x}_0$$ với bán kính $$\epsilon > 0$$ là:

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
