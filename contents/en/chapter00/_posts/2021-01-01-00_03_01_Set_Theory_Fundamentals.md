---
layout: post
title: 00-03-01 Set Theory Fundamentals
chapter: '00'
order: 12
owner: GitHub Copilot
lang: en
categories:
- chapter00
---

> *Every evaluation metric you compute in machine learning—precision, recall, Intersection-over-Union (IoU) in object detection, or train/validation/test splits—is literally an operation on sets. Mastering set notation and laws prevents subtle bugs like data leakage and miscalculated class metrics.*

## Background Knowledge Map: Set Operations in Machine Learning Workflows

1. **Sets, Subsets, and Partitions (Dataset Splitting)**: Formalizes how a full dataset $$D$$ must be partitioned into pairwise disjoint subsets $$D_{\text{train}}, D_{\text{val}}, D_{\text{test}}$$ whose union is $$D$$ and pairwise intersections are $$\emptyset$$ (zero leakage).
2. **Union, Intersection, and Set Difference (Evaluation Metrics)**: Directly defines True Positives ($$P \cap T$$), False Positives ($$P \setminus T$$), False Negatives ($$T \setminus P$$), and Jaccard / IoU similarity $$\frac{\lvert A \cap B\rvert}{\lvert A \cup B\rvert}$$.
3. **De Morgan's Laws & Complements (Logical Filtering)**: Translates complex boolean data filters—such as excluding samples that are either corrupted or unlabeled—into clean, bug-free set identities: $$(A \cup B)^c = A^c \cap B^c$$.
4. **Cartesian Products & Functions (Input–Output Spaces)**: Defines labeled datasets as subsets of $$\mathcal{X} \times \mathcal{Y}$$ and neural models as mappings $$f: \mathcal{X} \to \mathcal{Y}$$ between structured sets.

### Step-by-Step Worked Example (Read Slowly): Computing IoU and Verifying Inclusion–Exclusion on Detection Sets

Suppose an object detector predicts a set of foreground pixels $$P$$ of size $$\lvert P\rvert = 120$$, while the ground-truth bounding mask $$T$$ has $$\lvert T\rvert = 100$$ pixels, and they overlap on $$\lvert P \cap T\rvert = 80$$ pixels inside a $$500$$-pixel image $$U$$.

1. **Compute the Union via Inclusion–Exclusion**:
   $$\lvert P \cup T\rvert = \lvert P\rvert + \lvert T\rvert - \lvert P \cap T\rvert = 120 + 100 - 80 = 140$$
2. **Compute Intersection-over-Union (IoU / Jaccard Index)**:
   $$\text{IoU}(P, T) = \frac{\lvert P \cap T\rvert}{\lvert P \cup T\rvert} = \frac{80}{140} = \frac{4}{7} \approx 0.5714$$
3. **Decompose Errors Using Set Difference**:
   - False Positives: $$\lvert P \setminus T\rvert = \lvert P\rvert - \lvert P \cap T\rvert = 120 - 80 = 40$$.
   - False Negatives: $$\lvert T \setminus P\rvert = \lvert T\rvert - \lvert P \cap T\rvert = 100 - 80 = 20$$.
4. **Verify True Negatives via De Morgan's Law**:
   Background pixels correctly ignored are $$P^c \cap T^c = (P \cup T)^c$$, so:
   $$\lvert P^c \cap T^c\rvert = \lvert U\rvert - \lvert P \cup T\rvert = 500 - 140 = 360$$
   Partition check: $$\text{TP} + \text{FP} + \text{FN} + \text{TN} = 80 + 40 + 20 + 360 = 500 = \lvert U\rvert$$. Every pixel is accounted for exactly once.

This lesson covers fundamental concepts from set theory that provide the mathematical foundation for understanding deep-learning problems, constraints, and feasible regions.

---

## Introduction to Set Theory

Set theory provides the foundation for modern mathematics and is essential for understanding deep-learning concepts. A **set** is simply a collection of distinct objects, called elements or members.

### Basic Notation

- **Set notation:** $$A = \{1, 2, 3, 4\}$$
- **Element membership:** $$x \in A$$ (x is in A) or $$x \notin A$$ (x is not in A)
- **Empty set:** $$\emptyset = \{\}$$ (the set with no elements)
- **Set builder notation:** $$A = \{x : P(x)\}$$ (the set of all x such that property P(x) holds)

### Examples

- $$A = \{1, 2, 3, 4\}$$ (explicit listing)
- $$B = \{x \in \mathbb{R} : x^2 < 4\} = (-2, 2)$$ (set builder notation)
- $$C = \{x \in \mathbb{Z} : x \text{ is even}\}$$ (all even integers)

---

## Basic Set Operations

### Sets and Subsets

- **Subset:** $$A \subseteq B$$ means every element of $$A$$ is also in $$B$$
- **Proper Subset:** $$A \subset B$$ means $$A \subseteq B$$ and $$A \neq B$$
- **Set Equality:** $$A = B$$ if and only if $$A \subseteq B$$ and $$B \subseteq A$$

**Examples:**
- $$\{1, 2\} \subseteq \{1, 2, 3, 4\}$$
- $$\{1, 2\} \subset \{1, 2, 3, 4\}$$
- $$\emptyset \subseteq A$$ for any set $$A$$

### Union and Intersection

**Union ($$A \cup B$$):** All elements that are in $$A$$ or $$B$$ (or both)
$$A \cup B = \{x : x \in A \text{ or } x \in B\}$$

**Intersection ($$A \cap B$$):** All elements that are in both $$A$$ and $$B$$
$$A \cap B = \{x : x \in A \text{ and } x \in B\}$$

**Examples:**
- If $$A = \{1, 2, 3\}$$ and $$B = \{3, 4, 5\}$$, then:
  - $$A \cup B = \{1, 2, 3, 4, 5\}$$
  - $$A \cap B = \{3\}$$

**Disjoint Sets:** $$A$$ and $$B$$ are disjoint if $$A \cap B = \emptyset$$

### Complement and Difference

**Complement ($$A^c$$):** All elements not in $$A$$ (within some universal set $$U$$)
$$A^c = \{x \in U : x \notin A\}$$

**Set Difference ($$A \setminus B$$):** Elements in $$A$$ but not in $$B$$
$$A \setminus B = \{x : x \in A \text{ and } x \notin B\}$$

**Symmetric Difference:** $$(A \setminus B) \cup (B \setminus A) = (A \cup B) \setminus (A \cap B)$$

### Set Laws

**Commutative Laws:**
- $$A \cup B = B \cup A$$
- $$A \cap B = B \cap A$$

**Associative Laws:**
- $$(A \cup B) \cup C = A \cup (B \cup C)$$
- $$(A \cap B) \cap C = A \cap (B \cap C)$$

**Distributive Laws:**
- $$A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$$
- $$A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$$

**De Morgan's Laws:**
- $$(A \cup B)^c = A^c \cap B^c$$
- $$(A \cap B)^c = A^c \cup B^c$$

---

## Important Number Sets

Understanding the hierarchy of number systems is crucial for deep-learning:

### Natural Numbers
$$\mathbb{N} = \{1, 2, 3, 4, \ldots\}$$
(Sometimes includes 0: $$\mathbb{N}_0 = \{0, 1, 2, 3, \ldots\}$$)

### Integers
$$\mathbb{Z} = \{\ldots, -2, -1, 0, 1, 2, \ldots\}$$

### Rational Numbers
$$\mathbb{Q} = \left\{\frac{p}{q} : p, q \in \mathbb{Z}, q \neq 0\right\}$$

All numbers that can be expressed as fractions.

### Real Numbers
$$\mathbb{R}$$ includes all rational and irrational numbers (like $$\pi$$, $$e$$, $$\sqrt{2}$$).

### Complex Numbers
$$\mathbb{C} = \{a + bi : a, b \in \mathbb{R}, i^2 = -1\}$$

**Hierarchy:** $$\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$$

---

## Functions: Domain and Range

A **function** $$f: A \to B$$ is a rule that assigns to each element in set $$A$$ exactly one element in set $$B$$.

### Key Concepts

- **Domain:** The set $$A$$ of all possible input values
- **Codomain:** The set $$B$$ where outputs are taken from
- **Range (Image):** The set of all actual output values: $$\text{Range}(f) = \{f(x) : x \in A\} \subseteq B$$

**Example:** For $$f(x) = x^2$$ with $$f: \mathbb{R} \to \mathbb{R}$$:
- Domain: $$\mathbb{R}$$
- Codomain: $$\mathbb{R}$$
- Range: $$[0, \infty)$$

### Types of Functions

**Injective (One-to-One):** Each element in the range corresponds to exactly one element in the domain
$$f(x_1) = f(x_2) \implies x_1 = x_2$$

**Surjective (Onto):** Every element in the codomain is in the range
For every $$y \in B$$, there exists $$x \in A$$ such that $$f(x) = y$$

**Bijective:** Both injective and surjective
There's a perfect one-to-one correspondence between domain and codomain.

**Examples:**
- $$f(x) = 2x$$ on $$\mathbb{R}$$ is bijective
- $$f(x) = x^2$$ on $$\mathbb{R}$$ is neither injective nor surjective
- $$f(x) = x^2$$ on $$[0, \infty) \to [0, \infty)$$ is bijective

---

## Inequalities

Understanding inequalities is crucial for deep-learning, as constraints are often expressed as inequalities.

### Basic Inequality Symbols

- $$a < b$$: $$a$$ is strictly less than $$b$$
- $$a \leq b$$: $$a$$ is less than or equal to $$b$$
- $$a > b$$: $$a$$ is strictly greater than $$b$$
- $$a \geq b$$: $$a$$ is greater than or equal to $$b$$

### Properties of Inequalities

1. **Transitivity:** If $$a \leq b$$ and $$b \leq c$$, then $$a \leq c$$

2. **Addition:** If $$a \leq b$$, then $$a + c \leq b + c$$ for any $$c$$

3. **Multiplication by Positive:** If $$a \leq b$$ and $$c > 0$$, then $$ac \leq bc$$

4. **Multiplication by Negative:** If $$a \leq b$$ and $$c < 0$$, then $$ac \geq bc$$ (inequality flips!)

### Interval Notation

- **Open interval:** $$(a, b) = \{x \in \mathbb{R} : a < x < b\}$$
- **Closed interval:** $$[a, b] = \{x \in \mathbb{R} : a \leq x \leq b\}$$
- **Half-open intervals:** $$[a, b)$$, $$(a, b]$$
- **Unbounded intervals:** $$(-\infty, a)$$, $$[a, \infty)$$, $$(-\infty, \infty) = \mathbb{R}$$

---

## Applications in Deep Learning

Set theory concepts are fundamental to deep-learning:

### 1. Feasible Regions

The **feasible region** is the set of all points satisfying the constraints:
$$S = \{x \in \mathbb{R}^n : g_i(x) \leq 0, i = 1, \ldots, m; h_j(x) = 0, j = 1, \ldots, p\}$$

### 2. Level Sets

For a function $$f: \mathbb{R}^n \to \mathbb{R}$$, the **level set** at level $$c$$ is:
$$L_c = \{x \in \mathbb{R}^n : f(x) = c\}$$

### 3. Constraint Qualification

Understanding when constraint sets have "nice" properties (like being closed or having non-empty interior) affects the existence and characterization of optimal solutions.

### 4. Convergence Analysis

Sequences and limits are essential for analyzing whether deep-learning algorithms converge to optimal solutions.

### 5. Set Operations in Algorithms

- **Intersection:** Finding points that satisfy multiple constraints
- **Union:** Combining feasible regions from different scenarios
- **Complement:** Understanding infeasible regions

**Example:** In linear programming, the feasible region is:
$$S = \{x \in \mathbb{R}^n : Ax \leq b, x \geq 0\} = \bigcap_{i=1}^{m} \{x : a_i^T x \leq b_i\} \cap \{x : x \geq 0\}$$

This is the intersection of half-spaces, demonstrating how set operations naturally arise in deep-learning problem formulation.

Understanding set theory provides the rigorous mathematical foundation needed to formulate deep-learning problems precisely and analyze their properties systematically.

## Video tham khảo

- **xem trước** [Intro to Sets | Examples, Notation & Properties](https://www.youtube.com/watch?v=B1v2-nGXNzs) — Dr. Trefor Bazett, English. Set notation, operations, and what a function's domain and codomain are.
- [INTRODUCTION to SET THEORY - DISCRETE MATHEMATICS](https://www.youtube.com/watch?v=tyDKR4FG3Yw) — TrevTutor, English. A fuller pass over unions, intersections, and the number sets the lesson lists.
- [MaDL - Sets, Scalars, Vectors, Matrices and Tensors](https://www.youtube.com/watch?v=JzIjCkouIGg) — Tübingen Machine Learning, English. Connects those sets to the scalars, vectors, and tensors the rest of the course actually computes with.
