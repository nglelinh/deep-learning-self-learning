---
layout: post
title: 00-02-03 Eigenvalues and Eigenvectors
chapter: '00'
order: 10
owner: GitHub Copilot
lang: en
categories:
- chapter00
---

> *Train a 50-layer recurrent network with carelessly scaled weights and within seconds your gradients either collapse to `0.0` or explode to `NaN`. Why? Because multiplying by the same matrix $$W$$ fifty times scales every component along its eigenvectors by $$\lambda_i^{50}$$—and eigenvalues tell you at a glance whether a system will die, stabilize, or blow up.*

## Key ideas: Spectral Structure of Matrices

1. **Eigenvalues & Eigenvectors (Invariant Directions)**: Nonzero vectors $$\mathbf{v}$$ that a matrix $$A$$ merely stretches or flips ($$A\mathbf{v} = \lambda\mathbf{v}$$) without rotating, revealing the intrinsic coordinate axes of the transformation.
2. **Characteristic Polynomial & Trace/Determinant Checks**: Eigenvalues satisfy $$\det(A - \lambda I) = 0$$, and always obey two quick sanity checks: $$\sum_i \lambda_i = \text{tr}(A)$$ and $$\prod_i \lambda_i = \det(A)$$.
3. **Spectral Theorem for Symmetric Matrices**: Real symmetric matrices (such as covariance matrices in PCA and Hessians of smooth losses) always have real eigenvalues and an orthonormal eigenbasis, guaranteeing clean diagonalization $$A = Q\Lambda Q^\top$$.
4. **Condition Number & Optimization Speed**: The ratio $$\kappa = \lvert \lambda_{\max}\rvert / \lvert \lambda_{\min}\rvert$$ of the Hessian controls how elongated a loss valley is; large $$\kappa$$ forces vanilla gradient descent to crawl, motivating preconditioning and normalization.

### Worked example: Full Eigendecomposition and Condition Number of a Hessian

Consider the symmetric Hessian matrix $$H = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}$$ of a quadratic loss $$L(\mathbf{w}) = \frac{1}{2}\mathbf{w}^\top H \mathbf{w}$$.

1. **Solve the Characteristic Equation $$\det(H - \lambda I) = 0$$**:
   $$(5 - \lambda)(2 - \lambda) - (2)(2) = \lambda^2 - 7\lambda + 10 - 4 = \lambda^2 - 7\lambda + 6 = 0$$
   Factoring gives $$(\lambda - 6)(\lambda - 1) = 0$$, so $$\lambda_1 = 6$$ and $$\lambda_2 = 1$$.
   - Trace check: $$\lambda_1 + \lambda_2 = 6 + 1 = 7 = 5 + 2 = \text{tr}(H)$$.
   - Determinant check: $$\lambda_1 \lambda_2 = 6 \times 1 = 6 = (5)(2) - 4 = \det(H)$$.
2. **Find Orthonormal Eigenvectors**:
   - For $$\lambda_1 = 6$$: $$(H - 6I)\mathbf{v} = \begin{bmatrix} -1 & 2 \\ 2 & -4 \end{bmatrix}\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \mathbf{0} \implies v_1 = 2v_2$$.
     Normalized: $$\mathbf{q}_1 = \frac{1}{\sqrt{5}}\begin{bmatrix} 2 \\ 1 \end{bmatrix}$$.
   - For $$\lambda_2 = 1$$: $$(H - I)\mathbf{v} = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \mathbf{0} \implies v_2 = -2v_1$$.
     Normalized: $$\mathbf{q}_2 = \frac{1}{\sqrt{5}}\begin{bmatrix} 1 \\ -2 \end{bmatrix}$$.
   - Orthogonality check: $$\mathbf{q}_1^\top \mathbf{q}_2 = \frac{1}{5}(2 - 2) = 0$$.
3. **Optimization Implication (Condition Number & Max Safe Step Size)**:
   - Condition number: $$\kappa(H) = \frac{\lambda_1}{\lambda_2} = \frac{6}{1} = 6$$.
   - Maximum stable gradient descent learning rate: $$\eta_{\max} = \frac{2}{\lambda_{\max}} = \frac{2}{6} = \frac{1}{3} \approx 0.333$$. Any step size $$\eta > 1/3$$ will oscillate and diverge along $$\mathbf{q}_1$$!

This lesson covers eigenvalues and eigenvectors, which are crucial for understanding the behavior of linear transformations and quadratic functions in deep-learning.

---

## Definition and Intuition

When a matrix transforms a vector, it usually changes both the vector's direction and its length. However, **eigenvectors** are special vectors that, when transformed by a given matrix, only get scaled but do not change their direction.

### Mathematical Definition

For a square matrix $$\mathbf{A}$$ and a non-zero vector $$\mathbf{v}$$:

- $$\mathbf{v}$$ is an **eigenvector** of $$\mathbf{A}$$
- $$\lambda$$ is the corresponding **eigenvalue**

if they satisfy the **eigenvalue equation**:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

### Geometric Interpretation

- **Eigenvectors:** Non-zero vectors that maintain their direction under the transformation $$\mathbf{A}$$
- **Eigenvalues:** The scalar factors by which the eigenvectors are scaled

**Visual Understanding:**
- If $$\lambda > 1$$: The eigenvector is stretched
- If $$0 < \lambda < 1$$: The eigenvector is shrunk
- If $$\lambda < 0$$: The eigenvector is scaled and flipped
- If $$\lambda = 0$$: The eigenvector is mapped to the zero vector

---

## Finding Eigenvalues and Eigenvectors

### Step 1: Find Eigenvalues

Rearrange the eigenvalue equation:
$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$
$$\mathbf{A}\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$$

For a non-trivial solution ($$\mathbf{v} \neq \mathbf{0}$$), the matrix $$(\mathbf{A} - \lambda\mathbf{I})$$ must be singular, so:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

This is called the **characteristic equation**.

### Step 2: Find Eigenvectors

For each eigenvalue $$\lambda_i$$, solve the system:
$$(\mathbf{A} - \lambda_i\mathbf{I})\mathbf{v} = \mathbf{0}$$

The solutions form the **eigenspace** corresponding to $$\lambda_i$$.

---

## Detailed Example

Let's find the eigenvalues and eigenvectors of $$\mathbf{A} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$.

### Step 1: Find Eigenvalues

$$\mathbf{A} - \lambda\mathbf{I} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix} - \lambda\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix}$$

$$\det(\mathbf{A} - \lambda\mathbf{I}) = (3-\lambda)(2-\lambda) - (1)(0) = (3-\lambda)(2-\lambda) = 0$$

This gives us $$\lambda_1 = 3$$ and $$\lambda_2 = 2$$.

### Step 2: Find Eigenvectors

**For $$\lambda_1 = 3$$:**
$$(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives us $$v_2 = 0$$ and $$v_1$$ can be any non-zero value. So $$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ (or any scalar multiple).

**For $$\lambda_2 = 2$$:**
$$(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

This gives us $$v_1 + v_2 = 0$$, so $$v_2 = -v_1$$. Thus $$\mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$ (or any scalar multiple).

### Verification

Let's verify our results:
- $$\mathbf{A}\mathbf{v}_1 = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix} = 3\begin{pmatrix} 1 \\ 0 \end{pmatrix} = 3\mathbf{v}_1$$ ✓
- $$\mathbf{A}\mathbf{v}_2 = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 2 \\ -2 \end{pmatrix} = 2\begin{pmatrix} 1 \\ -1 \end{pmatrix} = 2\mathbf{v}_2$$ ✓

---

## Properties and Important Theorems

### Key Properties

1. **Sum of eigenvalues = trace of matrix:**
   $$\sum_{i=1}^n \lambda_i = \text{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}$$

2. **Product of eigenvalues = determinant of matrix:**
   $$\prod_{i=1}^n \lambda_i = \det(\mathbf{A})$$

3. **Eigenvectors corresponding to different eigenvalues are linearly independent**

4. **If $$\mathbf{A}$$ is symmetric, all eigenvalues are real and eigenvectors are orthogonal**

### Eigenvalue Multiplicity

- **Algebraic multiplicity:** How many times $$\lambda$$ appears as a root of the characteristic polynomial
- **Geometric multiplicity:** The dimension of the eigenspace (number of linearly independent eigenvectors)

For any eigenvalue: geometric multiplicity ≤ algebraic multiplicity

---

## Diagonalization

A matrix $$\mathbf{A}$$ is **diagonalizable** if it can be written as:

$$\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}$$

where:
- $$\mathbf{D}$$ is a diagonal matrix of eigenvalues
- $$\mathbf{P}$$ is a matrix whose columns are the corresponding eigenvectors

### Benefits of Diagonalization

1. **Easy computation of powers:** $$\mathbf{A}^k = \mathbf{P}\mathbf{D}^k\mathbf{P}^{-1}$$
2. **Understanding behavior:** The eigenvalues determine the transformation's behavior along each eigenvector direction

---

## Applications in Deep Learning

Eigenvalues and eigenvectors are crucial in deep-learning for several reasons:

### 1. Quadratic Forms and Definiteness

For a quadratic function $$f(\mathbf{x}) = \mathbf{x}^T\mathbf{Q}\mathbf{x}$$:

- **Positive definite** ($$f(\mathbf{x}) > 0$$ for $$\mathbf{x} \neq \mathbf{0}$$): All eigenvalues of $$\mathbf{Q}$$ are positive
- **Positive semidefinite** ($$f(\mathbf{x}) \geq 0$$): All eigenvalues are non-negative
- **Negative definite** ($$f(\mathbf{x}) < 0$$ for $$\mathbf{x} \neq \mathbf{0}$$): All eigenvalues are negative
- **Indefinite** ($$f(\mathbf{x})$$ can be positive or negative): Mixed positive and negative eigenvalues

### 2. Second-Order Optimality Conditions

For a function $$f(\mathbf{x})$$ at a critical point $$\mathbf{x}^*$$ (where $$\nabla f(\mathbf{x}^*) = \mathbf{0}$$):

- **Local minimum:** Hessian $$\nabla^2 f(\mathbf{x}^*)$$ is positive definite (all eigenvalues > 0)
- **Local maximum:** Hessian is negative definite (all eigenvalues < 0)
- **Saddle point:** Hessian is indefinite (mixed eigenvalues)

### 3. Principal Component Analysis (PCA)

PCA finds the directions of maximum variance in data:
- Eigenvectors of the covariance matrix give the principal directions
- Eigenvalues give the variance along each principal direction

### 4. Convergence Analysis

In iterative deep-learning algorithms:
- The **condition number** $$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$ affects convergence speed
- Large condition numbers lead to slow convergence

### 5. Newton's Method

Newton's method uses the inverse Hessian:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

The eigenvalues of the Hessian determine the method's behavior and convergence rate.

---

## Example: Deep Learning Application

Consider minimizing $$f(x, y) = 2x^2 + 3y^2 + 2xy$$.

The Hessian is: $$\mathbf{H} = \begin{pmatrix} 4 & 2 \\ 2 & 6 \end{pmatrix}$$

**Finding eigenvalues:**
$$\det(\mathbf{H} - \lambda\mathbf{I}) = (4-\lambda)(6-\lambda) - 4 = \lambda^2 - 10\lambda + 20 = 0$$

$$\lambda = \frac{10 \pm \sqrt{100-80}}{2} = \frac{10 \pm 2\sqrt{5}}{2} = 5 \pm \sqrt{5}$$

Since both eigenvalues are positive ($$\lambda_1 = 5 + \sqrt{5} > 0$$ and $$\lambda_2 = 5 - \sqrt{5} > 0$$), the Hessian is positive definite, confirming that the origin is a global minimum.

The condition number is $$\kappa = \frac{5 + \sqrt{5}}{5 - \sqrt{5}} \approx 4.24$$, indicating reasonably good conditioning for deep-learning algorithms.

Understanding eigenvalues and eigenvectors provides deep insights into the geometric and analytical properties of deep-learning problems, enabling better algorithm design and convergence analysis.

## Video tham khảo

- **xem trước** [Eigenvectors and eigenvalues | Chapter 14, Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g) — 3Blue1Brown, English. The geometric definition: Av is a scaled copy of v.
- [MaDL - Eigenvalue and Singular Value Decomposition](https://www.youtube.com/watch?v=-GcrsEqzmsA) — Tübingen Machine Learning, English. Computes eigenvalues and the SVD the lesson's diagonalization section leads into.
