---
layout: post
title: 00-02 Basic Linear Algebra
chapter: '00'
order: 7
owner: GitHub Copilot
lang: en
categories:
- chapter00
---

> *If you ever wonder why GPUs—chips originally built to rotate 3D triangles in video games—sparked the deep learning revolution, the answer is linear algebra. A neural network is, at its core, an alternating cascade of matrix multiplications and nonlinearities, and linear algebra gives us the geometry to understand what those matrices actually do to data.*

## Key ideas: The Three Pillars of Linear Algebra in Deep Learning

1. **Vectors & Vector Spaces (Feature Geometry)**: Represents words, images, and hidden states as points in $$\mathbb{R}^d$$, where inner products measure semantic similarity and norms measure signal energy.
2. **Matrices & Linear Transformations (Layer Operators)**: Views every weight matrix $$W \in \mathbb{R}^{m \times n}$$ as a geometric operator that rotates, scales, shears, or projects data from one representation space to another.
3. **Eigenvalues & Eigenvectors (Dynamical Stability)**: Reveals the invariant axes of a linear operator, governing PCA dimensionality reduction, Hessian conditioning, and why repeated matrix multiplication causes vanishing or exploding gradients in deep networks.

### Worked example: Tracing a Feature Vector Through a Linear Layer and Its Spectrum

Consider a 2D input feature vector $$\mathbf{x} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$$ passed through a symmetric weight matrix $$W = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$.

1. **Matrix-Vector Transformation (Forward Pass)**:
   $$\mathbf{h} = W\mathbf{x} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} 2(3) + 1(1) \\ 1(3) + 2(1) \end{bmatrix} = \begin{bmatrix} 7 \\ 5 \end{bmatrix}$$
2. **Find the Eigenvalues of $$W$$**:
   $$\det(W - \lambda I) = (2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 3)(\lambda - 1) = 0$$
   So $$\lambda_1 = 3$$ (eigenvector $$\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$$) and $$\lambda_2 = 1$$ (eigenvector $$\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$$).
3. **Decompose $$\mathbf{x}$$ in the Eigenbasis**:
   Notice $$\mathbf{x} = \begin{bmatrix} 3 \\ 1 \end{bmatrix} = 2\begin{bmatrix} 1 \\ 1 \end{bmatrix} + 1\begin{bmatrix} 1 \\ -1 \end{bmatrix}$$.
4. **Apply $$W$$ Spectrally and Verify**:
   $$W\mathbf{x} = 2(3)\begin{bmatrix} 1 \\ 1 \end{bmatrix} + 1(1)\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 6 \\ 6 \end{bmatrix} + \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 7 \\ 5 \end{bmatrix}$$
   After $$k$$ repeated linear layers, $$W^k \mathbf{x} = 2(3^k)\begin{bmatrix} 1 \\ 1 \end{bmatrix} + 1(1^k)\begin{bmatrix} 1 \\ -1 \end{bmatrix}$$, showing immediately why the dominant eigenvalue $$\lambda_1 = 3$$ exponentially overwhelms the smaller mode!

This lesson covers essential linear algebra concepts needed for deep-learning, organized into three main sections for systematic learning.

## Video tham khảo

- **xem trước** [Vectors | Chapter 1, Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs) — 3Blue1Brown, English. Vectors in R^n are the objects the linear-algebra block is about.
- [Linear transformations and matrices | Chapter 3, Essence of linear algebra](https://www.youtube.com/watch?v=kYB8IZa5AuE) — 3Blue1Brown, English. Matrices as linear maps are the second lesson in this block.
- [Eigenvectors and eigenvalues | Chapter 14, Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g) — 3Blue1Brown, English. Eigenvectors are the third lesson; this is the geometric picture before any algorithm.
