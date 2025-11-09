---
layout: post
title: 00-01-03 Gradient và Đạo Hàm Theo Hướng
chapter: '00'
order: 5
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

This lesson explores the gradient vector and directional derivatives, which are central concepts in deep-learning for understanding how functions change in different directions.

---

## Gradient Vector

The gradient $$\nabla f$$ is a vector composed of the partial derivatives of the function $$f$$ with respect to each of its variables. It indicates the direction of the steepest ascent of the function at a given point.

### Definition and Computation

For a function of two variables, $$f(x, y)$$, its gradient is:

$$ \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix} $$

For a function of $$n$$ variables, $$f(x_1, x_2, \ldots, x_n)$$:

$$ \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix} $$

### Example: Computing a Gradient

For $$f(x, y) = x^2 + 3xy + y^2$$:

$$\frac{\partial f}{\partial x} = 2x + 3y$$
$$\frac{\partial f}{\partial y} = 3x + 2y$$

Therefore: $$\nabla f = \begin{pmatrix} 2x + 3y \\ 3x + 2y \end{pmatrix}$$

At the point $$(1, 2)$$: $$\nabla f(1, 2) = \begin{pmatrix} 2(1) + 3(2) \\ 3(1) + 2(2) \end{pmatrix} = \begin{pmatrix} 8 \\ 7 \end{pmatrix}$$

---

## Directional Derivatives

The **directional derivative** measures the rate of change of $$f$$ when we move in any chosen direction $$\mathbf{u}$$. Here $$\mathbf{u}$$ must be a unit vector (length 1).

### Definition

For a function $$f(\mathbf{x})$$ and unit vector $$\mathbf{u} = \langle u_1, u_2, \ldots, u_n \rangle$$:

$$D_{\mathbf{u}}f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} u_i$$

### Geometric Interpretation

The directional derivative can be written as:

$$D_{\mathbf{u}}f = \lvert \nabla f \rvert \cos \theta$$

where $$\theta$$ is the angle between $$\nabla f$$ and $$\mathbf{u}$$, and $$\lvert \nabla f \rvert$$ is the magnitude of the gradient.

### Example: Computing Directional Derivatives

Using our previous example $$f(x, y) = x^2 + 3xy + y^2$$ at point $$(1, 2)$$ where $$\nabla f(1, 2) = \begin{pmatrix} 8 \\ 7 \end{pmatrix}$$:

**Direction 1:** $$\mathbf{u}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ (positive x-direction)
$$D_{\mathbf{u}_1}f(1, 2) = 8 \cdot 1 + 7 \cdot 0 = 8$$

**Direction 2:** $$\mathbf{u}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ (positive y-direction)
$$D_{\mathbf{u}_2}f(1, 2) = 8 \cdot 0 + 7 \cdot 1 = 7$$

**Direction 3:** $$\mathbf{u}_3 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ (45° diagonal)
$$D_{\mathbf{u}_3}f(1, 2) = 8 \cdot \frac{1}{\sqrt{2}} + 7 \cdot \frac{1}{\sqrt{2}} = \frac{15}{\sqrt{2}} \approx 10.61$$

---

## Maximum and Minimum Rates of Change

### Key Properties

From the formula $$D_{\mathbf{u}}f = \lvert \nabla f \rvert \cos \theta$$, we can determine:

1. **Maximum Rate of Change**: Occurs when $$\cos \theta = 1$$ (i.e., $$\theta = 0°$$)
   - Direction: $$\mathbf{u} = \frac{\nabla f}{\lvert \nabla f \rvert}$$ (same direction as gradient)
   - Maximum rate: $$D_{\max}f = \lvert \nabla f \rvert$$

2. **Minimum Rate of Change**: Occurs when $$\cos \theta = -1$$ (i.e., $$\theta = 180°$$)
   - Direction: $$\mathbf{u} = -\frac{\nabla f}{\lvert \nabla f \rvert}$$ (opposite to gradient)
   - Minimum rate: $$D_{\min}f = -\lvert \nabla f \rvert$$

3. **Zero Rate of Change**: Occurs when $$\cos \theta = 0$$ (i.e., $$\theta = 90°$$)
   - Direction: Any vector perpendicular to $$\nabla f$$

### Summary of Gradient Properties

- The gradient $$\nabla f$$ points in the direction of **steepest increase**
- The direction $$-\nabla f$$ points in the direction of **steepest decrease**
- The magnitude $$\lvert \nabla f \rvert$$ gives the **maximum rate of change**
- When $$\nabla f = \mathbf{0}$$, the point is a **critical point** (potential optimum)

---

## Relation to Level Curves

At any point on a level curve $$f(x, y) = c$$, the gradient vector $$\nabla f$$ is **orthogonal (perpendicular)** to the tangent line of the level curve at that point.

### Why This Matters

This orthogonality property is fundamental because:

1. **Level curves represent constant function values**: Moving along a level curve doesn't change the function value, so the directional derivative is zero.

2. **Gradient points to steepest increase**: The direction that increases the function value most rapidly must be perpendicular to the direction that doesn't change it at all.

3. **Deep Learning insight**: To find extrema, we look for points where the gradient is zero (critical points) or where the gradient is perpendicular to the constraint boundary.

### Applications in Deep Learning

Understanding gradients and directional derivatives is crucial for:

1. **Gradient Descent**: Moving in the direction $$-\nabla f$$ to minimize $$f$$
2. **Gradient Ascent**: Moving in the direction $$+\nabla f$$ to maximize $$f$$
3. **Constrained Deep Learning**: Using the relationship between gradients and level curves
4. **Convergence Analysis**: Understanding when algorithms will converge to optimal solutions
5. **Step Size Selection**: Determining how far to move in the gradient direction

The gradient provides both the direction to move and information about how quickly the function is changing, making it the foundation for most deep-learning algorithms.
