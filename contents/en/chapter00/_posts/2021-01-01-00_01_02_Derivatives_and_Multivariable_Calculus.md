---
layout: post
title: 00-01-02 Derivatives and Multivariable Calculus
chapter: '00'
order: 4
owner: GitHub Copilot
lang: en
categories:
- chapter00
---

> *When a neural network misclassifies an X-ray, a single scalar loss number has to send bespoke blame signals back to millions of weights simultaneously. Multivariable calculus—partial derivatives, Jacobians, Hessians, and the multivariable chain rule—is the exact bookkeeping system that makes that feat possible.*

## Key ideas: Multivariable Calculus for Neural Optimization

1. **Partial Derivatives (Axis-Aligned Sensitivities)**: Isolate the rate of change of a loss $$f(x_1, \dots, x_n)$$ with respect to one coordinate while holding all other parameters frozen.
2. **The Gradient & Jacobian (First-Order Linearization)**: Package all partial derivatives of a scalar function into a gradient vector $$\nabla f$$, or of a vector-valued layer $$h: \mathbb{R}^n \to \mathbb{R}^m$$ into an $$m \times n$$ Jacobian matrix $$J_h$$.
3. **The Hessian Matrix (Second-Order Curvature)**: Collects all second partial derivatives $$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$, distinguishing strict local minima (positive definite), maxima (negative definite), and saddle points (indefinite)—which dominate high-dimensional neural loss landscapes.
4. **Multivariable Chain Rule (The Core of Backpropagation)**: Composes local derivatives along every path in a computational graph, summing contributions whenever one variable feeds into multiple downstream nodes.

### Worked example: Gradient, Hessian, and Chain Rule on a Coupled Loss

Consider the two-variable objective $$f(x, y) = x^2 y + 3y^2$$ along the parameterized path $$x(t) = 2t$$, $$y(t) = t + 1$$, evaluated at $$t = 1$$ (so $$x = 2, y = 2$$).

1. **Compute the Gradient Vector $$\nabla f(x, y)$$**:
   $$\frac{\partial f}{\partial x} = 2xy, \qquad \frac{\partial f}{\partial y} = x^2 + 6y$$
   At $$(x, y) = (2, 2)$$:
   $$\nabla f(2, 2) = \begin{bmatrix} 2(2)(2) \\ 2^2 + 6(2) \end{bmatrix} = \begin{bmatrix} 8 \\ 16 \end{bmatrix}$$
2. **Compute the Hessian Matrix $$H_f(x, y)$$**:
   $$\frac{\partial^2 f}{\partial x^2} = 2y, \quad \frac{\partial^2 f}{\partial x \partial y} = 2x, \quad \frac{\partial^2 f}{\partial y^2} = 6$$
   At $$(2, 2)$$:
   $$H_f(2, 2) = \begin{bmatrix} 4 & 4 \\ 4 & 6 \end{bmatrix}$$
   Check definiteness via leading principal minors: $$D_1 = 4 > 0$$ and $$D_2 = (4)(6) - (4)(4) = 24 - 16 = 8 > 0$$. Thus $$H_f(2, 2)$$ is strictly positive definite (locally bowl-shaped).
3. **Apply the Multivariable Chain Rule for $$\frac{df}{dt}$$ at $$t = 1$$**:
   We have $$\frac{dx}{dt} = 2$$ and $$\frac{dy}{dt} = 1$$. Therefore:
   $$\frac{df}{dt}\Big|_{t=1} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt} = (8)(2) + (16)(1) = 16 + 16 = 32$$
4. **Independent Substitution Check**:
   Substitute directly: $$f(t) = (2t)^2(t+1) + 3(t+1)^2 = 4t^3 + 7t^2 + 6t + 3$$. Differentiating gives $$f'(t) = 12t^2 + 14t + 6$$, and at $$t = 1$$: $$f'(1) = 12 + 14 + 6 = 32$$. Exact match!

This lesson covers derivatives and essential multivariable calculus concepts that form the foundation for deep-learning theory and algorithms.

---

## Derivatives and Rate of Change

The derivative of a single variable function represents its instantaneous rate of change, which is fundamental to understanding how functions behave locally.

### Basic Derivative Concepts

**Slope between two points:**

$$\text{Slope} = \frac{y_2 - y_1}{x_2 - x_1}$$

**Derivative (instantaneous rate of change):**

$$f'(x_0) = \lim_{x_1 \to x_0} \frac{f(x_1) - f(x_0)}{x_1 - x_0} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

The derivative tells us how quickly the function is changing at any given point, which is essential for finding optimal points where the rate of change is zero.

### Level Curves of Functions

Level curves are a fundamental concept in multivariable calculus used to visualize functions of two variables, typically denoted as $$f(x, y)$$. They provide a way to represent a 3D surface in a 2D plane.

A **level curve** of a function $$f(x, y)$$ is the set of all points $$(x, y)$$ in the domain of $$f$$ where the function takes a constant value:

$$f(x, y) = c$$

**Examples:**
- For $$f(x, y) = x^2 + y^2$$, the level curves are circles: $$x^2 + y^2 = c$$
- For $$f(x, y) = x + y$$, the level curves are parallel lines: $$x + y = c$$

Level curves help us understand:
1. The topography of the function
2. Directions of steepest ascent and descent
3. Locations of potential optima

---

## Multivariable Calculus Key Concepts

### Partial Derivatives

For a function $$f(x_1, x_2, \ldots, x_n)$$, the **partial derivative** with respect to $$x_i$$ is:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

This measures how $$f$$ changes when only $$x_i$$ varies while all other variables remain fixed.

### Gradient Vector

The **gradient** is a vector composed of all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

The gradient points in the direction of steepest increase of the function and is perpendicular to level curves.

### Hessian Matrix

The **Hessian matrix** contains all second-order partial derivatives:

$$\nabla^2 f(\mathbf{x}) = \mathbf{H} = \begin{pmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

The Hessian provides information about the curvature of the function and is crucial for:
- Determining the nature of critical points (minimum, maximum, or saddle point)
- Second-order deep-learning methods like Newton's method

---

## Chain Rule for Multivariable Functions

The chain rule is fundamental for computing derivatives of composite functions, which frequently appear in deep-learning problems.

### Basic Chain Rule

For a function $$z = f(x, y)$$ where $$x = g(t)$$ and $$y = h(t)$$:

$$ \frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} $$

### General Chain Rule

For $$z = f(x_1, x_2, \ldots, x_n)$$ where each $$x_i = x_i(t_1, t_2, \ldots, t_m)$$:

$$ \frac{\partial z}{\partial t_j} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \frac{\partial x_i}{\partial t_j} $$

### Applications in Deep Learning

The chain rule is essential for:

1. **Gradient Computation**: Computing gradients of composite objective functions
2. **Constraint Handling**: Dealing with constraints that are functions of other variables
3. **Algorithm Implementation**: Backpropagation in neural networks and automatic differentiation
4. **Sensitivity Analysis**: Understanding how changes in parameters affect optimal solutions

### Example: Deep Learning with Constraints

Consider minimizing $$f(x, y) = x^2 + y^2$$ subject to $$g(x, y) = x + y - 1 = 0$$.

Using the constraint to eliminate one variable: $$y = 1 - x$$, so we minimize:
$$h(x) = f(x, 1-x) = x^2 + (1-x)^2$$

Using the chain rule:
$$h'(x) = \frac{\partial f}{\partial x} \cdot 1 + \frac{\partial f}{\partial y} \cdot \frac{d(1-x)}{dx} = 2x + 2(1-x)(-1) = 4x - 2$$

Setting $$h'(x) = 0$$ gives $$x = 1/2$$, so the optimal point is $$(1/2, 1/2)$$.

This demonstrates how multivariable calculus concepts work together to solve deep-learning problems systematically.

## Video tham khảo

- **xem trước** [The paradox of the derivative | Chapter 2, Essence of calculus](https://www.youtube.com/watch?v=9vKqVkMQHKk) — 3Blue1Brown, English. The one-variable derivative the lesson uses as the rate of change.
- [What are derivatives in 3D? Intro to Partial Derivatives](https://www.youtube.com/watch?v=EoEV5-_mLeM) — Dr. Trefor Bazett, English. Partial derivatives are the coordinate-wise rates the multivariable section adds once the input is a vector.
- [Visualizing the chain rule and product rule | Chapter 4, Essence of calculus](https://www.youtube.com/watch?v=YG15m2VwSjA) — 3Blue1Brown, English. The chain rule visualized here is the one-variable case; the lesson's multivariable chain rule is the same product of derivatives along a composition.
