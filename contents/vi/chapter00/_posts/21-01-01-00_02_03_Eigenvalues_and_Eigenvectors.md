---
layout: post
title: 00-02-03 Giá trị riêng và Vector riêng
chapter: '00'
order: 10
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

Bài học này bao gồm giá trị riêng và vector riêng, rất quan trọng để hiểu hành vi của các phép biến đổi tuyến tính và hàm bậc hai trong tối ưu hóa.

---

## Định nghĩa và Trực giác

Khi một ma trận biến đổi một vector, nó thường thay đổi cả hướng và độ dài của vector. Tuy nhiên, **vector riêng** là những vector đặc biệt mà khi được biến đổi bởi một ma trận cho trước, chỉ bị thay đổi tỉ lệ nhưng không thay đổi hướng.

### Định nghĩa Toán học

Với một ma trận vuông $$\mathbf{A}$$ và một vector khác không $$\mathbf{v}$$:

- $$\mathbf{v}$$ là một **vector riêng** của $$\mathbf{A}$$
- $$\lambda$$ là **giá trị riêng** tương ứng

nếu chúng thỏa mãn **phương trình giá trị riêng**:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

### Giải thích Hình học

- **Vector riêng:** Các vector khác không duy trì hướng của chúng dưới phép biến đổi $$\mathbf{A}$$
- **Giá trị riêng:** Các hệ số vô hướng mà các vector riêng được nhân với

**Hiểu biết Trực quan:**
- Nếu $$\lambda > 1$$: Vector riêng bị kéo dài
- Nếu $$0 < \lambda < 1$$: Vector riêng bị co lại
- Nếu $$\lambda < 0$$: Vector riêng bị nhân tỉ lệ và đảo ngược
- Nếu $$\lambda = 0$$: Vector riêng được ánh xạ thành vector không

---

## Tìm Giá trị riêng và Vector riêng

### Bước 1: Tìm Giá trị riêng

Sắp xếp lại phương trình giá trị riêng:
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

## Phân Tích Eigendecomposition

### Ma Trận Đối Xứng (Symmetric Matrices)

Ma trận đối xứng $$\mathbf{A} = \mathbf{A}^T$$ có những tính chất đặc biệt quan trọng:

1. **Tất cả giá trị riêng đều là số thực**
2. **Các vector riêng tương ứng với giá trị riêng khác nhau là trực giao**
3. **Luôn có thể chéo hóa được với ma trận trực giao**

Với ma trận đối xứng, ta có phân tích đặc biệt:

$$\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$$

trong đó:
- $$\mathbf{Q}$$ là ma trận trực giao ($$\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$$) với các cột là vector riêng chuẩn hóa
- $$\mathbf{\Lambda}$$ là ma trận chéo chứa các giá trị riêng

### Các Phép Toán Trên Eigendecomposition

Eigendecomposition cho phép thực hiện nhiều phép toán phức tạp một cách hiệu quả:

**1. Lũy thừa ma trận:**

$$\mathbf{A}^k = \mathbf{P}\mathbf{D}^k\mathbf{P}^{-1}$$

Điều này đặc biệt hữu ích khi $$k$$ lớn, vì việc lũy thừa ma trận chéo rất đơn giản:

$$\mathbf{D}^k = \begin{pmatrix} \lambda_1^k & 0 & \cdots & 0 \\ 0 & \lambda_2^k & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n^k \end{pmatrix}$$

**2. Ma trận nghịch đảo:**

Nếu tất cả giá trị riêng khác 0:

$$\mathbf{A}^{-1} = \mathbf{P}\mathbf{D}^{-1}\mathbf{P}^{-1}$$

với $$\mathbf{D}^{-1}$$ có các phần tử chéo là $$1/\lambda_i$$.

**3. Ma trận hàm mũ:**

$$e^{\mathbf{A}} = \mathbf{P}e^{\mathbf{D}}\mathbf{P}^{-1}$$

với $$e^{\mathbf{D}} = \text{diag}(e^{\lambda_1}, e^{\lambda_2}, \ldots, e^{\lambda_n})$$

**4. Định thức và vết (trace):**

- $$\det(\mathbf{A}) = \prod_i \lambda_i$$ (tích các giá trị riêng)
- $$\text{tr}(\mathbf{A}) = \sum_i \lambda_i$$ (tổng các giá trị riêng)

---

## Định Lý Gershgorin Circle

Định lý Gershgorin Circle cung cấp một cách đơn giản để ước lượng vị trí của các giá trị riêng mà không cần tính toán chính xác.

### Phát Biểu Định Lý

Cho ma trận $$\mathbf{A} = [a_{ij}]_{n \times n}$$. Định nghĩa:
- **Tâm (center):** $$a_{ii}$$ (phần tử chéo thứ $$i$$)
- **Bán kính (radius):** $$r_i = \sum_{j \neq i} |a_{ij}|$$ (tổng giá trị tuyệt đối các phần tử ngoài đường chéo ở hàng $$i$$)

**Định lý:** Mọi giá trị riêng của $$\mathbf{A}$$ nằm trong hợp của các đĩa Gershgorin:

$$D_i = \{z \in \mathbb{C} : |z - a_{ii}| \leq r_i\}$$

### Ví Dụ

Xét ma trận: $$\mathbf{A} = \begin{pmatrix} 4 & 0.3 & 0.2 \\ 0.3 & 5 & 0.1 \\ 0.2 & 0.1 & 3 \end{pmatrix}$$

Các đĩa Gershgorin:
- $$D_1$$: tâm = 4, bán kính = 0.5, nên $$3.5 \leq \lambda \leq 4.5$$
- $$D_2$$: tâm = 5, bán kính = 0.4, nên $$4.6 \leq \lambda \leq 5.4$$
- $$D_3$$: tâm = 3, bán kính = 0.3, nên $$2.7 \leq \lambda \leq 3.3$$

**Kết luận:** Tất cả giá trị riêng phải nằm trong $$[2.7, 5.4]$$.

### Ứng Dụng Thực Tế

1. **Kiểm tra nhanh tính positive definite:** Nếu tất cả đĩa Gershgorin nằm bên phải trục tưởng, ma trận có thể là positive definite
2. **Ước lượng condition number:** Giúp đánh giá độ ổn định số học
3. **Thiết kế ma trận:** Đảm bảo các thuộc tính mong muốn bằng cách điều chỉnh phần tử chéo và ngoài đường chéo

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

## Ứng Dụng: Hành Vi của Phép Nhân Ma Trận Lặp

Một ứng dụng quan trọng của eigenvalues là hiểu hành vi của việc nhân ma trận lặp lại: $$\mathbf{A}\mathbf{x}, \mathbf{A}^2\mathbf{x}, \mathbf{A}^3\mathbf{x}, \ldots$$

### Vector Riêng và Hành Vi Dài Hạn

Giả sử ma trận $$\mathbf{A}$$ có eigendecomposition:

$$\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}$$

Khi đó:

$$\mathbf{A}^k = \mathbf{P}\mathbf{D}^k\mathbf{P}^{-1} = \mathbf{P}\begin{pmatrix} \lambda_1^k & 0 & \cdots \\ 0 & \lambda_2^k & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}\mathbf{P}^{-1}$$

**Quan sát quan trọng:**

1. **Nếu $$|\lambda_{\max}| > 1$$:** Vector sẽ tăng theo cấp số nhân (explode)
2. **Nếu $$|\lambda_{\max}| < 1$$:** Vector sẽ giảm dần về 0 (vanish)
3. **Nếu $$|\lambda_{\max}| = 1$$:** Vector sẽ ổn định hoặc dao động

### Ví Dụ: Ma Trận Ngẫu Nhiên

Xét ma trận ngẫu nhiên $$5 \times 5$$ với các phần tử tuân theo phân phối Gaussian chuẩn (mean = 0, variance = 1).

Nếu ta áp dụng ma trận này lên một vector ngẫu nhiên nhiều lần, ta quan sát:
- Độ dài của vector tăng theo cấp số nhân
- Tốc độ tăng được chi phối bởi giá trị riêng lớn nhất $$\lambda_{\max}$$

**Định luật Circular Law:** Đối với ma trận ngẫu nhiên lớn $$n \times n$$ với các phần tử độc lập, mean = 0, variance = 1, giá trị riêng lớn nhất trung bình xấp xỉ $$\sqrt{n}$$.

### Chuẩn Hóa Ma Trận

Để kiểm soát hành vi này, ta có thể chuẩn hóa ma trận:

$$\mathbf{A}_{\text{normalized}} = \frac{\mathbf{A}}{\lambda_{\max}}$$

Sau khi chuẩn hóa:
- Giá trị riêng lớn nhất = 1
- Vector không tăng hoặc giảm theo cấp số nhân
- Hành vi ổn định, hội tụ về trạng thái cân bằng

### Ý Nghĩa Trong Deep Learning

Hiện tượng này có ý nghĩa quan trọng trong neural networks:

1. **Vanishing/Exploding Gradients:** 
   - Nếu các trọng số (weights) có eigenvalues quá lớn → gradients explode
   - Nếu eigenvalues quá nhỏ → gradients vanish
   
2. **Khởi Tạo Trọng Số (Weight Initialization):**
   - Cần khởi tạo sao cho eigenvalues gần 1
   - Các phương pháp như Xavier/He initialization dựa trên nguyên lý này
   
3. **Batch Normalization:**
   - Giúp kiểm soát phân phối của activations
   - Ngăn chặn exploding/vanishing trong quá trình training

4. **Residual Connections (ResNet):**
   - Thêm skip connections giúp gradient flow tốt hơn
   - Tương đương với việc thêm ma trận đơn vị, đảm bảo eigenvalue = 1

### Phân Tích Hội Tụ Của Thuật Toán

Trong các thuật toán lặp (như gradient descent, power iteration):

**Tốc độ hội tụ** phụ thuộc vào tỷ số của các eigenvalues:

$$\text{Convergence rate} \propto \left|\frac{\lambda_2}{\lambda_1}\right|$$

- Nếu $$|\lambda_2| \ll |\lambda_1|$$: Hội tụ nhanh
- Nếu $$|\lambda_2| \approx |\lambda_1|$$: Hội tụ chậm

**Power Iteration Algorithm:**
1. Bắt đầu với vector ngẫu nhiên $$\mathbf{v}_0$$
2. Lặp: $$\mathbf{v}_{k+1} = \frac{\mathbf{A}\mathbf{v}_k}{\|\mathbf{A}\mathbf{v}_k\|}$$
3. Vector sẽ hội tụ về eigenvector tương ứng với $$|\lambda_{\max}|$$

Đây là cơ sở cho nhiều thuật toán tính eigenvalues/eigenvectors hiệu quả.

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
