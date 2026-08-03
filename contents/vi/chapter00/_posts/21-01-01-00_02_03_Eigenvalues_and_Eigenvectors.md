---
layout: post
title: 00-02-03 Giá trị riêng và Vector riêng
chapter: '00'
order: 10
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Bài học này trình bày giá trị riêng (*eigenvalue*) và vector riêng (*eigenvector*) — những khái niệm then chốt để hiểu hành vi của phép biến đổi tuyến tính và các hàm bậc hai trong tối ưu hóa.

---

## Định nghĩa và trực giác

Khi một ma trận biến đổi một vector, nói chung cả hướng lẫn độ dài của vector đều thay đổi. Tuy nhiên, **vector riêng** là những vector đặc biệt: dưới tác động của ma trận cho trước, chúng chỉ bị co giãn theo tỉ lệ mà không đổi hướng.

### Định nghĩa toán học

Với ma trận vuông $$\mathbf{A}$$ và vector khác không $$\mathbf{v}$$:

- $$\mathbf{v}$$ là **vector riêng** của $$\mathbf{A}$$
- $$\lambda$$ là **giá trị riêng** tương ứng

nếu chúng thỏa mãn **phương trình giá trị riêng**:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

### Diễn giải hình học

- **Vector riêng:** các vector khác không giữ nguyên hướng dưới phép biến đổi $$\mathbf{A}$$
- **Giá trị riêng:** các hệ số vô hướng theo đó vector riêng bị co giãn

**Hình dung trực quan:**
- Nếu $$\lambda > 1$$: vector riêng bị kéo dãn
- Nếu $$0 < \lambda < 1$$: vector riêng bị co lại
- Nếu $$\lambda < 0$$: vector riêng bị co giãn và đảo chiều
- Nếu $$\lambda = 0$$: vector riêng được ánh xạ thành vector không

---

## Tìm giá trị riêng và vector riêng

### Bước 1: Tìm giá trị riêng

Biến đổi phương trình giá trị riêng:
$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$
$$\mathbf{A}\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$$

Để có nghiệm không tầm thường ($$\mathbf{v} \neq \mathbf{0}$$), ma trận $$(\mathbf{A} - \lambda\mathbf{I})$$ phải suy biến (*singular*), do đó:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

Đây được gọi là **phương trình đặc trưng** (*characteristic equation*).

### Bước 2: Tìm vector riêng

Với mỗi giá trị riêng $$\lambda_i$$, giải hệ:
$$(\mathbf{A} - \lambda_i\mathbf{I})\mathbf{v} = \mathbf{0}$$

Các nghiệm tạo thành **không gian riêng** (*eigenspace*) ứng với $$\lambda_i$$.

---

## Ví dụ chi tiết

Ta tìm giá trị riêng và vector riêng của $$\mathbf{A} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$.

### Bước 1: Tìm giá trị riêng

$$\mathbf{A} - \lambda\mathbf{I} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix} - \lambda\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix}$$

$$\det(\mathbf{A} - \lambda\mathbf{I}) = (3-\lambda)(2-\lambda) - (1)(0) = (3-\lambda)(2-\lambda) = 0$$

Ta được $$\lambda_1 = 3$$ và $$\lambda_2 = 2$$.

### Bước 2: Tìm vector riêng

**Với $$\lambda_1 = 3$$:**
$$(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

Suy ra $$v_2 = 0$$ và $$v_1$$ có thể nhận bất kỳ giá trị khác không nào. Vậy $$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ (hoặc mọi bội vô hướng của nó).

**Với $$\lambda_2 = 2$$:**
$$(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

Suy ra $$v_1 + v_2 = 0$$, nên $$v_2 = -v_1$$. Vậy $$\mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$ (hoặc mọi bội vô hướng của nó).

### Kiểm tra

Ta kiểm tra kết quả:
- $$\mathbf{A}\mathbf{v}_1 = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix} = 3\begin{pmatrix} 1 \\ 0 \end{pmatrix} = 3\mathbf{v}_1$$ ✓
- $$\mathbf{A}\mathbf{v}_2 = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 2 \\ -2 \end{pmatrix} = 2\begin{pmatrix} 1 \\ -1 \end{pmatrix} = 2\mathbf{v}_2$$ ✓

---

## Tính chất và các định lý quan trọng

### Các tính chất chính

1. **Tổng các giá trị riêng bằng vết của ma trận:**
   $$\sum_{i=1}^n \lambda_i = \text{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}$$

2. **Tích các giá trị riêng bằng định thức của ma trận:**
   $$\prod_{i=1}^n \lambda_i = \det(\mathbf{A})$$

3. **Các vector riêng ứng với các giá trị riêng khác nhau là độc lập tuyến tính**

4. **Nếu $$\mathbf{A}$$ đối xứng thì mọi giá trị riêng đều thực và các vector riêng trực giao với nhau**

### Bội số của giá trị riêng

- **Bội đại số** (*algebraic multiplicity*): số lần $$\lambda$$ xuất hiện như nghiệm của đa thức đặc trưng
- **Bội hình học** (*geometric multiplicity*): chiều của không gian riêng (số vector riêng độc lập tuyến tính)

Với mọi giá trị riêng: bội hình học ≤ bội đại số

---

## Chéo hóa (*diagonalization*)

Ma trận $$\mathbf{A}$$ được gọi là **chéo hóa được** nếu có thể viết dưới dạng:

$$\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}$$

trong đó:
- $$\mathbf{D}$$ là ma trận chéo gồm các giá trị riêng
- $$\mathbf{P}$$ là ma trận có các cột là các vector riêng tương ứng

### Lợi ích của chéo hóa

1. **Tính lũy thừa dễ dàng:** $$\mathbf{A}^k = \mathbf{P}\mathbf{D}^k\mathbf{P}^{-1}$$
2. **Hiểu hành vi biến đổi:** các giá trị riêng quyết định hành vi của phép biến đổi dọc theo từng hướng vector riêng

---

## Ứng dụng trong tối ưu hóa

Giá trị riêng và vector riêng đóng vai trò then chốt trong tối ưu hóa vì nhiều lý do sau.

### 1. Dạng bậc hai và tính xác định

Với hàm bậc hai $$f(\mathbf{x}) = \mathbf{x}^T\mathbf{Q}\mathbf{x}$$:

- **Xác định dương** (*positive definite*, $$f(\mathbf{x}) > 0$$ với $$\mathbf{x} \neq \mathbf{0}$$): mọi giá trị riêng của $$\mathbf{Q}$$ đều dương
- **Nửa xác định dương** (*positive semidefinite*, $$f(\mathbf{x}) \geq 0$$): mọi giá trị riêng đều không âm
- **Xác định âm** (*negative definite*, $$f(\mathbf{x}) < 0$$ với $$\mathbf{x} \neq \mathbf{0}$$): mọi giá trị riêng đều âm
- **Không xác định** (*indefinite*, $$f(\mathbf{x})$$ có thể dương hoặc âm): có cả giá trị riêng dương và âm

### 2. Điều kiện tối ưu bậc hai

Với hàm $$f(\mathbf{x})$$ tại điểm tới hạn $$\mathbf{x}^*$$ (nơi $$\nabla f(\mathbf{x}^*) = \mathbf{0}$$):

- **Cực tiểu địa phương:** Hessian $$\nabla^2 f(\mathbf{x}^*)$$ xác định dương (mọi giá trị riêng $$ > 0$$)
- **Cực đại địa phương:** Hessian xác định âm (mọi giá trị riêng $$ < 0$$)
- **Điểm yên ngựa:** Hessian không xác định (có cả giá trị riêng dương và âm)

### 3. Phân tích thành phần chính (PCA)

PCA tìm các hướng phương sai cực đại trong dữ liệu:
- Các vector riêng của ma trận hiệp phương sai cho các hướng chính
- Các giá trị riêng cho phương sai dọc theo từng hướng chính

### 4. Phân tích hội tụ

Trong các thuật toán tối ưu lặp:
- **Số điều kiện** (*condition number*) $$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$ ảnh hưởng đến tốc độ hội tụ
- Số điều kiện lớn dẫn đến hội tụ chậm

### 5. Phương pháp Newton

Phương pháp Newton dùng nghịch đảo của Hessian:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

Các giá trị riêng của Hessian quyết định hành vi và tốc độ hội tụ của phương pháp.

---

## Ví dụ: Ứng dụng trong tối ưu hóa

Xét bài toán cực tiểu hóa $$f(x, y) = 2x^2 + 3y^2 + 2xy$$.

Hessian là: $$\mathbf{H} = \begin{pmatrix} 4 & 2 \\ 2 & 6 \end{pmatrix}$$

**Tìm giá trị riêng:**
$$\det(\mathbf{H} - \lambda\mathbf{I}) = (4-\lambda)(6-\lambda) - 4 = \lambda^2 - 10\lambda + 20 = 0$$

$$\lambda = \frac{10 \pm \sqrt{100-80}}{2} = \frac{10 \pm 2\sqrt{5}}{2} = 5 \pm \sqrt{5}$$

Vì cả hai giá trị riêng đều dương ($$\lambda_1 = 5 + \sqrt{5} > 0$$ và $$\lambda_2 = 5 - \sqrt{5} > 0$$), Hessian xác định dương, xác nhận gốc tọa độ là cực tiểu toàn cục.

Số điều kiện là $$\kappa = \frac{5 + \sqrt{5}}{5 - \sqrt{5}} \approx 4{,}24$$, cho thấy bài toán có điều kiện số khá tốt đối với các thuật toán tối ưu.

Hiểu giá trị riêng và vector riêng mang lại cái nhìn sâu về các tính chất hình học và giải tích của bài toán tối ưu hóa, từ đó hỗ trợ thiết kế thuật toán và phân tích hội tụ tốt hơn.
