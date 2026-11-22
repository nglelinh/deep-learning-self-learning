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

Bài học này trình bày giá trị riêng (*eigenvalue*) và vector riêng (*eigenvector*) — những khái niệm cốt lõi để phân tích hành vi của phép biến đổi tuyến tính và các hàm bậc hai trong tối ưu hóa.

---

## Định nghĩa và trực giác

Khi một ma trận tác động lên vector, thông thường cả hướng lẫn độ dài của vector đều thay đổi. Tuy nhiên, **vector riêng** là những vector đặc biệt: dưới tác động của ma trận, phương của chúng không đổi mà chỉ bị co giãn theo một hệ số tỉ lệ.

### Định nghĩa toán học

Với ma trận vuông $$\mathbf{A}$$ và vector khác không $$\mathbf{v}$$:

- $$\mathbf{v}$$ là **vector riêng** của $$\mathbf{A}$$,
- $$\lambda$$ là **giá trị riêng** tương ứng,

nếu chúng thỏa mãn **phương trình giá trị riêng**:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

### Diễn giải hình học

- **Vector riêng:** các vector khác không giữ nguyên phương dưới phép biến đổi $$\mathbf{A}$$.
- **Giá trị riêng:** hệ số vô hướng quy định mức co giãn của vector riêng tương ứng.

**Ý nghĩa theo giá trị của $$\lambda$$:**
- Nếu $$\lambda > 1$$: vector riêng bị kéo giãn.
- Nếu $$0 < \lambda < 1$$: vector riêng bị co lại.
- Nếu $$\lambda < 0$$: vector riêng vừa bị co giãn vừa đổi chiều ngược lại.
- Nếu $$\lambda = 0$$: vector riêng bị ánh xạ về vector không.

---

## Cách tìm giá trị riêng và vector riêng

### Bước 1: Tìm giá trị riêng

Biến đổi phương trình giá trị riêng:
$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$
$$\mathbf{A}\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$$

Để hệ có nghiệm không tầm thường ($$\mathbf{v} \neq \mathbf{0}$$), ma trận $$(\mathbf{A} - \lambda\mathbf{I})$$ phải suy biến (*singular*), tức là:

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

Phương trình này được gọi là **phương trình đặc trưng** (*characteristic equation*).

### Bước 2: Tìm vector riêng

Với mỗi giá trị riêng $$\lambda_i$$, ta giải hệ phương trình tuyến tính thuần nhất:
$$(\mathbf{A} - \lambda_i\mathbf{I})\mathbf{v} = \mathbf{0}$$

Tập nghiệm của hệ tạo thành **không gian riêng** (*eigenspace*) ứng với $$\lambda_i$$.

---

## Ví dụ chi tiết

Tìm giá trị riêng và vector riêng của ma trận $$\mathbf{A} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$.

### Bước 1: Tìm giá trị riêng

$$\mathbf{A} - \lambda\mathbf{I} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix} - \lambda\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix}$$

$$\det(\mathbf{A} - \lambda\mathbf{I}) = (3-\lambda)(2-\lambda) - (1)(0) = (3-\lambda)(2-\lambda) = 0$$

Suy ra $$\lambda_1 = 3$$ và $$\lambda_2 = 2$$.

### Bước 2: Tìm vector riêng

**Với $$\lambda_1 = 3$$:**
$$(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

Hệ cho $$v_2 = 0$$ và $$v_1$$ tùy ý khác không. Chọn $$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ (mọi bội khác không của nó cũng là vector riêng).

**Với $$\lambda_2 = 2$$:**
$$(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

Hệ cho $$v_1 + v_2 = 0$$, hay $$v_2 = -v_1$$. Chọn $$\mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$ (hoặc mọi bội khác không của nó).

### Kiểm tra lại kết quả

- $$\mathbf{A}\mathbf{v}_1 = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix} = 3\begin{pmatrix} 1 \\ 0 \end{pmatrix} = 3\mathbf{v}_1$$ ✓
- $$\mathbf{A}\mathbf{v}_2 = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 2 \\ -2 \end{pmatrix} = 2\begin{pmatrix} 1 \\ -1 \end{pmatrix} = 2\mathbf{v}_2$$ ✓

---

## Tính chất và các định lý quan trọng

### Các tính chất chính

1. **Tổng các giá trị riêng bằng vết của ma trận:**
   $$\sum_{i=1}^n \lambda_i = \text{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}$$

2. **Tích các giá trị riêng bằng định thức của ma trận:**
   $$\prod_{i=1}^n \lambda_i = \det(\mathbf{A})$$

3. **Các vector riêng ứng với các giá trị riêng phân biệt thì độc lập tuyến tính.**

4. **Nếu $$\mathbf{A}$$ là ma trận thực đối xứng, mọi giá trị riêng đều là số thực và các vector riêng ứng với các giá trị riêng khác nhau trực giao với nhau.**

### Bội số của giá trị riêng

- **Bội đại số** (*algebraic multiplicity*): số lần $$\lambda$$ xuất hiện với tư cách nghiệm của đa thức đặc trưng.
- **Bội hình học** (*geometric multiplicity*): số chiều của không gian riêng (số vector riêng độc lập tuyến tính tối đa ứng với $$\lambda$$).

Luôn có bất đẳng thức: bội hình học ≤ bội đại số.

---

## Chéo hóa ma trận (*diagonalization*)

Ma trận vuông $$\mathbf{A}$$ được gọi là **chéo hóa được** nếu tồn tại ma trận khả nghịch $$\mathbf{P}$$ và ma trận đường chéo $$\mathbf{D}$$ sao cho:

$$\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1}$$

trong đó:
- $$\mathbf{D}$$ là ma trận đường chéo chứa các giá trị riêng của $$\mathbf{A}$$,
- các cột của $$\mathbf{P}$$ là các vector riêng tương ứng.

### Lợi ích của phép chéo hóa

1. **Tính lũy thừa nhanh chóng:** $$\mathbf{A}^k = \mathbf{P}\mathbf{D}^k\mathbf{P}^{-1}$$.
2. **Phân tách hành vi biến đổi:** từng giá trị riêng mô tả riêng rẽ tác động co giãn của phép biến đổi dọc theo từng hướng vector riêng.

---

## Ứng dụng trong tối ưu hóa

Giá trị riêng và vector riêng đóng vai trò trọng tâm trong nhiều bài toán tối ưu:

### 1. Dạng bậc hai và tính xác định

Với dạng bậc hai $$f(\mathbf{x}) = \mathbf{x}^T\mathbf{Q}\mathbf{x}$$ ($$\mathbf{Q}$$ đối xứng):

- **Xác định dương** (*positive definite*, $$f(\mathbf{x}) > 0$$ với mọi $$\mathbf{x} \neq \mathbf{0}$$): mọi giá trị riêng của $$\mathbf{Q}$$ đều dương.
- **Nửa xác định dương** (*positive semidefinite*, $$f(\mathbf{x}) \geq 0$$): mọi giá trị riêng đều không âm.
- **Xác định âm** (*negative definite*, $$f(\mathbf{x}) < 0$$ với mọi $$\mathbf{x} \neq \mathbf{0}$$): mọi giá trị riêng đều âm.
- **Không xác định** (*indefinite*, $$f(\mathbf{x})$$ nhận cả giá trị dương và âm): có cả giá trị riêng dương lẫn âm.

### 2. Điều kiện tối ưu bậc hai

Tại điểm tới hạn $$\mathbf{x}^*$$ (thỏa mãn $$\nabla f(\mathbf{x}^*) = \mathbf{0}$$):

- **Cực tiểu địa phương:** ma trận Hessian $$\nabla^2 f(\mathbf{x}^*)$$ xác định dương (mọi giá trị riêng $$> 0$$).
- **Cực đại địa phương:** ma trận Hessian xác định âm (mọi giá trị riêng $$< 0$$).
- **Điểm yên ngựa:** ma trận Hessian không xác định (có cả giá trị riêng dương và âm).

### 3. Phân tích thành phần chính (PCA)

PCA tìm các phương có phương sai dữ liệu lớn nhất:
- Các vector riêng của ma trận hiệp phương sai xác định các trục thành phần chính.
- Các giá trị riêng tương ứng đo phương sai dữ liệu dọc theo từng trục đó.

### 4. Phân tích tốc độ hội tụ

Trong các thuật toán tối ưu lặp (với Hessian đối xứng xác định dương):
- **Số điều kiện** (*condition number*) $$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$ chi phối trực tiếp tốc độ hội tụ.
- Số điều kiện càng lớn thì tốc độ hội tụ của các phương pháp bậc nhất càng chậm.

### 5. Phương pháp Newton

Phương pháp Newton sử dụng nghịch đảo của ma trận Hessian:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

Các giá trị riêng của Hessian quyết định tính khả nghịch, độ ổn định và tốc độ hội tụ của thuật toán.

---

## Ví dụ: Ứng dụng trong tối ưu hóa

Xét bài toán cực tiểu hóa $$f(x, y) = 2x^2 + 3y^2 + 2xy$$.

Ma trận Hessian của hàm là: $$\mathbf{H} = \begin{pmatrix} 4 & 2 \\ 2 & 6 \end{pmatrix}$$

**Tìm giá trị riêng:**
$$\det(\mathbf{H} - \lambda\mathbf{I}) = (4-\lambda)(6-\lambda) - 4 = \lambda^2 - 10\lambda + 20 = 0$$

$$\lambda = \frac{10 \pm \sqrt{100-80}}{2} = \frac{10 \pm 2\sqrt{5}}{2} = 5 \pm \sqrt{5}$$

Vì cả hai giá trị riêng đều dương ($$\lambda_1 = 5 + \sqrt{5} > 0$$ và $$\lambda_2 = 5 - \sqrt{5} > 0$$), ma trận Hessian xác định dương, khẳng định gốc tọa độ $$(0, 0)$$ là cực tiểu toàn cục.

Số điều kiện là $$\kappa = \frac{5 + \sqrt{5}}{5 - \sqrt{5}} \approx 2{,}62$$, cho thấy bài toán có điều kiện số khá tốt đối với các thuật toán tối ưu hóa.

Hiểu rõ giá trị riêng và vector riêng giúp nắm bắt sâu cấu trúc hình học và giải tích của bài toán tối ưu, từ đó lựa chọn thuật toán và đánh giá hội tụ hiệu quả hơn.

## Video tham khảo

- **Xem trước:** [Eigenvectors and eigenvalues | Chapter 14, Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g) — 3Blue1Brown, tiếng Anh. Trực quan hóa định nghĩa hình học: $$\mathbf{A}\mathbf{v}$$ chỉ là bản co giãn của $$\mathbf{v}$$ dọc theo cùng một phương.
- [MaDL - Eigenvalue and Singular Value Decomposition](https://www.youtube.com/watch?v=-GcrsEqzmsA) — Tübingen Machine Learning, tiếng Anh. Trình bày phân tích giá trị riêng và phân tích giá trị suy biến (SVD), nối tiếp phần chéo hóa trong bài.
- [Phân tích giá trị riêng | Khoa học dữ liệu cơ bản](https://www.youtube.com/watch?v=OnLGH_5Ovg4) — Mat-chan, tiếng Việt. Video tiếng Việt giải thích phân tích giá trị riêng gắn với ngữ cảnh khoa học dữ liệu.
