---
layout: post
title: 00-02-02 Ma trận và Phép biến đổi tuyến tính
chapter: '00'
order: 9
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Bài học này trình bày ma trận (*matrix*), các phép toán ma trận và phép biến đổi tuyến tính (*linear transformation*) — những công cụ cơ bản để biểu diễn và giải quyết các bài toán tối ưu hóa.

---

## Ma trận và các phép toán ma trận

### Ma trận là gì?

Một **ma trận** là một bảng chữ nhật các số được sắp xếp theo hàng và cột. Ma trận biểu diễn dữ liệu, phép biến đổi, hệ phương trình và các quan hệ giữa các biến.

**Dạng tổng quát:**
$$\mathbf{A} = \begin{pmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}$$

Đây là ma trận kích thước $$m \times n$$ ($$m$$ hàng, $$n$$ cột).

**Ví dụ:**
$$\mathbf{A} = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$ là ma trận $$2 \times 3$$.

### Phép cộng ma trận

Hai ma trận được cộng bằng cách cộng các phần tử tương ứng. Cả hai ma trận phải có cùng kích thước.

$$\mathbf{A} + \mathbf{B} = \begin{pmatrix} a_{11} + b_{11} & a_{12} + b_{12} \\ a_{21} + b_{21} & a_{22} + b_{22} \end{pmatrix}$$

**Ví dụ:**
$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} + \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$$

### Phép nhân vô hướng

Nhân mọi phần tử của ma trận với một vô hướng:

$$c\mathbf{A} = \begin{pmatrix} ca_{11} & ca_{12} \\ ca_{21} & ca_{22} \end{pmatrix}$$

### Phép nhân ma trận

Với các ma trận $$\mathbf{A}_{m \times n}$$ và $$\mathbf{B}_{n \times p}$$, tích $$\mathbf{C}_{m \times p}$$ được lập bằng cách lấy tích vô hướng giữa các hàng của $$\mathbf{A}$$ và các cột của $$\mathbf{B}$$:

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**Ví dụ:**
$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$$

**Lưu ý quan trọng:** Phép nhân ma trận **không giao hoán**: nói chung $$\mathbf{AB} \neq \mathbf{BA}$$.

---

## Phép biến đổi tuyến tính

Một **phép biến đổi tuyến tính** là hàm $$T: \mathbb{R}^n \to \mathbb{R}^m$$ bảo toàn phép cộng vector và phép nhân vô hướng. Mọi phép biến đổi tuyến tính đều có thể biểu diễn bằng một ma trận.

### Định nghĩa

Phép biến đổi $$T(\mathbf{v}) = \mathbf{Av}$$ là tuyến tính khi và chỉ khi:

1. **Tính cộng tính (*additivity*):** $$T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$$
2. **Tính thuần nhất (*homogeneity*):** $$T(c\mathbf{v}) = cT(\mathbf{v})$$

Hai điều kiện trên có thể gộp lại thành: $$T(c_1\mathbf{u} + c_2\mathbf{v}) = c_1T(\mathbf{u}) + c_2T(\mathbf{v})$$

### Phép nhân ma trận–vector

Nếu $$\mathbf{A}$$ là ma trận $$m \times n$$ và $$\mathbf{v}$$ là vector cột $$n \times 1$$, thì tích $$\mathbf{Av}$$ là vector cột $$m \times 1$$:

$$ \mathbf{w} = \mathbf{Av} = \begin{pmatrix} 
a_{11}v_1 + a_{12}v_2 + \cdots + a_{1n}v_n \\
a_{21}v_1 + a_{22}v_2 + \cdots + a_{2n}v_n \\
\vdots \\
a_{m1}v_1 + a_{m2}v_2 + \cdots + a_{mn}v_n
\end{pmatrix} $$

**Ví dụ:**
$$\begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} 4 \\ 5 \end{pmatrix} = \begin{pmatrix} 2 \cdot 4 + 1 \cdot 5 \\ 0 \cdot 4 + 3 \cdot 5 \end{pmatrix} = \begin{pmatrix} 13 \\ 15 \end{pmatrix}$$

---

## Các phép biến đổi hai chiều thông dụng

Hiểu các phép biến đổi hình học giúp trực quan hóa cách ma trận tác động lên vector.

### Phép co giãn (*scaling*)

**Ma trận co giãn:**
$$\mathbf{S} = \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}$$

- Co giãn tọa độ $$x$$ theo hệ số $$s_x$$ và tọa độ $$y$$ theo hệ số $$s_y$$
- **Ví dụ:** $$\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$$ nhân đôi các giá trị $$x$$ và nhân ba các giá trị $$y$$

### Phép quay (*rotation*)

**Ma trận quay (ngược chiều kim đồng hồ một góc $$\theta$$):**
$$\mathbf{R} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

- **Ví dụ:** quay $$90^\circ$$: $$\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$
- Ánh xạ $$(x, y) \mapsto (-y, x)$$

### Phép phản xạ (*reflection*)

**Phản xạ qua trục $$x$$:**
$$\mathbf{F}_x = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Phản xạ qua trục $$y$$:**
$$\mathbf{F}_y = \begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}$$

**Phản xạ qua đường thẳng $$y = x$$:**
$$\mathbf{F}_{y=x} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

### Phép cắt (*shearing*)

**Cắt theo phương ngang:**
$$\mathbf{H} = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$$

Ánh xạ $$(x, y) \mapsto (x + ky, y)$$

---

## Các loại ma trận đặc biệt

### Ma trận đơn vị (*identity matrix*)

**Ma trận đơn vị** $$\mathbf{I}$$ đóng vai trò giống như số 1 đối với phép nhân ma trận:

$$\mathbf{I}_n = \begin{pmatrix} 
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{pmatrix}$$

**Tính chất:** $$\mathbf{AI} = \mathbf{IA} = \mathbf{A}$$ với mọi ma trận $$\mathbf{A}$$ tương thích.

### Ma trận chuyển vị (*transpose*)

**Ma trận chuyển vị** $$\mathbf{A}^T$$ được lập bằng cách lật ma trận qua đường chéo chính:

$$\text{Nếu } \mathbf{A} = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, \text{ thì } \mathbf{A}^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$

**Các tính chất:**
- $$(\mathbf{A}^T)^T = \mathbf{A}$$
- $$(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$$
- $$(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$$

### Ma trận đối xứng (*symmetric matrix*)

Một ma trận được gọi là **đối xứng** nếu $$\mathbf{A} = \mathbf{A}^T$$:

$$\mathbf{A} = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{pmatrix}$$

Ma trận đối xứng có nhiều tính chất đặc biệt quan trọng trong tối ưu hóa.

### Ma trận nghịch đảo (*inverse*)

**Nghịch đảo** $$\mathbf{A}^{-1}$$ của ma trận vuông $$\mathbf{A}$$ thỏa mãn:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

**Với ma trận $$2 \times 2$$:**
$$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

trong đó $$\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$ và $$\det(\mathbf{A}) = ad - bc$$.

**Lưu ý:** Không phải mọi ma trận đều có nghịch đảo. Ma trận **khả nghịch** (*invertible*, hay *non-singular*) khi và chỉ khi định thức khác không.

---

## Ứng dụng trong tối ưu hóa

Ma trận và phép biến đổi tuyến tính mang tính nền tảng trong tối ưu hóa vì nhiều lý do sau.

### 1. Hệ phương trình tuyến tính

Nhiều bài toán tối ưu hóa liên quan đến việc giải $$\mathbf{Ax} = \mathbf{b}$$:
- **Nghiệm duy nhất:** $$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$ (khi $$\mathbf{A}$$ khả nghịch)
- **Bình phương tối thiểu:** cực tiểu hóa $$\|\mathbf{Ax} - \mathbf{b}\|^2$$ khi không tồn tại nghiệm chính xác

### 2. Dạng bậc hai (*quadratic form*)

Các hàm bậc hai xuất hiện thường xuyên trong tối ưu hóa:
$$f(\mathbf{x}) = \mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{c}^T\mathbf{x} + d$$

Ma trận $$\mathbf{Q}$$ quyết định các tính chất về độ cong của hàm.

### 3. Quy hoạch tuyến tính (*linear programming*)

Dạng chuẩn: cực tiểu hóa $$\mathbf{c}^T\mathbf{x}$$ với các ràng buộc $$\mathbf{Ax} = \mathbf{b}$$, $$\mathbf{x} \geq \mathbf{0}$$

### 4. Biểu diễn ràng buộc

- **Ràng buộc đẳng thức:** $$\mathbf{Ax} = \mathbf{b}$$
- **Ràng buộc bất đẳng thức:** $$\mathbf{Ax} \leq \mathbf{b}$$

### 5. Phép đổi biến

Đổi biến $$\mathbf{y} = \mathbf{T}\mathbf{x}$$ có thể đơn giản hóa các bài toán tối ưu hóa.

### Ví dụ: Tối ưu danh mục đầu tư

Trong tài chính, ta có thể cực tiểu hóa rủi ro danh mục:
$$\text{minimize } \mathbf{w}^T\mathbf{\Sigma}\mathbf{w}$$

trong đó $$\mathbf{w}$$ là vector trọng số danh mục và $$\mathbf{\Sigma}$$ là ma trận hiệp phương sai của lợi suất tài sản.

Nắm vững ma trận và phép biến đổi tuyến tính cung cấp công cụ để phát biểu, phân tích và giải hiệu quả một lớp rộng các bài toán tối ưu hóa.
