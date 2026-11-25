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

Nếu chỉ nhìn ma trận như một bảng Excel chứa đầy con số, phép nhân ma trận sẽ trông như một quy tắc nhân-rồi-cộng kỳ quặc và vô nghĩa. Nhưng ngay khi bạn đeo cặp kính hình học vào, mọi thứ bỗng bừng sáng: một ma trận $$m \times n$$ chính là một cỗ máy biến đổi toàn bộ không gian $$\mathbb{R}^n$$ sang $$\mathbb{R}^m$$—xoay, lật, kéo giãn hoặc chiếu bẹp nó xuống chiều thấp hơn. Mỗi tầng `Linear` hay tích chập trong mạng nơ-ron sâu đều là một phép biến đổi ma trận như vậy, và nhân nhiều ma trận liên tiếp chính là ghép nối các tầng mạng lại với nhau.

## Kiến thức cần nhớ: Bốn góc nhìn thiết yếu về Ma trận và Phép biến đổi Tuyến tính

1. **Phép nhân ma trận là phép hợp hàm (Composition of Transformations)**:
   Áp dụng ma trận $$\mathbf{A}$$ rồi đến $$\mathbf{B}$$ tương đương với một phép biến đổi duy nhất $$\mathbf{B}\mathbf{A}$$. Điều này giải thích ngay vì sao mạng nhiều tầng *không có hàm kích hoạt phi tuyến* chỉ tương đương một ma trận duy nhất!

2. **Tính không giao hoán và Chuyển vị (Non-commutativity & Transpose)**:
   Thứ tự biến đổi cực kỳ quan trọng ($$\mathbf{A}\mathbf{B} \ne \mathbf{B}\mathbf{A}$$); còn ma trận chuyển vị $$\mathbf{W}^\top$$ đóng vai trò truyền gradient ngược từ tầng sau về tầng trước trong lan truyền ngược.

3. **Hạng ma trận và Không gian ảnh/hạt nhân (Rank, Column Space, Null Space)**:
   Hạng (*rank*) đo số chiều thực sự sống sót sau phép biến đổi. Khi hạng bị sụt giảm (ma trận suy biến), thông tin bị nén mất không thể phục hồi—đây cũng chính là tư duy phía sau kỹ thuật nén mô hình **LoRA (Low-Rank Adaptation)**.

4. **Định thức và Ma trận nghịch đảo (Determinant & Inverse)**:
   Định thức $$\det(\mathbf{A})$$ đo hệ số phóng đại diện tích/thể tích của phép biến đổi; khi $$\det(\mathbf{A}) = 0$$, không gian bị ép phẳng và phép biến đổi không còn khả nghịch.

### Giải thích: Kiểm chứng bằng tay tính không giao hoán và sự ép chiều

Xét hai ma trận $$2 \times 2$$: ma trận xoay $$90^\circ$$ ngược chiều kim đồng hồ $$\mathbf{R} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$ và ma trận kéo giãn trục ngang gấp 2 lần $$\mathbf{S} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$$. Tác động lên điểm $$\mathbf{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$ theo hai thứ tự:

1. **Kéo giãn trước, xoay sau ($$\mathbf{R}\mathbf{S}$$):**
   $$\mathbf{R}\mathbf{S} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 2 & 0 \end{bmatrix} \implies (\mathbf{R}\mathbf{S})\mathbf{x} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}.$$
2. **Xoay trước, kéo giãn sau ($$\mathbf{S}\mathbf{R}$$):**
   $$\mathbf{S}\mathbf{R} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & -2 \\ 1 & 0 \end{bmatrix} \implies (\mathbf{S}\mathbf{R})\mathbf{x} = \begin{bmatrix} -2 \\ 1 \end{bmatrix}.$$
3. **Kiểm tra định thức (hệ số diện tích)**:
   $$\det(\mathbf{R}) = 0(0) - (-1)(1) = 1$$ (phép xoay bảo toàn diện tích), $$\det(\mathbf{S}) = 2(1) - 0 = 2$$ (tăng gấp đôi diện tích), và $$\det(\mathbf{R}\mathbf{S}) = 0 - (-2) = 2 = \det(\mathbf{R})\det(\mathbf{S})$$.

Thứ tự nhân ma trận hoàn toàn thay đổi điểm đến, nhưng tổng hệ số phóng đại diện tích vẫn luôn bằng tích các định thức thành phần.

Bài học này trình bày ma trận (*matrix*), các phép toán ma trận và phép biến đổi tuyến tính (*linear transformation*) — những công cụ cơ bản để biểu diễn và giải các bài toán tối ưu hóa.

---

## Ma trận và các phép toán ma trận

### Ma trận là gì?

Một **ma trận** là bảng chữ nhật gồm các số được sắp xếp theo hàng và cột. Ma trận dùng để biểu diễn dữ liệu, phép biến đổi, hệ phương trình và quan hệ giữa các biến.

**Dạng tổng quát:**
$$\mathbf{A} = \begin{pmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}$$

Đây là ma trận kích thước $$m \times n$$ ($$m$$ hàng, $$n$$ cột).

**Ví dụ:**
$$\mathbf{A} = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$ là ma trận kích thước $$2 \times 3$$.

### Phép cộng ma trận

Hai ma trận cùng kích thước được cộng bằng cách cộng từng cặp phần tử tương ứng:

$$\mathbf{A} + \mathbf{B} = \begin{pmatrix} a_{11} + b_{11} & a_{12} + b_{12} \\ a_{21} + b_{21} & a_{22} + b_{22} \end{pmatrix}$$

**Ví dụ:**
$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} + \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$$

### Phép nhân với vô hướng

Nhân mọi phần tử của ma trận với cùng một số thực $$c$$:

$$c\mathbf{A} = \begin{pmatrix} ca_{11} & ca_{12} \\ ca_{21} & ca_{22} \end{pmatrix}$$

### Phép nhân ma trận

Với hai ma trận $$\mathbf{A}_{m \times n}$$ và $$\mathbf{B}_{n \times p}$$, tích $$\mathbf{C}_{m \times p} = \mathbf{AB}$$ có phần tử ở hàng $$i$$, cột $$j$$ bằng tích vô hướng của hàng $$i$$ thuộc $$\mathbf{A}$$ với cột $$j$$ thuộc $$\mathbf{B}$$:

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**Ví dụ:**
$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$$

**Lưu ý quan trọng:** Phép nhân ma trận **không có tính giao hoán**; nhìn chung $$\mathbf{AB} \neq \mathbf{BA}$$.

---

## Phép biến đổi tuyến tính

Một **phép biến đổi tuyến tính** là ánh xạ $$T: \mathbb{R}^n \to \mathbb{R}^m$$ bảo toàn phép cộng vector và phép nhân vô hướng. Mọi phép biến đổi tuyến tính giữa các không gian hữu hạn chiều đều có thể biểu diễn bằng một ma trận.

### Định nghĩa

Ánh xạ $$T(\mathbf{v}) = \mathbf{Av}$$ là tuyến tính khi và chỉ khi thỏa mãn hai tính chất:

1. **Tính cộng tính (*additivity*):** $$T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$$
2. **Tính thuần nhất (*homogeneity*):** $$T(c\mathbf{v}) = cT(\mathbf{v})$$

Hai điều kiện này có thể gộp thành: $$T(c_1\mathbf{u} + c_2\mathbf{v}) = c_1T(\mathbf{u}) + c_2T(\mathbf{v})$$.

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

Xét các phép biến đổi hình học trong mặt phẳng giúp ta dễ hình dung cách ma trận tác động lên vector.

### Phép co giãn (*scaling*)

**Ma trận co giãn:**
$$\mathbf{S} = \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}$$

- Co giãn tọa độ $$x$$ theo hệ số $$s_x$$ và tọa độ $$y$$ theo hệ số $$s_y$$.
- **Ví dụ:** $$\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$$ nhân đôi hoành độ $$x$$ và nhân ba tung độ $$y$$.

### Phép quay (*rotation*)

**Ma trận quay ngược chiều kim đồng hồ một góc $$\theta$$:**
$$\mathbf{R} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

- **Ví dụ:** phép quay $$90^\circ$$ có ma trận $$\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$, ánh xạ $$(x, y) \mapsto (-y, x)$$.

### Phép phản xạ (*reflection*)

**Phản xạ qua trục $$x$$:**
$$\mathbf{F}_x = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Phản xạ qua trục $$y$$:**
$$\mathbf{F}_y = \begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}$$

**Phản xạ qua đường thẳng $$y = x$$:**
$$\mathbf{F}_{y=x} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

### Phép cắt (*shearing*)

**Phép cắt theo phương ngang:**
$$\mathbf{H} = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$$

Ánh xạ $$(x, y) \mapsto (x + ky, y)$$.

---

## Các loại ma trận đặc biệt

### Ma trận đơn vị (*identity matrix*)

**Ma trận đơn vị** $$\mathbf{I}$$ đóng vai trò phần tử đơn vị (tương tự số 1) trong phép nhân ma trận:

$$\mathbf{I}_n = \begin{pmatrix} 
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{pmatrix}$$

**Tính chất:** $$\mathbf{AI} = \mathbf{IA} = \mathbf{A}$$ với mọi ma trận $$\mathbf{A}$$ có kích thước phù hợp.

### Ma trận chuyển vị (*transpose*)

**Ma trận chuyển vị** $$\mathbf{A}^T$$ thu được bằng cách đổi hàng thành cột (đối xứng qua đường chéo chính):

$$\text{Nếu } \mathbf{A} = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, \text{ thì } \mathbf{A}^T = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$

**Các tính chất:**
- $$(\mathbf{A}^T)^T = \mathbf{A}$$
- $$(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$$
- $$(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$$

### Ma trận đối xứng (*symmetric matrix*)

Một ma trận vuông được gọi là **đối xứng** nếu $$\mathbf{A} = \mathbf{A}^T$$:

$$\mathbf{A} = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{pmatrix}$$

Ma trận đối xứng sở hữu nhiều tính chất đặc biệt quan trọng trong tối ưu hóa.

### Ma trận nghịch đảo (*inverse*)

**Ma trận nghịch đảo** $$\mathbf{A}^{-1}$$ của ma trận vuông $$\mathbf{A}$$ thỏa mãn:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

**Trường hợp ma trận $$2 \times 2$$:**
$$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

trong đó $$\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$ và $$\det(\mathbf{A}) = ad - bc$$.

**Lưu ý:** Không phải mọi ma trận vuông đều có nghịch đảo. Ma trận **khả nghịch** (*invertible*, hay không suy biến — *non-singular*) khi và chỉ khi định thức của nó khác không.

---

## Ứng dụng trong tối ưu hóa

Ma trận và phép biến đổi tuyến tính xuất hiện xuyên suốt trong tối ưu hóa:

### 1. Hệ phương trình tuyến tính

Nhiều bài toán tối ưu quy về việc giải hệ $$\mathbf{Ax} = \mathbf{b}$$:
- **Nghiệm duy nhất:** $$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$ (khi $$\mathbf{A}$$ khả nghịch).
- **Bình phương tối thiểu:** cực tiểu hóa $$\|\mathbf{Ax} - \mathbf{b}\|^2$$ khi hệ không có nghiệm chính xác.

### 2. Dạng bậc hai (*quadratic form*)

Hàm mục tiêu bậc hai xuất hiện rất phổ biến trong tối ưu hóa:
$$f(\mathbf{x}) = \mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{c}^T\mathbf{x} + d$$

Ma trận $$\mathbf{Q}$$ quyết định độ cong và tính lồi của hàm số.

### 3. Quy hoạch tuyến tính (*linear programming*)

Dạng chuẩn: cực tiểu hóa $$\mathbf{c}^T\mathbf{x}$$ với các ràng buộc $$\mathbf{Ax} = \mathbf{b}$$, $$\mathbf{x} \geq \mathbf{0}$$.

### 4. Biểu diễn ràng buộc

- **Ràng buộc đẳng thức:** $$\mathbf{Ax} = \mathbf{b}$$
- **Ràng buộc bất đẳng thức:** $$\mathbf{Ax} \leq \mathbf{b}$$

### 5. Phép đổi biến

Phép đổi biến tuyến tính $$\mathbf{y} = \mathbf{T}\mathbf{x}$$ thường giúp đơn giản hóa cấu trúc của bài toán tối ưu.

### Ví dụ: Tối ưu danh mục đầu tư

Trong tài chính, bài toán cực tiểu hóa rủi ro danh mục đầu tư có dạng:
$$\text{minimize } \mathbf{w}^T\mathbf{\Sigma}\mathbf{w}$$

trong đó $$\mathbf{w}$$ là vector trọng số phân bổ vốn và $$\mathbf{\Sigma}$$ là ma trận hiệp phương sai lợi suất của các tài sản.

Nắm vững ma trận và phép biến đổi tuyến tính giúp ta mô hình hóa, phân tích và giải hiệu quả nhiều lớp bài toán tối ưu hóa khác nhau.

## Video tham khảo

- **Xem trước:** [Linear transformations and matrices | Chapter 3, Essence of linear algebra](https://www.youtube.com/watch?v=kYB8IZa5AuE) — 3Blue1Brown, tiếng Anh. Giải thích bản chất ma trận như một phép biến đổi tuyến tính chứ không chỉ là bảng số.
- [Matrix multiplication as composition | Chapter 4, Essence of linear algebra](https://www.youtube.com/watch?v=XkY2DOUCWMU) — 3Blue1Brown, tiếng Anh. Minh họa phép nhân ma trận chính là phép hợp hai phép biến đổi tuyến tính liên tiếp.
- [Inverse matrices, column space and null space | Chapter 7, Essence of linear algebra](https://www.youtube.com/watch?v=uQhTuRlWMxw) — 3Blue1Brown, tiếng Anh. Trình bày trực quan về ma trận nghịch đảo, không gian cột và không gian hạch, gắn với điều kiện giải hệ phương trình ở cuối bài.
