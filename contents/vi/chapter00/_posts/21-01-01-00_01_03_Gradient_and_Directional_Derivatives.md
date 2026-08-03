---
layout: post
title: 00-01-03 Gradient và Đạo hàm theo hướng
chapter: '00'
order: 5
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Bài học này trình bày vector gradient (*gradient*) và đạo hàm theo hướng (*directional derivative*) — hai khái niệm trung tâm trong tối ưu hóa, giúp ta hiểu hàm số biến thiên theo các hướng khác nhau như thế nào.

---

## Vector gradient

Gradient $$\nabla f$$ là một vector gồm các đạo hàm riêng của hàm $$f$$ theo từng biến. Tại một điểm cho trước, gradient chỉ theo hướng tăng dốc nhất của hàm số.

### Định nghĩa và cách tính

Với hàm hai biến $$f(x, y)$$, gradient được cho bởi:

$$ \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix} $$

Với hàm $$n$$ biến $$f(x_1, x_2, \ldots, x_n)$$:

$$ \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix} $$

### Ví dụ: Tính gradient

Xét $$f(x, y) = x^2 + 3xy + y^2$$:

$$\frac{\partial f}{\partial x} = 2x + 3y$$
$$\frac{\partial f}{\partial y} = 3x + 2y$$

Do đó: $$\nabla f = \begin{pmatrix} 2x + 3y \\ 3x + 2y \end{pmatrix}$$

Tại điểm $$(1, 2)$$: $$\nabla f(1, 2) = \begin{pmatrix} 2(1) + 3(2) \\ 3(1) + 2(2) \end{pmatrix} = \begin{pmatrix} 8 \\ 7 \end{pmatrix}$$

---

## Đạo hàm theo hướng

**Đạo hàm theo hướng** đo tốc độ biến thiên của $$f$$ khi ta di chuyển theo một hướng bất kỳ $$\mathbf{u}$$. Ở đây $$\mathbf{u}$$ phải là vector đơn vị (độ dài bằng 1).

### Định nghĩa

Với hàm $$f(\mathbf{x})$$ và vector đơn vị $$\mathbf{u} = \langle u_1, u_2, \ldots, u_n \rangle$$:

$$D_{\mathbf{u}}f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} u_i$$

### Diễn giải hình học

Đạo hàm theo hướng có thể viết dưới dạng:

$$D_{\mathbf{u}}f = \lvert \nabla f \rvert \cos \theta$$

trong đó $$\theta$$ là góc giữa $$\nabla f$$ và $$\mathbf{u}$$, còn $$\lvert \nabla f \rvert$$ là độ lớn của gradient.

### Ví dụ: Tính đạo hàm theo hướng

Dùng ví dụ trước $$f(x, y) = x^2 + 3xy + y^2$$ tại điểm $$(1, 2)$$ với $$\nabla f(1, 2) = \begin{pmatrix} 8 \\ 7 \end{pmatrix}$$:

**Hướng 1:** $$\mathbf{u}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ (hướng trục $$x$$ dương)
$$D_{\mathbf{u}_1}f(1, 2) = 8 \cdot 1 + 7 \cdot 0 = 8$$

**Hướng 2:** $$\mathbf{u}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ (hướng trục $$y$$ dương)
$$D_{\mathbf{u}_2}f(1, 2) = 8 \cdot 0 + 7 \cdot 1 = 7$$

**Hướng 3:** $$\mathbf{u}_3 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ (đường chéo $$45^\circ$$)
$$D_{\mathbf{u}_3}f(1, 2) = 8 \cdot \frac{1}{\sqrt{2}} + 7 \cdot \frac{1}{\sqrt{2}} = \frac{15}{\sqrt{2}} \approx 10{,}61$$

---

## Tốc độ biến thiên cực đại và cực tiểu

### Các tính chất chính

Từ công thức $$D_{\mathbf{u}}f = \lvert \nabla f \rvert \cos \theta$$, ta suy ra:

1. **Tốc độ biến thiên cực đại**: xảy ra khi $$\cos \theta = 1$$ (tức $$\theta = 0^\circ$$)
   - Hướng: $$\mathbf{u} = \frac{\nabla f}{\lvert \nabla f \rvert}$$ (cùng hướng với gradient)
   - Tốc độ cực đại: $$D_{\max}f = \lvert \nabla f \rvert$$

2. **Tốc độ biến thiên cực tiểu**: xảy ra khi $$\cos \theta = -1$$ (tức $$\theta = 180^\circ$$)
   - Hướng: $$\mathbf{u} = -\frac{\nabla f}{\lvert \nabla f \rvert}$$ (ngược hướng gradient)
   - Tốc độ cực tiểu: $$D_{\min}f = -\lvert \nabla f \rvert$$

3. **Tốc độ biến thiên bằng không**: xảy ra khi $$\cos \theta = 0$$ (tức $$\theta = 90^\circ$$)
   - Hướng: mọi vector vuông góc với $$\nabla f$$

### Tóm tắt các tính chất của gradient

- Gradient $$\nabla f$$ chỉ theo hướng **tăng dốc nhất**
- Hướng $$-\nabla f$$ chỉ theo hướng **giảm dốc nhất**
- Độ lớn $$\lvert \nabla f \rvert$$ cho **tốc độ biến thiên cực đại**
- Khi $$\nabla f = \mathbf{0}$$, điểm đó là **điểm tới hạn** (*critical point*) — ứng viên cho cực trị

---

## Quan hệ với đường mức

Tại mọi điểm trên đường mức $$f(x, y) = c$$, vector gradient $$\nabla f$$ **trực giao (vuông góc)** với tiếp tuyến của đường mức tại điểm đó.

### Vì sao tính chất này quan trọng

Tính trực giao này mang tính cơ bản vì:

1. **Đường mức biểu diễn giá trị hàm không đổi**: di chuyển dọc theo đường mức không làm thay đổi giá trị hàm, nên đạo hàm theo hướng bằng không.

2. **Gradient chỉ hướng tăng dốc nhất**: hướng làm tăng giá trị hàm nhanh nhất phải vuông góc với hướng không làm thay đổi giá trị hàm.

3. **Ý nghĩa trong tối ưu hóa**: để tìm cực trị, ta tìm các điểm có gradient bằng không (điểm tới hạn), hoặc các điểm mà gradient vuông góc với biên ràng buộc.

### Ứng dụng trong tối ưu hóa

Hiểu gradient và đạo hàm theo hướng là then chốt cho:

1. **Gradient descent**: di chuyển theo hướng $$-\nabla f$$ để cực tiểu hóa $$f$$
2. **Gradient ascent**: di chuyển theo hướng $$+\nabla f$$ để cực đại hóa $$f$$
3. **Tối ưu có ràng buộc**: tận dụng quan hệ giữa gradient và đường mức
4. **Phân tích hội tụ**: hiểu khi nào thuật toán hội tụ về nghiệm tối ưu
5. **Chọn độ dài bước**: xác định khoảng di chuyển dọc theo hướng gradient

Gradient cung cấp đồng thời hướng cần di chuyển và thông tin về tốc độ biến thiên của hàm, do đó là nền tảng của hầu hết các thuật toán tối ưu hóa.
