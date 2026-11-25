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

Giữa màn sương dày đặc trên sườn núi, bạn không thấy đỉnh núi cũng chẳng thấy thung lũng, nhưng đôi chân bạn cảm nhận được mặt đất đang dốc về hướng nào. Nếu muốn xuống núi nhanh nhất có thể trong bước tiếp theo, bạn nên bước theo hướng nào? Câu hỏi hình học đơn giản ấy dẫn thẳng tới **vector gradient** và **đạo hàm theo hướng**—hai khái niệm định nghĩa nên chiếc la bàn vạn năng của toàn bộ ngành Học sâu hiện đại.

## Kiến thức cần nhớ: Bốn góc nhìn cốt lõi về Gradient và Đạo hàm theo hướng

1. **Vector Gradient ($$\nabla f$$)**:
   Gom mọi đạo hàm riêng thành một vector duy nhất sống trong không gian tham số, đóng vai trò mũi tên chỉ chính xác hướng tăng dốc nhất của hàm số.

2. **Đạo hàm theo hướng (Directional Derivative)**:
   Đo tốc độ biến thiên của $$f$$ khi đi theo một vector đơn vị $$\mathbf{u}$$ bất kỳ bằng tích vô hướng $$D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \|\nabla f(\mathbf{x})\|\cos\theta$$.

3. **Tính vuông góc với đường mức (Orthogonality to Contours)**:
   Tại mọi điểm, vector gradient luôn vuông góc với đường mức đi qua điểm đó—lý giải vì sao trên thung lũng hình elip dẹt, Gradient Descent lại chạy zig-zag qua lại hai vách núi thay vì đi thẳng tới đáy.

4. **Hướng giảm nhanh nhất (Steepest Descent)**:
   Vì $$\cos\theta$$ đạt cực tiểu bằng $$-1$$ khi $$\theta = 180^\circ$$, bước đi theo hướng $$\mathbf{u} = -\frac{\nabla f}{\|\nabla f\|}$$ đảm bảo hàm mất mát giảm nhanh nhất trong lân cận vô cùng bé.

### Giải thích: So sánh tốc độ giảm khi bước theo nhiều hướng khác nhau

Xét hàm mất mát hai biến $$f(x, y) = x^2 + 4y^2$$ (một thung lũng elip) tại điểm hiện tại $$\mathbf{p} = (2, 1)$$, nơi $$f(2, 1) = 4 + 4 = 8$$:

1. **Tính vector gradient tại $$\mathbf{p} = (2, 1)$$**:
   $$\nabla f(x, y) = (2x, 8y) \implies \nabla f(2, 1) = (4, 8).$$
   Độ lớn gradient là $$\|\nabla f(2, 1)\| = \sqrt{4^2 + 8^2} = \sqrt{80} \approx 8.944$$.
2. **Thử đi theo 3 hướng đơn vị khác nhau**:
   - **Hướng sang trái** $$\mathbf{u}_1 = (-1, 0)$$: $$D_{\mathbf{u}_1} f = (4, 8) \cdot (-1, 0) = -4$$ (hàm giảm với tốc độ 4).
   - **Hướng đi thẳng về gốc tọa độ** $$\mathbf{u}_2 = \frac{(-2, -1)}{\sqrt{5}} \approx (-0.894, -0.447)$$: $$D_{\mathbf{u}_2} f = \frac{4(-2) + 8(-1)}{\sqrt{5}} = \frac{-16}{\sqrt{5}} \approx -7.155$$.
   - **Hướng ngược gradient** $$\mathbf{u}_3 = \frac{(-4, -8)}{\sqrt{80}} \approx (-0.447, -0.894)$$: $$D_{\mathbf{u}_3} f = -\sqrt{80} \approx -8.944$$ (giảm nhanh nhất!).
3. **Điểm tinh tế đáng suy ngẫm**: Hướng ngược gradient $$\mathbf{u}_3$$ giảm nhanh nhất *tại chỗ*, nhưng lại **không** chỉ thẳng về đáy $$(0, 0)$$ (là hướng $$\mathbf{u}_2$$) do trục $$y$$ dốc gấp 4 lần trục $$x$$. Đây chính là nguồn gốc của hiện tượng zig-zag và lý do ra đời của Momentum và Adam sau này!

Bài học này trình bày vector gradient (*gradient*) và đạo hàm theo hướng (*directional derivative*) — hai khái niệm trung tâm trong tối ưu hóa, giúp ta xác định tốc độ biến thiên của hàm số theo từng hướng khác nhau.

---

## Vector gradient

Gradient $$\nabla f$$ là vector gồm tất cả các đạo hàm riêng của hàm $$f$$ theo từng biến. Tại một điểm cho trước, gradient chỉ theo hướng tăng dốc nhất của hàm số.

### Định nghĩa và cách tính

Với hàm hai biến $$f(x, y)$$, gradient được xác định bởi:

$$ \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix} $$

Với hàm $$n$$ biến $$f(x_1, x_2, \ldots, x_n)$$:

$$ \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix} $$

### Ví dụ: Tính gradient

Xét hàm số $$f(x, y) = x^2 + 3xy + y^2$$:

$$\frac{\partial f}{\partial x} = 2x + 3y$$
$$\frac{\partial f}{\partial y} = 3x + 2y$$

Do đó: $$\nabla f = \begin{pmatrix} 2x + 3y \\ 3x + 2y \end{pmatrix}$$

Tại điểm $$(1, 2)$$: $$\nabla f(1, 2) = \begin{pmatrix} 2(1) + 3(2) \\ 3(1) + 2(2) \end{pmatrix} = \begin{pmatrix} 8 \\ 7 \end{pmatrix}$$

---

## Đạo hàm theo hướng

**Đạo hàm theo hướng** đo tốc độ biến thiên của $$f$$ khi di chuyển theo một hướng bất kỳ $$\mathbf{u}$$, trong đó $$\mathbf{u}$$ là vector đơn vị (có độ dài bằng 1).

### Định nghĩa

Với hàm $$f(\mathbf{x})$$ và vector đơn vị $$\mathbf{u} = \langle u_1, u_2, \ldots, u_n \rangle$$:

$$D_{\mathbf{u}}f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} u_i$$

### Diễn giải hình học

Đạo hàm theo hướng có thể biểu diễn dưới dạng:

$$D_{\mathbf{u}}f = \lvert \nabla f \rvert \cos \theta$$

trong đó $$\theta$$ là góc giữa $$\nabla f$$ và $$\mathbf{u}$$, còn $$\lvert \nabla f \rvert$$ là độ lớn của gradient.

### Ví dụ: Tính đạo hàm theo hướng

Tiếp tục xét $$f(x, y) = x^2 + 3xy + y^2$$ tại điểm $$(1, 2)$$ với $$\nabla f(1, 2) = \begin{pmatrix} 8 \\ 7 \end{pmatrix}$$:

**Hướng 1:** $$\mathbf{u}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ (chiều dương trục $$x$$)
$$D_{\mathbf{u}_1}f(1, 2) = 8 \cdot 1 + 7 \cdot 0 = 8$$

**Hướng 2:** $$\mathbf{u}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ (chiều dương trục $$y$$)
$$D_{\mathbf{u}_2}f(1, 2) = 8 \cdot 0 + 7 \cdot 1 = 7$$

**Hướng 3:** $$\mathbf{u}_3 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ (hướng đường chéo $$45^\circ$$)
$$D_{\mathbf{u}_3}f(1, 2) = 8 \cdot \frac{1}{\sqrt{2}} + 7 \cdot \frac{1}{\sqrt{2}} = \frac{15}{\sqrt{2}} \approx 10{,}61$$

---

## Tốc độ biến thiên cực đại và cực tiểu

### Các tính chất chính

Từ công thức $$D_{\mathbf{u}}f = \lvert \nabla f \rvert \cos \theta$$, ta suy ra:

1. **Tốc độ biến thiên cực đại:** đạt được khi $$\cos \theta = 1$$ (tức $$\theta = 0^\circ$$).
   - Hướng: $$\mathbf{u} = \frac{\nabla f}{\lvert \nabla f \rvert}$$ (cùng hướng với gradient).
   - Giá trị cực đại: $$D_{\max}f = \lvert \nabla f \rvert$$.

2. **Tốc độ biến thiên cực tiểu:** đạt được khi $$\cos \theta = -1$$ (tức $$\theta = 180^\circ$$).
   - Hướng: $$\mathbf{u} = -\frac{\nabla f}{\lvert \nabla f \rvert}$$ (ngược hướng gradient).
   - Giá trị cực tiểu: $$D_{\min}f = -\lvert \nabla f \rvert$$.

3. **Tốc độ biến thiên bằng không:** xảy ra khi $$\cos \theta = 0$$ (tức $$\theta = 90^\circ$$).
   - Hướng: mọi vector vuông góc với $$\nabla f$$.

### Tóm tắt các tính chất của gradient

- Gradient $$\nabla f$$ chỉ theo hướng **tăng dốc nhất**.
- Hướng $$-\nabla f$$ chỉ theo hướng **giảm dốc nhất**.
- Độ lớn $$\lvert \nabla f \rvert$$ là **tốc độ biến thiên cực đại**.
- Điểm thỏa mãn $$\nabla f = \mathbf{0}$$ được gọi là **điểm tới hạn** (*critical point*) — vị trí tiềm năng của cực trị.

---

## Quan hệ với đường mức

Tại mọi điểm trên đường mức $$f(x, y) = c$$, vector gradient $$\nabla f$$ luôn **trực giao (vuông góc)** với tiếp tuyến của đường mức tại điểm đó.

### Ý nghĩa của tính trực giao

Tính chất này đặc biệt quan trọng vì:

1. **Đường mức biểu diễn giá trị hàm không đổi:** di chuyển dọc theo đường mức không làm thay đổi giá trị hàm, nên đạo hàm theo hướng tiếp tuyến bằng không.
2. **Gradient chỉ hướng tăng dốc nhất:** hướng làm giá trị hàm tăng nhanh nhất phải vuông góc với hướng mà hàm không đổi.
3. **Ứng dụng trong tối ưu hóa:** để tìm cực trị, ta tìm các điểm có gradient bằng không (điểm tới hạn), hoặc các điểm mà gradient vuông góc với biên của miền ràng buộc.

### Ứng dụng trong tối ưu hóa

Hiểu rõ gradient và đạo hàm theo hướng giúp ta làm việc hiệu quả với:

1. **Gradient descent:** di chuyển theo hướng $$-\nabla f$$ để cực tiểu hóa $$f$$.
2. **Gradient ascent:** di chuyển theo hướng $$+\nabla f$$ để cực đại hóa $$f$$.
3. **Tối ưu có ràng buộc:** khai thác quan hệ hình học giữa gradient và đường mức.
4. **Phân tích hội tụ:** xác định điều kiện để thuật toán hội tụ về nghiệm tối ưu.
5. **Chọn độ dài bước:** quyết định khoảng di chuyển phù hợp dọc theo hướng gradient.

Vì cung cấp đồng thời cả hướng di chuyển lẫn tốc độ biến thiên của hàm số, gradient là nền tảng của hầu hết các thuật toán tối ưu hóa.

## Video tham khảo

- **Xem trước:** [Directional Derivatives | What's the slope in any direction?](https://www.youtube.com/watch?v=GJODOGq7cAY) — Dr. Trefor Bazett, tiếng Anh. Trực quan hóa đạo hàm theo hướng như độ dốc theo một hướng bất kỳ và vai trò của gradient là hướng tăng nhanh nhất.
- [MaDL - Vector Calculus](https://www.youtube.com/watch?v=rh885pI4lPA) — Tübingen Machine Learning, tiếng Anh. Trình bày gradient, đạo hàm theo hướng và ma trận Jacobian theo hệ thống ký hiệu dùng cho ánh xạ nhận giá trị vector trong khóa học.
- [Continuity vs Partial Derivatives vs Differentiability | My Favorite Multivariable Function](https://www.youtube.com/watch?v=6Wi1kT9kR1M) — Dr. Trefor Bazett, tiếng Anh. Minh họa vì sao sự tồn tại của đạo hàm riêng là điều kiện yếu hơn khả vi toàn phần — điểm phân biệt quan trọng cho phần đường mức.
