---
layout: post
title: 00-01-02 Đạo hàm và Giải tích đa biến
chapter: '00'
order: 4
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

Bài học này bao gồm đạo hàm và các khái niệm giải tích đa biến thiết yếu tạo nền tảng cho lý thuyết và thuật toán tối ưu hóa.

---

## Đạo hàm và Tốc độ Thay đổi

Đạo hàm của một hàm một biến thể hiện tốc độ thay đổi tức thời của nó, điều này rất cơ bản để hiểu cách các hàm số hoạt động cục bộ.

### Các Khái niệm Đạo hàm Cơ bản

**Độ dốc giữa hai điểm:**

$$\text{Độ dốc} = \frac{y_2 - y_1}{x_2 - x_1}$$

**Đạo hàm (tốc độ thay đổi tức thời):**

$$f'(x_0) = \lim_{x_1 \to x_0} \frac{f(x_1) - f(x_0)}{x_1 - x_0} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

Đạo hàm cho chúng ta biết hàm số thay đổi nhanh như thế nào tại bất kỳ điểm nào, điều này rất quan trọng để tìm các điểm tối ưu nơi tốc độ thay đổi bằng không.

### Đường mức của Hàm số

Đường mức là một khái niệm cơ bản trong giải tích đa biến được sử dụng để trực quan hóa các hàm hai biến, thường được ký hiệu là $$f(x, y)$$. Chúng cung cấp cách biểu diễn một bề mặt 3D trong mặt phẳng 2D.

Một **đường mức** của hàm số $$f(x, y)$$ là tập hợp tất cả các điểm $$(x, y)$$ trong miền xác định của $$f$$ nơi hàm số nhận giá trị hằng số:

$$f(x, y) = c$$

**Ví dụ:**
- Với $$f(x, y) = x^2 + y^2$$, các đường mức là các hình tròn: $$x^2 + y^2 = c$$
- Với $$f(x, y) = x + y$$, các đường mức là các đường thẳng song song: $$x + y = c$$

Đường mức giúp chúng ta hiểu:
1. Địa hình của hàm số
2. Hướng tăng và giảm dốc nhất
3. Vị trí của các điểm tối ưu tiềm năng

---

## Các Khái niệm Chính của Giải tích Đa biến

### Đạo hàm Riêng

Với một hàm số $$f(x_1, x_2, \ldots, x_n)$$, **đạo hàm riêng** theo $$x_i$$ là:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

Điều này đo lường cách $$f$$ thay đổi khi chỉ có $$x_i$$ biến thiên trong khi tất cả các biến khác giữ cố định.

### Vector Gradient

**Gradient** là một vector gồm tất cả các đạo hàm riêng:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

Gradient chỉ theo hướng tăng dốc nhất của hàm số và vuông góc với các đường mức.

### Ma trận Hessian

**Ma trận Hessian** chứa tất cả các đạo hàm riêng bậc hai:

$$\nabla^2 f(\mathbf{x}) = \mathbf{H} = \begin{pmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

Hessian cung cấp thông tin về độ cong của hàm số và rất quan trọng cho:
- Xác định bản chất của các điểm tới hạn (cực tiểu, cực đại, hoặc điểm yên ngựa)
- Các phương pháp tối ưu hóa bậc hai như phương pháp Newton

---

## Quy tắc Dây chuyền cho Hàm Đa biến

Quy tắc dây chuyền là cơ bản để tính đạo hàm của các hàm hợp thành, thường xuất hiện trong các bài toán tối ưu hóa.

### Quy tắc Dây chuyền Cơ bản

Với hàm số $$z = f(x, y)$$ nơi $$x = g(t)$$ và $$y = h(t)$$:

$$ \frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} $$

### Quy tắc Dây chuyền Tổng quát

Với $$z = f(x_1, x_2, \ldots, x_n)$$ nơi mỗi $$x_i = x_i(t_1, t_2, \ldots, t_m)$$:

$$ \frac{\partial z}{\partial t_j} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \frac{\partial x_i}{\partial t_j} $$

### Ứng dụng trong Tối ưu hóa

Quy tắc dây chuyền rất thiết yếu cho:

1. **Tính toán Gradient**: Tính gradient của các hàm mục tiêu hợp thành
2. **Xử lý Ràng buộc**: Xử lý các ràng buộc là hàm của các biến khác
3. **Triển khai Thuật toán**: Lan truyền ngược trong mạng nơ-ron và vi phân tự động
4. **Phân tích Độ nhạy**: Hiểu cách thay đổi tham số ảnh hưởng đến các nghiệm tối ưu

### Ví dụ: Tối ưu hóa với Ràng buộc

Xem xét việc tối thiểu hóa $$f(x, y) = x^2 + y^2$$ với điều kiện $$g(x, y) = x + y - 1 = 0$$.

Sử dụng ràng buộc để loại bỏ một biến: $$y = 1 - x$$, vậy chúng ta tối thiểu hóa:
$$h(x) = f(x, 1-x) = x^2 + (1-x)^2$$

Sử dụng quy tắc dây chuyền:
$$h'(x) = \frac{\partial f}{\partial x} \cdot 1 + \frac{\partial f}{\partial y} \cdot \frac{d(1-x)}{dx} = 2x + 2(1-x)(-1) = 4x - 2$$

Đặt $$h'(x) = 0$$ cho $$x = 1/2$$, vậy điểm tối ưu là $$(1/2, 1/2)$$.

Điều này minh họa cách các khái niệm giải tích đa biến làm việc cùng nhau để giải quyết các bài toán tối ưu hóa một cách hệ thống.
