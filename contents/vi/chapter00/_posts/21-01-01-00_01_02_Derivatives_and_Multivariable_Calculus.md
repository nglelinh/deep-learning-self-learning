---
layout: post
title: 00-01-02 Đạo hàm và Giải tích đa biến
chapter: '00'
order: 4
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Khi huấn luyện một mạng nơ-ron nhỏ nhất cũng có hàng nghìn trọng số cùng lúc tác động lên hàm mất mát, ta không còn đứng trên một đường cong 2D nữa mà đang đứng giữa một địa hình núi non nghìn chiều. Làm thế nào để biết việc tăng trọng số $$w_{17}$$ thêm $$0.001$$ sẽ làm giảm sai số hay đẩy nó tăng vọt, trong khi mọi trọng số khác vẫn đang đồng thời thay đổi? Câu trả lời nằm ở **đạo hàm riêng**, **quy tắc dây chuyền đa biến** và **ma trận Jacobian**—những viên gạch trực tiếp kiến tạo nên thuật toán lan truyền ngược.

## Bản đồ kiến thức nền: Bốn khái niệm trụ cột của Giải tích Đa biến

1. **Đạo hàm riêng (Partial Derivatives)**:
   Đóng băng tất cả các biến khác và chỉ đo độ nhạy của hàm theo duy nhất một tọa độ $$x_i$$, tương ứng với độ dốc khi cắt mặt cong bằng một mặt phẳng song song trục tọa độ.

2. **Đường mức (Level Curves / Contours)**:
   Tập hợp các điểm có cùng giá trị hàm $$f(x, y) = c$$. Nhìn vào bản đồ đường mức (giống bản đồ địa hình), ta nhận diện ngay đáy thung lũng, yên ngựa và vùng khe hẹp khó hội tụ.

3. **Ma trận Jacobian và Hessian**:
   Jacobian tổng hợp toàn bộ đạo hàm riêng bậc nhất cho hàm nhận vector và trả về vector (như một lớp mạng nơ-ron), còn Hessian lưu đạo hàm bậc hai mô tả độ cong lồi/lõm của bề mặt mất mát.

4. **Quy tắc dây chuyền đa biến (Multivariable Chain Rule)**:
   Cộng dồn mọi đường lan truyền ảnh hưởng từ một biến đầu vào qua nhiều biến trung gian tới đầu ra cuối cùng—chính xác là cơ chế tính gradient qua các tầng nối tiếp và kết nối tắt (*skip connections*) của ResNet.

### Khung giải thích (nên đọc chậm): Tính tay đạo hàm riêng và quy tắc dây chuyền cho mạng 2 nhánh

Xét một hàm hợp mô phỏng mạng hai nhánh hội tụ: $$z = u^2 + 3uv$$, trong đó hai nhánh trung gian phụ thuộc vào cùng một tham số $$t$$ theo $$u(t) = 2t$$ và $$v(t) = t + 1$$. Hãy tính $$\frac{dz}{dt}$$ tại $$t = 1$$ theo đúng quy tắc dây chuyền đa biến:

1. **Tính giá trị lan truyền xuôi tại $$t = 1$$**:
   $$u(1) = 2(1) = 2, \quad v(1) = 1 + 1 = 2 \implies z(1) = 2^2 + 3(2)(2) = 4 + 12 = 16.$$
2. **Tính các đạo hàm riêng cục bộ**:
   $$\frac{\partial z}{\partial u} = 2u + 3v = 2(2) + 3(2) = 10, \quad \frac{\partial z}{\partial v} = 3u = 3(2) = 6.$$
   $$\frac{du}{dt} = 2, \quad \frac{dv}{dt} = 1.$$
3. **Cộng đóng góp từ cả hai nhánh (Chain Rule)**:
   $$\frac{dz}{dt} = \frac{\partial z}{\partial u}\frac{du}{dt} + \frac{\partial z}{\partial v}\frac{dv}{dt} = 10 \times 2 + 6 \times 1 = 26.$$
4. **Kiểm tra chéo bằng cách thế trực tiếp**: $$z(t) = (2t)^2 + 3(2t)(t+1) = 4t^2 + 6t^2 + 6t = 10t^2 + 6t \implies z'(t) = 20t + 6 \implies z'(1) = 26$$. Khớp hoàn toàn! Đây chính là lý do trong đồ thị tính toán, mỗi khi một biến rẽ nhánh, gradient chảy ngược về nó phải được **cộng dồn**.

Bài học này trình bày khái niệm đạo hàm và các công cụ giải tích đa biến cốt lõi làm nền tảng cho lý thuyết và thuật toán tối ưu hóa.

---

## Đạo hàm và tốc độ biến thiên

Đạo hàm của hàm một biến biểu thị tốc độ biến thiên tức thời của hàm số, giúp ta mô tả chính xác hành vi cục bộ của hàm quanh một điểm.

### Các khái niệm cơ bản về đạo hàm

**Độ dốc giữa hai điểm:**

$$\text{Độ dốc} = \frac{y_2 - y_1}{x_2 - x_1}$$

**Đạo hàm (tốc độ biến thiên tức thời):**

$$f'(x_0) = \lim_{x_1 \to x_0} \frac{f(x_1) - f(x_0)}{x_1 - x_0} = \lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0)}{\Delta x}$$

Đạo hàm cho biết hàm số thay đổi nhanh hay chậm tại từng điểm, từ đó hỗ trợ tìm các điểm tối ưu — nơi tốc độ biến thiên bằng không.

### Đường mức của hàm số

Đường mức là công cụ quen thuộc trong giải tích đa biến để trực quan hóa hàm hai biến $$f(x, y)$$, cho phép biểu diễn một bề mặt trong không gian 3D trên mặt phẳng 2D.

Một **đường mức** của hàm số $$f(x, y)$$ là tập hợp tất cả các điểm $$(x, y)$$ trong miền xác định sao cho hàm số nhận cùng một giá trị hằng số:

$$f(x, y) = c$$

**Ví dụ:**
- Với $$f(x, y) = x^2 + y^2$$, các đường mức là những đường tròn đồng tâm: $$x^2 + y^2 = c$$.
- Với $$f(x, y) = x + y$$, các đường mức là những đường thẳng song song: $$x + y = c$$.

Đường mức giúp ta nhận diện:
1. Hình dạng địa hình của hàm số.
2. Hướng tăng và hướng giảm dốc nhất.
3. Vị trí của các điểm cực trị tiềm năng.

---

## Các khái niệm chính của giải tích đa biến

### Đạo hàm riêng

Với hàm số $$f(x_1, x_2, \ldots, x_n)$$, **đạo hàm riêng** theo biến $$x_i$$ được định nghĩa là:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

Đại lượng này đo mức độ biến thiên của $$f$$ khi chỉ riêng $$x_i$$ thay đổi trong khi tất cả các biến còn lại được giữ cố định.

### Vector gradient

**Gradient** là vector tổng hợp tất cả các đạo hàm riêng của hàm số:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

Gradient chỉ theo hướng tăng dốc nhất của hàm số và luôn vuông góc với các đường mức.

### Ma trận Hessian

**Ma trận Hessian** tập hợp toàn bộ các đạo hàm riêng bậc hai của hàm số:

$$\nabla^2 f(\mathbf{x}) = \mathbf{H} = \begin{pmatrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

Ma trận Hessian phản ánh độ cong của hàm số và đóng vai trò quan trọng trong:
- Phân loại các điểm tới hạn (cực tiểu, cực đại hoặc điểm yên ngựa).
- Các thuật toán tối ưu hóa bậc hai như phương pháp Newton.

---

## Quy tắc dây chuyền cho hàm đa biến

Quy tắc dây chuyền (*chain rule*) là công cụ cơ bản để tính đạo hàm của hàm hợp — dạng hàm xuất hiện rất thường xuyên trong các bài toán tối ưu hóa.

### Quy tắc dây chuyền cơ bản

Với hàm số $$z = f(x, y)$$ trong đó $$x = g(t)$$ và $$y = h(t)$$:

$$ \frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} $$

### Quy tắc dây chuyền tổng quát

Với $$z = f(x_1, x_2, \ldots, x_n)$$ trong đó mỗi $$x_i = x_i(t_1, t_2, \ldots, t_m)$$:

$$ \frac{\partial z}{\partial t_j} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \frac{\partial x_i}{\partial t_j} $$

### Ứng dụng trong tối ưu hóa

Quy tắc dây chuyền được sử dụng trực tiếp trong:

1. **Tính gradient:** tính gradient của các hàm mục tiêu dạng hàm hợp.
2. **Xử lý ràng buộc:** xử lý các biến phụ thuộc thông qua phương trình ràng buộc.
3. **Cài đặt thuật toán:** thuật toán lan truyền ngược (*backpropagation*) trong mạng nơ-ron và vi phân tự động.
4. **Phân tích độ nhạy:** đánh giá tác động của việc thay đổi tham số lên nghiệm tối ưu.

### Ví dụ: Tối ưu hóa có ràng buộc

Xét bài toán cực tiểu hóa $$f(x, y) = x^2 + y^2$$ với ràng buộc $$g(x, y) = x + y - 1 = 0$$.

Từ phương trình ràng buộc, ta rút $$y = 1 - x$$ để đưa về bài toán một biến:
$$h(x) = f(x, 1-x) = x^2 + (1-x)^2$$

Áp dụng quy tắc dây chuyền:
$$h'(x) = \frac{\partial f}{\partial x} \cdot 1 + \frac{\partial f}{\partial y} \cdot \frac{d(1-x)}{dx} = 2x + 2(1-x)(-1) = 4x - 2$$

Giải phương trình $$h'(x) = 0$$ ta được $$x = 1/2$$, suy ra điểm tối ưu là $$(1/2, 1/2)$$.

Ví dụ này cho thấy cách các khái niệm giải tích đa biến phối hợp với nhau để giải bài toán tối ưu hóa một cách có hệ thống.

## Video tham khảo

- **Xem trước:** [The paradox of the derivative | Chapter 2, Essence of calculus](https://www.youtube.com/watch?v=9vKqVkMQHKk) — 3Blue1Brown, tiếng Anh. Giới thiệu đạo hàm một biến dưới góc nhìn tốc độ biến thiên.
- [What are derivatives in 3D? Intro to Partial Derivatives](https://www.youtube.com/watch?v=EoEV5-_mLeM) — Dr. Trefor Bazett, tiếng Anh. Giải thích đạo hàm riêng như tốc độ biến thiên theo từng trục tọa độ khi đầu vào là vector.
- [Visualizing the chain rule and product rule | Chapter 4, Essence of calculus](https://www.youtube.com/watch?v=YG15m2VwSjA) — 3Blue1Brown, tiếng Anh. Minh họa trực quan quy tắc dây chuyền ở trường hợp một biến; dạng đa biến trong bài là sự mở rộng trực tiếp của quy tắc này dọc theo phép hợp hàm.
