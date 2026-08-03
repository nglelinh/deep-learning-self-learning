---
layout: post
title: 00-05 Bài tập trắc nghiệm - Khái niệm cơ bản
chapter: '00'
order: 24
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: quiz
---

Bài tập trắc nghiệm này kiểm tra hiểu biết của bạn về các khái niệm cơ bản trong giải tích và đại số tuyến tính, là nền tảng cho tối ưu hóa.

---

## 📚 Ôn tập lý thuyết

Trước khi làm bài tập, hãy ôn lại các khái niệm chính trong chương này:

### 🔢 **Giải tích cơ bản**

**Đạo hàm và đạo hàm riêng:**
- Đạo hàm: $$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
- Đạo hàm riêng: $$\frac{\partial f}{\partial x_i}$$ - đạo hàm theo $$x_i$$ khi các biến khác cố định

**Gradient và đạo hàm có hướng:**
- Gradient: $$\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$ - vector chỉ hướng tăng dốc nhất
- Đạo hàm có hướng: $$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos \theta$$

**Chuỗi Taylor:**
- Bậc 1: $$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0)$$
- Bậc 2: $$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_0)^T \nabla^2 f(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)$$

### 📐 **Đại số tuyến tính**

**Vector và không gian vector:**
- Vector: danh sách có thứ tự các số $$\mathbf{v} = \begin{pmatrix} v_1 \\ \vdots \\ v_n \end{pmatrix}$$
- Không gian $$\mathbb{R}^n$$: tập tất cả vector có $$n$$ thành phần

**Các phép toán vector:**
- Cộng vector: $$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ \vdots \\ u_n + v_n \end{pmatrix}$$
- Nhân vô hướng: $$c\mathbf{v} = \begin{pmatrix} cv_1 \\ \vdots \\ cv_n \end{pmatrix}$$
- Tích vô hướng: $$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta$$

**Chuẩn vector:**
- Chuẩn L2 (Euclid): $$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$$
- Chuẩn L1 (Manhattan): $$\|\mathbf{x}\|_1 = \sum_{i=1}^n \lvert x_i \rvert$$
- Chuẩn L∞ (Max): $$\|\mathbf{x}\|_\infty = \max_i \lvert x_i \rvert$$

**Ma trận:**
- Phép nhân ma trận: $$(AB)_{ij} = \sum_{k} A_{ik} B_{kj}$$
- Ma trận nghịch đảo: $$AA^{-1} = I$$ (nếu tồn tại)
- Định thức: $$\det(A)$$ - đo "thể tích" biến đổi tuyến tính

**Eigenvalue và eigenvector:**
- $$A\mathbf{v} = \lambda \mathbf{v}$$ với $$\mathbf{v} \neq \mathbf{0}$$
- $$\lambda$$: eigenvalue, $$\mathbf{v}$$: eigenvector tương ứng

### 📊 **Lý thuyết tập hợp và giải tích thực**

**Tập hợp cơ bản:**
- Tập mở: mọi điểm đều có lân cận nằm trong tập
- Tập đóng: chứa tất cả điểm biên
- Tập compact: đóng và bị chặn trong $$\mathbb{R}^n$$

**Tính liên tục:**
- Hàm liên tục tại $$x_0$$: $$\lim_{x \to x_0} f(x) = f(x_0)$$
- Liên tục đều: $$\forall \epsilon > 0, \exists \delta > 0$$ sao cho $$\lvert x-y \rvert < \delta \Rightarrow \lvert f(x)-f(y) \rvert < \epsilon$$

### 🎲 **Xác suất cơ bản**

**Khái niệm cơ bản:**
- Xác suất: $$P(A) \in [0,1]$$, $$P(\Omega) = 1$$
- Biến ngẫu nhiên: hàm từ không gian mẫu đến số thực
- Kỳ vọng: $$E[X] = \int x f(x) dx$$ (liên tục) hoặc $$E[X] = \sum x P(X=x)$$ (rời rạc)

**Phân phối thông dụng:**
- Chuẩn: $$X \sim N(\mu, \sigma^2)$$
- Đều: $$X \sim U(a,b)$$
- Bernoulli: $$X \sim \text{Ber}(p)$$

---

💡 **Mẹo:** Các khái niệm này là nền tảng cho tối ưu hóa. Gradient chỉ hướng tăng nhanh nhất, ma trận Hessian mô tả độ cong, và chuỗi Taylor giúp xấp xỉ hàm cục bộ.

---

<div id="quiz-container">
    <div id="quiz-header">
        <h2>Bài tập trắc nghiệm: Khái niệm cơ bản</h2>
        <p>Chọn đáp án đúng nhất cho mỗi câu hỏi. Bạn sẽ nhận được kết quả ngay sau khi hoàn thành.</p>
        <div id="progress-bar">
            <div id="progress-fill"></div>
        </div>
        <p id="progress-text">Câu hỏi 1/25</p>
    </div>

    <div id="quiz-questions">
        <!-- Câu hỏi 1: Đạo hàm cơ bản -->
        <div class="question" id="q1">
            <h3>Câu 1: Đạo hàm của hàm số $$f(x) = x^3 + 2x^2 - 5x + 1$$ tại $$x = 1$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q1" value="a"> A) $$4$$</label>
                <label><input type="radio" name="q1" value="b"> B) $$2$$</label>
                <label><input type="radio" name="q1" value="c"> C) $$-1$$</label>
                <label><input type="radio" name="q1" value="d"> D) $$6$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: B) $$2$$</strong><br>
                $$f'(x) = 3x^2 + 4x - 5$$<br>
                $$f'(1) = 3(1)^2 + 4(1) - 5 = 3 + 4 - 5 = 2$$
            </div>
        </div>

        <!-- Câu hỏi 2: Đạo hàm riêng -->
        <div class="question" id="q2" style="display: none;">
            <h3>Câu 2: Cho $$f(x,y) = x^2y + 3xy^2$$. Đạo hàm riêng $$\frac{\partial f}{\partial x}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q2" value="a"> A) $$2xy + 3y^2$$</label>
                <label><input type="radio" name="q2" value="b"> B) $$x^2 + 6xy$$</label>
                <label><input type="radio" name="q2" value="c"> C) $$2xy + 6xy$$</label>
                <label><input type="radio" name="q2" value="d"> D) $$x^2 + 3y^2$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$2xy + 3y^2$$</strong><br>
                Khi tính đạo hàm riêng theo $$x$$, ta coi $$y$$ là hằng số:<br>
                $$\frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(x^2y + 3xy^2) = 2xy + 3y^2$$
            </div>
        </div>

        <!-- Câu hỏi 3: Gradient -->
        <div class="question" id="q3" style="display: none;">
            <h3>Câu 3: Gradient của hàm $$f(x,y) = x^2 + 2xy + y^2$$ tại điểm $$(1,1)$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q3" value="a"> A) $$\begin{pmatrix} 4 \\ 4 \end{pmatrix}$$</label>
                <label><input type="radio" name="q3" value="b"> B) $$\begin{pmatrix} 2 \\ 2 \end{pmatrix}$$</label>
                <label><input type="radio" name="q3" value="c"> C) $$\begin{pmatrix} 3 \\ 3 \end{pmatrix}$$</label>
                <label><input type="radio" name="q3" value="d"> D) $$\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\begin{pmatrix} 4 \\ 4 \end{pmatrix}$$</strong><br>
                $$\frac{\partial f}{\partial x} = 2x + 2y$$, $$\frac{\partial f}{\partial y} = 2x + 2y$$<br>
                Tại $$(1,1)$$: $$\nabla f(1,1) = \begin{pmatrix} 2(1) + 2(1) \\ 2(1) + 2(1) \end{pmatrix} = \begin{pmatrix} 4 \\ 4 \end{pmatrix}$$
            </div>
        </div>

        <!-- Câu hỏi 4: Đạo hàm có hướng -->
        <div class="question" id="q4" style="display: none;">
            <h3>Câu 4: Cho gradient $$\nabla f = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$ và vector đơn vị $$\mathbf{u} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix}$$. Đạo hàm có hướng $$D_{\mathbf{u}}f$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q4" value="a"> A) $$5$$</label>
                <label><input type="radio" name="q4" value="b"> B) $$5.0$$</label>
                <label><input type="radio" name="q4" value="c"> C) $$3.2$$</label>
                <label><input type="radio" name="q4" value="d"> D) $$7$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$5$$</strong><br>
                $$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = 3(0.6) + 4(0.8) = 1.8 + 3.2 = 5$$
            </div>
        </div>

        <!-- Câu hỏi 5: Hướng tăng dốc nhất -->
        <div class="question" id="q5" style="display: none;">
            <h3>Câu 5: Hướng tăng dốc nhất của hàm số luôn là:</h3>
            <div class="options">
                <label><input type="radio" name="q5" value="a"> A) Hướng của vector gradient</label>
                <label><input type="radio" name="q5" value="b"> B) Hướng ngược với vector gradient</label>
                <label><input type="radio" name="q5" value="c"> C) Hướng vuông góc với vector gradient</label>
                <label><input type="radio" name="q5" value="d"> D) Hướng của trục x</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) Hướng của vector gradient</strong><br>
                Gradient luôn chỉ theo hướng tăng dốc nhất của hàm số. Đây là một tính chất cơ bản của gradient.
            </div>
        </div>

        <!-- Câu hỏi 6: Phép cộng vector -->
        <div class="question" id="q6" style="display: none;">
            <h3>Câu 6: Cho $$\mathbf{u} = \begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}$$ và $$\mathbf{v} = \begin{pmatrix} 1 \\ 4 \\ -2 \end{pmatrix}$$. Vector $$\mathbf{u} + \mathbf{v}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q6" value="a"> A) $$\begin{pmatrix} 3 \\ 3 \\ 1 \end{pmatrix}$$</label>
                <label><input type="radio" name="q6" value="b"> B) $$\begin{pmatrix} 1 \\ -5 \\ 5 \end{pmatrix}$$</label>
                <label><input type="radio" name="q6" value="c"> C) $$\begin{pmatrix} 2 \\ 4 \\ -6 \end{pmatrix}$$</label>
                <label><input type="radio" name="q6" value="d"> D) $$\begin{pmatrix} 3 \\ -3 \\ 1 \end{pmatrix}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\begin{pmatrix} 3 \\ 3 \\ 1 \end{pmatrix}$$</strong><br>
                $$\mathbf{u} + \mathbf{v} = \begin{pmatrix} 2+1 \\ -1+4 \\ 3+(-2) \end{pmatrix} = \begin{pmatrix} 3 \\ 3 \\ 1 \end{pmatrix}$$
            </div>
        </div>

        <!-- Câu hỏi 7: Tích vô hướng -->
        <div class="question" id="q7" style="display: none;">
            <h3>Câu 7: Tích vô hướng của $$\mathbf{a} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$$ và $$\mathbf{b} = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q7" value="a"> A) $$32$$</label>
                <label><input type="radio" name="q7" value="b"> B) $$14$$</label>
                <label><input type="radio" name="q7" value="c"> C) $$18$$</label>
                <label><input type="radio" name="q7" value="d"> D) $$24$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$32$$</strong><br>
                $$\mathbf{a} \cdot \mathbf{b} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32$$
            </div>
        </div>

        <!-- Câu hỏi 8: Chuẩn Euclid -->
        <div class="question" id="q8" style="display: none;">
            <h3>Câu 8: Chuẩn Euclid của vector $$\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q8" value="a"> A) $$5$$</label>
                <label><input type="radio" name="q8" value="b"> B) $$7$$</label>
                <label><input type="radio" name="q8" value="c"> C) $$25$$</label>
                <label><input type="radio" name="q8" value="d"> D) $$\sqrt{7}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$5$$</strong><br>
                $$\|\mathbf{v}\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$
            </div>
        </div>

        <!-- Câu hỏi 9: Vector trực giao -->
        <div class="question" id="q9" style="display: none;">
            <h3>Câu 9: Hai vector $$\mathbf{u}$$ và $$\mathbf{v}$$ được gọi là trực giao khi:</h3>
            <div class="options">
                <label><input type="radio" name="q9" value="a"> A) $$\mathbf{u} \cdot \mathbf{v} = 0$$</label>
                <label><input type="radio" name="q9" value="b"> B) $$\mathbf{u} \cdot \mathbf{v} = 1$$</label>
                <label><input type="radio" name="q9" value="c"> C) $$\|\mathbf{u}\| = \|\mathbf{v}\|$$</label>
                <label><input type="radio" name="q9" value="d"> D) $$\mathbf{u} + \mathbf{v} = \mathbf{0}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\mathbf{u} \cdot \mathbf{v} = 0$$</strong><br>
                Hai vector trực giao (vuông góc) khi và chỉ khi tích vô hướng của chúng bằng 0.
            </div>
        </div>

        <!-- Câu hỏi 10: Phép nhân ma trận -->
        <div class="question" id="q10" style="display: none;">
            <h3>Câu 10: Tích của hai ma trận $$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q10" value="a"> A) $$\begin{pmatrix} 4 & 6 \\ 10 & 12 \end{pmatrix}$$</label>
                <label><input type="radio" name="q10" value="b"> B) $$\begin{pmatrix} 4 & 6 \\ 10 & 12 \end{pmatrix}$$</label>
                <label><input type="radio" name="q10" value="c"> C) $$\begin{pmatrix} 2 & 0 \\ 3 & 12 \end{pmatrix}$$</label>
                <label><input type="radio" name="q10" value="d"> D) $$\begin{pmatrix} 4 & 6 \\ 10 & 12 \end{pmatrix}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\begin{pmatrix} 4 & 6 \\ 10 & 12 \end{pmatrix}$$</strong><br>
                Phần tử $$(1,1)$$: $$1 \cdot 2 + 2 \cdot 1 = 4$$<br>
                Phần tử $$(1,2)$$: $$1 \cdot 0 + 2 \cdot 3 = 6$$<br>
                Phần tử $$(2,1)$$: $$3 \cdot 2 + 4 \cdot 1 = 10$$<br>
                Phần tử $$(2,2)$$: $$3 \cdot 0 + 4 \cdot 3 = 12$$
            </div>
        </div>

        <!-- Câu hỏi 11: Ma trận đơn vị -->
        <div class="question" id="q11" style="display: none;">
            <h3>Câu 11: Ma trận đơn vị $$2 \times 2$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q11" value="a"> A) $$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$</label>
                <label><input type="radio" name="q11" value="b"> B) $$\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$</label>
                <label><input type="radio" name="q11" value="c"> C) $$\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$</label>
                <label><input type="radio" name="q11" value="d"> D) $$\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$</strong><br>
                Ma trận đơn vị có các phần tử trên đường chéo chính bằng 1 và các phần tử khác bằng 0.
            </div>
        </div>

        <!-- Câu hỏi 12: Ma trận chuyển vị -->
        <div class="question" id="q12" style="display: none;">
            <h3>Câu 12: Chuyển vị của ma trận $$\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q12" value="a"> A) $$\begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$</label>
                <label><input type="radio" name="q12" value="b"> B) $$\begin{pmatrix} 6 & 5 & 4 \\ 3 & 2 & 1 \end{pmatrix}$$</label>
                <label><input type="radio" name="q12" value="c"> C) $$\begin{pmatrix} 4 & 1 \\ 5 & 2 \\ 6 & 3 \end{pmatrix}$$</label>
                <label><input type="radio" name="q12" value="d"> D) $$\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}$$</strong><br>
                Chuyển vị của ma trận được tạo bằng cách hoán đổi hàng và cột.
            </div>
        </div>

        <!-- Câu hỏi 13: Định thức 2x2 -->
        <div class="question" id="q13" style="display: none;">
            <h3>Câu 13: Định thức của ma trận $$\begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q13" value="a"> A) $$10$$</label>
                <label><input type="radio" name="q13" value="b"> B) $$14$$</label>
                <label><input type="radio" name="q13" value="c"> C) $$2$$</label>
                <label><input type="radio" name="q13" value="d"> D) $$12$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$10$$</strong><br>
                $$\det\begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix} = 3 \cdot 4 - 1 \cdot 2 = 12 - 2 = 10$$
            </div>
        </div>

        <!-- Câu hỏi 14: Độc lập tuyến tính -->
        <div class="question" id="q14" style="display: none;">
            <h3>Câu 14: Hai vector $$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$ và $$\mathbf{v}_2 = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q14" value="a"> A) Độc lập tuyến tính</label>
                <label><input type="radio" name="q14" value="b"> B) Phụ thuộc tuyến tính</label>
                <label><input type="radio" name="q14" value="c"> C) Trực giao</label>
                <label><input type="radio" name="q14" value="d"> D) Không thể xác định</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: B) Phụ thuộc tuyến tính</strong><br>
                $$\mathbf{v}_2 = 2\mathbf{v}_1$$, do đó hai vector này phụ thuộc tuyến tính.
            </div>
        </div>

        <!-- Câu hỏi 15: Chuẩn Manhattan -->
        <div class="question" id="q15" style="display: none;">
            <h3>Câu 15: Chuẩn Manhattan (L1) của vector $$\mathbf{v} = \begin{pmatrix} -2 \\ 3 \\ -1 \end{pmatrix}$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q15" value="a"> A) $$6$$</label>
                <label><input type="radio" name="q15" value="b"> B) $$0$$</label>
                <label><input type="radio" name="q15" value="c"> C) $$4$$</label>
                <label><input type="radio" name="q15" value="d"> D) $$\sqrt{14}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$6$$</strong><br>
                $$\|\mathbf{v}\|_1 = |-2| + |3| + |-1| = 2 + 3 + 1 = 6$$
            </div>
        </div>

        <!-- Câu hỏi 16: Quy tắc dây chuyền -->
        <div class="question" id="q16" style="display: none;">
            <h3>Câu 16: Cho $$z = x^2 + y^2$$ với $$x = 2t$$ và $$y = 3t$$. Đạo hàm $$\frac{dz}{dt}$$ tại $$t = 1$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q16" value="a"> A) $$26$$</label>
                <label><input type="radio" name="q16" value="b"> B) $$13$$</label>
                <label><input type="radio" name="q16" value="c"> C) $$10$$</label>
                <label><input type="radio" name="q16" value="d"> D) $$5$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$26$$</strong><br>
                $$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt} = 2x \cdot 2 + 2y \cdot 3$$<br>
                Tại $$t = 1$$: $$x = 2, y = 3$$, nên $$\frac{dz}{dt} = 2(2)(2) + 2(3)(3) = 8 + 18 = 26$$
            </div>
        </div>

        <!-- Câu hỏi 17: Điểm tới hạn -->
        <div class="question" id="q17" style="display: none;">
            <h3>Câu 17: Điểm tới hạn của hàm số là điểm mà:</h3>
            <div class="options">
                <label><input type="radio" name="q17" value="a"> A) Gradient bằng vector không</label>
                <label><input type="radio" name="q17" value="b"> B) Hàm số đạt giá trị lớn nhất</label>
                <label><input type="radio" name="q17" value="c"> C) Hàm số đạt giá trị nhỏ nhất</label>
                <label><input type="radio" name="q17" value="d"> D) Hàm số không xác định</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) Gradient bằng vector không</strong><br>
                Điểm tới hạn là điểm mà gradient của hàm số bằng vector không, tức là $$\nabla f = \mathbf{0}$$.
            </div>
        </div>

        <!-- Câu hỏi 18: Ma trận nghịch đảo -->
        <div class="question" id="q18" style="display: none;">
            <h3>Câu 18: Ma trận $$\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$$ có nghịch đảo là:</h3>
            <div class="options">
                <label><input type="radio" name="q18" value="a"> A) $$\begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix}$$</label>
                <label><input type="radio" name="q18" value="b"> B) $$\begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}$$</label>
                <label><input type="radio" name="q18" value="c"> C) $$\begin{pmatrix} 2 & -1 \\ -1 & 1 \end{pmatrix}$$</label>
                <label><input type="radio" name="q18" value="d"> D) $$\begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 1 \end{pmatrix}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix}$$</strong><br>
                $$\det(A) = 2 \cdot 1 - 1 \cdot 1 = 1$$<br>
                $$A^{-1} = \frac{1}{1}\begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix}$$
            </div>
        </div>

        <!-- Câu hỏi 19: Đường mức -->
        <div class="question" id="q19" style="display: none;">
            <h3>Câu 19: Gradient của hàm số tại một điểm luôn:</h3>
            <div class="options">
                <label><input type="radio" name="q19" value="a"> A) Vuông góc với đường mức tại điểm đó</label>
                <label><input type="radio" name="q19" value="b"> B) Song song với đường mức tại điểm đó</label>
                <label><input type="radio" name="q19" value="c"> C) Tạo góc 45° với đường mức</label>
                <label><input type="radio" name="q19" value="d"> D) Không có mối quan hệ với đường mức</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) Vuông góc với đường mức tại điểm đó</strong><br>
                Gradient luôn vuông góc với đường mức vì đường mức là tập hợp các điểm có cùng giá trị hàm số.
            </div>
        </div>

        <!-- Câu hỏi 20: Ứng dụng trong tối ưu hóa -->
        <div class="question" id="q20" style="display: none;">
            <h3>Câu 20: Trong thuật toán gradient descent, chúng ta di chuyển theo hướng:</h3>
            <div class="options">
                <label><input type="radio" name="q20" value="a"> A) Ngược với hướng gradient</label>
                <label><input type="radio" name="q20" value="b"> B) Cùng hướng với gradient</label>
                <label><input type="radio" name="q20" value="c"> C) Vuông góc với gradient</label>
                <label><input type="radio" name="q20" value="d"> D) Hướng ngẫu nhiên</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) Ngược với hướng gradient</strong><br>
                Gradient descent di chuyển theo hướng $$-\nabla f$$ để tìm điểm cực tiểu của hàm số.
            </div>
        </div>

        <!-- Câu hỏi 21: Tính toán đạo hàm bậc hai -->
        <div class="question" id="q21" style="display: none;">
            <h3>Câu 21: Đạo hàm bậc hai của hàm $$f(x) = 3x^4 - 2x^3 + x^2 - 5x + 1$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q21" value="a"> A) $$36x^2 - 12x + 2$$</label>
                <label><input type="radio" name="q21" value="b"> B) $$12x^3 - 6x^2 + 2x - 5$$</label>
                <label><input type="radio" name="q21" value="c"> C) $$36x^2 - 12x + 1$$</label>
                <label><input type="radio" name="q21" value="d"> D) $$12x^2 - 6x + 2$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$36x^2 - 12x + 2$$</strong><br>
                $$f'(x) = 12x^3 - 6x^2 + 2x - 5$$<br>
                $$f''(x) = 36x^2 - 12x + 2$$
            </div>
        </div>

        <!-- Câu hỏi 22: Ứng dụng ma trận Hessian -->
        <div class="question" id="q22" style="display: none;">
            <h3>Câu 22: Cho $$f(x,y) = x^2 + 2xy + 3y^2$$. Ma trận Hessian tại điểm $$(1,1)$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q22" value="a"> A) $$\begin{pmatrix} 2 & 2 \\ 2 & 6 \end{pmatrix}$$</label>
                <label><input type="radio" name="q22" value="b"> B) $$\begin{pmatrix} 1 & 2 \\ 2 & 3 \end{pmatrix}$$</label>
                <label><input type="radio" name="q22" value="c"> C) $$\begin{pmatrix} 2 & 1 \\ 1 & 6 \end{pmatrix}$$</label>
                <label><input type="radio" name="q22" value="d"> D) $$\begin{pmatrix} 4 & 4 \\ 4 & 8 \end{pmatrix}$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\begin{pmatrix} 2 & 2 \\ 2 & 6 \end{pmatrix}$$</strong><br>
                $$\frac{\partial^2 f}{\partial x^2} = 2$$, $$\frac{\partial^2 f}{\partial y^2} = 6$$, $$\frac{\partial^2 f}{\partial x \partial y} = 2$$<br>
                Ma trận Hessian không phụ thuộc vào điểm cụ thể trong trường hợp này.
            </div>
        </div>

        <!-- Câu hỏi 23: Tính toán thực tế với vector -->
        <div class="question" id="q23" style="display: none;">
            <h3>Câu 23: Khoảng cách từ điểm $$A(1,2)$$ đến điểm $$B(4,6)$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q23" value="a"> A) $$5$$</label>
                <label><input type="radio" name="q23" value="b"> B) $$7$$</label>
                <label><input type="radio" name="q23" value="c"> C) $$\sqrt{25}$$</label>
                <label><input type="radio" name="q23" value="d"> D) $$3$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$5$$</strong><br>
                Khoảng cách = $$\sqrt{(4-1)^2 + (6-2)^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$
            </div>
        </div>

        <!-- Câu hỏi 24: Ứng dụng eigenvalue -->
        <div class="question" id="q24" style="display: none;">
            <h3>Câu 24: Ma trận $$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$ có eigenvalue là:</h3>
            <div class="options">
                <label><input type="radio" name="q24" value="a"> A) $$\lambda_1 = 3, \lambda_2 = 2$$</label>
                <label><input type="radio" name="q24" value="b"> B) $$\lambda_1 = 2, \lambda_2 = 3$$</label>
                <label><input type="radio" name="q24" value="c"> C) $$\lambda_1 = 1, \lambda_2 = 5$$</label>
                <label><input type="radio" name="q24" value="d"> D) $$\lambda_1 = 0, \lambda_2 = 5$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$\lambda_1 = 3, \lambda_2 = 2$$</strong><br>
                Đối với ma trận tam giác trên, eigenvalue chính là các phần tử trên đường chéo chính.
            </div>
        </div>

        <!-- Câu hỏi 25: Tối ưu hóa đơn giản -->
        <div class="question" id="q25" style="display: none;">
            <h3>Câu 25: Điểm cực tiểu của hàm $$f(x) = x^2 - 4x + 5$$ là:</h3>
            <div class="options">
                <label><input type="radio" name="q25" value="a"> A) $$x = 2$$</label>
                <label><input type="radio" name="q25" value="b"> B) $$x = -2$$</label>
                <label><input type="radio" name="q25" value="c"> C) $$x = 4$$</label>
                <label><input type="radio" name="q25" value="d"> D) $$x = 0$$</label>
            </div>
            <div class="explanation" style="display: none;">
                <strong>Đáp án đúng: A) $$x = 2$$</strong><br>
                $$f'(x) = 2x - 4 = 0 \Rightarrow x = 2$$<br>
                $$f''(x) = 2 > 0$$, nên $$x = 2$$ là điểm cực tiểu.
            </div>
        </div>
    </div>

    <div id="quiz-navigation">
        <button id="prev-btn" onclick="previousQuestion()" disabled>Câu trước</button>
        <button id="next-btn" onclick="nextQuestion()">Câu tiếp</button>
        <button id="submit-btn" onclick="submitQuiz()" style="display: none;">Nộp bài</button>
    </div>

    <div id="quiz-results" style="display: none;">
        <h3>Kết quả bài tập</h3>
        <div id="score-display"></div>
        <div id="detailed-results"></div>
        <button onclick="restartQuiz()">Làm lại</button>
    </div>
</div>

<style>
#quiz-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

#quiz-header {
    text-align: center;
    margin-bottom: 30px;
}

#progress-bar {
    width: 100%;
    height: 10px;
    background-color: #e0e0e0;
    border-radius: 5px;
    margin: 20px 0;
    overflow: hidden;
}

#progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #45a049);
    width: 5%;
    transition: width 0.3s ease;
}

.question {
    background: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.question h3 {
    color: #333;
    margin-bottom: 20px;
    font-size: 1.1em;
    line-height: 1.4;
}

.options {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.options label {
    display: flex;
    align-items: center;
    padding: 12px;
    background: white;
    border: 2px solid #e0e0e0;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 1em;
}

.options label:hover {
    border-color: #4CAF50;
    background-color: #f0f8f0;
}

.options input[type="radio"] {
    margin-right: 12px;
    transform: scale(1.2);
}

.options label.selected {
    border-color: #4CAF50;
    background-color: #e8f5e8;
}

.explanation {
    margin-top: 20px;
    padding: 15px;
    background-color: #e3f2fd;
    border-left: 4px solid #2196F3;
    border-radius: 4px;
    font-size: 0.95em;
    line-height: 1.5;
}

.explanation strong {
    color: #1976D2;
}

#quiz-navigation {
    text-align: center;
    margin: 30px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

#quiz-navigation button {
    background: #4CAF50;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1em;
    transition: background-color 0.2s ease;
}

#quiz-navigation button:hover:not(:disabled) {
    background: #45a049;
}

#quiz-navigation button:disabled {
    background: #cccccc;
    cursor: not-allowed;
}

#quiz-results {
    text-align: center;
    padding: 30px;
    background: #f0f8f0;
    border-radius: 8px;
    border: 2px solid #4CAF50;
}

#score-display {
    font-size: 1.5em;
    font-weight: bold;
    margin: 20px 0;
    color: #2E7D32;
}

#detailed-results {
    text-align: left;
    margin: 20px 0;
    max-height: 400px;
    overflow-y: auto;
}

.result-item {
    padding: 10px;
    margin: 5px 0;
    border-radius: 4px;
    border-left: 4px solid;
}

.result-item.correct {
    background-color: #e8f5e8;
    border-left-color: #4CAF50;
}

.result-item.incorrect {
    background-color: #ffebee;
    border-left-color: #f44336;
}

@media (max-width: 600px) {
    #quiz-container {
        padding: 10px;
    }
    
    .question {
        padding: 15px;
    }
    
    #quiz-navigation {
        flex-direction: column;
        gap: 10px;
    }
    
    #quiz-navigation button {
        width: 100%;
    }
}
</style>

<script>
let currentQuestion = 1;
const totalQuestions = 25;
let userAnswers = {};
let quizSubmitted = false;

// Đáp án đúng cho từng câu hỏi
const correctAnswers = {
    q1: 'b', q2: 'a', q3: 'a', q4: 'a', q5: 'a',
    q6: 'a', q7: 'a', q8: 'a', q9: 'a', q10: 'a',
    q11: 'a', q12: 'a', q13: 'a', q14: 'b', q15: 'a',
    q16: 'a', q17: 'a', q18: 'a', q19: 'a', q20: 'a',
    q21: 'a', q22: 'a', q23: 'a', q24: 'a', q25: 'a'
};

/**
 * Cập nhật thanh tiến trình và hiển thị câu hỏi hiện tại
 */
function updateProgress() {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    const percentage = (currentQuestion / totalQuestions) * 100;
    progressFill.style.width = percentage + '%';
    progressText.textContent = `Câu hỏi ${currentQuestion}/${totalQuestions}`;
}

/**
 * Hiển thị câu hỏi theo số thứ tự
 */
function showQuestion(questionNum) {
    // Ẩn tất cả câu hỏi
    for (let i = 1; i <= totalQuestions; i++) {
        document.getElementById(`q${i}`).style.display = 'none';
    }
    
    // Hiển thị câu hỏi hiện tại
    document.getElementById(`q${questionNum}`).style.display = 'block';
    
    // Cập nhật trạng thái nút điều hướng
    document.getElementById('prev-btn').disabled = (questionNum === 1);
    document.getElementById('next-btn').style.display = (questionNum === totalQuestions) ? 'none' : 'inline-block';
    document.getElementById('submit-btn').style.display = (questionNum === totalQuestions) ? 'inline-block' : 'none';
    
    updateProgress();
}

/**
 * Chuyển đến câu hỏi tiếp theo
 */
function nextQuestion() {
    if (currentQuestion < totalQuestions) {
        currentQuestion++;
        showQuestion(currentQuestion);
    }
}

/**
 * Quay lại câu hỏi trước
 */
function previousQuestion() {
    if (currentQuestion > 1) {
        currentQuestion--;
        showQuestion(currentQuestion);
    }
}

/**
 * Lưu đáp án của người dùng
 */
function saveAnswer(questionId, answer) {
    userAnswers[questionId] = answer;
    
    // Cập nhật giao diện để hiển thị đáp án đã chọn
    const labels = document.querySelectorAll(`#${questionId} .options label`);
    labels.forEach(label => {
        label.classList.remove('selected');
        if (label.querySelector('input').value === answer) {
            label.classList.add('selected');
        }
    });
}

/**
 * Nộp bài và hiển thị kết quả
 */
function submitQuiz() {
    if (quizSubmitted) return;
    
    quizSubmitted = true;
    let correctCount = 0;
    let detailedResults = '';
    
    // Tính điểm và tạo báo cáo chi tiết
    for (let i = 1; i <= totalQuestions; i++) {
        const questionId = `q${i}`;
        const userAnswer = userAnswers[questionId];
        const correctAnswer = correctAnswers[questionId];
        const isCorrect = userAnswer === correctAnswer;
        
        if (isCorrect) correctCount++;
        
        // Hiển thị giải thích cho tất cả câu hỏi
        const explanation = document.querySelector(`#${questionId} .explanation`);
        if (explanation) {
            explanation.style.display = 'block';
        }
        
        // Tạo báo cáo chi tiết
        detailedResults += `
            <div class="result-item ${isCorrect ? 'correct' : 'incorrect'}">
                <strong>Câu ${i}:</strong> ${isCorrect ? 'Đúng' : 'Sai'}
                ${!isCorrect ? `<br><small>Đáp án của bạn: ${userAnswer || 'Chưa trả lời'} | Đáp án đúng: ${correctAnswer}</small>` : ''}
            </div>
        `;
    }
    
    // Hiển thị kết quả
    const scorePercentage = Math.round((correctCount / totalQuestions) * 100);
    document.getElementById('score-display').innerHTML = `
        <div>Điểm số: ${correctCount}/${totalQuestions} (${scorePercentage}%)</div>
        <div style="margin-top: 10px; font-size: 0.9em;">
            ${scorePercentage >= 80 ? '🎉 Xuất sắc!' : 
              scorePercentage >= 60 ? '👍 Khá tốt!' : 
              scorePercentage >= 40 ? '📚 Cần ôn tập thêm' : '💪 Hãy cố gắng hơn!'}
        </div>
    `;
    
    document.getElementById('detailed-results').innerHTML = detailedResults;
    document.getElementById('quiz-results').style.display = 'block';
    document.getElementById('quiz-navigation').style.display = 'none';
    
    // Cuộn đến kết quả
    document.getElementById('quiz-results').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Khởi động lại bài tập
 */
function restartQuiz() {
    currentQuestion = 1;
    userAnswers = {};
    quizSubmitted = false;
    
    // Ẩn kết quả và hiển thị lại điều hướng
    document.getElementById('quiz-results').style.display = 'none';
    document.getElementById('quiz-navigation').style.display = 'flex';
    
    // Ẩn tất cả giải thích
    document.querySelectorAll('.explanation').forEach(exp => {
        exp.style.display = 'none';
    });
    
    // Xóa tất cả lựa chọn
    document.querySelectorAll('input[type="radio"]').forEach(input => {
        input.checked = false;
    });
    
    document.querySelectorAll('.options label').forEach(label => {
        label.classList.remove('selected');
    });
    
    // Hiển thị câu hỏi đầu tiên
    showQuestion(1);
    
    // Cuộn lên đầu
    document.getElementById('quiz-header').scrollIntoView({ behavior: 'smooth' });
}

// Khởi tạo bài tập khi trang được tải
document.addEventListener('DOMContentLoaded', function() {
    showQuestion(1);
    
    // Thêm event listener cho tất cả radio button
    document.querySelectorAll('input[type="radio"]').forEach(input => {
        input.addEventListener('change', function() {
            const questionId = this.name;
            const answer = this.value;
            saveAnswer(questionId, answer);
        });
    });
    
    // Render MathJax sau khi DOM được tải
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
});

// Xử lý phím tắt
document.addEventListener('keydown', function(event) {
    if (quizSubmitted) return;
    
    switch(event.key) {
        case 'ArrowLeft':
            if (currentQuestion > 1) previousQuestion();
            break;
        case 'ArrowRight':
            if (currentQuestion < totalQuestions) nextQuestion();
            break;
        case 'Enter':
            if (currentQuestion === totalQuestions) submitQuiz();
            break;
    }
});
</script>
