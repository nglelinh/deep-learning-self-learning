---
layout: post
title: 00-03-02 Tôpô trong Giải tích thực
chapter: '00'
order: 13
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Bài học này trình bày các khái niệm tôpô (*topology*) cốt yếu từ giải tích thực — nền tảng để hiểu cấu trúc vùng khả thi, tính liên tục và sự tồn tại nghiệm tối ưu trong các bài toán tối ưu hóa.

---

## Giới thiệu về tôpô

Tôpô nghiên cứu những tính chất của không gian được bảo toàn dưới các biến dạng liên tục. Trong tối ưu hóa, các khái niệm tôpô giúp ta hiểu cấu trúc vùng khả thi và hành vi của hàm số, đặc biệt liên quan đến sự tồn tại và đặc trưng của nghiệm tối ưu.

### Không gian metric và khoảng cách

Trước khi bàn về tôpô, ta cần khái niệm khoảng cách. Trong $$\mathbb{R}^n$$, **khoảng cách Euclid** chuẩn giữa hai điểm $$\mathbf{x}$$ và $$\mathbf{y}$$ là:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

### Hình cầu mở và lân cận

Một **hình cầu mở** (*open ball*) tâm $$\mathbf{x}_0$$ bán kính $$\epsilon > 0$$ là:

$$B(\mathbf{x}_0, \epsilon) = \{\mathbf{y} \in \mathbb{R}^n : d(\mathbf{x}_0, \mathbf{y}) < \epsilon\}$$

Tập này gồm mọi điểm cách $$\mathbf{x}_0$$ một khoảng nhỏ hơn $$\epsilon$$.

**Ví dụ:**
- Trong $$\mathbb{R}$$: $$B(0, 1) = (-1, 1)$$ (khoảng mở)
- Trong $$\mathbb{R}^2$$: $$B(\mathbf{0}, 1) = \{(x, y) : x^2 + y^2 < 1\}$$ (đĩa đơn vị mở)

---

## Tập mở

Một **tập mở** (*open set*) được đặc trưng bởi tính chất: nó **không chứa điểm biên nào** của chính nó.

### Định nghĩa hình thức

Một tập $$S$$ trong $$\mathbb{R}^n$$ được gọi là **mở** nếu với mọi điểm $$\mathbf{x} \in S$$, tồn tại số thực dương $$\epsilon > 0$$ sao cho hình cầu mở $$B(\mathbf{x}, \epsilon)$$ nằm hoàn toàn trong $$S$$:

$$\forall \mathbf{x} \in S, \exists \epsilon > 0 : B(\mathbf{x}, \epsilon) \subseteq S$$

### Hiểu theo trực giác

Tập mở có tính chất: nếu ta đang ở bên trong tập, ta có thể di chuyển một khoảng nhỏ theo mọi hướng mà vẫn còn trong tập. Xung quanh mọi điểm luôn có một “khoảng trống an toàn”.

### Ví dụ về tập mở

**Trong $$\mathbb{R}$$:**
- $$(0, 1) = \{x : 0 < x < 1\}$$
- $$(-\infty, 5) = \{x : x < 5\}$$
- chính $$\mathbb{R}$$

**Trong $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 < 1\}$$ (đĩa đơn vị mở)
- $$\{(x, y) : x > 0, y > 0\}$$ (góc phần tư thứ nhất, không gồm các trục)
- chính $$\mathbb{R}^2$$

**Trong $$\mathbb{R}^n$$:**
- Mọi hình cầu mở $$B(\mathbf{x}_0, r)$$
- chính $$\mathbb{R}^n$$
- $$\emptyset$$ (tập rỗng — mở một cách hiển nhiên theo định nghĩa)

### Tính chất của tập mở

1. Hợp tùy ý của các tập mở là tập mở
2. Giao hữu hạn các tập mở là tập mở
3. $$\mathbb{R}^n$$ và $$\emptyset$$ đều là tập mở

---

## Tập đóng

Một **tập đóng** (*closed set*) được định nghĩa là tập chứa tất cả các điểm biên của nó. Tương đương, tập $$S$$ đóng nếu phần bù $$\mathbb{R}^n \setminus S$$ là một **tập mở**.

### Định nghĩa hình thức

Tập $$S$$ được gọi là **đóng** nếu nó chứa mọi điểm giới hạn của nó. Nghĩa là, nếu một dãy điểm $$(\mathbf{x}_n)$$ trong $$S$$ hội tụ về điểm $$\mathbf{x}$$, thì $$\mathbf{x}$$ cũng phải thuộc $$S$$:

$$\text{Nếu } \mathbf{x}_n \in S \text{ với mọi } n \text{ và } \lim_{n \to \infty} \mathbf{x}_n = \mathbf{x}, \text{ thì } \mathbf{x} \in S$$

### Ví dụ về tập đóng

**Trong $$\mathbb{R}$$:**
- $$[0, 1] = \{x : 0 \leq x \leq 1\}$$
- $$[a, \infty) = \{x : x \geq a\}$$
- $$\{0\}$$ (điểm đơn)
- $$\mathbb{Z}$$ (tập số nguyên)

**Trong $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 \leq 1\}$$ (đĩa đơn vị đóng)
- $$\{(x, y) : x \geq 0, y \geq 0\}$$ (góc phần tư thứ nhất, gồm cả các trục)
- $$\{(0, 0)\}$$ (điểm đơn)

**Trong $$\mathbb{R}^n$$:**
- Mọi hình cầu đóng $$\overline{B}(\mathbf{x}_0, r) = \{\mathbf{x} : d(\mathbf{x}, \mathbf{x}_0) \leq r\}$$
- chính $$\mathbb{R}^n$$
- $$\emptyset$$ (tập rỗng)
- Mọi tập hữu hạn

### Tính chất của tập đóng

1. Giao tùy ý của các tập đóng là tập đóng
2. Hợp hữu hạn các tập đóng là tập đóng
3. $$\mathbb{R}^n$$ và $$\emptyset$$ đều là tập đóng

### Lưu ý quan trọng

Một tập có thể:
- **Mở nhưng không đóng:** $$(0, 1)$$
- **Đóng nhưng không mở:** $$[0, 1]$$
- **Vừa mở vừa đóng:** $$\mathbb{R}^n$$, $$\emptyset$$
- **Không mở cũng không đóng:** $$[0, 1)$$, $$(0, 1]$$

---

## Biên, phần trong và bao đóng

### Biên

**Biên** (*boundary*) của tập $$S$$, ký hiệu $$\partial S$$, gồm các điểm “nằm trên mép” của tập. Điểm $$\mathbf{x}$$ là **điểm biên** của $$S$$ nếu mọi hình cầu mở tâm $$\mathbf{x}$$ đều giao cả $$S$$ lẫn phần bù $$S^c$$:

$$\partial S = \{\mathbf{x} : \forall \epsilon > 0, B(\mathbf{x}, \epsilon) \cap S \neq \emptyset \text{ và } B(\mathbf{x}, \epsilon) \cap S^c \neq \emptyset\}$$

### Phần trong

**Phần trong** (*interior*) của tập $$S$$, ký hiệu $$S^\circ$$ hoặc $$\text{int}(S)$$, gồm mọi điểm nằm “bên trong” tập một cách chặt, không gồm biên:

$$S^\circ = \{\mathbf{x} \in S : \exists \epsilon > 0, B(\mathbf{x}, \epsilon) \subseteq S\}$$

### Bao đóng

**Bao đóng** (*closure*) của tập $$S$$, ký hiệu $$\overline{S}$$ hoặc $$\text{cl}(S)$$, là tập đóng nhỏ nhất chứa $$S$$:

$$\overline{S} = S \cup \partial S$$

### Phân tích ví dụ

Với khoảng $$S = [0, 1)$$ trong $$\mathbb{R}$$:
- **Phần trong:** $$S^\circ = (0, 1)$$
- **Biên:** $$\partial S = \{0, 1\}$$
- **Bao đóng:** $$\overline{S} = [0, 1]$$

Với đĩa mở $$S = \{(x, y) : x^2 + y^2 < 1\}$$ trong $$\mathbb{R}^2$$:
- **Phần trong:** $$S^\circ = S$$ (tập đã mở sẵn)
- **Biên:** $$\partial S = \{(x, y) : x^2 + y^2 = 1\}$$ (đường tròn đơn vị)
- **Bao đóng:** $$\overline{S} = \{(x, y) : x^2 + y^2 \leq 1\}$$ (đĩa đơn vị đóng)

---

## Tập compact

**Tập compact** (*compact set*) là một trong những khái niệm quan trọng nhất trong lý thuyết tối ưu hóa.

### Định nghĩa trong không gian Euclid

**Định lý Heine–Borel:** Trong không gian Euclid ($$\mathbb{R}^n$$), một tập compact khi và chỉ khi nó vừa **đóng** vừa **bị chặn**.

- **Bị chặn:** tập $$S$$ bị chặn nếu nó nằm trong một hình cầu mở đủ lớn: $$\exists M > 0, \mathbf{x}_0$$ sao cho $$S \subseteq B(\mathbf{x}_0, M)$$
- **Đóng:** như định nghĩa ở trên

### Ví dụ về tập compact

**Trong $$\mathbb{R}$$:**
- $$[a, b]$$ (mọi đoạn đóng, bị chặn)
- $$\{0\}$$ (điểm đơn)
- Mọi tập hữu hạn

**Trong $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 \leq 1\}$$ (đĩa đơn vị đóng)
- $$[0, 1] \times [0, 1]$$ (hình vuông đơn vị)
- Mọi tập hữu hạn các điểm

**Trong $$\mathbb{R}^n$$:**
- Mọi hình cầu đóng $$\overline{B}(\mathbf{x}_0, r)$$
- Mọi hình hộp chữ nhật đóng, bị chặn $$[a_1, b_1] \times [a_2, b_2] \times \cdots \times [a_n, b_n]$$

### Các tập không compact

- $$(0, 1)$$ (bị chặn nhưng không đóng)
- $$[0, \infty)$$ (đóng nhưng không bị chặn)
- $$\mathbb{R}^n$$ (không bị chặn)
- $$\{1, 1/2, 1/3, 1/4, \ldots\}$$ (bị chặn nhưng không đóng, vì $$0$$ là điểm giới hạn không thuộc tập)

---

## Tính liên tục của hàm số

### Liên tục tại một điểm

Hàm $$f: A \to \mathbb{R}$$ **liên tục tại điểm** $$\mathbf{c} \in A$$ nếu với mọi $$\varepsilon > 0$$, tồn tại $$\delta > 0$$ sao cho với mọi $$\mathbf{x} \in A$$:

$$\|\mathbf{x} - \mathbf{c}\| < \delta \implies |f(\mathbf{x}) - f(\mathbf{c})| < \varepsilon$$

**Ý nghĩa trực giác:** thay đổi nhỏ của đầu vào dẫn đến thay đổi nhỏ của đầu ra.

### Liên tục toàn cục

$$f$$ **liên tục trên $$A$$** nếu nó liên tục tại mọi điểm của $$A$$.

### Đặc trưng theo dãy

$$f$$ liên tục tại $$\mathbf{c}$$ khi và chỉ khi với mọi dãy $$(\mathbf{x}_n)$$ trong $$A$$ hội tụ về $$\mathbf{c}$$:

$$\lim_{n \to \infty} f(\mathbf{x}_n) = f(\mathbf{c})$$

---

## Các định lý quan trọng cho tối ưu hóa

### Định lý giá trị cực trị (*Extreme Value Theorem*)

**Nếu $$f$$ liên tục trên tập compact $$K$$ thì $$f$$ đạt cực đại và cực tiểu trên $$K$$.**

Đây là định lý nền tảng trong tối ưu hóa: nó đảm bảo rằng hàm mục tiêu liên tục có nghiệm tối ưu trên vùng khả thi compact.

**Ý tưởng chứng minh:** tính compact bảo đảm rằng supremum và infimum của $$f$$ trên $$K$$ thực sự đạt được tại các điểm thuộc $$K$$.

### Định lý giá trị trung gian (*Intermediate Value Theorem*)

**Nếu $$f$$ liên tục trên $$[a, b]$$ và $$y$$ nằm giữa $$f(a)$$ và $$f(b)$$, thì tồn tại $$c \in [a, b]$$ sao cho $$f(c) = y$$.**

Định lý này giúp thiết lập sự tồn tại nghiệm của phương trình $$f(x) = 0$$.

### Định lý Bolzano–Weierstrass

**Mọi dãy bị chặn trong $$\mathbb{R}^n$$ đều có dãy con hội tụ.**

Định lý này then chốt khi chứng minh hội tụ của các thuật toán tối ưu.

### Định lý xấp xỉ Weierstrass

**Mọi hàm liên tục trên đoạn đóng đều có thể được xấp xỉ đều bởi đa thức.**

Định lý này biện minh cho việc dùng xấp xỉ đa thức trong các thuật toán tối ưu.

---

## Ứng dụng trong tối ưu hóa

### 1. Sự tồn tại nghiệm

**Vùng khả thi compact bảo đảm tồn tại nghiệm tối ưu:**
- Nếu vùng khả thi $$S$$ compact và hàm mục tiêu $$f$$ liên tục, thì bài toán tối ưu $$\min_{\mathbf{x} \in S} f(\mathbf{x})$$ có nghiệm.

### 2. Điều kiện quy chuẩn ràng buộc (*constraint qualification*)

Hiểu các tính chất tôpô của tập ràng buộc:
- **Điểm chính quy:** các điểm tại đó gradient của ràng buộc độc lập tuyến tính
- **Phương pháp điểm trong:** đòi hỏi vùng khả thi có phần trong khác rỗng

### 3. Phân tích hội tụ

Phân tích liệu thuật toán tối ưu có hội tụ hay không:
- **Tập đóng:** bảo đảm các điểm giới hạn của dãy hội tụ vẫn khả thi
- **Tính compact:** bảo đảm tồn tại dãy con hội tụ

### 4. Cực trị địa phương và toàn cục

Dùng lân cận để định nghĩa tính tối ưu:
- **Cực tiểu địa phương:** $$f(\mathbf{x}^*) \leq f(\mathbf{x})$$ với mọi $$\mathbf{x}$$ trong một lân cận nào đó của $$\mathbf{x}^*$$
- **Cực tiểu toàn cục:** $$f(\mathbf{x}^*) \leq f(\mathbf{x})$$ với mọi $$\mathbf{x}$$ trong vùng khả thi

### 5. Phân tích vùng khả thi

Xác định tính chất của tập ràng buộc:
- **Ràng buộc tuyến tính:** xác định các tập đóng (nửa không gian)
- **Ràng buộc phi tuyến:** có thể tạo ra các tập vừa không mở vừa không đóng
- **Vùng khả thi compact:** bảo đảm tồn tại nghiệm tối ưu

### Ví dụ: Tối ưu danh mục đầu tư

Xét bài toán cực tiểu hóa rủi ro danh mục với các ràng buộc:

$$\begin{align}
\min_{\mathbf{w}} \quad & \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \\
\text{s.t.} \quad & \mathbf{1}^T \mathbf{w} = 1 \\
& \mathbf{w} \geq \mathbf{0}
\end{align}$$

Vùng khả thi $$S = \{\mathbf{w} : \mathbf{1}^T \mathbf{w} = 1, \mathbf{w} \geq \mathbf{0}\}$$ là:
- **Đóng:** giao của các tập đóng
- **Bị chặn:** ràng buộc $$\mathbf{1}^T \mathbf{w} = 1$$ cùng với $$\mathbf{w} \geq \mathbf{0}$$ chặn vùng khả thi
- **Compact:** vừa đóng vừa bị chặn trong $$\mathbb{R}^n$$

Vì hàm mục tiêu $$\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$ liên tục và $$S$$ compact, Định lý giá trị cực trị bảo đảm tồn tại danh mục tối ưu.

Hiểu tôpô và giải tích thực cung cấp nền tảng chặt chẽ để chứng minh sự tồn tại nghiệm của bài toán tối ưu và khả năng thuật toán tìm được nghiệm đó. Các khái niệm này thiết yếu cho cả phân tích lý thuyết lẫn thiết kế thuật toán thực tiễn.
