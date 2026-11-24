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

Trong bảo mật AI hiện đại, một bức ảnh gấu trúc chỉ cần cộng thêm một nhiễu vô hình có chuẩn nhỏ hơn $$\varepsilon = 8/255$$ đã có thể đánh lừa mạng nơ-ron nhận nhầm thành vượn cáo. Câu hỏi *"tất cả những bức ảnh nằm trong bán kính $$\varepsilon$$ quanh ảnh gốc có hình thù thế nào?"* chính là một câu hỏi của **tôpô trong không gian metric**. Bằng cách hiểu rõ hình cầu mở, tập đóng, điểm biên, tính lồi và tính compact, bạn sẽ nhìn thấy khung hình học đứng sau cả tối ưu hóa lẫn độ bền vững của mạng nơ-ron.

## Bản đồ kiến thức nền: Bốn khái niệm Tôpô thiết yếu trong Giải tích Thực

1. **Không gian Metric và Hình cầu mở (Metric Spaces & Open Balls)**:
   Tổng quát hóa khái niệm khoảng cách (Euclid $$L_2$$, Manhattan $$L_1$$, Chebyshev $$L_\infty$$). Mỗi chuẩn tạo ra một "hình cầu" khác nhau: hình tròn ($$L_2$$), hình thoi ($$L_1$$), và hình vuông/siêu lập phương ($$L_\infty$$—chuẩn phổ biến nhất trong tấn công đối kháng).

2. **Tập mở, Tập đóng và Điểm biên (Open, Closed, Boundary)**:
   Tập mở cho phép di chuyển tự do một chút theo mọi hướng mà không rời khỏi tập (điều kiện để tính đạo hàm nội điểm); tập đóng chứa trọn đường biên của nó nên mọi dãy hội tụ trong tập không bị "rơi ra ngoài".

3. **Tính Compact (Định lý Heine–Borel trong $$\mathbb{R}^n$$)**:
   Tập đóng và bị chặn đảm bảo mọi dãy đều có dãy con hội tụ và mọi hàm liên tục đều đạt cực tiểu/cực đại toàn cục trên tập đó.

4. **Liên hệ với Tối ưu hóa Lồi và Regularization**:
   Ràng buộc trọng số $$\|\mathbf{w}\|_2 \le C$$ hay $$\|\mathbf{w}\|_1 \le C$$ chính là các tập compact lồi; nghiệm tối ưu thường rơi đúng lên **biên** của tập này (và ở các góc nhọn của hình thoi $$L_1$$, tạo nên trọng số bằng $$0$$).

### Khung giải thích (nên đọc chậm): So sánh tay 3 hình cầu bán kính $$r = 1$$ quanh gốc tọa độ

Xét điểm thử $$\mathbf{p} = (0.7, 0.7)$$ trong $$\mathbb{R}^2$$. Hãy kiểm tra xem $$\mathbf{p}$$ có nằm trong hình cầu mở đơn vị $$B(\mathbf{0}, 1) = \{\mathbf{x} : \|\mathbf{x}\| < 1\}$$ ứng với ba chuẩn khác nhau hay không:

| Chuẩn (Metric) | Công thức khoảng cách từ $$\mathbf{0}$$ tới $$\mathbf{p} = (0.7, 0.7)$$ | Kết quả số | Thuộc hình cầu mở $$B(\mathbf{0}, 1)$$? |
| :--- | :--- | :---: | :---: |
| **Chuẩn $$L_1$$ (Manhattan)** | $$\lvert 0.7 \rvert + \lvert 0.7 \rvert$$ | $$1.40$$ | **Không** ($$1.40 > 1$$, nằm ngoài hình thoi) |
| **Chuẩn $$L_2$$ (Euclid)** | $$\sqrt{0.7^2 + 0.7^2} = \sqrt{0.98}$$ | $$\approx 0.990$$ | **Có** ($$0.990 < 1$$, nằm sát biên trong hình tròn) |
| **Chuẩn $$L_\infty$$ (Chebyshev)** | $$\max(\lvert 0.7 \rvert, \lvert 0.7 \rvert)$$ | $$0.70$$ | **Có** ($$0.70 < 1$$, nằm sâu trong hình vuông) |

Từ phép tính nhỏ này ta thấy ngay quan hệ bao hàm tôpô tổng quát trên $$\mathbb{R}^n$$: $$B_{L_1}(\mathbf{0}, 1) \subset B_{L_2}(\mathbf{0}, 1) \subset B_{L_\infty}(\mathbf{0}, 1)$$. Đó cũng là lý do cùng một ngân sách nhiễu $$\varepsilon$$, không gian tấn công theo chuẩn $$L_\infty$$ rộng lớn hơn rất nhiều so với $$L_2$$ và $$L_1$$.

Bài học này trình bày các khái niệm tôpô (*topology*) cốt lõi trong giải tích thực — nền tảng để phân tích cấu trúc miền khả thi, tính liên tục và sự tồn tại nghiệm tối ưu.

---

## Giới thiệu về tôpô

Tôpô nghiên cứu các tính chất của không gian được bảo toàn qua những phép biến dạng liên tục. Trong tối ưu hóa, ngôn ngữ tôpô giúp ta mô tả chính xác cấu trúc của miền khả thi và hành vi của hàm mục tiêu, đặc biệt khi xét sự tồn tại và đặc trưng của nghiệm tối ưu.

### Không gian metric và khoảng cách

Trước khi xét các khái niệm tôpô, ta cần định nghĩa khoảng cách. Trong $$\mathbb{R}^n$$, **khoảng cách Euclid** chuẩn giữa hai điểm $$\mathbf{x}$$ và $$\mathbf{y}$$ là:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

### Hình cầu mở và lân cận

Một **hình cầu mở** (*open ball*) tâm $$\mathbf{x}_0$$ bán kính $$\epsilon > 0$$ được định nghĩa là:

$$B(\mathbf{x}_0, \epsilon) = \{\mathbf{y} \in \mathbb{R}^n : d(\mathbf{x}_0, \mathbf{y}) < \epsilon\}$$

Tập này chứa mọi điểm có khoảng cách tới $$\mathbf{x}_0$$ nhỏ hơn ngặt $$\epsilon$$.

**Ví dụ:**
- Trong $$\mathbb{R}$$: $$B(0, 1) = (-1, 1)$$ (khoảng mở).
- Trong $$\mathbb{R}^2$$: $$B(\mathbf{0}, 1) = \{(x, y) : x^2 + y^2 < 1\}$$ (hình tròn đơn vị mở).

---

## Tập mở

Một **tập mở** (*open set*) được đặc trưng bởi tính chất không chứa bất kỳ điểm biên nào của chính nó.

### Định nghĩa hình thức

Tập $$S \subseteq \mathbb{R}^n$$ được gọi là **tập mở** nếu với mọi điểm $$\mathbf{x} \in S$$, tồn tại $$\epsilon > 0$$ sao cho hình cầu mở $$B(\mathbf{x}, \epsilon)$$ nằm trọn trong $$S$$:

$$\forall \mathbf{x} \in S, \exists \epsilon > 0 : B(\mathbf{x}, \epsilon) \subseteq S$$

### Trực giác

Nếu đứng tại bất kỳ điểm nào thuộc tập mở, ta luôn có thể dịch chuyển một đoạn đủ nhỏ theo mọi hướng mà vẫn nằm bên trong tập đó. Xung quanh mỗi điểm luôn có một "vùng đệm" thuộc tập.

### Ví dụ về tập mở

**Trong $$\mathbb{R}$$:**
- $$(0, 1) = \{x : 0 < x < 1\}$$
- $$(-\infty, 5) = \{x : x < 5\}$$
- Chính $$\mathbb{R}$$

**Trong $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 < 1\}$$ (hình tròn đơn vị mở)
- $$\{(x, y) : x > 0, y > 0\}$$ (góc phần tư thứ nhất, không tính hai trục tọa độ)
- Chính $$\mathbb{R}^2$$

**Trong $$\mathbb{R}^n$$:**
- Mọi hình cầu mở $$B(\mathbf{x}_0, r)$$
- Chính $$\mathbb{R}^n$$
- Tập rỗng $$\emptyset$$ (hiển nhiên thỏa mãn định nghĩa)

### Tính chất của tập mở

1. Hợp tùy ý các tập mở là tập mở.
2. Giao hữu hạn các tập mở là tập mở.
3. $$\mathbb{R}^n$$ và $$\emptyset$$ đều là tập mở.

---

## Tập đóng

Một **tập đóng** (*closed set*) là tập chứa toàn bộ các điểm biên của nó. Nói cách khác, $$S$$ là tập đóng khi và chỉ khi phần bù $$\mathbb{R}^n \setminus S$$ là một **tập mở**.

### Định nghĩa hình thức

Tập $$S$$ là **tập đóng** nếu nó chứa mọi điểm giới hạn của nó: bất cứ khi nào một dãy $$(\mathbf{x}_n)$$ trong $$S$$ hội tụ về $$\mathbf{x}$$, ta đều có $$\mathbf{x} \in S$$:

$$\text{Nếu } \mathbf{x}_n \in S \text{ với mọi } n \text{ và } \lim_{n \to \infty} \mathbf{x}_n = \mathbf{x}, \text{ thì } \mathbf{x} \in S$$

### Ví dụ về tập đóng

**Trong $$\mathbb{R}$$:**
- $$[0, 1] = \{x : 0 \leq x \leq 1\}$$
- $$[a, \infty) = \{x : x \geq a\}$$
- $$\{0\}$$ (tập một điểm)
- $$\mathbb{Z}$$ (tập số nguyên)

**Trong $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 \leq 1\}$$ (hình tròn đơn vị đóng)
- $$\{(x, y) : x \geq 0, y \geq 0\}$$ (góc phần tư thứ nhất, bao gồm cả hai trục tọa độ)
- $$\{(0, 0)\}$$ (tập một điểm)

**Trong $$\mathbb{R}^n$$:**
- Mọi hình cầu đóng $$\overline{B}(\mathbf{x}_0, r) = \{\mathbf{x} : d(\mathbf{x}, \mathbf{x}_0) \leq r\}$$
- Chính $$\mathbb{R}^n$$
- Tập rỗng $$\emptyset$$
- Mọi tập hữu hạn

### Tính chất của tập đóng

1. Giao tùy ý các tập đóng là tập đóng.
2. Hợp hữu hạn các tập đóng là tập đóng.
3. $$\mathbb{R}^n$$ và $$\emptyset$$ đều là tập đóng.

### Lưu ý quan trọng

Một tập hợp có thể rơi vào một trong bốn trường hợp:
- **Mở nhưng không đóng:** $$(0, 1)$$
- **Đóng nhưng không mở:** $$[0, 1]$$
- **Vừa mở vừa đóng:** $$\mathbb{R}^n$$, $$\emptyset$$
- **Không mở cũng không đóng:** $$[0, 1)$$, $$(0, 1]$$

---

## Biên, phần trong và bao đóng

### Biên

**Biên** (*boundary*) của tập $$S$$, ký hiệu $$\partial S$$, là tập hợp các điểm nằm trên "mép" của $$S$$. Điểm $$\mathbf{x}$$ là **điểm biên** của $$S$$ nếu mọi hình cầu mở tâm $$\mathbf{x}$$ đều giao đồng thời với cả $$S$$ và phần bù $$S^c$$:

$$\partial S = \{\mathbf{x} : \forall \epsilon > 0, B(\mathbf{x}, \epsilon) \cap S \neq \emptyset \text{ và } B(\mathbf{x}, \epsilon) \cap S^c \neq \emptyset\}$$

### Phần trong

**Phần trong** (*interior*) của tập $$S$$, ký hiệu $$S^\circ$$ hoặc $$\text{int}(S)$$, gồm mọi điểm nằm hoàn toàn bên trong $$S$$ (không nằm trên biên):

$$S^\circ = \{\mathbf{x} \in S : \exists \epsilon > 0, B(\mathbf{x}, \epsilon) \subseteq S\}$$

### Bao đóng

**Bao đóng** (*closure*) của tập $$S$$, ký hiệu $$\overline{S}$$ hoặc $$\text{cl}(S)$$, là tập đóng nhỏ nhất chứa $$S$$:

$$\overline{S} = S \cup \partial S$$

### Ví dụ minh họa

Với nửa khoảng $$S = [0, 1)$$ trong $$\mathbb{R}$$:
- **Phần trong:** $$S^\circ = (0, 1)$$
- **Biên:** $$\partial S = \{0, 1\}$$
- **Bao đóng:** $$\overline{S} = [0, 1]$$

Với hình tròn mở $$S = \{(x, y) : x^2 + y^2 < 1\}$$ trong $$\mathbb{R}^2$$:
- **Phần trong:** $$S^\circ = S$$ (vì $$S$$ vốn là tập mở)
- **Biên:** $$\partial S = \{(x, y) : x^2 + y^2 = 1\}$$ (đường tròn đơn vị)
- **Bao đóng:** $$\overline{S} = \{(x, y) : x^2 + y^2 \leq 1\}$$ (hình tròn đơn vị đóng)

---

## Tập compact

**Tập compact** (*compact set*) là một trong những khái niệm trung tâm của lý thuyết tối ưu hóa.

### Đặc trưng trong không gian Euclid

**Định lý Heine–Borel:** Trong không gian Euclid $$\mathbb{R}^n$$, một tập hợp là compact khi và chỉ khi nó vừa **đóng** vừa **bị chặn**.

- **Bị chặn:** tập $$S$$ bị chặn nếu nó nằm trọn trong một hình cầu mở đủ lớn: $$\exists M > 0, \mathbf{x}_0$$ sao cho $$S \subseteq B(\mathbf{x}_0, M)$$.
- **Đóng:** như đã định nghĩa ở mục trên.

### Ví dụ về tập compact

**Trong $$\mathbb{R}$$:**
- $$[a, b]$$ (mọi đoạn đóng, bị chặn)
- $$\{0\}$$ (tập một điểm)
- Mọi tập hữu hạn

**Trong $$\mathbb{R}^2$$:**
- $$\{(x, y) : x^2 + y^2 \leq 1\}$$ (hình tròn đơn vị đóng)
- $$[0, 1] \times [0, 1]$$ (hình vuông đơn vị đóng)
- Mọi tập hữu hạn điểm

**Trong $$\mathbb{R}^n$$:**
- Mọi hình cầu đóng $$\overline{B}(\mathbf{x}_0, r)$$
- Mọi hình hộp chữ nhật đóng, bị chặn $$[a_1, b_1] \times [a_2, b_2] \times \cdots \times [a_n, b_n]$$

### Các tập không compact

- $$(0, 1)$$ (bị chặn nhưng không đóng)
- $$[0, \infty)$$ (đóng nhưng không bị chặn)
- $$\mathbb{R}^n$$ (không bị chặn)
- $$\{1, 1/2, 1/3, 1/4, \ldots\}$$ (bị chặn nhưng không đóng, vì điểm giới hạn $$0$$ không thuộc tập)

---

## Tính liên tục của hàm số

### Liên tục tại một điểm

Hàm số $$f: A \to \mathbb{R}$$ **liên tục tại điểm** $$\mathbf{c} \in A$$ nếu với mọi $$\varepsilon > 0$$, tồn tại $$\delta > 0$$ sao cho với mọi $$\mathbf{x} \in A$$:

$$\|\mathbf{x} - \mathbf{c}\| < \delta \implies |f(\mathbf{x}) - f(\mathbf{c})| < \varepsilon$$

**Ý nghĩa trực quan:** thay đổi nhỏ ở đầu vào chỉ gây ra thay đổi nhỏ ở đầu ra.

### Liên tục trên tập hợp

Hàm $$f$$ **liên tục trên $$A$$** nếu nó liên tục tại mọi điểm thuộc $$A$$.

### Đặc trưng theo dãy

Hàm $$f$$ liên tục tại $$\mathbf{c}$$ khi và chỉ khi với mọi dãy $$(\mathbf{x}_n)$$ trong $$A$$ hội tụ về $$\mathbf{c}$$:

$$\lim_{n \to \infty} f(\mathbf{x}_n) = f(\mathbf{c})$$

---

## Các định lý quan trọng cho tối ưu hóa

### Định lý giá trị cực trị (*Extreme Value Theorem*)

**Nếu $$f$$ liên tục trên tập compact khác rỗng $$K$$, thì $$f$$ đạt cả giá trị lớn nhất và giá trị nhỏ nhất trên $$K$$.**

Đây là định lý nền tảng của tối ưu hóa vì nó bảo đảm mọi hàm mục tiêu liên tục trên miền khả thi compact đều có nghiệm tối ưu.

**Ý tưởng chứng minh:** tính compact bảo đảm supremum và infimum của $$f$$ trên $$K$$ thực sự đạt được tại các điểm nằm trong $$K$$.

### Định lý giá trị trung gian (*Intermediate Value Theorem*)

**Nếu $$f$$ liên tục trên đoạn $$[a, b]$$ và $$y$$ nằm giữa $$f(a)$$ và $$f(b)$$, thì tồn tại $$c \in [a, b]$$ sao cho $$f(c) = y$$.**

Định lý này thường được dùng để chứng minh sự tồn tại nghiệm của phương trình $$f(x) = 0$$.

### Định lý Bolzano–Weierstrass

**Mọi dãy bị chặn trong $$\mathbb{R}^n$$ đều trích ra được ít nhất một dãy con hội tụ.**

Đây là công cụ then chốt trong các chứng minh hội tụ của thuật toán tối ưu.

### Định lý xấp xỉ Weierstrass

**Mọi hàm liên tục trên một đoạn đóng bị chặn đều có thể xấp xỉ đều với độ chính xác tùy ý bởi đa thức.**

Định lý này tạo cơ sở lý thuyết cho việc sử dụng xấp xỉ đa thức trong các thuật toán tối ưu hóa.

---

## Ứng dụng trong tối ưu hóa

### 1. Sự tồn tại nghiệm

**Miền khả thi compact bảo đảm tồn tại nghiệm tối ưu:** nếu miền khả thi $$S$$ là tập compact khác rỗng và hàm mục tiêu $$f$$ liên tục, thì bài toán $$\min_{\mathbf{x} \in S} f(\mathbf{x})$$ chắc chắn có nghiệm.

### 2. Điều kiện chính quy ràng buộc (*constraint qualification*)

Các tính chất tôpô của tập ràng buộc quyết định tính hợp lệ của điều kiện tối ưu:
- **Điểm chính quy:** các điểm mà tại đó gradient của các ràng buộc kích hoạt độc lập tuyến tính.
- **Phương pháp điểm trong:** đòi hỏi miền khả thi có phần trong khác rỗng.

### 3. Phân tích hội tụ

Để đánh giá khả năng hội tụ của thuật toán lặp:
- **Tính đóng:** bảo đảm điểm giới hạn của một dãy điểm khả thi vẫn là điểm khả thi.
- **Tính compact:** bảo đảm dãy lặp bị chặn luôn tồn tại dãy con hội tụ.

### 4. Cực trị địa phương và toàn cục

Khái niệm lân cận cho phép định nghĩa chặt chẽ hai loại cực trị:
- **Cực tiểu địa phương:** $$f(\mathbf{x}^*) \leq f(\mathbf{x})$$ với mọi $$\mathbf{x}$$ khả thi nằm trong một lân cận đủ nhỏ của $$\mathbf{x}^*$$.
- **Cực tiểu toàn cục:** $$f(\mathbf{x}^*) \leq f(\mathbf{x})$$ với mọi $$\mathbf{x}$$ thuộc toàn bộ miền khả thi.

### 5. Phân tích miền khả thi

Tính chất tôpô của miền khả thi phụ thuộc trực tiếp vào dạng ràng buộc:
- **Ràng buộc tuyến tính (bất đẳng thức yếu):** xác định các nửa không gian đóng.
- **Ràng buộc phi tuyến:** có thể tạo ra tập không đóng hoặc không mở tùy theo dạng bất đẳng thức.
- **Miền khả thi compact:** điều kiện đủ để hàm liên tục đạt cực trị.

### Ví dụ: Tối ưu danh mục đầu tư

Xét bài toán cực tiểu hóa rủi ro danh mục với các ràng buộc:

$$\begin{align}
\min_{\mathbf{w}} \quad & \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \\
\text{s.t.} \quad & \mathbf{1}^T \mathbf{w} = 1 \\
& \mathbf{w} \geq \mathbf{0}
\end{align}$$

Miền khả thi $$S = \{\mathbf{w} : \mathbf{1}^T \mathbf{w} = 1, \mathbf{w} \geq \mathbf{0}\}$$ có các tính chất:
- **Đóng:** là giao của các tập đóng.
- **Bị chặn:** ràng buộc $$\mathbf{1}^T \mathbf{w} = 1$$ kết hợp với $$\mathbf{w} \geq \mathbf{0}$$ suy ra mọi thành phần nằm trong $$[0, 1]$$.
- **Compact:** vừa đóng vừa bị chặn trong $$\mathbb{R}^n$$.

Vì hàm mục tiêu $$\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$ liên tục và $$S$$ compact, Định lý giá trị cực trị bảo đảm luôn tồn tại một danh mục tối ưu.

Các khái niệm tôpô và giải tích thực cung cấp cơ sở chặt chẽ để chứng minh sự tồn tại nghiệm cũng như phân tích hội tụ cho cả lý thuyết lẫn thuật toán tối ưu hóa thực tiễn.

## Video tham khảo

- **Xem trước:** [Intro to Open Sets (with Examples) | Real Analysis](https://www.youtube.com/watch?v=pnWgj8jjs3w) — Wrath of Math, tiếng Anh. Trình bày định nghĩa tập mở bằng hình cầu mở quanh từng điểm, khớp với phần mở đầu của bài.
- [Identifying Open, Closed, and Compact Sets | Real Analysis Exercises](https://www.youtube.com/watch?v=0_mottmadEU) — Wrath of Math, tiếng Anh. Hướng dẫn qua ví dụ cách phân biệt tập mở, tập đóng và tập compact, kèm khái niệm biên và bao đóng.
- [Open Covers, Finite Subcovers, and Compact Sets | Real Analysis](https://www.youtube.com/watch?v=dAz9ax-bjzU) — Wrath of Math, tiếng Anh. Giới thiệu định nghĩa tính compact qua phủ mở — nền tảng cho các định lý cực trị trong bài.
