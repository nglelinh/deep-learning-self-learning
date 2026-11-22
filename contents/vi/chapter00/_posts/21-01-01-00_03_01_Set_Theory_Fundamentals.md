---
layout: post
title: 00-03-01 Cơ sở Lý thuyết tập hợp
chapter: '00'
order: 12
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Bài học này trình bày các khái niệm cơ bản của lý thuyết tập hợp, làm nền tảng toán học để phát biểu bài toán tối ưu hóa, ràng buộc và miền khả thi.

---

## Giới thiệu về lý thuyết tập hợp

Lý thuyết tập hợp là nền móng của toán học hiện đại và đóng vai trò thiết yếu khi mô tả các bài toán tối ưu hóa. Một **tập hợp** là một bộ sưu tập các đối tượng phân biệt, gọi là các phần tử.

### Ký hiệu cơ bản

- **Ký hiệu liệt kê:** $$A = \{1, 2, 3, 4\}$$
- **Quan hệ thuộc:** $$x \in A$$ ($$x$$ thuộc $$A$$) hoặc $$x \notin A$$ ($$x$$ không thuộc $$A$$)
- **Tập rỗng:** $$\emptyset = \{\}$$ (tập hợp không chứa phần tử nào)
- **Ký hiệu theo tính chất đặc trưng:** $$A = \{x : P(x)\}$$ (tập hợp mọi phần tử $$x$$ thỏa mãn tính chất $$P(x)$$)

### Ví dụ

- $$A = \{1, 2, 3, 4\}$$ (liệt kê tường minh)
- $$B = \{x \in \mathbb{R} : x^2 < 4\} = (-2, 2)$$ (chỉ ra tính chất đặc trưng)
- $$C = \{x \in \mathbb{Z} : x \text{ là số chẵn}\}$$ (tập hợp các số nguyên chẵn)

---

## Các phép toán tập hợp cơ bản

### Tập hợp và tập con

- **Tập con:** $$A \subseteq B$$ nghĩa là mọi phần tử của $$A$$ đều thuộc $$B$$.
- **Tập con thực sự:** $$A \subset B$$ nghĩa là $$A \subseteq B$$ và $$A \neq B$$.
- **Hai tập hợp bằng nhau:** $$A = B$$ khi và chỉ khi $$A \subseteq B$$ và $$B \subseteq A$$.

**Ví dụ:**
- $$\{1, 2\} \subseteq \{1, 2, 3, 4\}$$
- $$\{1, 2\} \subset \{1, 2, 3, 4\}$$
- $$\emptyset \subseteq A$$ với mọi tập hợp $$A$$

### Phép hợp và phép giao

**Hợp ($$A \cup B$$):** tập hợp tất cả các phần tử thuộc $$A$$ hoặc thuộc $$B$$ (hoặc cả hai):
$$A \cup B = \{x : x \in A \text{ hoặc } x \in B\}$$

**Giao ($$A \cap B$$):** tập hợp tất cả các phần tử thuộc đồng thời cả $$A$$ và $$B$$:
$$A \cap B = \{x : x \in A \text{ và } x \in B\}$$

**Ví dụ:**
- Nếu $$A = \{1, 2, 3\}$$ và $$B = \{3, 4, 5\}$$, thì:
  - $$A \cup B = \{1, 2, 3, 4, 5\}$$
  - $$A \cap B = \{3\}$$

**Hai tập hợp rời nhau:** $$A$$ và $$B$$ rời nhau nếu $$A \cap B = \emptyset$$.

### Phần bù và hiệu tập hợp

**Phần bù ($$A^c$$):** tập hợp tất cả các phần tử không thuộc $$A$$ (trong một tập vũ trụ $$U$$ cho trước):
$$A^c = \{x \in U : x \notin A\}$$

**Hiệu tập hợp ($$A \setminus B$$):** tập hợp các phần tử thuộc $$A$$ nhưng không thuộc $$B$$:
$$A \setminus B = \{x : x \in A \text{ và } x \notin B\}$$

**Hiệu đối xứng:** $$(A \setminus B) \cup (B \setminus A) = (A \cup B) \setminus (A \cap B)$$

### Các luật đại số tập hợp

**Luật giao hoán:**
- $$A \cup B = B \cup A$$
- $$A \cap B = B \cap A$$

**Luật kết hợp:**
- $$(A \cup B) \cup C = A \cup (B \cup C)$$
- $$(A \cap B) \cap C = A \cap (B \cap C)$$

**Luật phân phối:**
- $$A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$$
- $$A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$$

**Luật De Morgan:**
- $$(A \cup B)^c = A^c \cap B^c$$
- $$(A \cap B)^c = A^c \cup B^c$$

---

## Các tập hợp số quan trọng

Nắm rõ hệ thống phân cấp các tập số giúp xác định chính xác miền biến thiên của bài toán tối ưu:

### Số tự nhiên
$$\mathbb{N} = \{1, 2, 3, 4, \ldots\}$$
(Đôi khi quy ước gồm cả số 0: $$\mathbb{N}_0 = \{0, 1, 2, 3, \ldots\}$$)

### Số nguyên
$$\mathbb{Z} = \{\ldots, -2, -1, 0, 1, 2, \ldots\}$$

### Số hữu tỉ
$$\mathbb{Q} = \left\{\frac{p}{q} : p, q \in \mathbb{Z}, q \neq 0\right\}$$

Gồm tất cả các số biểu diễn được dưới dạng phân số của hai số nguyên.

### Số thực
$$\mathbb{R}$$ bao gồm tất cả các số hữu tỉ và số vô tỉ (như $$\pi$$, $$e$$, $$\sqrt{2}$$).

### Số phức
$$\mathbb{C} = \{a + bi : a, b \in \mathbb{R}, i^2 = -1\}$$

**Quan hệ bao hàm:** $$\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$$

---

## Hàm số: Miền xác định và miền giá trị

Một **hàm số** $$f: A \to B$$ là quy tắc gán cho mỗi phần tử thuộc tập $$A$$ đúng một phần tử thuộc tập $$B$$.

### Các khái niệm chính

- **Miền xác định (*domain*):** tập hợp $$A$$ gồm tất cả các giá trị đầu vào hợp lệ.
- **Tập đích (*codomain*):** tập hợp $$B$$ chứa các giá trị đầu ra có thể có.
- **Miền giá trị / Ảnh (*range/image*):** tập hợp tất cả các giá trị đầu ra thực sự nhận được: $$\text{Range}(f) = \{f(x) : x \in A\} \subseteq B$$.

**Ví dụ:** Với $$f(x) = x^2$$ xét như ánh xạ $$f: \mathbb{R} \to \mathbb{R}$$:
- Miền xác định: $$\mathbb{R}$$
- Tập đích: $$\mathbb{R}$$
- Miền giá trị: $$[0, \infty)$$

### Phân loại ánh xạ

**Đơn ánh (*injective*):** hai đầu vào khác nhau luôn cho hai đầu ra khác nhau:
$$f(x_1) = f(x_2) \implies x_1 = x_2$$

**Toàn ánh (*surjective*):** mọi phần tử của tập đích đều có ít nhất một tạo ảnh trong miền xác định:
Với mọi $$y \in B$$, tồn tại $$x \in A$$ sao cho $$f(x) = y$$.

**Song ánh (*bijective*):** ánh xạ vừa đơn ánh vừa toàn ánh, thiết lập tương ứng một-một giữa miền xác định và tập đích.

**Ví dụ:**
- $$f(x) = 2x$$ từ $$\mathbb{R}$$ vào $$\mathbb{R}$$ là song ánh.
- $$f(x) = x^2$$ từ $$\mathbb{R}$$ vào $$\mathbb{R}$$ không đơn ánh và cũng không toàn ánh.
- $$f(x) = x^2$$ từ $$[0, \infty)$$ vào $$[0, \infty)$$ là song ánh.

---

## Bất đẳng thức

Bất đẳng thức xuất hiện rất phổ biến trong tối ưu hóa vì phần lớn ràng buộc thực tế được biểu diễn dưới dạng bất đẳng thức.

### Các ký hiệu cơ bản

- $$a < b$$: $$a$$ nhỏ hơn ngặt $$b$$
- $$a \leq b$$: $$a$$ nhỏ hơn hoặc bằng $$b$$
- $$a > b$$: $$a$$ lớn hơn ngặt $$b$$
- $$a \geq b$$: $$a$$ lớn hơn hoặc bằng $$b$$

### Tính chất của bất đẳng thức

1. **Tính bắc cầu:** nếu $$a \leq b$$ và $$b \leq c$$, thì $$a \leq c$$.
2. **Phép cộng:** nếu $$a \leq b$$, thì $$a + c \leq b + c$$ với mọi $$c \in \mathbb{R}$$.
3. **Nhân với số dương:** nếu $$a \leq b$$ và $$c > 0$$, thì $$ac \leq bc$$.
4. **Nhân với số âm:** nếu $$a \leq b$$ và $$c < 0$$, thì $$ac \geq bc$$ (bất đẳng thức đổi chiều).

### Ký hiệu khoảng và đoạn

- **Khoảng mở:** $$(a, b) = \{x \in \mathbb{R} : a < x < b\}$$
- **Đoạn đóng:** $$[a, b] = \{x \in \mathbb{R} : a \leq x \leq b\}$$
- **Nửa khoảng:** $$[a, b)$$, $$(a, b]$$
- **Khoảng không bị chặn:** $$(-\infty, a)$$, $$[a, \infty)$$, $$(-\infty, \infty) = \mathbb{R}$$

---

## Ứng dụng trong tối ưu hóa

Các khái niệm của lý thuyết tập hợp được dùng trực tiếp trong việc mô hình hóa bài toán tối ưu:

### 1. Miền khả thi

**Miền khả thi** (*feasible region*) là tập hợp tất cả các điểm thỏa mãn toàn bộ ràng buộc của bài toán:
$$S = \{x \in \mathbb{R}^n : g_i(x) \leq 0, i = 1, \ldots, m; h_j(x) = 0, j = 1, \ldots, p\}$$

### 2. Tập mức

Với hàm số $$f: \mathbb{R}^n \to \mathbb{R}$$, **tập mức** tại mức $$c$$ là:
$$L_c = \{x \in \mathbb{R}^n : f(x) = c\}$$

### 3. Tính chất của tập ràng buộc

Việc tập ràng buộc có các tính chất tôpô thuận lợi (như tính đóng hay có phần trong khác rỗng) ảnh hưởng trực tiếp đến sự tồn tại và đặc trưng của nghiệm tối ưu.

### 4. Phân tích hội tụ

Khái niệm dãy và giới hạn trên tập hợp là công cụ cần thiết để kiểm tra xem thuật toán tối ưu có hội tụ về nghiệm mong muốn hay không.

### 5. Phép toán tập hợp trong biểu diễn bài toán

- **Giao:** xác định tập các điểm thỏa mãn đồng thời nhiều ràng buộc.
- **Hợp:** gộp các miền khả thi từ nhiều kịch bản khác nhau.
- **Phần bù:** xác định miền không khả thi.

**Ví dụ:** Trong quy hoạch tuyến tính, miền khả thi có dạng:
$$S = \{x \in \mathbb{R}^n : Ax \leq b, x \geq 0\} = \bigcap_{i=1}^{m} \{x : a_i^T x \leq b_i\} \cap \{x : x \geq 0\}$$

Đây là giao của các nửa không gian, minh họa cách các phép toán tập hợp xuất hiện tự nhiên khi phát biểu bài toán tối ưu.

Nắm vững lý thuyết tập hợp giúp đặt bài toán tối ưu hóa một cách chặt chẽ và phân tích tính chất của bài toán một cách có hệ thống.

## Video tham khảo

- **Xem trước:** [Intro to Sets | Examples, Notation & Properties](https://www.youtube.com/watch?v=B1v2-nGXNzs) — Dr. Trefor Bazett, tiếng Anh. Giới thiệu ký hiệu tập hợp, các phép toán cơ bản cùng miền xác định và miền giá trị của hàm số.
- [INTRODUCTION to SET THEORY - DISCRETE MATHEMATICS](https://www.youtube.com/watch?v=tyDKR4FG3Yw) — TrevTutor, tiếng Anh. Trình bày đầy đủ hơn về phép hợp, phép giao và các tập số xuất hiện trong bài.
- [MaDL - Sets, Scalars, Vectors, Matrices and Tensors](https://www.youtube.com/watch?v=JzIjCkouIGg) — Tübingen Machine Learning, tiếng Anh. Kết nối ngôn ngữ tập hợp với vô hướng, vector, ma trận và tensor dùng xuyên suốt khóa học.
