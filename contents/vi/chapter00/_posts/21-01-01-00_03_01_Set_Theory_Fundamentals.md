---
layout: post
title: 00-03-01 Cơ sở Lý thuyết tập hợp
chapter: '00'
order: 12
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

Bài học này bao gồm các khái niệm cơ bản từ lý thuyết tập hợp cung cấp nền tảng toán học để hiểu các bài toán tối ưu hóa, ràng buộc và vùng khả thi.

---

## Giới thiệu về Lý thuyết tập hợp

Lý thuyết tập hợp cung cấp nền tảng cho toán học hiện đại và rất cần thiết để hiểu các khái niệm tối ưu hóa. Một **tập hợp** đơn giản là một bộ sưu tập các đối tượng riêng biệt, được gọi là các phần tử hoặc thành viên.

### Ký hiệu cơ bản

- **Ký hiệu tập hợp:** $$A = \{1, 2, 3, 4\}$$
- **Quan hệ thành viên:** $$x \in A$$ (x thuộc A) hoặc $$x \notin A$$ (x không thuộc A)
- **Tập rỗng:** $$\emptyset = \{\}$$ (tập hợp không có phần tử nào)
- **Ký hiệu xây dựng tập hợp:** $$A = \{x : P(x)\}$$ (tập hợp tất cả x sao cho tính chất P(x) đúng)

### Ví dụ

- $$A = \{1, 2, 3, 4\}$$ (liệt kê tường minh)
- $$B = \{x \in \mathbb{R} : x^2 < 4\} = (-2, 2)$$ (ký hiệu xây dựng tập hợp)
- $$C = \{x \in \mathbb{Z} : x \text{ là số chẵn}\}$$ (tất cả số nguyên chẵn)

---

## Các phép toán tập hợp cơ bản

### Tập hợp và Tập con

- **Tập con:** $$A \subseteq B$$ có nghĩa là mọi phần tử của $$A$$ cũng thuộc $$B$$
- **Tập con thực sự:** $$A \subset B$$ có nghĩa là $$A \subseteq B$$ và $$A \neq B$$
- **Bằng nhau giữa các tập hợp:** $$A = B$$ khi và chỉ khi $$A \subseteq B$$ và $$B \subseteq A$$

**Ví dụ:**
- $$\{1, 2\} \subseteq \{1, 2, 3, 4\}$$
- $$\{1, 2\} \subset \{1, 2, 3, 4\}$$
- $$\emptyset \subseteq A$$ với mọi tập hợp $$A$$

### Hợp và Giao

**Hợp ($$A \cup B$$):** Tất cả các phần tử thuộc $$A$$ hoặc $$B$$ (hoặc cả hai)
$$A \cup B = \{x : x \in A \text{ hoặc } x \in B\}$$

**Giao ($$A \cap B$$):** Tất cả các phần tử thuộc cả $$A$$ và $$B$$
$$A \cap B = \{x : x \in A \text{ và } x \in B\}$$

**Ví dụ:**
- Nếu $$A = \{1, 2, 3\}$$ và $$B = \{3, 4, 5\}$$, thì:
  - $$A \cup B = \{1, 2, 3, 4, 5\}$$
  - $$A \cap B = \{3\}$$

**Tập hợp rời nhau:** $$A$$ và $$B$$ rời nhau nếu $$A \cap B = \emptyset$$

### Phần bù và Hiệu

**Phần bù ($$A^c$$):** Tất cả các phần tử không thuộc $$A$$ (trong một tập hợp toàn thể $$U$$ nào đó)
$$A^c = \{x \in U : x \notin A\}$$

**Hiệu tập hợp ($$A \setminus B$$):** Các phần tử thuộc $$A$$ nhưng không thuộc $$B$$
$$A \setminus B = \{x : x \in A \text{ và } x \notin B\}$$

**Hiệu đối xứng:** $$(A \setminus B) \cup (B \setminus A) = (A \cup B) \setminus (A \cap B)$$

### Các luật tập hợp

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

## Các tập số quan trọng

Hiểu biết về hệ thống phân cấp các tập số là rất quan trọng cho tối ưu hóa:

### Số tự nhiên
$$\mathbb{N} = \{1, 2, 3, 4, \ldots\}$$
(Thỉnh thoảng bao gồm 0: $$\mathbb{N}_0 = \{0, 1, 2, 3, \ldots\}$$)

### Số nguyên
$$\mathbb{Z} = \{\ldots, -2, -1, 0, 1, 2, \ldots\}$$

### Số hữu tỉ
$$\mathbb{Q} = \left\{\frac{p}{q} : p, q \in \mathbb{Z}, q \neq 0\right\}$$

Tất cả các số có thể biểu diễn dưới dạng phân số.

### Số thực
$$\mathbb{R}$$ bao gồm tất cả các số hữu tỉ và vô tỉ (như $$\pi$$, $$e$$, $$\sqrt{2}$$).

### Số phức
$$\mathbb{C} = \{a + bi : a, b \in \mathbb{R}, i^2 = -1\}$$

**Phân cấp:** $$\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$$

---

## Hàm số: Miền xác định và Miền giá trị

Một **hàm số** $$f: A \to B$$ là một quy tắc gán cho mỗi phần tử trong tập hợp $$A$$ đúng một phần tử trong tập hợp $$B$$.

### Các khái niệm chính

- **Miền xác định:** Tập hợp $$A$$ của tất cả các giá trị đầu vào có thể
- **Miền đồng biến:** Tập hợp $$B$$ nơi các giá trị đầu ra được lấy từ
- **Miền giá trị (Ảnh):** Tập hợp tất cả các giá trị đầu ra thực tế: $$\text{Range}(f) = \{f(x) : x \in A\} \subseteq B$$

**Ví dụ:** Với $$f(x) = x^2$$ và $$f: \mathbb{R} \to \mathbb{R}$$:
- Miền xác định: $$\mathbb{R}$$
- Miền đồng biến: $$\mathbb{R}$$
- Miền giá trị: $$[0, \infty)$$

### Các loại hàm số

**Đơn ánh (Một-đến-Một):** Mỗi phần tử trong miền giá trị tương ứng với đúng một phần tử trong miền xác định
$$f(x_1) = f(x_2) \implies x_1 = x_2$$

**Toàn ánh (Lên):** Mọi phần tử trong miền đồng biến đều nằm trong miền giá trị
Với mọi $$y \in B$$, tồn tại $$x \in A$$ sao cho $$f(x) = y$$

**Song ánh:** Vừa là đơn ánh vừa là toàn ánh
Có một tương ứng hoàn hảo một-đến-một giữa miền xác định và miền đồng biến.

**Ví dụ:**
- $$f(x) = 2x$$ trên $$\mathbb{R}$$ là song ánh
- $$f(x) = x^2$$ trên $$\mathbb{R}$$ không là đơn ánh cũng không là toàn ánh
- $$f(x) = x^2$$ trên $$[0, \infty) \to [0, \infty)$$ là song ánh

---

## Bất đẳng thức

Hiểu biết về bất đẳng thức là rất quan trọng cho tối ưu hóa, vì các ràng buộc thường được biểu diễn dưới dạng bất đẳng thức.

### Các ký hiệu bất đẳng thức cơ bản

- $$a < b$$: $$a$$ nhỏ hơn $$b$$ một cách nghiêm ngặt
- $$a \leq b$$: $$a$$ nhỏ hơn hoặc bằng $$b$$
- $$a > b$$: $$a$$ lớn hơn $$b$$ một cách nghiêm ngặt
- $$a \geq b$$: $$a$$ lớn hơn hoặc bằng $$b$$

### Tính chất của bất đẳng thức

1. **Tính bắc cầu:** Nếu $$a \leq b$$ và $$b \leq c$$, thì $$a \leq c$$

2. **Phép cộng:** Nếu $$a \leq b$$, thì $$a + c \leq b + c$$ với mọi $$c$$

3. **Nhân với số dương:** Nếu $$a \leq b$$ và $$c > 0$$, thì $$ac \leq bc$$

4. **Nhân với số âm:** Nếu $$a \leq b$$ và $$c < 0$$, thì $$ac \geq bc$$ (bất đẳng thức đảo chiều!)

### Ký hiệu khoảng

- **Khoảng mở:** $$(a, b) = \{x \in \mathbb{R} : a < x < b\}$$
- **Khoảng đóng:** $$[a, b] = \{x \in \mathbb{R} : a \leq x \leq b\}$$
- **Khoảng nửa mở:** $$[a, b)$$, $$(a, b]$$
- **Khoảng không bị chặn:** $$(-\infty, a)$$, $$[a, \infty)$$, $$(-\infty, \infty) = \mathbb{R}$$

---

## Ứng dụng trong Tối ưu hóa

Các khái niệm lý thuyết tập hợp là cơ bản cho tối ưu hóa:

### 1. Vùng khả thi

**Vùng khả thi** là tập hợp tất cả các điểm thỏa mãn các ràng buộc:
$$S = \{x \in \mathbb{R}^n : g_i(x) \leq 0, i = 1, \ldots, m; h_j(x) = 0, j = 1, \ldots, p\}$$

### 2. Tập mức

Với một hàm số $$f: \mathbb{R}^n \to \mathbb{R}$$, **tập mức** tại mức $$c$$ là:
$$L_c = \{x \in \mathbb{R}^n : f(x) = c\}$$

### 3. Điều kiện ràng buộc

Hiểu biết khi nào các tập ràng buộc có các tính chất "tốt" (như đóng hoặc có phần trong không rỗng) ảnh hưởng đến sự tồn tại và đặc trưng của các nghiệm tối ưu.

### 4. Phân tích hội tụ

Dãy số và giới hạn là thiết yếu để phân tích liệu các thuật toán tối ưu hóa có hội tụ về nghiệm tối ưu hay không.

### 5. Các phép toán tập hợp trong thuật toán

- **Giao:** Tìm các điểm thỏa mãn nhiều ràng buộc
- **Hợp:** Kết hợp các vùng khả thi từ các kịch bản khác nhau
- **Phần bù:** Hiểu biết về các vùng không khả thi

**Ví dụ:** Trong quy hoạch tuyến tính, vùng khả thi là:
$$S = \{x \in \mathbb{R}^n : Ax \leq b, x \geq 0\} = \bigcap_{i=1}^{m} \{x : a_i^T x \leq b_i\} \cap \{x : x \geq 0\}$$

Đây là giao của các nửa không gian, minh họa cách các phép toán tập hợp xuất hiện tự nhiên trong việc xây dựng bài toán tối ưu hóa.

Hiểu biết về lý thuyết tập hợp cung cấp nền tảng toán học chặt chẽ cần thiết để xây dựng các bài toán tối ưu hóa một cách chính xác và phân tích các tính chất của chúng một cách hệ thống.
