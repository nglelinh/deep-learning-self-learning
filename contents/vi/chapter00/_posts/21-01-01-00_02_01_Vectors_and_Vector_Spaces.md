---
layout: post
title: 00-02-01 Vector và Không gian Vector
chapter: '00'
order: 8
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

Bài học này giới thiệu vector, không gian vector, và các khái niệm cơ bản tạo nền tảng để hiểu đại số tuyến tính trong ngữ cảnh tối ưu hóa.

---

## Vector và Không gian Vector ($$\mathbb{R}^n$$)

### Vector là gì?

- **Vector:** Hãy nghĩ về vector như một mũi tên trong không gian, biểu diễn cả hướng và độ lớn (độ dài). Về mặt toán học, nó là một danh sách có thứ tự các số, giống như tọa độ. Ví dụ, một vector trong không gian 2D có thể là $$\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$, có nghĩa là 3 đơn vị dọc theo trục x và 4 đơn vị dọc theo trục y.

- **Góc nhìn Hình học vs Đại số:**
  - **Hình học:** Vector là mũi tên có hướng và độ lớn
  - **Đại số:** Vector là danh sách có thứ tự các số thực

### Không gian Vector

- **Không gian Vector ($$\mathbb{R}^n$$):** Đây là tập hợp tất cả các vector có thể có $$n$$ thành phần (số). Ví dụ, $$\mathbb{R}^2$$ bao gồm tất cả vector 2 thành phần, biểu diễn tất cả các điểm hoặc mũi tên trong mặt phẳng 2D.

- **Ví dụ:**
  - $$\mathbb{R}^2 = \left\{\begin{pmatrix} x \\ y \end{pmatrix} : x, y \in \mathbb{R}\right\}$$ (mặt phẳng)
  - $$\mathbb{R}^3 = \left\{\begin{pmatrix} x \\ y \\ z \end{pmatrix} : x, y, z \in \mathbb{R}\right\}$$ (không gian 3D)

### Các Phép toán Vector

**Phép Cộng Vector:**
$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{pmatrix} + \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

**Phép Nhân Vô hướng:**
$$c\mathbf{v} = c \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{pmatrix}$$

---

## Độc lập Tuyến tính, Cơ sở, và Chiều

### Độc lập Tuyến tính

Một tập hợp vector $$\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$$ là **độc lập tuyến tính** nếu nghiệm duy nhất của:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$

là $$c_1 = c_2 = \cdots = c_k = 0$$.

**Hiểu biết Trực quan:** Một tập hợp vector "độc lập tuyến tính" nếu không có vector nào trong tập có thể được tạo ra bằng cách chia tỷ lệ và cộng các vector khác trong tập. Tất cả chúng đều chỉ theo các hướng "đủ khác nhau".

**Ví dụ trong $$\mathbb{R}^2$$:**
- $$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ và $$\mathbf{v}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ là độc lập tuyến tính
- $$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$ và $$\mathbf{v}_2 = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$$ là phụ thuộc tuyến tính (vì $$\mathbf{v}_2 = 2\mathbf{v}_1$$)

### Cơ sở

Một **cơ sở** cho một không gian vector là một tập hợp tối thiểu các vector độc lập tuyến tính có thể được kết hợp (chia tỷ lệ và cộng) để tạo ra *bất kỳ* vector nào khác trong không gian đó. Nó giống như một tập hợp các khối xây dựng cơ bản.

**Tính chất của một Cơ sở:**
1. Các vector độc lập tuyến tính
2. Chúng sinh ra toàn bộ không gian vector
3. Mọi vector trong không gian có thể được viết duy nhất như một tổ hợp tuyến tính của các vector cơ sở

**Cơ sở Chuẩn cho $$\mathbb{R}^n$$:**
$$\mathbf{e}_1 = \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \mathbf{e}_2 = \begin{pmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{pmatrix}, \ldots, \mathbf{e}_n = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{pmatrix}$$

### Chiều

**Chiều** của một không gian vector đơn giản là số lượng vector trong bất kỳ cơ sở nào của nó. Nó cho bạn biết cần bao nhiêu hướng độc lập để mô tả không gian.

- $$\dim(\mathbb{R}^2) = 2$$
- $$\dim(\mathbb{R}^3) = 3$$
- $$\dim(\mathbb{R}^n) = n$$

---

## Chuẩn của Vector

**Chuẩn** là một hàm gán "độ dài" hoặc "kích thước" cho một vector. Nó tổng quát hóa khái niệm khoảng cách từ gốc tọa độ.

### Tính chất của Chuẩn

Bất kỳ chuẩn $$\|\cdot\|$$ nào cũng phải thỏa mãn ba tính chất:

1. **Không âm:** $$\|\mathbf{x}\| \geq 0$$, và $$\|\mathbf{x}\| = 0$$ khi và chỉ khi $$\mathbf{x} = \mathbf{0}$$
2. **Đồng nhất:** $$\|t\mathbf{x}\| = |t| \|\mathbf{x}\|$$ với bất kỳ vô hướng $$t$$
3. **Bất đẳng thức Tam giác:** $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$$

### Các Chuẩn Thông dụng

**Chuẩn Euclid (Chuẩn L2):**
$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

Đây là khoảng cách "thông thường" mà chúng ta quen thuộc.

**Chuẩn Manhattan (Chuẩn L1):**
$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| = |x_1| + |x_2| + \cdots + |x_n|$$

Còn được gọi là "chuẩn taxi" - khoảng cách mà một chiếc taxi sẽ đi trong thành phố có bố cục dạng lưới.

**Chuẩn Tối đa (Chuẩn L∞):**
$$\|\mathbf{x}\|_\infty = \max_{i} |x_i|$$

Thành phần lớn nhất theo giá trị tuyệt đối.

**Ví dụ:** Với $$\mathbf{x} = \begin{pmatrix} 3 \\ -4 \\ 1 \end{pmatrix}$$:
- $$\|\mathbf{x}\|_2 = \sqrt{3^2 + (-4)^2 + 1^2} = \sqrt{26} \approx 5.1$$
- $$\|\mathbf{x}\|_1 = |3| + |-4| + |1| = 8$$
- $$\|\mathbf{x}\|_\infty = \max\{|3|, |-4|, |1|\} = 4$$

---

## Tích Vô hướng (Tích Chấm)

**Tích chấm** (hoặc tích vô hướng) là cách phổ biến nhất để nhân hai vector, tạo ra kết quả vô hướng.

### Định nghĩa

Với hai vector $$\mathbf{x}$$ và $$\mathbf{y}$$ trong $$\mathbb{R}^n$$:

$$\mathbf{x}^T \mathbf{y} = \mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^n x_i y_i = x_1 y_1 + x_2 y_2 + \cdots + x_n y_n$$

### Diễn giải Hình học

$$\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos \theta$$

trong đó $$\theta$$ là góc giữa các vector.

### Tính chất

1. **Giao hoán:** $$\mathbf{x} \cdot \mathbf{y} = \mathbf{y} \cdot \mathbf{x}$$
2. **Phân phối:** $$\mathbf{x} \cdot (\mathbf{y} + \mathbf{z}) = \mathbf{x} \cdot \mathbf{y} + \mathbf{x} \cdot \mathbf{z}$$
3. **Đồng nhất:** $$(c\mathbf{x}) \cdot \mathbf{y} = c(\mathbf{x} \cdot \mathbf{y})$$

### Trường hợp Đặc biệt

- **Vector trực giao:** $$\mathbf{x} \cdot \mathbf{y} = 0$$ (vuông góc)
- **Vector song song:** $$\mathbf{x} \cdot \mathbf{y} = \pm \|\mathbf{x}\| \|\mathbf{y}\|$$
- **Tích chấm bản thân:** $$\mathbf{x} \cdot \mathbf{x} = \|\mathbf{x}\|_2^2$$

### Ví dụ

Với $$\mathbf{x} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$ và $$\mathbf{y} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$:

$$\mathbf{x}^T \mathbf{y} = (1)(3) + (2)(4) = 3 + 8 = 11$$

---

## Ứng dụng trong Tối ưu hóa

Hiểu về vector và không gian vector là rất quan trọng cho tối ưu hóa vì:

1. **Biến Quyết định:** Các bài toán tối ưu hóa thường liên quan đến việc tìm giá trị tốt nhất cho nhiều biến, được biểu diễn tự nhiên như vector.

2. **Gradient:** Gradient của một hàm là một vector chỉ theo hướng tăng dốc nhất.

3. **Ràng buộc:** Ràng buộc tuyến tính trong tối ưu hóa có thể được biểu diễn bằng tích chấm: $$\mathbf{a}^T \mathbf{x} \leq b$$.

4. **Khoảng cách và Độ tương tự:** Các chuẩn khác nhau cung cấp các cách khác nhau để đo khoảng cách giữa các nghiệm hoặc kích thước của các thay đổi.

5. **Tính trực giao:** Nhiều khái niệm tối ưu hóa dựa vào tính vuông góc, chẳng hạn như mối quan hệ giữa gradient và đường mức.

6. **Tổ hợp Tuyến tính:** Vùng khả thi thường được định nghĩa như tổ hợp tuyến tính của vector (bao lồi, hình nón, v.v.).

Khung không gian vector cung cấp nền tảng toán học để xây dựng và giải quyết các bài toán tối ưu hóa một cách có hệ thống.
