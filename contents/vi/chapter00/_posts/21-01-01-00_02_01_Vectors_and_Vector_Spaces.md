---
layout: post
title: 00-02-01 Vector và Không gian Vector
chapter: '00'
order: 8
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Trong những mô hình ngôn ngữ nổi tiếng, phép toán huyền thoại $$\text{vector}(\text{"Vua"}) - \text{vector}(\text{"Nam"}) + \text{vector}(\text{"Nữ"}) \approx \text{vector}(\text{"Hoàng hậu"})$$ từng khiến cả giới công nghệ kinh ngạc. Điều kỳ diệu ấy không phải phép thuật—nó là hệ quả trực tiếp của cấu trúc **không gian vector**, nơi ngữ nghĩa được ánh xạ thành hướng và khoảng cách hình học. Bài học này xây dựng nền tảng chặt chẽ về vector, tổ hợp tuyến tính, độc lập tuyến tính, chuẩn và tích vô hướng để bạn làm chủ ngôn ngữ biểu diễn dữ liệu của Deep Learning.

## Bản đồ kiến thức nền: Bốn khái niệm xương sống của Không gian Vector

1. **Tổ hợp tuyến tính và Không gian sinh (Linear Combination & Span)**:
   Mô tả tập hợp tất cả các điểm có thể tạo ra từ một bộ vector cơ sở; một lớp mạng tuyến tính thực chất đang lấy tổ hợp tuyến tính các cột của ma trận trọng số.

2. **Độc lập tuyến tính và Số chiều (Linear Independence & Dimension)**:
   Phát hiện sự dư thừa trong dữ liệu: nếu một đặc trưng là tổ hợp tuyến tính của các đặc trưng khác, không gian biểu diễn thực tế có chiều nhỏ hơn số cột bạn tưởng.

3. **Chuẩn vector (Vector Norms — $$L_1, L_2, L_\infty$$)**:
   Định nghĩa "độ dài" và "khoảng cách": chuẩn $$L_2$$ gắn với khoảng cách Euclid và Weight Decay (Ridge), còn chuẩn $$L_1$$ tạo ra tính thưa (*sparsity*), đẩy nhiều trọng số về đúng $$0$$.

4. **Tích vô hướng và Độ tương đồng Cosine (Dot Product & Cosine Similarity)**:
   Liên kết đại số với hình học qua công thức $$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos\theta$$—phép toán trung tâm của cả phép chiếu, phân loại tuyến tính lẫn cơ chế Self-Attention trong Transformer.

### Khung giải thích (nên đọc chậm): Tính tay chuẩn $$L_1, L_2$$ và độ tương đồng Cosine giữa hai embedding

Giả sử hai câu văn được mã hóa thành hai vector embedding 3 chiều: $$\mathbf{u} = (1, 2, 2)$$ và $$\mathbf{v} = (2, 1, -2)$$. Hãy tính tay toàn bộ các đại lượng hình học quan trọng:

1. **Chuẩn $$L_1$$ và $$L_2$$**:
   - $$\|\mathbf{u}\|_1 = \lvert 1 \rvert + \lvert 2 \rvert + \lvert 2 \rvert = 5$$; $$\|\mathbf{u}\|_2 = \sqrt{1^2 + 2^2 + 2^2} = \sqrt{9} = 3$$.
   - $$\|\mathbf{v}\|_1 = \lvert 2 \rvert + \lvert 1 \rvert + \lvert -2 \rvert = 5$$; $$\|\mathbf{v}\|_2 = \sqrt{2^2 + 1^2 + (-2)^2} = \sqrt{9} = 3$$.
2. **Tích vô hướng**:
   $$\mathbf{u} \cdot \mathbf{v} = 1(2) + 2(1) + 2(-2) = 2 + 2 - 4 = 0.$$
3. **Góc và Độ tương đồng Cosine**:
   $$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{0}{3 \times 3} = 0 \implies \theta = 90^\circ.$$
4. **Khoảng cách Euclid giữa hai điểm**:
   $$\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{(1-2)^2 + (2-1)^2 + (2 - (-2))^2} = \sqrt{1 + 1 + 16} = \sqrt{18} \approx 4.243.$$

Dù hai vector có cùng độ dài ($$3$$), tích vô hướng bằng $$0$$ chứng tỏ chúng **trực giao** (độc lập về mặt hướng)—trong hệ thống tìm kiếm ngữ nghĩa (RAG), cặp văn bản này sẽ bị chấm điểm tương đồng bằng $$0$$.

Bài học này giới thiệu vector, không gian vector và các khái niệm nền tảng của đại số tuyến tính trong bối cảnh tối ưu hóa.

---

## Vector và không gian vector ($$\mathbb{R}^n$$)

### Vector là gì?

- **Vector:** Có thể hình dung vector như một mũi tên trong không gian, mang cả hướng lẫn độ lớn (độ dài). Về mặt đại số, vector là một bộ số thực có thứ tự, tương tự như tọa độ. Chẳng hạn, trong mặt phẳng 2D, vector $$\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$ biểu diễn dịch chuyển 3 đơn vị theo trục $$x$$ và 4 đơn vị theo trục $$y$$.
- **Hai cách nhìn — hình học và đại số:**
  - **Hình học:** vector là mũi tên có hướng và độ lớn.
  - **Đại số:** vector là một bộ có thứ tự các số thực.

### Không gian vector

- **Không gian vector ($$\mathbb{R}^n$$):** là tập hợp tất cả các vector gồm $$n$$ thành phần thực. Chẳng hạn, $$\mathbb{R}^2$$ gồm mọi vector 2 thành phần, tương ứng với các điểm hoặc mũi tên trên mặt phẳng 2D.
- **Ví dụ:**
  - $$\mathbb{R}^2 = \left\{\begin{pmatrix} x \\ y \end{pmatrix} : x, y \in \mathbb{R}\right\}$$ (mặt phẳng)
  - $$\mathbb{R}^3 = \left\{\begin{pmatrix} x \\ y \\ z \end{pmatrix} : x, y, z \in \mathbb{R}\right\}$$ (không gian 3D)

### Các phép toán trên vector

**Phép cộng vector:**
$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 \\ u_2 \\ \vdots \\ u_n \end{pmatrix} + \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

**Phép nhân với vô hướng:**
$$c\mathbf{v} = c \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix} = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{pmatrix}$$

---

## Độc lập tuyến tính, cơ sở và số chiều

### Độc lập tuyến tính

Một hệ vector $$\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$$ được gọi là **độc lập tuyến tính** nếu phương trình

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$

chỉ có nghiệm duy nhất $$c_1 = c_2 = \cdots = c_k = 0$$.

**Ý nghĩa trực quan:** Một hệ vector độc lập tuyến tính khi không có vector nào trong hệ có thể biểu diễn dưới dạng tổ hợp tuyến tính (nhân vô hướng rồi cộng) của các vector còn lại. Mỗi vector đóng góp thêm một hướng độc lập mới.

**Ví dụ trong $$\mathbb{R}^2$$:**
- $$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ và $$\mathbf{v}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ là độc lập tuyến tính.
- $$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$ và $$\mathbf{v}_2 = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$$ là phụ thuộc tuyến tính (vì $$\mathbf{v}_2 = 2\mathbf{v}_1$$).

### Cơ sở

Một **cơ sở** (*basis*) của không gian vector là tập hợp tối thiểu các vector độc lập tuyến tính sao cho *mọi* vector trong không gian đó đều có thể biểu diễn bằng tổ hợp tuyến tính của chúng.

**Tính chất của một cơ sở:**
1. Các vector trong cơ sở độc lập tuyến tính.
2. Chúng sinh ra toàn bộ không gian vector.
3. Mọi vector trong không gian đều có biểu diễn duy nhất dưới dạng tổ hợp tuyến tính của các vector cơ sở.

**Cơ sở chính tắc của $$\mathbb{R}^n$$:**
$$\mathbf{e}_1 = \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \mathbf{e}_2 = \begin{pmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{pmatrix}, \ldots, \mathbf{e}_n = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{pmatrix}$$

### Số chiều

**Số chiều** (*dimension*) của một không gian vector là số lượng vector có trong một cơ sở bất kỳ của nó, cho biết cần bao nhiêu hướng độc lập để xác định mọi điểm trong không gian.

- $$\dim(\mathbb{R}^2) = 2$$
- $$\dim(\mathbb{R}^3) = 3$$
- $$\dim(\mathbb{R}^n) = n$$

---

## Chuẩn của vector

**Chuẩn** (*norm*) là một hàm gán "độ dài" hay "độ lớn" cho mỗi vector, mở rộng khái niệm khoảng cách từ gốc tọa độ.

### Tính chất của chuẩn

Mọi chuẩn $$\|\cdot\|$$ đều phải thỏa mãn ba tính chất:

1. **Tính không âm:** $$\|\mathbf{x}\| \geq 0$$, và $$\|\mathbf{x}\| = 0$$ khi và chỉ khi $$\mathbf{x} = \mathbf{0}$$.
2. **Tính thuần nhất tuyệt đối:** $$\|t\mathbf{x}\| = |t| \|\mathbf{x}\|$$ với mọi số thực $$t$$.
3. **Bất đẳng thức tam giác:** $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$$.

### Các chuẩn thông dụng

**Chuẩn Euclid (chuẩn $$L_2$$):**
$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$$

Đây là khoảng cách hình học quen thuộc theo đường thẳng.

**Chuẩn Manhattan (chuẩn $$L_1$$):**
$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| = |x_1| + |x_2| + \cdots + |x_n|$$

Còn được gọi là khoảng cách taxi — quãng đường di chuyển theo các trục vuông góc trên mạng lưới đường phố dạng ô bàn cờ.

**Chuẩn cực đại (chuẩn $$L_\infty$$):**
$$\|\mathbf{x}\|_\infty = \max_{i} |x_i|$$

Đây là giá trị tuyệt đối lớn nhất trong các thành phần của vector.

**Ví dụ:** Với $$\mathbf{x} = \begin{pmatrix} 3 \\ -4 \\ 1 \end{pmatrix}$$:
- $$\|\mathbf{x}\|_2 = \sqrt{3^2 + (-4)^2 + 1^2} = \sqrt{26} \approx 5{,}1$$
- $$\|\mathbf{x}\|_1 = |3| + |-4| + |1| = 8$$
- $$\|\mathbf{x}\|_\infty = \max\{|3|, |-4|, |1|\} = 4$$

---

## Tích vô hướng

**Tích vô hướng** (*dot product* hay *inner product*) là phép nhân phổ biến nhất giữa hai vector, trả về một số thực.

### Định nghĩa

Với hai vector $$\mathbf{x}$$ và $$\mathbf{y}$$ trong $$\mathbb{R}^n$$:

$$\mathbf{x}^T \mathbf{y} = \mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^n x_i y_i = x_1 y_1 + x_2 y_2 + \cdots + x_n y_n$$

### Diễn giải hình học

$$\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos \theta$$

trong đó $$\theta$$ là góc giữa hai vector.

### Tính chất

1. **Giao hoán:** $$\mathbf{x} \cdot \mathbf{y} = \mathbf{y} \cdot \mathbf{x}$$
2. **Phân phối:** $$\mathbf{x} \cdot (\mathbf{y} + \mathbf{z}) = \mathbf{x} \cdot \mathbf{y} + \mathbf{x} \cdot \mathbf{z}$$
3. **Thuần nhất:** $$(c\mathbf{x}) \cdot \mathbf{y} = c(\mathbf{x} \cdot \mathbf{y})$$

### Các trường hợp đặc biệt

- **Hai vector trực giao (vuông góc):** $$\mathbf{x} \cdot \mathbf{y} = 0$$
- **Hai vector cùng phương (song song):** $$\mathbf{x} \cdot \mathbf{y} = \pm \|\mathbf{x}\| \|\mathbf{y}\|$$
- **Tích vô hướng với chính nó:** $$\mathbf{x} \cdot \mathbf{x} = \|\mathbf{x}\|_2^2$$

### Ví dụ

Với $$\mathbf{x} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$ và $$\mathbf{y} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$$:

$$\mathbf{x}^T \mathbf{y} = (1)(3) + (2)(4) = 3 + 8 = 11$$

---

## Ứng dụng trong tối ưu hóa

Vector và không gian vector là công cụ nền tảng trong tối ưu hóa ở nhiều khía cạnh:

1. **Biến quyết định:** bài toán tối ưu nhiều biến biểu diễn tự nhiên dưới dạng một vector tham số.
2. **Gradient:** gradient của hàm mục tiêu là vector chỉ theo hướng tăng dốc nhất.
3. **Ràng buộc:** ràng buộc tuyến tính được viết gọn bằng tích vô hướng: $$\mathbf{a}^T \mathbf{x} \leq b$$.
4. **Khoảng cách và độ tương đồng:** các loại chuẩn khác nhau giúp đo khoảng cách giữa các nghiệm hoặc độ lớn của bước cập nhật.
5. **Tính trực giao:** nhiều tính chất tối ưu dựa trên quan hệ vuông góc, chẳng hạn gradient vuông góc với đường mức.
6. **Tổ hợp tuyến tính:** miền khả thi thường được mô tả qua tổ hợp tuyến tính của các vector (bao lồi, nón lồi, v.v.).

Khung lý thuyết về không gian vector cung cấp ngôn ngữ toán học chặt chẽ để phát biểu và giải quyết các bài toán tối ưu hóa một cách có hệ thống.

## Video tham khảo

- **Xem trước:** [Vectors | Chapter 1, Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs) — 3Blue1Brown, tiếng Anh. Giới thiệu khái niệm vector trước khi bước vào không gian sinh (*span*), cơ sở và số chiều.
- [Linear combinations, span, and basis vectors | Chapter 2, Essence of linear algebra](https://www.youtube.com/watch?v=k7RM-ot2NWY) — 3Blue1Brown, tiếng Anh. Trực quan hóa tổ hợp tuyến tính, không gian sinh và cơ sở, tương ứng với phần độc lập tuyến tính và số chiều trong bài.
- [Dot products and duality | Chapter 9, Essence of linear algebra](https://www.youtube.com/watch?v=LyGKycYT2v0) — 3Blue1Brown, tiếng Anh. Giải thích tích vô hướng và mối liên hệ với chuẩn Euclid — độ dài được cảm sinh trực tiếp từ tích vô hướng.
