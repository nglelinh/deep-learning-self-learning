---
layout: post
title: 04-01-02 Trực giác Tích chập và Ví dụ
chapter: '04'
order: 4
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

Làm thế nào một mạng chỉ toàn các bộ lọc bé xíu $$3 \times 3$$ — mỗi bộ lọc chỉ nhìn thấy 9 điểm ảnh — lại có thể nhận ra cả một chiếc ô tô hay một khuôn mặt người nằm trải dài hàng trăm pixel? Câu trả lời không nằm ở việc phóng to từng bộ lọc, mà nằm ở nghệ thuật **xếp chồng** để mở rộng tầm nhìn và lắp ghép đặc trưng theo từng tầng bậc.

## Kiến thức cần nhớ: Ba trực giác cốt lõi để "đọc vị" một mạng CNN

1. **Bộ lọc như một máy đo độ tương đồng mẫu (Template Matching)**:
   Phép tích vô hướng giữa cửa sổ ảnh và bộ lọc đạt cực đại khi vùng ảnh có hoa văn giống hệt bộ lọc (ví dụ Sobel dọc phản hồi cực đại tại ranh giới sáng–tối theo phương thẳng đứng).
2. **Phân cấp đặc trưng tự phát (Emergent Hierarchy)**:
   Khi trực quan hóa trọng số của một CNN đã huấn luyện trên ImageNet, tầng 1 luôn tự học các bộ lọc Gabor (cạnh và chuyển màu), tầng 2–3 ghép cạnh thành đường cong và kết cấu (lưới, lông thú), còn tầng 4–5 phản hồi với bánh xe, mắt người hay mặt chó.
3. **Sự mở rộng của Trường tiếp nhận (Receptive Field)**:
   Mỗi neuron ở tầng $$L$$ tổng hợp tín hiệu từ một vùng $$k \times k$$ của tầng $$L-1$$. Nhờ hiệu ứng cộng dồn (và nhân lên khi có stride/pooling), trường tiếp nhận hiệu dụng mở rộng nhanh theo chiều sâu.
4. **Trường tiếp nhận lý thuyết vs. Trường tiếp nhận hiệu dụng (ERF)**:
   Dù trường tiếp nhận lý thuyết phủ rộng, các điểm ảnh ở tâm đóng góp nhiều đường đi hơn nên trọng số ảnh hưởng thường phân bố dạng hình chuông Gaussian tập trung quanh trung tâm.

### Giải thích: Tính tay sự mở rộng trường tiếp nhận qua 3 tầng tích chập

Giả sử ta xếp chồng các tầng tích chập $$3 \times 3$$ với `padding=1`. Công thức truy hồi trường tiếp nhận $$r_l$$ và bước nhảy tích lũy $$j_l$$ là:
$$r_l = r_{l-1} + (k_l - 1) \cdot j_{l-1}, \qquad j_l = j_{l-1} \cdot s_l$$
với giá trị khởi đầu tại ảnh gốc: $$r_0 = 1, \; j_0 = 1$$.

| Tầng $$l$$ | Cấu hình tầng | Bước nhảy tích lũy $$j_{l-1}$$ | Công thức tính $$r_l$$ | Trường tiếp nhận $$r_l$$ |
| :---: | :--- | :---: | :--- | :---: |
| 0 | Ảnh đầu vào | — | — | $$1 \times 1$$ |
| 1 | Conv $$3\times 3, s=1$$ | 1 | $$1 + (3-1)\cdot 1$$ | **$$3 \times 3$$** |
| 2 | Conv $$3\times 3, s=2$$ | 1 | $$3 + (3-1)\cdot 1$$ | **$$5 \times 5$$** (và $$j_2 = 2$$) |
| 3 | Conv $$3\times 3, s=1$$ | 2 | $$5 + (3-1)\cdot 2$$ | **$$9 \times 9$$** |

Chỉ sau 3 tầng lọc nhỏ $$3\times 3$$ (với một bước stride 2 ở giữa), mỗi điểm đặc trưng đã "bao quát" một vùng ảnh gốc $$9 \times 9$$, nhưng tổng số trọng số trên mỗi kênh lại nhỏ hơn rất nhiều so với một bộ lọc $$9 \times 9$$ duy nhất ($$3 \times 9 = 27$$ so với $$81$$).

## 3. Ví dụ / Trực giác

### Ví dụ 1: Phát hiện Cạnh bằng Bộ lọc Thủ công

Trước khi học sâu phổ biến, thị giác máy tính dựa nhiều vào các bộ lọc thiết kế thủ công. Tìm hiểu cách hoạt động của chúng giúp hình dung rõ hơn những gì CNN tự học trong quá trình huấn luyện.

**Bộ dò cạnh đứng** (kiểu Sobel):

$$K_{vertical} = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}$$

**Cơ chế hoạt động:**
- Cột bên trái mang trọng số dương, đo cường độ sáng phía trái.
- Cột bên phải mang trọng số âm, đo cường độ sáng phía phải.
- Nếu bên trái sáng và bên phải tối, tổng cho giá trị dương lớn (báo hiệu có cạnh đứng).
- Nếu vùng ảnh đồng đều, phần dương và phần âm triệt tiêu nhau nên đầu ra gần bằng 0.

**Bộ dò cạnh ngang:**

$$K_{horizontal} = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}$$

**Ví dụ tính tay:**

Xét ảnh đầu vào mô phỏng một cạnh đứng (nửa trái sáng 100, nửa phải tối 0):
$$I = \begin{bmatrix} 
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0
\end{bmatrix}$$

**Áp dụng nhân cạnh đứng tại vị trí (1,1) (vùng đồng đều):**

$$\begin{align}
S[1,1] &= (100 \times 1) + (100 \times 0) + (100 \times -1) \\
&+ (100 \times 2) + (100 \times 0) + (100 \times -2) \\
&+ (100 \times 1) + (100 \times 0) + (100 \times -1) \\
&= 100 - 100 + 200 - 200 + 100 - 100 = 0
\end{align}$$

**Tại vị trí (1,2) (nằm đúng trên cạnh):**

$$\begin{align}
S[1,2] &= (100 \times 1) + (100 \times 0) + (0 \times -1) \\
&+ (100 \times 2) + (100 \times 0) + (0 \times -2) \\
&+ (100 \times 1) + (100 \times 0) + (0 \times -1) \\
&= 100 + 0 + 0 + 200 + 0 + 0 + 100 + 0 + 0 = 400
\end{align}$$

**Kết quả:** Giá trị kích hoạt tăng cao (400) đúng tại vị trí ranh giới sáng–tối.

### Ví dụ 2: Cách CNN Xây dựng Biểu diễn Phân cấp

Một đặc trưng quan trọng của CNN là khả năng tự động học hệ thống đặc trưng theo nhiều tầng trừu tượng:

**Tầng 1 (Các tầng đầu — Đặc trưng cơ bản):**
- Cạnh theo nhiều góc nghiêng khác nhau
- Vệt màu và sự chuyển đổi độ sáng (gradient)
- Kết cấu đơn giản
- Ví dụ bộ lọc: $$\begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$$, $$\begin{bmatrix} 1 & 1 & 1 \\ 0 & 0 & 0 \\ -1 & -1 & -1 \end{bmatrix}$$

**Tầng 2–3 (Các tầng giữa — Hình dạng sơ cấp):**
- Góc (nơi các cạnh giao nhau)
- Hình dạng cơ bản (đường cong, hình tròn, hình chữ nhật)
- Hoa văn lặp lại như sọc, lưới, chấm

**Tầng 4–5 (Các tầng sâu hơn — Bộ phận đối tượng):**
- Mắt, mũi, tai (trong bài toán khuôn mặt)
- Bánh xe, cửa sổ, đèn pha (trong bài toán nhận diện phương tiện)
- Cánh hoa, lá, thân (trong bài toán phân loại thực vật)

**Các tầng cuối (Mức cao — Đối tượng hoàn chỉnh):**
- Khuôn mặt hoàn chỉnh ở nhiều góc nhìn
- Hình dáng trọn vẹn của phương tiện hoặc động vật

**Hệ phân cấp này tự hình thành qua quá trình huấn luyện:** mạng tự phát hiện rằng việc tổ hợp cạnh thành góc, góc thành bộ phận và bộ phận thành đối tượng giúp giảm hàm mất mát hiệu quả nhất.

### Ví dụ 3: Sự Mở rộng của Trường Tiếp nhận

**Trường tiếp nhận** (*receptive field*) là vùng trên ảnh đầu vào gốc có ảnh hưởng trực tiếp tới giá trị của một phần tử đầu ra nhất định.

**Một tầng tích chập 3×3:** Mỗi điểm ở đầu ra quan sát vùng 3×3 của đầu vào.

**Hai tầng tích chập 3×3 xếp chồng:** Mỗi điểm ở đầu ra tầng 2 quan sát vùng 3×3 của đầu ra tầng 1; nhưng mỗi điểm trong vùng đó lại đại diện cho vùng 3×3 của đầu vào gốc. Kết hợp lại, đầu ra tầng 2 bao phủ vùng **5×5** trên ảnh gốc.

**Công thức trường tiếp nhận với $$n$$ tầng tích chập 3×3 (stride = 1):**

$$RF = 1 + 2n$$

| Số tầng | Trường tiếp nhận | Tương đương một nhân |
|--------|-----------------|-------------------------|
| 1 | 3×3 | 3×3 |
| 2 | 5×5 | 5×5 |
| 3 | 7×7 | 7×7 |
| 5 | 11×11 | 11×11 |
| 10 | 21×21 | 21×21 |

**Vì sao nên xếp chồng nhiều bộ lọc nhỏ thay vì dùng một bộ lọc lớn?**

So sánh ba tầng tích chập 3×3 với một tầng tích chập 7×7 (giả sử cùng số kênh):

| Chỉ số | Ba tầng 3×3 | Một tầng 7×7 |
|--------|-----------|---------|
| Trường tiếp nhận | 7×7 | 7×7 |
| Tham số mỗi cặp kênh | 3×(3×3) = 27 | 7×7 = 49 |
| Số lớp phi tuyến | 3 ReLU | 1 ReLU |

Ba tầng 3×3 xếp chồng đạt cùng trường tiếp nhận 7×7 nhưng:
- **Ít tham số hơn** (27 so với 49 trên mỗi cặp kênh)
- **Nhiều phép biến đổi phi tuyến hơn** (3 lớp ReLU so với 1)
- **Sức biểu diễn cao hơn** nhờ có thêm các lớp kích hoạt trung gian

Nhận xét này từ kiến trúc VGGNet (2014) đã ảnh hưởng sâu rộng đến cách thiết kế các mạng CNN hiện đại.

---

## Video tham khảo

- **Xem trước:** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, tiếng Anh. Diễn giải trực quan bộ lọc như bộ phát hiện cạnh và mẫu cục bộ.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Minh họa các bộ lọc dò đặc trưng trên mảng ảnh cụ thể.
- [CNN là gì? | Hiểu Nhanh Mạng Nơ-ron Tích Chập \[PODCAST\]](https://www.youtube.com/watch?v=YlgdXkzjo3Q) — Gà AI, tiếng Việt. Giải thích trực giác cửa sổ trượt bằng tiếng Việt.
