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

## 3. Ví dụ / Trực giác

### Ví dụ 1: Phát hiện Cạnh (Bộ lọc Thủ công)

Trước học sâu, thị giác máy tính dựa vào các bộ lọc thủ công. Hiểu chúng giúp xây dựng trực giác về những gì CNN học tự động.

**Bộ dò cạnh đứng** (kiểu Sobel):

$$K_{vertical} = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}$$

**Vì sao nó hoạt động**: 
- Cột trái có trọng số dương → đo độ sáng bên trái
- Cột phải có trọng số âm → đo độ sáng bên phải
- Nếu trái sáng và phải tối: đầu ra dương lớn (có cạnh!)
- Nếu đồng đều: dương và âm triệt tiêu → đầu ra gần zero

**Bộ dò cạnh ngang**:

$$K_{horizontal} = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix}$$

**Ví dụ tính tay**:

**Ảnh đầu vào** (mô phỏng cạnh đứng):
$$I = \begin{bmatrix} 
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0 \\
100 & 100 & 100 & 0 & 0
\end{bmatrix}$$

**Áp dụng nhân cạnh đứng tại vị trí (1,1)**:

$$\begin{align}
S[1,1] &= (100 \times 1) + (100 \times 0) + (100 \times -1) \\
&+ (100 \times 2) + (100 \times 0) + (100 \times -2) \\
&+ (100 \times 1) + (100 \times 0) + (100 \times -1) \\
&= 100 - 100 + 200 - 200 + 100 - 100 = 0
\end{align}$$

**Tại vị trí (1,2)** (trên cạnh):

$$\begin{align}
S[1,2] &= (100 \times 1) + (100 \times 0) + (0 \times -1) \\
&+ (100 \times 2) + (100 \times 0) + (0 \times -2) \\
&+ (100 \times 1) + (100 \times 0) + (0 \times -1) \\
&= 100 + 0 + 0 + 200 + 0 + 0 + 100 + 0 + 0 = 400
\end{align}$$

**Kết quả**: Kích hoạt cao (400) đúng tại vị trí cạnh đứng!

### Ví dụ 2: CNN Xây dựng Biểu diễn Phân cấp như thế nào

Một trong những khía cạnh đẹp nhất của CNN là cách chúng tự động học đặc trưng phân cấp:

**Tầng 1 (Tầng sớm - Đặc trưng đơn giản)**:
- Cạnh có hướng ở nhiều góc
- Vệt màu và gradient
- Kết cấu cơ bản
- Ví dụ bộ lọc: $$\begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$$, $$\begin{bmatrix} 1 & 1 & 1 \\ 0 & 0 & 0 \\ -1 & -1 & -1 \end{bmatrix}$$

**Tầng 2–3 (Tầng giữa - Các bộ phận)**:
- Góc (tổ hợp các cạnh gặp nhau)
- Hình dạng đơn giản (đường cong, tròn, chữ nhật)
- Kết cấu (mẫu cạnh lặp lại)
- Lưới, sọc, đốm

**Tầng 4–5 (Tầng sâu hơn - Bộ phận đối tượng)**:
- Mắt, mũi, tai (cho nhận diện khuôn mặt)
- Bánh xe, cửa sổ, đèn pha (cho nhận diện xe)
- Cánh hoa, lá, thân (cho nhận diện hoa)

**Tầng cuối (Cấp cao - Đối tượng)**:
- Khuôn mặt đầy đủ từ nhiều góc
- Hình dáng xe hoàn chỉnh
- Toàn bộ động vật

**Hệ phân cấp này xuất hiện tự động từ huấn luyện** — mạng khám phá rằng cạnh hữu ích để xây góc, góc để xây hình dạng, và hình dạng để xây đối tượng. Điều này phản ánh cách các nhà khoa học thần kinh tin rằng vỏ thị giác xử lý thông tin.

### Ví dụ 3: Sự Tăng của Trường Tiếp nhận

**Trường tiếp nhận** (*receptive field*) là vùng của ảnh đầu vào ảnh hưởng đến một neuron đầu ra đơn lẻ.

**Một tầng conv 3×3**: Mỗi điểm ảnh đầu ra “nhìn” vùng 3×3 của đầu vào.

**Hai tầng conv 3×3 xếp chồng**: Mỗi điểm ảnh đầu ra ở tầng 2 nhìn vùng 3×3 của đầu ra tầng 1. Nhưng mỗi điểm trong vùng 3×3 đó lại nhìn vùng 3×3 của đầu vào. Kết hợp: đầu ra tầng 2 nhìn vùng **5×5** của đầu vào gốc.

**Công thức trường tiếp nhận với $$n$$ tầng tích chập 3×3**:

$$RF = 1 + 2n$$

| Số tầng | Trường tiếp nhận | Tương đương một nhân |
|--------|-----------------|-------------------------|
| 1 | 3×3 | 3×3 |
| 2 | 5×5 | 5×5 |
| 3 | 7×7 | 7×7 |
| 5 | 11×11 | 11×11 |
| 10 | 21×21 | 21×21 |

**Vì sao xếp chồng bộ lọc nhỏ thay vì dùng một bộ lọc lớn?**

So sánh: Ba tầng 3×3 so với một tầng 7×7

| Chỉ số | Ba 3×3 | Một 7×7 |
|--------|-----------|---------|
| Trường tiếp nhận | 7×7 | 7×7 |
| Tham số mỗi kênh | 3×(3×3) = 27 | 7×7 = 49 |
| Phi tuyến | 3 ReLU | 1 ReLU |

Ba tích chập 3×3 xếp chồng có:
- **Ít tham số hơn** (27 so với 49)
- **Nhiều phi tuyến hơn** (3 ReLU so với 1)
- **Sức biểu diễn mạnh hơn** (có thể biểu diễn hàm phức tạp hơn)

Hiểu biết này từ VGGNet (2014) đã cách mạng hóa thiết kế kiến trúc CNN.

---
