---
layout: post
title: 04-01-01 Toán học Tích chập và Kích thước
chapter: '04'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

# Tầng Tích chập: Khối Xây dựng của Thị giác Máy tính

![CNN](/deep-learning-self-learning/img/chapter_img/chapter04/conv1.jpg)
*Tổng quan kiến trúc Mạng Neuron Tích chập (CNN). Nguồn: AnalyticsVidhya*

## 1. Tổng quan khái niệm
**Tầng tích chập** (*convolutional layers*) là các tầng mạng neuron chuyên biệt để xử lý dữ liệu dạng lưới, đặc biệt là ảnh. Thay vì nối mọi đầu vào với mọi neuron (kết nối đầy đủ), tầng tích chập dùng các bộ lọc nhỏ, học được, trượt trên đầu vào để phát hiện các mẫu cục bộ.

**Vì sao CNN quan trọng**:
- **Hiệu quả tham số**: Ít hơn hàng triệu tham số so với mạng kết nối đầy đủ
- **Bất biến tịnh tiến**: Phát hiện đặc trưng bất kể vị trí trên ảnh
- **Học phân cấp**: Tự động học đặc trưng thấp → trung → cao
- **Tiên tiến**: Hiệu năng tốt nhất trên hầu hết tác vụ thị giác

**Hiểu biết then chốt**: Ảnh có cấu trúc không gian — các điểm ảnh gần nhau có liên quan. Tầng tích chập khai thác cấu trúc này qua **kết nối cục bộ** và **chia sẻ tham số**.

**Phép ẩn dụ**: Hãy nghĩ tích chập như trượt một kính lúp (bộ lọc) trên ảnh để tìm các mẫu cụ thể (cạnh, kết cấu, hình dạng). Mỗi bộ lọc chuyên phát hiện một kiểu mẫu, và bạn dùng cùng một kính lúp ở mọi vị trí trên ảnh thay vì một kính khác cho mỗi vị trí.

![CNN](/deep-learning-self-learning/img/chapter_img/chapter04/conv2.jpg)
*Tổng quan kiến trúc Mạng Neuron Tích chập (CNN). Nguồn: AnalyticsVidhya*

### Cảm hứng Sinh học

![CNN Inspiration from Human Eye](/deep-learning-self-learning/img/chapter_img/chapter04/conv3.jpg)
*CNN được lấy cảm hứng trực tiếp từ cách não người xử lý thông tin thị giác. Mắt bạn phát hiện cạnh trước, rồi ghép thành hình dạng, rồi khớp mẫu với trí nhớ. CNN làm đúng điều đó. Nguồn: Analytics Vidhya*

Năm 1959, các nhà thần kinh sinh lý David Hubel và Torsten Wiesel phát hiện rằng neuron trong vỏ thị giác đáp ứng với các mẫu cụ thể trong các vùng cục bộ của trường thị giác (gọi là “trường tiếp nhận” — *receptive fields*). Một số neuron đáp ứng với cạnh theo hướng nhất định, số khác với chuyển động theo hướng cụ thể. Quá trình xử lý phân cấp này — từ đặc trưng đơn giản đến đối tượng phức tạp — đã trực tiếp truyền cảm hứng cho thiết kế mạng neuron tích chập.

---

## 2. Nền tảng toán học
### Phân biệt Tích chập và Tương quan chéo

Đây là sự phân biệt quan trọng thường gây nhầm lẫn cho người mới. Hãy hiểu cẩn thận cả hai phép toán.

![Convolution Operation](https://miro.medium.com/v2/resize:fit:1400/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)
*Minh họa phép tích chập 2D trên ảnh. Nguồn: Medium*

![The Convolution Filter Sliding](/deep-learning-self-learning/img/chapter_img/chapter04/conv4.jpg)
*Một cửa sổ nhỏ gọi là bộ lọc trượt trên ảnh, quét từng vùng. Tại mỗi vị trí nó hỏi: “Đặc trưng này có ở đây không?” Các bộ lọc khác nhau phát hiện mẫu khác nhau — cạnh, làm mờ, làm nét. CNN tự học hàng trăm bộ lọc trong quá trình huấn luyện. Nguồn: Analytics Vidhya*

#### Tích chập Đích thực (Định nghĩa Xử lý Tín hiệu)

Trong xử lý tín hiệu và toán học, **tích chập đích thực** bao gồm **lật nhân** (xoay 180°) trước khi trượt trên đầu vào. Với ảnh 2D $$I$$ và nhân $$K$$:

$$(I * K)[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[m, n] \cdot K[i-m, j-n]$$

Với tín hiệu rời rạc hữu hạn và nhân kích thước $$k \times k$$:

$$(I * K)[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i-m, j-n] \cdot K[m,n]$$

Tính chất then chốt ở đây là phép **trừ** trong chỉ số ($$i-m, j-n$$), vốn hiệu quả là lật nhân.

**Vì sao tích chập lật nhân?** Tích chập ban đầu được thiết kế để mô tả cách hệ thống đáp ứng với đầu vào theo thời gian. Việc lật đảm bảo tích chập:
1. **Giao hoán**: $$I * K = K * I$$
2. **Kết hợp**: $$(I * K_1) * K_2 = I * (K_1 * K_2)$$

Các tính chất này thiết yếu trong xử lý tín hiệu để phân tích hệ thống tuyến tính bất biến theo thời gian.

#### Tương quan chéo (Những gì Học sâu Thực sự Dùng)

**Tương quan chéo** (*cross-correlation*) tương tự nhưng **không lật nhân**:

$$S[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n] \cdot K[m,n]$$

Lưu ý phép **cộng** trong chỉ số ($$i+m, j+n$$) — ta đơn giản trượt nhân nguyên trạng trên ảnh.

#### So sánh Trực quan: Tích chập và Tương quan chéo

Hãy làm cụ thể bằng một ví dụ:

**Nhân** $$K$$:
$$K = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

**Nhân đã lật** (cho tích chập đích thực):
$$K_{flipped} = \begin{bmatrix} 9 & 8 & 7 \\ 6 & 5 & 4 \\ 3 & 2 & 1 \end{bmatrix}$$

**Với nhân đối xứng** (như nhiều bộ dò cạnh):
$$K = \begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$$

Lật nhân này cho:
$$K_{flipped} = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$$

Lưu ý phiên bản đã lật phát hiện cạnh theo **hướng ngược lại**!

#### Vì sao Học sâu Dùng Tương quan chéo (và Gọi là “Tích chập”)

Hiểu biết then chốt: **Trong học sâu, ta không quan tâm đến việc lật vì trọng số nhân được học**.

Xét lý luận sau:

1. **Nếu dùng tích chập đích thực**: Mạng học trọng số $$W$$, rồi áp dụng chúng đã lật trong lượt xuôi
2. **Nếu dùng tương quan chéo**: Mạng học trọng số $$W'$$, áp dụng không lật

Vì mạng học trọng số từ dữ liệu dù sao, $$W'$$ đơn giản học thành phiên bản đã lật của những gì $$W$$ sẽ là. **Mạng sẽ học cùng hàm bất kể cách nào**.

**Lợi ích của tương quan chéo**:
- **Cài đặt đơn giản hơn**: Không cần lật nhân
- **Trực quan hơn**: “Mẫu” của nhân khớp trực tiếp với mẫu nó phát hiện
- **Cùng khả năng học**: Mạng có thể học bất kỳ hàm nào dù sao

**Ghi chú lịch sử**: Cộng đồng học sâu chấp nhận thuật ngữ “tích chập” dù về kỹ thuật ta dùng tương quan chéo. Đây giờ là thuật ngữ chuẩn, nhưng hiểu sự phân biệt giúp khi đọc tài liệu xử lý tín hiệu hoặc cài đặt phép toán tùy chỉnh.

### Ví dụ Tích chập Từng bước

Hãy lần theo một ví dụ đầy đủ để củng cố hiểu biết:

**Ảnh đầu vào** $$I$$ (5×5):
$$I = \begin{bmatrix} 
1 & 2 & 3 & 0 & 1 \\
0 & 1 & 2 & 3 & 1 \\
1 & 2 & 1 & 0 & 0 \\
0 & 1 & 2 & 3 & 2 \\
2 & 1 & 0 & 1 & 1
\end{bmatrix}$$

**Nhân** $$K$$ (3×3):
$$K = \begin{bmatrix} 
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}$$

**Tính đầu ra tại vị trí (0,0)** bằng tương quan chéo:

Ta trích vùng 3×3 bắt đầu tại (0,0):
$$\text{region} = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

Nhân theo phần tử rồi cộng:
$$\begin{align}
S[0,0] &= (1 \times 1) + (2 \times 0) + (3 \times -1) \\
&+ (0 \times 1) + (1 \times 0) + (2 \times -1) \\
&+ (1 \times 1) + (2 \times 0) + (1 \times -1) \\
&= 1 + 0 - 3 + 0 + 0 - 2 + 1 + 0 - 1 \\
&= -4
\end{align}$$

**Tính đầu ra tại vị trí (0,1)**:

Vùng bắt đầu tại (0,1):
$$\text{region} = \begin{bmatrix} 2 & 3 & 0 \\ 1 & 2 & 3 \\ 2 & 1 & 0 \end{bmatrix}$$

$$\begin{align}
S[0,1] &= (2 \times 1) + (3 \times 0) + (0 \times -1) \\
&+ (1 \times 1) + (2 \times 0) + (3 \times -1) \\
&+ (2 \times 1) + (1 \times 0) + (0 \times -1) \\
&= 2 + 0 + 0 + 1 + 0 - 3 + 2 + 0 + 0 \\
&= 2
\end{align}$$

Tiếp tục quá trình này cho mọi vị trí hợp lệ để có đầu ra đầy đủ.

### Tích chập Đa kênh

Ảnh thực có nhiều kênh (RGB có 3; các tầng CNN trung gian có thể có 64, 128, 256 hoặc hơn). Tích chập mở rộng như sau:

**Với đầu vào có $$C_{in}$$ kênh** (ví dụ $$H \times W \times C_{in}$$) và **bộ lọc kích thước $$k \times k \times C_{in}$$**:

$$S[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n, c] \cdot K[m,n,c] + b$$

Trong đó:
- $$C_{in}$$: số kênh đầu vào (3 với RGB)
- $$b$$: hệ số bias (một vô hướng cho mỗi bộ lọc)
- Mỗi bộ lọc tạo **một** kênh đầu ra

**Để tạo $$C_{out}$$ kênh đầu ra**, ta cần $$C_{out}$$ bộ lọc, mỗi bộ kích thước $$k \times k \times C_{in}$$.

**Trực giác hình ảnh**: Với ảnh RGB, mỗi bộ lọc thực sự là một khối 3D (ví dụ 3×3×3). Bộ lọc có trọng số riêng cho mỗi kênh màu, và tất cả được cộng lại để tạo một giá trị đầu ra. Nhờ đó bộ lọc có thể học các mẫu phụ thuộc màu (ví dụ phát hiện bầu trời xanh so với cỏ xanh).

### Công thức Kích thước Đầu ra

Hiểu cách kích thước đầu vào biến đổi qua các tầng tích chập là thiết yếu khi thiết kế kiến trúc.

**Cho trước**:
- Kích thước đầu vào: $$H_{in} \times W_{in}$$
- Kích thước nhân: $$k \times k$$
- Bước nhảy (*stride*): $$s$$ (số điểm ảnh di chuyển giữa các vị trí)
- Đệm (*padding*): $$p$$ (số không thêm quanh biên)

**Kích thước đầu ra**:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W_{in} + 2p - k}{s} \right\rfloor + 1$$

**Các cấu hình thường gặp**:

| Cấu hình | Mục đích | Kết quả |
|--------|---------|----------------|
| $$k=3, s=1, p=1$$ | Đệm “same” | $$n_{out} = n_{in}$$ |
| $$k=3, s=2, p=1$$ | Giảm mẫu 2 lần | $$n_{out} = \lceil n_{in}/2 \rceil$$ |
| $$k=1, s=1, p=0$$ | Tích chập 1×1 | $$n_{out} = n_{in}$$ |
| $$k=7, s=2, p=3$$ | Giảm mẫu mạnh | $$n_{out} = \lceil n_{in}/2 \rceil$$ |

### Phân tích Số tham số

Với một tầng tích chập:

$$\text{Parameters} = (k \times k \times C_{in}) \times C_{out} + C_{out}$$

Số hạng đầu là trọng số (mỗi bộ lọc có $$k \times k \times C_{in}$$ trọng số, và ta có $$C_{out}$$ bộ lọc). Số hạng thứ hai là bias (một cho mỗi kênh đầu ra).

**Ví dụ tính toán**:
- Đầu vào: $$32 \times 32 \times 3$$ (ảnh RGB CIFAR-10)
- Bộ lọc: $$3 \times 3$$, 64 bộ lọc
- Tham số: $$(3 \times 3 \times 3) \times 64 + 64 = 27 \times 64 + 64 = 1,792$$

**So với tầng kết nối đầy đủ**:
- Đầu vào: $$32 \times 32 \times 3 = 3,072$$ neuron
- Đầu ra: 64 neuron
- Tham số: $$3,072 \times 64 + 64 = 196,672$$

**Hệ số giảm**: $$196,672 / 1,792 \approx 110\times$$ ít tham số hơn!

Sự giảm mạnh này đến từ hai tính chất:
1. **Kết nối cục bộ**: Mỗi đầu ra chỉ nối với một vùng nhỏ của đầu vào
2. **Chia sẻ tham số**: Cùng trọng số bộ lọc dùng ở mọi vị trí không gian

---
