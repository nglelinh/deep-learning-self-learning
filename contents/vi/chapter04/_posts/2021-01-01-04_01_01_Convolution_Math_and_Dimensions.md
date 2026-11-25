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

Trong phòng thí nghiệm giải phẫu thần kinh năm 1959, David Hubel và Torsten Wiesel tình cờ phát hiện một điều kinh ngạc: tế bào vỏ não thị giác mèo không hề phản ứng với một đốm sáng tròn, nhưng lại phát xung dữ dội ngay khi mép tấm kính trượt tạo thành một **vạch thẳng nghiêng** lướt ngang trường nhìn. Hơn nửa thế kỷ sau, phép toán mô phỏng lại cơ chế quét mẫu cục bộ ấy — tích chập 2D đa kênh — đã trở thành ngôn ngữ nền tảng của toàn bộ thị giác máy tính.

## Kiến thức cần nhớ: Bốn trụ cột toán học của Tầng Tích chập

1. **Tích chập (Convolution) vs. Tương quan chéo (Cross-Correlation)**:
   Xử lý tín hiệu lật ngược bộ lọc 180° trước khi trượt để phép toán có tính giao hoán; Deep Learning bỏ bước lật vì trọng số do mạng tự học nên lật hay không cho cùng không gian hàm.
2. **Tích chập đa kênh (Multi-Channel Convolution)**:
   Với ảnh RGB ($$C_{\text{in}} = 3$$), mỗi bộ lọc thực chất là một khối 3D kích thước $$K_h \times K_w \times C_{\text{in}}$$ cộng gộp kết quả trên cả 3 kênh để sinh ra **một** bản đồ đặc trưng 2D; muốn có $$C_{\text{out}}$$ kênh đầu ra thì cần $$C_{\text{out}}$$ khối lọc độc lập.
3. **Hình học đầu ra với Padding ($$P$$) và Stride ($$S$$)**:
   Padding giữ nguyên thông tin biên ảnh và kiểm soát kích thước; Stride lớn hơn 1 giúp giảm độ phân giải ngay trong phép tích chập mà không cần thêm tầng gộp.
4. **Ngân sách tham số và độ phức tạp tính toán**:
   Số tham số độc lập với độ phân giải ảnh $$(H, W)$$, nhưng số phép tính FLOPs tỷ lệ thuận với $$H_{\text{out}} \cdot W_{\text{out}}$$ — đây là lý do ảnh độ phân giải cao tiêu tốn bộ nhớ kích hoạt rất lớn dù mô hình ít tham số.

### Giải thích: Tính tay tương quan chéo 2D với Padding và Stride

Xét ma trận đầu vào $$I$$ kích thước $$4 \times 4$$ và bộ lọc phát hiện cạnh dọc $$K$$ kích thước $$2 \times 2$$ với $$P = 0, S = 2$$:

$$I = \begin{bmatrix} 1 & 3 & 0 & 2 \\ 2 & 4 & 1 & 0 \\ 0 & 1 & 5 & 2 \\ 3 & 2 & 1 & 4 \end{bmatrix}, \qquad K = \begin{bmatrix} -1 & 1 \\ -1 & 1 \end{bmatrix}$$

1. **Kiểm tra kích thước đầu ra**: $$H_{\text{out}} = \lfloor (4 + 0 - 2)/2 \rfloor + 1 = 2$$, nên đầu ra $$O$$ có kích thước $$2 \times 2$$.
2. **Tính từng phần tử của $$O$$**:
   - $$O[0,0]$$ (cửa sổ góc trên trái): $$(-1)(1) + (1)(3) + (-1)(2) + (1)(4) = -1 + 3 - 2 + 4 = 4$$.
   - $$O[0,1]$$ (dịch phải $$S=2$$ cột): $$(-1)(0) + (1)(2) + (-1)(1) + (1)(0) = 0 + 2 - 1 + 0 = 1$$.
   - $$O[1,0]$$ (dịch xuống $$S=2$$ hàng): $$(-1)(0) + (1)(1) + (-1)(3) + (1)(2) = 0 + 1 - 3 + 2 = 0$$.
   - $$O[1,1]$$ (góc dưới phải): $$(-1)(5) + (1)(2) + (-1)(1) + (1)(4) = -5 + 2 - 1 + 4 = 0$$.

Kết quả $$O = \begin{bmatrix} 4 & 1 \\ 0 & 0 \end{bmatrix}$$ cho thấy phản hồi mạnh nhất ($$4$$) nằm đúng ở góc trên trái — nơi giá trị pixel nhảy vọt từ cột trái (1, 2) sang cột phải (3, 4).

# Tầng Tích chập: Khối Xây dựng của Thị giác Máy tính

![CNN](/deep-learning-self-learning/img/chapter_img/chapter04/conv1.jpg)
*Tổng quan kiến trúc Mạng Neuron Tích chập (CNN). Nguồn: AnalyticsVidhya*

## 1. Tổng quan khái niệm

**Tầng tích chập** (*convolutional layers*) là thành phần chuyên dụng của mạng neuron dùng để xử lý dữ liệu dạng lưới, đặc biệt là ảnh. Thay vì nối mọi đầu vào với mọi neuron như tầng liên kết đầy đủ, tầng tích chập sử dụng các bộ lọc nhỏ có thể học được, trượt qua toàn bộ đầu vào để phát hiện các mẫu cục bộ.

**Các ưu điểm chính của CNN:**
- **Hiệu quả tham số:** Giảm hàng triệu tham số so với mạng liên kết đầy đủ.
- **Tính bất biến tịnh tiến:** Có khả năng phát hiện cùng một đặc trưng ở bất kỳ vị trí nào trên ảnh.
- **Học đặc trưng phân cấp:** Tự động trích xuất dần từ đặc trưng mức thấp đến mức trung bình và mức cao.
- **Hiệu quả thực tiễn cao:** Đạt kết quả tốt trên nhiều bài toán thị giác máy tính.

**Ý tưởng cốt lõi:** Ảnh có cấu trúc không gian chặt chẽ — các điểm ảnh lân cận thường liên hệ mật thiết với nhau. Tầng tích chập khai thác tính chất này thông qua **kết nối cục bộ** và **chia sẻ tham số**.

**Trực giác:** Có thể hình dung phép tích chập giống như việc trượt một kính lúp (bộ lọc) khắp bức ảnh để tìm một mẫu cụ thể (như cạnh, kết cấu hay hình dạng). Mỗi bộ lọc chuyên phát hiện một loại đặc trưng, và ta dùng cùng một bộ lọc đó cho mọi vị trí thay vì học riêng từng bộ trọng số cho từng vùng ảnh.

![CNN](/deep-learning-self-learning/img/chapter_img/chapter04/conv2.jpg)
*Tổng quan kiến trúc Mạng Neuron Tích chập (CNN). Nguồn: AnalyticsVidhya*

### Cảm hứng Sinh học

![CNN Inspiration from Human Eye](/deep-learning-self-learning/img/chapter_img/chapter04/conv3.jpg)
*CNN lấy cảm hứng từ cách hệ thống thị giác của não người xử lý hình ảnh: phát hiện cạnh trước, ghép thành hình dạng rồi nhận diện đối tượng. Nguồn: Analytics Vidhya*

Năm 1959, hai nhà thần kinh sinh lý học David Hubel và Torsten Wiesel phát hiện rằng các neuron trong vỏ não thị giác chỉ phản ứng với những mẫu nhất định nằm trong một vùng cục bộ của trường nhìn — gọi là "trường tiếp nhận" (*receptive field*). Một số neuron nhạy với cạnh theo hướng cụ thể, trong khi các lớp sâu hơn kết hợp chúng để nhận diện cấu trúc phức tạp hơn. Cơ chế xử lý phân cấp này đã trực tiếp gợi ý cho thiết kế của mạng neuron tích chập.

---

## 2. Nền tảng toán học

### Phân biệt Tích chập và Tương quan chéo

Đây là điểm khác biệt về thuật ngữ và định nghĩa mà người mới học thường dễ nhầm lẫn.

![Convolution Operation](https://miro.medium.com/v2/resize:fit:1400/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif)
*Minh họa phép tích chập 2D trên ảnh. Nguồn: Medium*

![The Convolution Filter Sliding](/deep-learning-self-learning/img/chapter_img/chapter04/conv4.jpg)
*Một cửa sổ nhỏ gọi là bộ lọc trượt qua từng vùng của ảnh để kiểm tra sự hiện diện của một đặc trưng nhất định (cạnh, làm mờ, làm nét). Trong quá trình huấn luyện, CNN tự học hàng trăm bộ lọc như vậy. Nguồn: Analytics Vidhya*

#### Tích chập Đúng nghĩa (Định nghĩa trong Xử lý Tín hiệu)

Trong toán học và xử lý tín hiệu, **tích chập** (*convolution*) bao gồm bước **lật nhân** (xoay 180°) trước khi trượt trên đầu vào. Với ảnh 2D $$I$$ và nhân $$K$$:

$$(I * K)[i,j] = \sum_{m}\sum_{n} I[m, n] \cdot K[i-m, j-n]$$

Khi viết theo chỉ số của nhân kích thước $$k \times k$$:

$$(I * K)[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i-m, j-n] \cdot K[m,n]$$

Đặc trưng nhận diện là dấu **trừ** ở chỉ số ($$i-m, j-n$$), tương đương với thao tác lật nhân theo cả hai trục.

**Vì sao xử lý tín hiệu cần lật nhân?** Việc lật nhân giúp phép tích chập thỏa mãn hai tính chất đại số quan trọng cho hệ thống tuyến tính bất biến theo thời gian (LTI):
1. **Tính giao hoán:** $$I * K = K * I$$
2. **Tính kết hợp:** $$(I * K_1) * K_2 = I * (K_1 * K_2)$$

#### Tương quan chéo (Phép toán Học sâu Thực tế Sử dụng)

**Tương quan chéo** (*cross-correlation*) có cấu trúc tương tự nhưng **không lật nhân**:

$$S[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n] \cdot K[m,n]$$

Chú ý dấu **cộng** trong chỉ số ($$i+m, j+n$$) — ta trượt trực tiếp nhân nguyên trạng qua ảnh.

#### Ví dụ So sánh: Tích chập và Tương quan chéo

Xét nhân $$K$$:
$$K = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

Nhân sau khi lật 180° (dùng cho tích chập đúng nghĩa):
$$K_{flipped} = \begin{bmatrix} 9 & 8 & 7 \\ 6 & 5 & 4 \\ 3 & 2 & 1 \end{bmatrix}$$

Với một bộ dò cạnh đứng:
$$K = \begin{bmatrix} 1 & 0 & -1 \\ 1 & 0 & -1 \\ 1 & 0 & -1 \end{bmatrix}$$

Khi lật nhân, dấu hai cột đổi chỗ cho nhau:
$$K_{flipped} = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$$

Như vậy, bản đã lật sẽ phát hiện sự chuyển sáng–tối theo **chiều ngược lại**.

#### Vì sao Học sâu Dùng Tương quan chéo nhưng Vẫn Gọi là "Tích chập"

Điểm mấu chốt là **trong học sâu, trọng số của nhân được học tự động từ dữ liệu**:

1. **Nếu cài đặt tích chập có lật:** Mạng học ma trận trọng số $$W$$ rồi lật nó khi tính lượt xuôi.
2. **Nếu cài đặt tương quan chéo:** Mạng học trực tiếp ma trận trọng số $$W'$$ mà không cần bước lật.

Vì $$W'$$ chỉ đơn giản học thành phiên bản lật của $$W$$, **khả năng biểu diễn của mạng là hoàn toàn tương đương**. Dùng tương quan chéo vừa gọn hơn khi cài đặt, vừa trực quan hơn khi quan sát bộ lọc vì hình dạng của nhân khớp trực tiếp với mẫu mà nó phát hiện.

Trong cộng đồng học sâu, thuật ngữ "tích chập" đã trở thành cách gọi quen thuộc dù đa số thư viện cài đặt phép tương quan chéo. Nắm rõ sự khác biệt này vẫn hữu ích khi đọc tài liệu xử lý tín hiệu hoặc tự viết phép toán tùy chỉnh.

### Ví dụ Tính Tương quan chéo Từng bước

Xét ảnh đầu vào $$I$$ kích thước 5×5:
$$I = \begin{bmatrix} 
1 & 2 & 3 & 0 & 1 \\
0 & 1 & 2 & 3 & 1 \\
1 & 2 & 1 & 0 & 0 \\
0 & 1 & 2 & 3 & 2 \\
2 & 1 & 0 & 1 & 1
\end{bmatrix}$$

và nhân $$K$$ kích thước 3×3:
$$K = \begin{bmatrix} 
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}$$

**Tính đầu ra tại vị trí (0,0):**

Trích vùng 3×3 bắt đầu từ góc (0,0):
$$\text{region} = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

Nhân từng phần tử tương ứng rồi cộng tổng:
$$\begin{align}
S[0,0] &= (1 \times 1) + (2 \times 0) + (3 \times -1) \\
&+ (0 \times 1) + (1 \times 0) + (2 \times -1) \\
&+ (1 \times 1) + (2 \times 0) + (1 \times -1) \\
&= 1 + 0 - 3 + 0 + 0 - 2 + 1 + 0 - 1 \\
&= -4
\end{align}$$

**Tính đầu ra tại vị trí (0,1):**

Trích vùng 3×3 bắt đầu từ (0,1):
$$\text{region} = \begin{bmatrix} 2 & 3 & 0 \\ 1 & 2 & 3 \\ 2 & 1 & 0 \end{bmatrix}$$

$$\begin{align}
S[0,1] &= (2 \times 1) + (3 \times 0) + (0 \times -1) \\
&+ (1 \times 1) + (2 \times 0) + (3 \times -1) \\
&+ (2 \times 1) + (1 \times 0) + (0 \times -1) \\
&= 2 + 0 + 0 + 1 + 0 - 3 + 2 + 0 + 0 \\
&= 2
\end{align}$$

Lặp lại phép tính này trên mọi vị trí hợp lệ sẽ thu được toàn bộ bản đồ đặc trưng đầu ra.

### Tích chập Đa kênh

Ảnh màu thực tế có nhiều kênh (ảnh RGB có 3 kênh; các tầng trung gian trong CNN có thể có 64, 128, 256 kênh hoặc nhiều hơn). Khi đó, phép tích chập được mở rộng như sau:

**Với đầu vào có $$C_{in}$$ kênh** (kích thước $$H \times W \times C_{in}$$) và **một bộ lọc kích thước $$k \times k \times C_{in}$$**:

$$S[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} I[i+m, j+n, c] \cdot K[m,n,c] + b$$

Trong đó:
- $$C_{in}$$: số kênh đầu vào (bằng 3 với ảnh RGB)
- $$b$$: hệ số độ lệch (một số vô hướng cho mỗi bộ lọc)
- Mỗi bộ lọc tạo ra **một** kênh đầu ra

**Để tạo $$C_{out}$$ kênh đầu ra**, tầng tích chập sử dụng $$C_{out}$$ bộ lọc độc lập, mỗi bộ lọc có kích thước $$k \times k \times C_{in}$$.

**Trực giác:** Với ảnh RGB, mỗi bộ lọc là một khối 3D (ví dụ 3×3×3) có trọng số riêng cho từng kênh màu rồi cộng gộp kết quả lại thành một số vô hướng tại mỗi vị trí. Nhờ đó bộ lọc có thể học các đặc trưng kết hợp thông tin màu sắc (ví dụ phân biệt màu xanh da trời và màu xanh của cỏ).

### Công thức Kích thước Đầu ra

Nắm chắc cách thay đổi kích thước không gian qua từng tầng là điều cần thiết khi thiết kế CNN.

**Các tham số:**
- Kích thước đầu vào: $$H_{in} \times W_{in}$$
- Kích thước nhân: $$k \times k$$
- Bước nhảy (*stride*): $$s$$ (số điểm ảnh dịch chuyển giữa hai vị trí liên tiếp)
- Đệm (*padding*): $$p$$ (số hàng/cột giá trị 0 thêm vào mỗi cạnh)

**Kích thước đầu ra:**

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W_{in} + 2p - k}{s} \right\rfloor + 1$$

**Các cấu hình thường gặp:**

| Cấu hình | Mục đích | Kết quả |
|--------|---------|----------------|
| $$k=3, s=1, p=1$$ | Đệm "same" (giữ nguyên kích thước) | $$n_{out} = n_{in}$$ |
| $$k=3, s=2, p=1$$ | Giảm kích thước một nửa | $$n_{out} = \lceil n_{in}/2 \rceil$$ |
| $$k=1, s=1, p=0$$ | Tích chập 1×1 | $$n_{out} = n_{in}$$ |
| $$k=7, s=2, p=3$$ | Giảm kích thước ở tầng đầu | $$n_{out} = \lceil n_{in}/2 \rceil$$ |

### Phân tích Số Tham số

Tổng số tham số của một tầng tích chập:

$$\text{Parameters} = (k \times k \times C_{in}) \times C_{out} + C_{out}$$

Số hạng thứ nhất là số trọng số ($$C_{out}$$ bộ lọc, mỗi bộ gồm $$k \times k \times C_{in}$$ trọng số), và số hạng thứ hai là số hệ số độ lệch (một cho mỗi kênh đầu ra).

**Ví dụ minh họa:**
- Đầu vào: $$32 \times 32 \times 3$$ (ảnh RGB CIFAR-10)
- Tầng tích chập: 64 bộ lọc kích thước $$3 \times 3$$
- Số tham số: $$(3 \times 3 \times 3) \times 64 + 64 = 27 \times 64 + 64 = 1,792$$

**So sánh với tầng liên kết đầy đủ:**
- Đầu vào: $$32 \times 32 \times 3 = 3,072$$ giá trị
- Đầu ra: 64 neuron
- Số tham số: $$3,072 \times 64 + 64 = 196,672$$

**Mức giảm:** $$196,672 / 1,792 \approx 110$$ lần ít tham số hơn.

Hiệu quả vượt trội về tham số này bắt nguồn trực tiếp từ hai cơ chế:
1. **Kết nối cục bộ:** Mỗi phần tử đầu ra chỉ phụ thuộc vào một vùng nhỏ của đầu vào.
2. **Chia sẻ tham số:** Cùng một bộ trọng số được áp dụng trên toàn bộ không gian ảnh.

---

## Video tham khảo

- **Xem trước:** [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown, tiếng Anh. Giải thích bản chất của tổng định nghĩa tích chập.
- [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, tiếng Anh. Trình bày công thức tính chiều cao và chiều rộng đầu ra theo kích thước nhân, padding và stride.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Ví dụ số trên ảnh nhỏ giúp tự kiểm chứng công thức kích thước bằng tay.
