---
layout: post
title: 00-01-01 Tính liên tục và Tính liên tục đều
chapter: '00'
order: 3
owner: GitHub Copilot
lang: vi
categories:
- chapter00
lesson_type: required
---

Bài học này giới thiệu các khái niệm cơ bản về tính liên tục và tính liên tục đều, những khái niệm quan trọng để hiểu hành vi của các hàm số trong tối ưu hóa.

---

## Tính liên tục và Tính liên tục đều

**Tính liên tục** và **Tính liên tục đều** là những khái niệm cơ bản mô tả hành vi của các hàm số, đặc biệt liên quan đến tính "mượt mà" hoặc "có thể dự đoán được" của chúng. Mặc dù có liên quan chặt chẽ, chúng thể hiện các tính chất khác biệt, với tính liên tục đều là điều kiện mạnh hơn so với tính liên tục thông thường.

### Định nghĩa Tính liên tục

Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục tại một điểm** $$c \in A$$ nếu, với mọi số thực dương $$\varepsilon > 0$$, tồn tại một số thực dương $$\delta > 0$$ sao cho với mọi $$x \in A$$, nếu 

$$\lvert x - c \rvert < \delta$$

thì 

$$\lvert f(x) - f(c) \rvert < \varepsilon$$ 

Một cách trực quan, điều này có nghĩa là với bất kỳ mức độ chính xác mong muốn $$\varepsilon$$ nào trong đầu ra $$f(x)$$, chúng ta có thể tìm được một khoảng đủ nhỏ xung quanh $$c$$ (có độ rộng $$2\delta$$) sao cho tất cả các giá trị $$x$$ trong khoảng này ánh xạ tới các giá trị $$f(x)$$ nằm trong khoảng $$\varepsilon$$ xung quanh $$f(c)$$. Khía cạnh quan trọng ở đây là việc chọn $$\delta$$ thường phụ thuộc không chỉ vào $$\varepsilon$$ mà còn vào điểm cụ thể $$c$$.

Một hàm số **liên tục trên một tập hợp** $$A$$ nếu nó liên tục tại mọi điểm $$c \in A$$.

**Ví dụ về các hàm liên tục** bao gồm tất cả đa thức (ví dụ: $$f(x) = x^2 + 3x - 1$$), các hàm lượng giác như $$\sin(x)$$ và $$\cos(x)$$, và các hàm mũ $$e^x$$ trên miền xác định tương ứng của chúng. 

### Định nghĩa Tính liên tục đều

**Tính liên tục đều**, mặt khác, áp đặt một điều kiện nghiêm ngặt hơn. Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục đều trên một tập hợp** $$A$$ nếu, với mọi số thực dương $$\varepsilon > 0$$, tồn tại một số thực dương $$\delta > 0$$ sao cho với mọi $$x, y \in A$$, nếu 

$$\lvert x - y \rvert < \delta$$

thì 

$$\lvert f(x) - f(y) \rvert < \varepsilon$$

**Sự khác biệt chính** so với tính liên tục theo điểm nằm ở thứ tự của các lượng từ: đối với tính liên tục **đều**, $$\delta$$ chỉ phụ thuộc vào $$\varepsilon$$ và độc lập với các điểm cụ thể $$x$$ và $$y$$ trong miền xác định.

### Định nghĩa Tính liên tục Lipschitz

**Tính liên tục Lipschitz** cung cấp một khái niệm cụ thể và định lượng hơn về tính liên tục. Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục Lipschitz** (hoặc **L-Lipschitz**) trên một tập hợp $$A$$ nếu tồn tại một hằng số $$L \geq 0$$ sao cho với mọi $$x, y \in A$$:

$$\lvert f(x) - f(y) \rvert \leq L \lvert x - y \rvert$$

Hằng số nhỏ nhất $$L$$ như vậy được gọi là **hằng số Lipschitz** hoặc **mô-đun Lipschitz** của $$f$$.

**Các tính chất chính của Tính liên tục Lipschitz:**

1. **Tốc độ thay đổi bị chặn**: Điều kiện Lipschitz đảm bảo rằng hàm số không thể thay đổi nhanh hơn một tốc độ tuyến tính được xác định bởi $$L$$.

2. **Tính liên tục đều**: Mọi hàm liên tục Lipschitz đều là liên tục đều (chọn $$\delta = \varepsilon/L$$ với $$L > 0$$).

3. **Khả vi hầu khắp nơi**: Các hàm liên tục Lipschitz khả vi hầu khắp nơi, và tại nơi đạo hàm tồn tại, $$\lvert f'(x) \rvert \leq L$$.

**Ví dụ:**
- $$f(x) = \lvert x \rvert$$ là 1-Lipschitz trên $$\mathbb{R}$$
- $$f(x) = \sin(x)$$ là 1-Lipschitz trên $$\mathbb{R}$$ (vì $$\lvert \cos(x) \rvert \leq 1$$)
- $$f(x) = x^2$$ không phải Lipschitz trên $$\mathbb{R}$$ nhưng là Lipschitz trên bất kỳ khoảng bị chặn nào

### Sự khác biệt chính và Thứ bậc

Ba loại tính liên tục tạo thành một thứ bậc của các điều kiện ngày càng mạnh:

**Tính liên tục ⊆ Tính liên tục đều ⊆ Tính liên tục Lipschitz**

1. **Theo điểm so với Toàn cục**: 
   - **Tính liên tục**: Tính chất cục bộ (kiểm tra tại mỗi điểm)
   - **Tính liên tục đều**: Tính chất toàn cục của toàn bộ hàm số
   - **Tính liên tục Lipschitz**: Tính chất toàn cục với các ràng buộc định lượng

2. **Lựa chọn $$\delta$$**: 
   - **Tính liên tục**: $$\delta$$ có thể phụ thuộc vào cả $$\varepsilon$$ và điểm cụ thể $$c$$
   - **Tính liên tục đều**: $$\delta$$ chỉ phụ thuộc vào $$\varepsilon$$, hoạt động cho tất cả các điểm đồng thời
   - **Tính liên tục Lipschitz**: $$\delta = \varepsilon/L$$ cung cấp mối quan hệ rõ ràng

3. **Kiểm soát Tốc độ Thay đổi**:
   - **Tính liên tục**: Không kiểm soát tốc độ thay đổi
   - **Tính liên tục đều**: Đảm bảo biến thiên bị chặn trên các khoảng nhỏ
   - **Tính liên tục Lipschitz**: Cung cấp ràng buộc tuyến tính rõ ràng cho tốc độ thay đổi

4. **Mối quan hệ Độ mạnh**: 
   - Mọi hàm liên tục Lipschitz đều là liên tục đều
   - Mọi hàm liên tục đều đều là liên tục
   - Các mệnh đề ngược lại nói chung không đúng

### Ví dụ Chi tiết và So sánh

**Ví dụ 1: $$f(x) = x^2$$**
- **Trên $$\mathbb{R}$$**: Liên tục nhưng không liên tục đều (tốc độ thay đổi $$\lvert f'(x) \rvert = 2\lvert x \rvert$$ không bị chặn)
- **Trên $$[0,1]$$**: Liên tục, liên tục đều, và Lipschitz với $$L = 2$$

**Ví dụ 2: $$f(x) = \sin(x)$$**
- **Trên $$\mathbb{R}$$**: Liên tục, liên tục đều, và 1-Lipschitz (vì $$\lvert \cos(x) \rvert \leq 1$$)

**Ví dụ 3: $$f(x) = \lvert x \rvert$$**
- **Trên $$\mathbb{R}$$**: Liên tục, liên tục đều, và 1-Lipschitz
- **Lưu ý**: Không khả vi tại $$x = 0$$, nhưng vẫn là Lipschitz

**Ví dụ 4: $$f(x) = \sqrt{x}$$**
- **Trên $$[0,1]$$**: Liên tục và liên tục đều, nhưng không phải Lipschitz (đạo hàm không bị chặn gần $$x = 0$$)
- **Trên $$[a,1]$$ với $$a > 0$$**: Lipschitz với $$L = 1/(2\sqrt{a})$$
