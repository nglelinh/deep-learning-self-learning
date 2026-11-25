---
layout: post
title: 00-01-01 Tính liên tục và Tính liên tục đều
chapter: '00'
order: 3
owner: GitHub Copilot
lang: vi
categories:
- chapter00
---

Tưởng tượng bạn đang tinh chỉnh núm vặn âm lượng: chỉ xoay nhẹ 1 milimet mà loa bất ngờ rú lên gấp nghìn lần, bạn sẽ không bao giờ chỉnh được mức nghe vừa ý. Trong Deep Learning, mạng nơ-ron gặp đúng thảm họa đó nếu hàm mất mát hay các tầng biến đổi thiếu **tính liên tục**, **liên tục đều**, và đặc biệt là **tính liên tục Lipschitz**—khái niệm từng cứu sống kiến trúc Wasserstein GAN (WGAN) khỏi cảnh bất ổn định nổi tiếng. Bài học này sẽ bóc tách ba tầng "độ mượt" ngày càng chặt chẽ của hàm số để bạn nhìn rõ chúng bảo vệ quá trình huấn luyện như thế nào.

## Kiến thức cần nhớ: Ba cấp độ mượt của hàm số trong tối ưu hóa

1. **Liên tục tại điểm và trên miền (Pointwise Continuity)**:
   Đảm bảo hàm không bị đứt gãy đột ngột; tuy nhiên bán kính an toàn $$\delta$$ có thể phải thu nhỏ vô hạn khi đi tới vùng hàm dốc đứng.

2. **Liên tục đều (Uniform Continuity)**:
   Áp đặt một bán kính $$\delta$$ dùng chung cho *mọi* điểm trên toàn miền xác định, triệt tiêu tình huống hàm ngày càng dốc vô hạn ở gần biên (như $$1/x$$ gần $$0$$).

3. **Liên tục Lipschitz (Lipschitz Continuity)**:
   Mạnh hơn cả liên tục đều: chặn trên trực tiếp tốc độ biến thiên $$\lvert f(x) - f(y) \rvert \le L \lvert x - y \rvert$$ bằng một hằng số $$L$$ cố định. Trong tối ưu hóa, gradient $$L$$-Lipschitz cho phép chọn bước học an toàn $$\eta \le 1/L$$.

4. **Vai trò trong Deep Learning hiện đại**:
   Các kỹ thuật như Spectral Normalization, Gradient Clipping hay Weight Decay đều xoay quanh mục tiêu kiểm soát hằng số Lipschitz của mạng để chống bùng nổ gradient và tăng khả năng chống nhiễu đối kháng (*adversarial robustness*).

### Giải thích: So sánh tay $$f(x) = x^2$$ trên $$[0, 2]$$ và trên $$\mathbb{R}$$

Hãy kiểm tra trực tiếp bằng số vì sao $$f(x) = x^2$$ là Lipschitz trên đoạn đóng $$[0, 2]$$ nhưng **không** liên tục đều trên toàn trục thực $$\mathbb{R}$$:

1. **Trên đoạn $$[0, 2]$$**: Với mọi $$x, y \in [0, 2]$$, ta có:
   $$\lvert x^2 - y^2 \rvert = \lvert x + y \rvert \cdot \lvert x - y \rvert \le (2 + 2)\lvert x - y \rvert = 4\lvert x - y \rvert.$$
   Vậy $$f(x) = x^2$$ là **Lipschitz với hằng số $$L = 4$$** trên $$[0, 2]$$. Nếu muốn $$\lvert f(x) - f(y) \rvert < 0.04$$, chỉ cần chọn duy nhất $$\delta = 0.01$$ cho mọi cặp điểm.
2. **Trên toàn $$\mathbb{R}$$**: Giả sử thử cố định bước nhỏ $$\delta = 0.01$$ và đặt $$y = x + 0.01$$:
   - Tại $$x = 1$$: $$\lvert f(1.01) - f(1) \rvert = \lvert 1.0201 - 1 \rvert = 0.0201$$.
   - Tại $$x = 100$$: $$\lvert f(100.01) - f(100) \rvert = \lvert 10002.0001 - 10000 \rvert = 2.0001$$.
   - Tại $$x = 10000$$: hiệu vọt lên $$200.0001$$!
3. **Kết luận trực giác**: Dù $$x^2$$ rất mượt ở từng điểm, độ dốc $$2x$$ tăng không giới hạn khi $$x \to \infty$$, khiến một bước học cố định sẽ làm sai số nổ tung ở vùng trọng số lớn—đó là lý do chuẩn hóa dữ liệu và ràng buộc trọng số (như Weight Decay) cực kỳ quan trọng trong Deep Learning.

Bài học này giới thiệu các khái niệm cơ bản về tính liên tục, tính liên tục đều và tính liên tục Lipschitz — những công cụ quan trọng để phân tích hành vi của hàm số trong tối ưu hóa.

---

## Tính liên tục và tính liên tục đều

**Tính liên tục** và **tính liên tục đều** đều mô tả mức độ "mượt" hay "ổn định" trong sự biến thiên của hàm số. Dù liên quan chặt chẽ với nhau, hai khái niệm này mang bản chất khác biệt: tính liên tục đều là điều kiện mạnh hơn so với tính liên tục thông thường.

### Định nghĩa tính liên tục

Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục tại điểm** $$c \in A$$ nếu với mọi số thực dương $$\varepsilon > 0$$, tồn tại số thực dương $$\delta > 0$$ sao cho với mọi $$x \in A$$, hễ

$$\lvert x - c \rvert < \delta$$

thì

$$\lvert f(x) - f(c) \rvert < \varepsilon.$$

Về mặt trực quan, với bất kỳ sai số cho phép $$\varepsilon$$ nào ở đầu ra $$f(x)$$, ta luôn tìm được một lân cận đủ nhỏ quanh $$c$$ (bán kính $$\delta$$) sao cho mọi điểm $$x$$ trong lân cận đó đều có giá trị $$f(x)$$ sai khác $$f(c)$$ nhỏ hơn $$\varepsilon$$. Điểm mấu chốt ở đây là $$\delta$$ nhìn chung phụ thuộc vào cả $$\varepsilon$$ lẫn điểm cụ thể $$c$$.

Một hàm số được gọi là **liên tục trên tập hợp** $$A$$ nếu nó liên tục tại mọi điểm $$c \in A$$.

**Một số ví dụ về hàm liên tục** gồm các đa thức (như $$f(x) = x^2 + 3x - 1$$), các hàm lượng giác $$\sin(x)$$, $$\cos(x)$$ và hàm mũ $$e^x$$ trên miền xác định tương ứng của chúng.

### Định nghĩa tính liên tục đều

Ngược lại, **tính liên tục đều** đặt ra yêu cầu chặt chẽ hơn. Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục đều trên tập hợp** $$A$$ nếu với mọi số thực dương $$\varepsilon > 0$$, tồn tại số thực dương $$\delta > 0$$ sao cho với mọi $$x, y \in A$$, hễ

$$\lvert x - y \rvert < \delta$$

thì

$$\lvert f(x) - f(y) \rvert < \varepsilon.$$

**Điểm khác biệt cốt lõi** so với tính liên tục từng điểm nằm ở thứ tự các lượng từ: với tính liên tục **đều**, giá trị $$\delta$$ chỉ phụ thuộc vào $$\varepsilon$$ và áp dụng đồng thời cho mọi cặp điểm $$x, y$$ trên toàn miền xác định.

### Định nghĩa tính liên tục Lipschitz

**Tính liên tục Lipschitz** đưa ra một điều kiện định lượng cụ thể hơn nữa. Một hàm số $$f: A \to \mathbb{R}$$ được gọi là **liên tục Lipschitz** (hay **$$L$$-Lipschitz**) trên tập hợp $$A$$ nếu tồn tại hằng số $$L \geq 0$$ sao cho với mọi $$x, y \in A$$:

$$\lvert f(x) - f(y) \rvert \leq L \lvert x - y \rvert.$$

Hằng số $$L$$ nhỏ nhất thỏa mãn bất đẳng thức trên được gọi là **hằng số Lipschitz** (hay **mô-đun Lipschitz**) của $$f$$.

**Các tính chất chính của tính liên tục Lipschitz:**

1. **Tốc độ biến thiên bị chặn:** Điều kiện Lipschitz đảm bảo hàm số không thể biến thiên nhanh hơn một tốc độ tuyến tính xác định bởi $$L$$.
2. **Suy ra tính liên tục đều:** Mọi hàm liên tục Lipschitz đều liên tục đều (chỉ cần chọn $$\delta = \varepsilon/L$$ khi $$L > 0$$).
3. **Khả vi hầu khắp nơi:** Hàm liên tục Lipschitz khả vi hầu khắp nơi, và tại mọi điểm có đạo hàm thì $$\lvert f'(x) \rvert \leq L$$.

**Ví dụ:**
- $$f(x) = \lvert x \rvert$$ là hàm 1-Lipschitz trên $$\mathbb{R}$$.
- $$f(x) = \sin(x)$$ là hàm 1-Lipschitz trên $$\mathbb{R}$$ (vì $$\lvert \cos(x) \rvert \leq 1$$).
- $$f(x) = x^2$$ không liên tục Lipschitz trên toàn $$\mathbb{R}$$, nhưng liên tục Lipschitz trên mọi khoảng bị chặn.

### So sánh và quan hệ bao hàm

Ba lớp hàm trên tạo thành một quan hệ bao hàm theo chiều mạnh dần của điều kiện:

**Tính liên tục Lipschitz ⊆ Tính liên tục đều ⊆ Tính liên tục**

1. **Tính chất cục bộ so với toàn cục:**
   - **Tính liên tục:** tính chất cục bộ (xét riêng tại từng điểm).
   - **Tính liên tục đều:** tính chất toàn cục trên cả miền xác định.
   - **Tính liên tục Lipschitz:** tính chất toàn cục kèm chặn định lượng rõ ràng.

2. **Sự phụ thuộc của $$\delta$$:**
   - **Tính liên tục:** $$\delta$$ có thể phụ thuộc vào cả $$\varepsilon$$ và điểm $$c$$.
   - **Tính liên tục đều:** $$\delta$$ chỉ phụ thuộc vào $$\varepsilon$$, dùng chung cho mọi điểm.
   - **Tính liên tục Lipschitz:** có công thức tường minh $$\delta = \varepsilon/L$$.

3. **Kiểm soát tốc độ biến thiên:**
   - **Tính liên tục:** không kiểm soát tốc độ biến thiên.
   - **Tính liên tục đều:** đảm bảo mức biến thiên bị chặn trên các khoảng đủ nhỏ.
   - **Tính liên tục Lipschitz:** áp đặt chặn trên tuyến tính rõ ràng lên tốc độ biến thiên.

4. **Quan hệ suy diễn:**
   - Mọi hàm liên tục Lipschitz đều liên tục đều.
   - Mọi hàm liên tục đều đều liên tục.
   - Các chiều ngược lại nhìn chung không đúng.

### Ví dụ minh họa và so sánh

**Ví dụ 1: $$f(x) = x^2$$**
- **Trên $$\mathbb{R}$$:** liên tục nhưng không liên tục đều (vì tốc độ biến thiên $$\lvert f'(x) \rvert = 2\lvert x \rvert$$ không bị chặn).
- **Trên $$[0,1]$$:** liên tục, liên tục đều và Lipschitz với $$L = 2$$.

**Ví dụ 2: $$f(x) = \sin(x)$$**
- **Trên $$\mathbb{R}$$:** liên tục, liên tục đều và 1-Lipschitz (vì $$\lvert \cos(x) \rvert \leq 1$$).

**Ví dụ 3: $$f(x) = \lvert x \rvert$$**
- **Trên $$\mathbb{R}$$:** liên tục, liên tục đều và 1-Lipschitz.
- **Lưu ý:** hàm không khả vi tại $$x = 0$$ nhưng vẫn thỏa mãn điều kiện Lipschitz.

**Ví dụ 4: $$f(x) = \sqrt{x}$$**
- **Trên $$[0,1]$$:** liên tục và liên tục đều, nhưng không liên tục Lipschitz (đạo hàm tiến ra vô cùng khi $$x \to 0^+$$).
- **Trên $$[a,1]$$ với $$a > 0$$:** liên tục Lipschitz với $$L = 1/(2\sqrt{a})$$.

## Video tham khảo

- **Xem trước:** [Continuity at a point | Limits and continuity | AP Calculus AB | Khan Academy](https://www.youtube.com/watch?v=3KMqU5j7irw) — Khan Academy, tiếng Anh. Giải thích định nghĩa liên tục tại một điểm, là điểm khởi đầu của bài học này.
- [Limits, L'Hôpital's rule, and epsilon delta definitions | Chapter 7, Essence of calculus](https://www.youtube.com/watch?v=kfF40MiS7zA) — 3Blue1Brown, tiếng Anh. Nửa sau video trình bày ngôn ngữ $$\varepsilon\text{-}\delta$$, công cụ dùng để mở rộng từ liên tục từng điểm lên liên tục đều.
- [Real Analysis | Intro to uniform continuity.](https://www.youtube.com/watch?v=gioaYgOQ3AQ) — Michael Penn, tiếng Anh. Tập trung trực tiếp vào tính liên tục đều: một giá trị $$\delta$$ duy nhất dùng chung cho cả tập hợp.
