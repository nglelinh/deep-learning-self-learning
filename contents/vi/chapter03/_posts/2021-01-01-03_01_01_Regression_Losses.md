---
layout: post
title: 03-01-01 Hàm Mất mát cho Hồi quy
chapter: '03'
order: 3
owner: Deep Learning Course
lang: vi
categories:
- chapter03
---

Tưởng tượng bạn xây mô hình dự báo giá căn hộ: 99 căn hộ bình thường có giá quanh 3 tỷ đồng, nhưng lọt vào tập dữ liệu đúng 1 căn biệt thự nhập nhầm giá 300 tỷ đồng. Nếu huấn luyện bằng **MSE**, chỉ duy nhất một điểm ngoại lai đó có thể kéo lệch toàn bộ mô hình vì sai số bị bình phương lên tới hàng chục nghìn lần! Ngược lại, **MAE** miễn nhiễm tốt với ngoại lai nhưng lại có độ dốc không đổi khiến tối ưu dễ rung lắc quanh đáy. Bài học này phân tích bộ ba **MSE, MAE và Huber Loss** để bạn biết chọn thước đo hồi quy tối ưu cho từng đặc thù nhiễu dữ liệu.

## Kiến thức cần nhớ: Bốn góc nhìn cốt lõi về Hàm Mất mát Hồi quy

1. **MSE (Mean Squared Error — $$L_2$$ Loss)**:
   Phạt theo bình phương sai số $$(y - \hat{y})^2$$, có đạo hàm tỷ lệ thuận với độ lớn sai số $$2(\hat{y} - y)$$ nên mượt quanh $$0$$ và tương ứng MLE của nhiễu Gaussian; nghiệm tối ưu hằng số là **trung bình (mean)**.

2. **MAE (Mean Absolute Error — $$L_1$$ Loss)**:
   Phạt tuyến tính theo $$\lvert y - \hat{y} \rvert$$, bền vững trước ngoại lai (*outliers*) với nghiệm tối ưu hằng số là **trung vị (median)**, nhưng không khả vi tại $$0$$ và gradient luôn có độ lớn $$\pm 1$$ kể cả khi rất gần nghiệm.

3. **Huber Loss (Smooth $$L_1$$)**:
   Kết hợp ưu điểm của cả hai: hành xử như parabol (MSE) khi sai số nhỏ $$\lvert e \rvert \le \delta$$ để hội tụ êm quanh đáy, và chuyển sang đường thẳng (MAE) khi $$\lvert e \rvert > \delta$$ để chặn tác động của ngoại lai—tiêu chuẩn vàng trong hồi quy khung bao Object Detection và DQN.

4. **Log-Cosh và Biến đổi Miền (RMSE, RMSLE)**:
   RMSE đưa đơn vị về lại thang đo gốc của $$y$$, còn RMSLE phạt sai số theo tỷ lệ phần trăm thay vì hiệu tuyệt đối—rất hữu ích khi giá trị mục tiêu trải rộng qua nhiều bậc độ lớn.

### Giải thích: Tính tay MSE, MAE và Huber Loss khi có 1 điểm ngoại lai

Xét 4 mẫu dữ liệu với sai số $$e_i = \hat{y}_i - y_i$$ lần lượt là: $$e_1 = 0.5,\; e_2 = -0.5,\; e_3 = 0.2,\; e_4 = 10.0$$ (mẫu 4 là ngoại lai). Ta tính mức phạt và độ lớn đạo hàm trên mẫu ngoại lai $$e_4 = 10$$ với Huber ngưỡng $$\delta = 1.0$$:

| Hàm mất mát | Công thức từng mẫu | Giá trị trên 3 mẫu sạch ($$e_1,e_2,e_3$$) | Giá trị trên ngoại lai ($$e_4=10$$) | Độ lớn gradient $$\lvert \partial L/\partial e_4 \rvert$$ |
| :--- | :--- | :---: | :---: | :---: |
| **MSE** | $$\frac{1}{2}e^2$$ | $$0.125,\; 0.125,\; 0.020$$ | **$$50.00$$** (chiếm 99.5% tổng loss!) | **$$10.0$$** |
| **MAE** | $$\lvert e \rvert$$ | $$0.500,\; 0.500,\; 0.200$$ | $$10.00$$ | **$$1.0$$** |
| **Huber ($$\delta=1$$)** | $$\frac{1}{2}e^2$$ nếu $$\lvert e \rvert \le 1$$; $$\lvert e \rvert - 0.5$$ nếu $$>1$$ | $$0.125,\; 0.125,\; 0.020$$ | $$9.50$$ | **$$1.0$$** |

Nhìn vào cột cuối: tại 3 mẫu sạch, Huber giữ gradient mượt bằng $$e$$ (như MSE), nhưng tại mẫu ngoại lai $$e_4 = 10$$, Huber cắt trần độ lớn gradient đúng bằng $$\delta = 1.0$$—ngăn không cho một điểm dữ liệu lỗi kéo hỏng cả mini-batch!

Bài này giới thiệu các hàm mất mát (*loss functions*, còn gọi là hàm chi phí hoặc hàm mục tiêu) dùng để định lượng mức độ sai lệch của mạng neuron, tập trung vào nhóm bài toán hồi quy.

---

## Hàm Mất mát là gì?

Một **hàm mất mát** $$\mathcal{L}$$ đo lường sự chênh lệch giữa đầu ra dự đoán $$\hat{y}$$ và giá trị thực tế $$y$$. Mục tiêu của quá trình huấn luyện là tìm bộ tham số $$\theta$$ (gồm trọng số và độ lệch) sao cho giá trị mất mát này đạt cực tiểu.

### Mất mát trên Một Mẫu

Với một mẫu huấn luyện đơn lẻ:

$$\mathcal{L}(\hat{y}, y)$$

### Hàm Chi phí (Tổng Mất mát)

Trên tập dữ liệu gồm $$m$$ mẫu, **hàm chi phí** $$J$$ thường được định nghĩa là giá trị mất mát trung bình:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

Trong thực tế, hàm chi phí có thể cộng thêm các hạng tử chính quy hóa (*regularization*) sẽ được trình bày ở các chương sau.

## Hàm Mất mát cho Hồi quy

### 1. Sai số Bình phương Trung bình (MSE)

**Công thức:**

$$\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{2}(y - \hat{y})^2$$

**Hàm chi phí:**

$$J_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2}(y^{(i)} - \hat{y}^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

**Lưu ý:** Hệ số $$\frac{1}{2}$$ được thêm vào để rút gọn hệ số 2 khi lấy đạo hàm.

**Tính chất:**
- Luôn không âm
- Phạt nặng các sai số lớn theo hàm bậc hai
- Nhạy với các điểm ngoại lai (*outliers*)
- Trơn và khả vi tại mọi điểm

**Đạo hàm:**

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}} = \hat{y} - y$$

**Trường hợp sử dụng:**
- **Bài toán hồi quy:** Dự đoán các đại lượng liên tục
- Khi nhiễu sai số có dạng gần phân phối chuẩn
- Khi các sai số có mức độ quan trọng tương đương nhau

**Ưu điểm:**
- Công thức đơn giản và trực quan
- Đạo hàm trơn, dễ tối ưu
- Có nền tảng lý thuyết rõ ràng

**Nhược điểm:**
- Rất nhạy với điểm ngoại lai do sai số lớn bị bình phương
- Ngầm giả định sai số tuân theo phân phối chuẩn

### 2. Sai số Tuyệt đối Trung bình (MAE)

**Công thức:**

$$\mathcal{L}_{\text{MAE}}(\hat{y}, y) = |y - \hat{y}|$$

**Hàm chi phí:**

$$J_{\text{MAE}} = \frac{1}{m} \sum_{i=1}^{m} |y^{(i)} - \hat{y}^{(i)}|$$

**Tính chất:**
- Phạt sai số theo tỷ lệ tuyến tính
- Ít nhạy với điểm ngoại lai hơn MSE
- Không khả vi tại điểm $$\hat{y} = y$$

**Đạo hàm:**

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{y}} = \begin{cases} 1 & \text{if } \hat{y} > y \\ -1 & \text{if } \hat{y} < y \\ \text{undefined} & \text{if } \hat{y} = y \end{cases}$$

(Trong thực tế, ta sử dụng dưới đạo hàm — *subgradient* — hoặc các hàm xấp xỉ trơn.)

**Trường hợp sử dụng:**
- Bài toán hồi quy có nhiều điểm ngoại lai
- Khi muốn phạt sai số tỷ lệ thuận với độ lớn thay vì bình phương

**Ưu điểm:**
- Bền vững trước các điểm ngoại lai
- Dễ diễn giải trực tiếp theo đơn vị của biến mục tiêu

**Nhược điểm:**
- Không khả vi tại $$\hat{y} = y$$
- Có thể hội tụ chậm hơn MSE
- Độ lớn gradient không đổi (luôn bằng 1) có thể khiến tham số dao động quanh cực tiểu

### 3. Mất mát Huber

**Công thức:**

$$\mathcal{L}_{\text{Huber}}(\hat{y}, y) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

trong đó $$\delta$$ là siêu tham số ngưỡng.

**Tính chất:**
- Kết hợp ưu điểm của cả MSE và MAE
- Hành xử bậc hai với sai số nhỏ và tuyến tính với sai số lớn
- Trơn và khả vi tại mọi điểm

**Trường hợp sử dụng:**
- Bài toán hồi quy có khả năng xuất hiện điểm ngoại lai
- Khi cần đồng thời gradient trơn ở gần cực tiểu và tính bền vững trước ngoại lai

**Ưu điểm:**
- Ít chịu ảnh hưởng bởi ngoại lai hơn MSE
- Có gradient trơn, khắc phục nhược điểm không khả vi của MAE
- Linh hoạt điều chỉnh ngưỡng chuyển tiếp qua $$\delta$$

**Nhược điểm:**
- Cần tinh chỉnh thêm siêu tham số $$\delta$$

## Video tham khảo

- **Xem trước:** [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown, tiếng Anh. Minh họa trực quan sai số bình phương giữa dự đoán và nhãn thật — chính là hàm mất mát hồi quy trong bài.
- [Lecture 3 | Loss Functions and Optimization](https://www.youtube.com/watch?v=h7iBpEHGVNc) — Stanford University School of Engineering, tiếng Anh. Đặt MSE trong bức tranh tổng thể cùng các hàm mất mát khác.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — StatQuest with Josh Starmer, tiếng Anh. Minh họa từng bước sự thay đổi của mặt mất mát hồi quy khi cập nhật tham số.
