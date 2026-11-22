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
