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

Bài này trình bày hàm mất mát (*loss functions*, còn gọi là hàm chi phí hoặc hàm mục tiêu), dùng để định lượng mức độ tốt của mạng neuron.

---

## Hàm Mất mát là gì?

Một **hàm mất mát** $$\mathcal{L}$$ đo sự chênh lệch giữa đầu ra dự đoán $$\hat{y}$$ và đầu ra thật $$y$$. Mục tiêu của huấn luyện là tìm tham số $$\theta$$ (trọng số và độ lệch) cực tiểu hóa mất mát này.

### Mất mát trên Một Mẫu

Với một mẫu huấn luyện:

$$\mathcal{L}(\hat{y}, y)$$

### Hàm Chi phí (Tổng Mất mát)

Với tập dữ liệu gồm $$m$$ mẫu, **hàm chi phí** $$J$$ thường là mất mát trung bình:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

Một số công thức còn bao gồm các hạng tử chính quy hóa (trình bày sau).

## Hàm Mất mát cho Hồi quy

### 1. Sai số Bình phương Trung bình (MSE)

**Công thức:**

$$\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{2}(y - \hat{y})^2$$

**Hàm chi phí:**

$$J_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2}(y^{(i)} - \hat{y}^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

**Lưu ý**: Hệ số $$\frac{1}{2}$$ được đưa vào vì tiện toán học (đơn giản hóa đạo hàm).

**Tính chất:**
- Luôn không âm
- Phạt nặng sai số lớn (phạt bậc hai)
- Nhạy với điểm ngoại lai (*outliers*)
- Trơn và khả vi mọi nơi

**Đạo hàm:**

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}} = \hat{y} - y$$

**Trường hợp dùng:**
- **Tác vụ hồi quy**: Dự đoán giá trị liên tục
- Khi sai số phân phối chuẩn
- Khi mọi sai số nên được trọng số hóa như nhau

**Ưu điểm:**
- Đơn giản và trực quan
- Gradient trơn
- Được hiểu rõ về lý thuyết

**Nhược điểm:**
- Rất nhạy với điểm ngoại lai (sai số lớn bị phạt nặng)
- Giả định sai số phân phối chuẩn

### 2. Sai số Tuyệt đối Trung bình (MAE)

**Công thức:**

$$\mathcal{L}_{\text{MAE}}(\hat{y}, y) = |y - \hat{y}|$$

**Hàm chi phí:**

$$J_{\text{MAE}} = \frac{1}{m} \sum_{i=1}^{m} |y^{(i)} - \hat{y}^{(i)}|$$

**Tính chất:**
- Phạt tuyến tính cho sai số
- Bền vững hơn MSE với điểm ngoại lai
- Không khả vi tại $$\hat{y} = y$$

**Đạo hàm:**

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{y}} = \begin{cases} 1 & \text{if } \hat{y} > y \\ -1 & \text{if } \hat{y} < y \\ \text{undefined} & \text{if } \hat{y} = y \end{cases}$$

(Trong thực tế, ta dùng dưới-gradient hoặc xấp xỉ trơn)

**Trường hợp dùng:**
- Hồi quy có điểm ngoại lai
- Khi muốn phạt bằng nhau cho mọi độ lớn sai số

**Ưu điểm:**
- Bền vững với điểm ngoại lai
- Diễn giải trực quan (sai số tuyệt đối trung bình)

**Nhược điểm:**
- Không khả vi tại không
- Có thể hội tụ chậm hơn
- Gradient hằng có thể gây vấn đề gần cực tiểu

### 3. Mất mát Huber

**Công thức:**

$$\mathcal{L}_{\text{Huber}}(\hat{y}, y) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

trong đó $$\delta$$ là tham số ngưỡng.

**Tính chất:**
- Kết hợp ưu điểm của MSE và MAE
- Bậc hai với sai số nhỏ, tuyến tính với sai số lớn
- Trơn và khả vi mọi nơi

**Trường hợp dùng:**
- Hồi quy có thể có điểm ngoại lai
- Khi muốn gradient trơn nhưng bền vững với ngoại lai

**Ưu điểm:**
- Ít nhạy với điểm ngoại lai hơn MSE
- Gradient trơn (không như MAE)
- Có thể cấu hình qua $$\delta$$

**Nhược điểm:**
- Cần tinh chỉnh siêu tham số $$\delta$$
