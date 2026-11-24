---
layout: post
title: 24-01 Diễn giải và Giải thích Mạng Neuron
chapter: '24'
order: 2
owner: Deep Learning Course
lang: vi
categories:
- chapter24
---

Đặt niềm tin mù quáng vào một con số độ chính xác 98% giống như lái máy bay mà không có bảng đồng hồ đo bên trong buồng lái: mọi thứ có thể êm đẹp trong điều kiện chuẩn, nhưng ngay khi gặp nhiễu lạ ngoài thực tế, bạn sẽ không biết mô hình hỏng ở đâu. Mục tổng quan nền tảng này kết nối chặt chẽ phần lý thuyết quy kết (**24-01-01**) với bài cài đặt thực chiến (**24-01-02**).

## Bản đồ kiến thức nền: Hệ quy chiếu phân loại các phương pháp Diễn giải Mô hình

- **Giải thích nội tại (Intrinsic) vs. Hậu kiểm (Post-hoc):** Phân biệt mô hình tự dễ hiểu (cây quyết định nông, hồi quy tuyến tính) với việc giải thích mạng sâu sau khi huấn luyện (Grad-CAM, SHAP).
- **Cục bộ (Local) vs. Toàn cục (Global):** Trả lời "vì sao ảnh này bị gán nhãn mèo?" (local) khác với "mô hình học quy luật tổng quát nào trên toàn tập dữ liệu?" (global).
- **Phụ thuộc mô hình (Model-Specific) vs. Độc lập mô hình (Model-Agnostic):** Các phương pháp gradient tận dụng cấu trúc khả vi của mạng nơ-ron cực nhanh, còn LIME/KernelSHAP chạy được với mọi hộp đen nhưng tốn nhiều lượt truy vấn.
- **Tiêu chuẩn chọn công cụ theo miền dữ liệu:** Grad-CAM/IG cho ảnh, SHAP cho dữ liệu bảng tài chính/y tế, và phân tích mạch/SAE cho mô hình ngôn ngữ.

### Khung giải thích (nên đọc chậm): So sánh chi phí tính toán: Grad-CAM vs. KernelSHAP trên một bức ảnh $$224 \times 224$$

Giả sử một lượt suy luận (forward) qua ResNet-50 mất $$5\text{ ms}$$ và một lượt lan truyền ngược (backward) mất $$10\text{ ms}$$.

1. **Grad-CAM:** Chỉ cần đúng **1 lượt forward + 1 lượt backward** → tổng thời gian $$= 5 + 10 = 15\text{ ms}$$ (chạy được theo thời gian thực trên luồng video 60 fps!).
2. **Integrated Gradients (50 bước tích phân):** Cần $$50$$ cặp forward/backward → $$50 \times 15 = 750\text{ ms}$$.
3. **LIME / KernelSHAP (chia ảnh thành 100 siêu điểm ảnh và lấy $$1{,}000$$ mẫu nhiễu):** Cần $$1{,}000$$ lượt forward → $$1{,}000 \times 5 = 5{,}000\text{ ms} = 5{,}0\text{ giây/ảnh}$$.
   Con số này giải thích trực quan vì sao kỹ sư thường dùng Grad-CAM cho chẩn đoán nhanh ảnh/video, và dành SHAP/LIME cho kiểm toán chuyên sâu dữ liệu bảng hoặc các ca quan trọng.

Chủ đề diễn giải mô hình (*model interpretability*) được chia thành hai phần:

1. **Lý thuyết** — bản đồ nổi bật (*saliency*), SHAP, LRP và trực giác hoạt động.
2. **Cài đặt** — mã minh họa Grad-CAM và SHAP, các bài báo nền tảng và những lưu ý thường gặp.

## Video tham khảo

- **Xem trước** [Lecture 12 | Visualizing and Understanding](https://www.youtube.com/watch?v=6wcs6szJWMY) — Stanford University School of Engineering, tiếng Anh. Phân biệt giữa việc quan sát bên trong mạng và việc giải thích một dự đoán cụ thể.
- [SHAP values for beginners | What they mean and their applications](https://www.youtube.com/watch?v=MQ6fFDwjuco) — A Data Odyssey, tiếng Anh. Minh họa cách dùng giá trị SHAP để giải thích từng dự đoán riêng lẻ.
- [Explainable AI explained! | #3 LIME](https://www.youtube.com/watch?v=d6j6bofhj2M) — DeepFindr, tiếng Anh. Giới thiệu LIME như một cách tiếp cận giải thích cục bộ bổ sung cho SHAP.
