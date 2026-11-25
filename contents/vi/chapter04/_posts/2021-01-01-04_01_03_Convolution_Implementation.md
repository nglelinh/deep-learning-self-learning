---
layout: post
title: 04-01-03 Cài đặt Tích chập
chapter: '04'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter04
---

Một dòng code `nn.Conv2d(3, 64, kernel_size=3, padding=1)` trong PyTorch chạy chỉ trong vài phần nghìn giây, nhưng chính sự tiện lợi ấy lại khiến nhiều lập trình viên gặp khó khăn khi phải debug lỗi lệch kích thước tensor, tối ưu tốc độ suy luận, hay viết lại tầng tích chập trên thiết bị nhúng không có PyTorch. Muốn làm chủ CNN, ta cần nhìn xuyên qua lớp vỏ API bằng cách tự cài đặt từ đầu bằng NumPy rồi mới đối chiếu với PyTorch.

## Kiến thức cần nhớ: Ba lớp kỹ thuật khi triển khai Tầng Tích chập

1. **Tư duy chỉ số mảng và Padding tường minh (NumPy)**:
   Tự dùng `np.pad` và cắt lát `image[i*s : i*s+kh, j*s : j*s+kw]` giúp bạn kiểm soát tuyệt đối từng tọa độ, hiểu rõ vì sao `output_shape` lấy phần nguyên sàn (`//`) khi chia cho stride.
2. **Quy ước chiều tensor (NCHW vs. NHWC)**:
   PyTorch dùng `(Batch, Channels, Height, Width)` — gọi là NCHW — tối ưu cho nhân cuDNN; trong khi ảnh đọc từ NumPy/OpenCV thường là `(Height, Width, Channels)`. Quên chuyển trục là nguyên nhân số 1 gây lỗi im lặng hoặc crash.
3. **Từ tích chập đơn lẻ đến Khối CNN tiêu chuẩn (Conv Block)**:
   Trong thực tế, ít khi dùng `Conv2d` đứng một mình; đơn vị xây dựng chuẩn là cụm `Conv2d(bias=False) -> BatchNorm2d -> ReLU` (hoặc SiLU/GELU) kết hợp với tích chập $$1\times 1$$ để nén hoặc mở rộng số kênh.

### Giải thích: Lần theo chỉ số cắt lát (slicing) trong vòng lặp NumPy

Giả sử ảnh đã pad có kích thước $$5 \times 5$$, nhân $$K$$ kích thước $$3 \times 3$$, bước nhảy $$s = 2$$. Khi đó $$H_{\text{out}} = \lfloor (5 - 3)/2 \rfloor + 1 = 2$$. Với mỗi cặp chỉ số đầu ra $$(i, j) \in \{0, 1\} \times \{0, 1\}$$, vùng ảnh được cắt lát là `padded[i*2 : i*2+3, j*2 : j*2+3]`:

| Chỉ số đầu ra $$(i, j)$$ | Phạm vi hàng `r_start : r_end` | Phạm vi cột `c_start : c_end` | Kích thước lát cắt |
| :---: | :---: | :---: | :---: |
| $$(0, 0)$$ | `0 : 3` (hàng 0, 1, 2) | `0 : 3` (cột 0, 1, 2) | $$3 \times 3$$ |
| $$(0, 1)$$ | `0 : 3` (hàng 0, 1, 2) | `2 : 5` (cột 2, 3, 4) | $$3 \times 3$$ |
| $$(1, 0)$$ | `2 : 5` (hàng 2, 3, 4) | `0 : 3` (cột 0, 1, 2) | $$3 \times 3$$ |
| $$(1, 1)$$ | `2 : 5` (hàng 2, 3, 4) | `2 : 5` (cột 2, 3, 4) | $$3 \times 3$$ |

Chú ý rằng chỉ số kết thúc lớn nhất là $$1\times 2 + 3 = 5$$, vừa khít chiều dài mảng `5`, không bao giờ vượt biên (`IndexError`).

Phần này hướng dẫn cài đặt các tầng tích chập theo hai mức độ:

1. **Cài đặt bằng NumPy** — viết phép tích chập 2D từ đầu kèm các ví dụ minh họa
2. **PyTorch và Khối CNN** — sử dụng `nn.Conv2d`, xây dựng khối CNN chuẩn và chạy một bước huấn luyện mẫu

## Video tham khảo

- **Xem trước:** [PyTorch Tutorial 14 - Convolutional Neural Network (CNN)](https://www.youtube.com/watch?v=pDdP0TFzsoQ) — Patrick Loeber, tiếng Anh. Hướng dẫn sử dụng `nn.Conv2d` tương ứng với phần PyTorch của bài.
- [Lecture 5 | Convolutional Neural Networks](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford University School of Engineering, tiếng Anh. Giải thích các quy ước về kích thước tensor khi gọi tầng tích chập.
- [Neural Networks Part 8: Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — StatQuest with Josh Starmer, tiếng Anh. Minh họa trực tiếp phép trượt và tính tổng mà phiên bản NumPy đang cài đặt.
