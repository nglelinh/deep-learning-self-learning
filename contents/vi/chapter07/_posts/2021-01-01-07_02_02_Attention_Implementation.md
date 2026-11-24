---
layout: post
title: 07-02-02 Cài đặt Attention
chapter: '07'
order: 5
owner: Deep Learning Course
lang: vi
categories:
- chapter07
---

Trong thực tế lập trình Deep Learning, 90% lỗi liên quan đến Attention không nằm ở chỗ viết sai công thức toán, mà nằm ở ba chỗ âm thầm: **nhầm chiều `transpose` khi tách head**, **áp sai chiều broadcast của mask**, hoặc **quên chia cho $$\sqrt{d_k}$$**. Danh mục kiểm tra và lộ trình cài đặt ở mục này giúp bạn viết code Attention sạch, dễ kiểm thử và không dính bẫy ngầm.

## Bản đồ kiến thức nền: Ba trụ cột khi triển khai mã nguồn Attention

1. **Từ vòng lặp tường minh (Bahdanau) đến vector hóa theo batch (Scaled Dot-Product)**:
   Viết bản Bahdanau tường minh giúp nhìn thấy rõ bảng gióng hàng giữa từng từ nguồn–đích, sau đó chuyển sang phép nhân tensor 4D `(B, H, L, d_k)` của Multi-Head Attention.
2. **Chiến lược biến đổi hình dạng tensor (`view` + `transpose`)**:
   Tách head đúng chuẩn là `x.view(B, L, H, d_k).transpose(1, 2)` để đưa trục `H` lên trước `L` (thành `(B, H, L, d_k)`), **tuyệt đối không** dùng trực tiếp `x.view(B, H, L, d_k)` vì sẽ xáo trộn dữ liệu giữa các token!
3. **Bộ kiểm thử nhanh (Smoke Tests) bắt buộc**:
   Luôn kiểm tra 3 bất biến: tổng trọng số softmax trên trục cuối bằng `1.0`, các vị trí bị che (causal/padding) có trọng số bằng đúng `0.0`, và thay đổi token tương lai không làm thay đổi đầu ra của token quá khứ.

### Khung giải thích (nên đọc chậm): Minh họa bằng số vì sao `view(B, H, L, d_k)` sai hoàn toàn so với `view(B, L, H, d_k).transpose(1, 2)`

Giả sử $$B=1, \; L=2$$ token, $$d_{\text{model}}=4$$, tách thành $$H=2$$ head (mỗi head $$d_k=2$$). Tensor gốc `x` gồm 2 token:
- Token 0: `[a1, a2, a3, a4]` (trong đó `[a1, a2]` thuộc Head 0; `[a3, a4]` thuộc Head 1)
- Token 1: `[b1, b2, b3, b4]` (trong đó `[b1, b2]` thuộc Head 0; `[b3, b4]` thuộc Head 1)

| Cách biến đổi shape | Dữ liệu mà **Head 0** nhận được | Dữ liệu mà **Head 1** nhận được | Kết luận |
| :--- | :--- | :--- | :--- |
| **ĐÚNG**: `view(1, 2, 2, 2).transpose(1, 2)` | `[[a1, a2], [b1, b2]]` | `[[a3, a4], [b3, b4]]` | Head 0 thấy đủ cả 2 token trên 2 chiều đầu — **Chính xác!** |
| **SAI**: `view(1, 2, 2, 2)` trực tiếp | `[[a1, a2], [a3, a4]]` | `[[b1, b2], [b3, b4]]` | Head 0 chỉ thấy Token 0 bị cắt đôi, Head 1 chỉ thấy Token 1 — **Hỏng hoàn toàn!** |

Lỗi `view` trực tiếp không hề báo exception khi chạy PyTorch, nên tự kiểm tra bằng tay trên mảng 8 phần tử như trên sẽ cứu bạn khỏi hàng giờ debug vô ích!

# 07-02-02 Cài đặt Attention

Phần toán học xác định rõ attention tính những đại lượng nào, còn phần cài đặt đảm bảo phép toán đó được hiện thực hóa chính xác — từ kích thước tensor, mặt nạ, độ ổn định số học cho tới cách dùng API.

## Các bài trong mục này

1. **Cài đặt cốt lõi**  
   Cài đặt attention Bahdanau và scaled self-attention từ đầu, chạy ví dụ minh họa kiểu dịch máy và đối chiếu phiên bản NumPy với PyTorch.

2. **Multi-Head, Papers và Cạm bẫy**  
   Xây dựng module multi-head attention, cài đặt causal mask, phác thảo mô hình seq2seq có attention, điểm qua các bài báo nền tảng và danh mục kiểm tra gỡ lỗi thực tế.

## Danh mục kiểm tra khi cài đặt

| Hạng mục kiểm tra | Lý do quan trọng |
|-------------------|------------------|
| Softmax trên trục **key** | Trọng số phải tạo thành phân phối xác suất trên các vị trí bộ nhớ |
| Chia tỉ lệ cho $$\sqrt{d_k}$$ | Tránh bão hòa softmax khi số chiều lớn |
| Áp mask **trước** softmax bằng số âm lớn / $$-\infty$$ | Đảm bảo trọng số bằng 0 và gradient chính xác |
| Causal mask cho decoder tự hồi quy | Ngăn rò rỉ thông tin từ các token tương lai |
| Padding mask khi huấn luyện theo batch | Tránh phân bổ trọng số vào các token `PAD` |
| Reshape multi-head: `(B,L,H,d)` ↔ `(B,H,L,d)` | Hoán vị sai chiều làm trộn lẫn đặc trưng dù chương trình vẫn chạy không báo lỗi |

Hãy bắt đầu từ bài **Cài đặt cốt lõi**, sau đó tiếp tục sang phần multi-head và các cạm bẫy thường gặp.

## Video tham khảo

- **Xem trước:** [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown (tiếng Anh). Ôn nhanh công thức toán học trước khi chuyển sang phần cài đặt.
- [A Dive Into Multihead Attention, Self-Attention and Cross-Attention](https://www.youtube.com/watch?v=mmzRYGCfTzc) — Machine Learning Studio (tiếng Anh). Sơ đồ phân biệt các module self-attention, cross-attention và multi-head.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy (tiếng Anh). Cài đặt trực tiếp cơ chế attention bằng Python, sát với mục tiêu thực hành của mục này.
